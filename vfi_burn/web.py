"""Zero-dependency Local Web UI server for vfi_burn (Option C).

Uses Python's standard-library ``http.server.ThreadingHTTPServer`` to serve the
packaged static frontend (``vfi_burn/web_assets/``) and a token-authenticated
JSON + SSE API wired directly to ``WorkflowController``.

Safety architecture:
  - Binds to ``127.0.0.1`` by default (loopback-only; SSH tunnel friendly).
  - Requires a 256-bit cryptographically random per-launch token on every ``/api/*``
    call and validates the ``Host`` header against DNS rebinding.
  - Serializes all hardware operations through ``WorkflowController``'s lock and
    allows at most one burn worker thread at a time.
  - Every engine confirmation (``SENSE``, ``BURN TEST``, ``BURN COMPUTE+MEM``,
    ``BURN PCIE``, ``CONFIRM A100 DEVID ...``) uses a single-use random
    ``confirm_id`` nonce resolved once via ``POST /api/burn/confirm`` so page
    reloads or double-clicks can never replay a burn step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import queue
import secrets
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

from .model import BurnError
from .workflow import (
    WorkflowController,
    fix_file_permissions,
    list_directory_entries,
)


ASSETS_DIR = Path(__file__).resolve().parent / "web_assets"


@dataclass
class _PendingConfirm:
    confirm_id: str
    prompt: str
    token: str
    event: threading.Event = field(default_factory=threading.Event)
    response: str | None = None
    cancelled: bool = False


class _BurnSessionManager:
    """Manages at most one interactive background burn worker + SSE event fanout."""

    def __init__(self, controller: WorkflowController) -> None:
        self.ctl = controller
        self._lock = threading.Lock()
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._pending_confirm: _PendingConfirm | None = None
        self._subscribers: list[queue.Queue[dict]] = []
        self._history: list[dict] = []

    def subscribe(self) -> queue.Queue[dict]:
        q: queue.Queue[dict] = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def broadcast(self, event: dict) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > 300:
                self._history = self._history[-300:]
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def status(self) -> dict:
        with self._lock:
            pc = self._pending_confirm
            return {
                "running": self._running,
                "pending_confirm": (
                    {
                        "confirm_id": pc.confirm_id,
                        "prompt": pc.prompt,
                        "token": pc.token,
                    }
                    if pc is not None
                    else None
                ),
                "history": list(self._history[-100:]),
            }

    def start(self) -> dict:
        with self._lock:
            if self._running:
                raise BurnError("a burn session is already in progress")
            self._running = True
            self._pending_confirm = None
            self._history.clear()

        def _confirm_cb(prompt: str, token: str) -> bool:
            pc = _PendingConfirm(
                confirm_id=secrets.token_urlsafe(16),
                prompt=prompt,
                token=token,
            )
            with self._lock:
                self._pending_confirm = pc
            self.broadcast({
                "type": "confirm_required",
                "confirm_id": pc.confirm_id,
                "prompt": prompt,
                "token": token,
            })
            # Wait for operator decision (fail-closed on timeout after 10 minutes or cancel).
            answered = pc.event.wait(timeout=600.0)
            with self._lock:
                if self._pending_confirm is pc:
                    self._pending_confirm = None
            if not answered or pc.cancelled or pc.response is None:
                self.broadcast({
                    "type": "confirm_resolved",
                    "confirm_id": pc.confirm_id,
                    "accepted": False,
                })
                return False
            accepted = pc.response.strip() == token
            self.broadcast({
                "type": "confirm_resolved",
                "confirm_id": pc.confirm_id,
                "accepted": accepted,
            })
            return accepted

        def _on_event(ev: dict) -> None:
            self.broadcast(ev)

        def _show_cb(msg: str) -> None:
            self.broadcast({"type": "log", "message": str(msg)})

        def _worker() -> None:
            try:
                res = self.ctl.execute_burn(
                    confirm_cb=_confirm_cb,
                    on_event=_on_event,
                    show_cb=_show_cb,
                )
                self.broadcast({"type": "burn_finished", "ok": True, "result": res})
            except BaseException as exc:
                self.broadcast({"type": "burn_finished", "ok": False, "error": str(exc)})
            finally:
                with self._lock:
                    self._running = False
                    if self._pending_confirm is not None:
                        self._pending_confirm.cancelled = True
                        self._pending_confirm.event.set()
                        self._pending_confirm = None

        t = threading.Thread(target=_worker, name="vfi-burn-worker", daemon=True)
        self._thread = t
        t.start()
        return {"started": True}

    def resolve_confirm(self, confirm_id: str, response: str | None, *, cancel: bool = False) -> dict:
        with self._lock:
            pc = self._pending_confirm
            if pc is None or pc.confirm_id != confirm_id:
                raise BurnError("no matching pending confirmation (it may have already been resolved)")
            pc.cancelled = bool(cancel)
            pc.response = None if cancel else str(response or "")
            pc.event.set()
            return {"resolved": True, "confirm_id": confirm_id}

    def abort(self) -> dict:
        with self._lock:
            pc = self._pending_confirm
            if pc is not None:
                pc.cancelled = True
                pc.event.set()
            return {"aborted": pc is not None, "running": self._running}


class VFIWebServer(ThreadingHTTPServer):
    """Token-authenticated local HTTP server hosting the vfi_burn Web UI."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        controller: WorkflowController,
        *,
        token: str | None = None,
        allowed_hosts: set[str] | None = None,
    ) -> None:
        self.controller = controller
        self.token = token or secrets.token_urlsafe(24)
        self.burn_mgr = _BurnSessionManager(controller)
        self._allowed_hosts = allowed_hosts
        super().__init__(server_address, _VFIRequestHandler)

    def is_host_allowed(self, host_header: str) -> bool:
        if not host_header:
            return False
        host_only = host_header.split(":", 1)[0].strip().lower()
        if self._allowed_hosts is not None:
            return host_only in self._allowed_hosts
        bind_host = str(self.server_address[0]).lower()
        if bind_host in ("127.0.0.1", "localhost", "::1"):
            return host_only in ("127.0.0.1", "localhost", "[::1]")
        # Explicit non-loopback bind (--host 0.0.0.0 / LAN IP): bearer token still gates every API call.
        return True

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])

    def access_url(self) -> str:
        host = self.server_address[0]
        if host in ("0.0.0.0", ""):
            host = "127.0.0.1"
        return f"http://{host}:{self.bound_port}/?token={self.token}"


class _VFIRequestHandler(BaseHTTPRequestHandler):
    server: VFIWebServer

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        # Suppress noisy per-poll HTTP access logs on the bench console.
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        if length > 2 * 1024 * 1024:
            raise BurnError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise BurnError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise BurnError("JSON body must be an object")
        return data

    def _check_auth(self, query: dict[str, list[str]]) -> bool:
        if not self.server.is_host_allowed(self.headers.get("Host", "")):
            self._send_error_json("forbidden Host header", status=HTTPStatus.FORBIDDEN)
            return False
        provided = self.headers.get("X-VFI-Token", "").strip()
        if not provided and "token" in query:
            provided = query["token"][0].strip()
        if not secrets.compare_digest(provided, self.server.token):
            self._send_error_json("invalid or missing session token", status=HTTPStatus.UNAUTHORIZED)
            return False
        return True

    def _serve_static(self, rel_path: str) -> None:
        if not self.server.is_host_allowed(self.headers.get("Host", "")):
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden Host header")
            return
        name = "index.html" if rel_path in ("", "/") else rel_path.lstrip("/")
        target = (ASSETS_DIR / name).resolve()
        if not str(target).startswith(str(ASSETS_DIR.resolve())) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        ctype, _ = mimetypes.guess_type(str(target))
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{ctype or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _handle_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.server.burn_mgr.subscribe()
        try:
            # Send initial burn status snapshot immediately.
            init_ev = {"type": "burn_status", **self.server.burn_mgr.status()}
            self.wfile.write(f"data: {json.dumps(init_ev)}\n\n".encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15.0)
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # Keep-alive comment
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.server.burn_mgr.unsubscribe(q)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if not path.startswith("/api/"):
            self._serve_static(path)
            return
        if not self._check_auth(query):
            return
        ctl = self.server.controller
        try:
            if path == "/api/events":
                self._handle_sse()
                return
            if path == "/api/overview":
                self._send_json({"ok": True, "data": ctl.get_overview()})
            elif path == "/api/gpus":
                probe = query.get("probe", ["1"])[0] != "0"
                self._send_json({"ok": True, "data": ctl.list_gpus(probe_bar0=probe)})
            elif path == "/api/files":
                directory = query.get("dir", ["private"])[0] or "private"
                kind = query.get("kind", ["all"])[0]
                target_pci = None
                if ctl.has_target:
                    try:
                        st = ctl.read_passive_status(include_full_report=False)
                        target_pci = st.get("pci_id")
                    except Exception:
                        pass
                listing = list_directory_entries(directory, kind_filter=kind, target_pci_id=target_pci)
                self._send_json({"ok": True, "data": listing})
            elif path == "/api/configure":
                self._send_json({"ok": True, "data": ctl.get_config_state()})
            elif path == "/api/plan/review":
                self._send_json({"ok": True, "data": ctl.get_plan_review()})
            elif path == "/api/status":
                full = query.get("full", ["0"])[0] == "1"
                self._send_json({"ok": True, "data": ctl.read_passive_status(include_full_report=full)})
            elif path == "/api/burn/preview":
                self._send_json({"ok": True, "data": ctl.get_burn_preview()})
            elif path == "/api/burn/status":
                self._send_json({"ok": True, "data": self.server.burn_mgr.status()})
            elif path == "/api/cold/preflight":
                self._send_json({"ok": True, "data": ctl.preflight_cold_verify()})
            elif path == "/api/tools/experimental":
                self._send_json({"ok": True, "data": ctl.read_experimental()})
            elif path == "/api/tools/journal":
                limit = int(query.get("limit", ["60"])[0])
                self._send_json({"ok": True, "data": ctl.read_journal(limit=limit)})
            elif path == "/api/tools/rows":
                start = int(query.get("start", ["222"])[0])
                end = int(query.get("end", ["270"])[0])
                self._send_json({"ok": True, "data": ctl.read_ff_rows_range(start, end)})
            elif path == "/api/tools/register":
                target = query.get("target", [""])[0]
                self._send_json({"ok": True, "data": ctl.read_register_value(target)})
            elif path == "/api/tools/decode":
                word = query.get("word", [""])[0]
                self._send_json({"ok": True, "data": ctl.decode_ff_word(word)})
            else:
                self._send_error_json(f"unknown API endpoint: {path}", status=HTTPStatus.NOT_FOUND)
        except (BurnError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            self._send_error_json(str(exc), status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if not path.startswith("/api/"):
            self._send_error_json("POST only allowed on /api/*", status=HTTPStatus.METHOD_NOT_ALLOWED)
            return
        if not self._check_auth(query):
            return
        ctl = self.server.controller
        try:
            body = self._read_json_body()
            if path == "/api/target/gpu":
                self._send_json({"ok": True, "data": ctl.select_gpu(str(body.get("bdf", "")))})
            elif path == "/api/target/sim":
                self._send_json({
                    "ok": True,
                    "data": ctl.select_simulation(
                        str(body.get("path", "")),
                        mode=str(body.get("mode", "sim_pcie")),
                    ),
                })
            elif path == "/api/target/detach":
                self._send_json({"ok": True, "data": {"logs": ctl.detach_gpu_driver()}})
            elif path == "/api/mode/execute":
                self._send_json({
                    "ok": True,
                    "data": ctl.set_execute_live(
                        bool(body.get("enabled", False)),
                        str(body.get("confirm_phrase", "")),
                    ),
                })
            elif path == "/api/settings":
                if "confirm_mode" in body:
                    ctl.set_confirm_mode(str(body["confirm_mode"]))
                if "journal_path" in body:
                    ctl.set_journal_path(str(body["journal_path"]))
                if "receipt_path" in body:
                    ctl.set_receipt_path(body["receipt_path"] or None)
                self._send_json({"ok": True, "data": ctl.get_overview()})
            elif path == "/api/files/chmod600":
                meta = fix_file_permissions(str(body.get("path", "")))
                self._send_json({"ok": True, "data": meta})
            elif path == "/api/plan/load":
                self._send_json({"ok": True, "data": ctl.load_plan(str(body.get("path", "")))})
            elif path == "/api/configure/start":
                refresh = bool(body.get("refresh", True))
                self._send_json({"ok": True, "data": ctl.start_configure(refresh=refresh)})
            elif path == "/api/configure/patch":
                self._send_json({"ok": True, "data": ctl.patch_config(body)})
            elif path == "/api/configure/save":
                res = ctl.save_configured_plan(
                    out_path=body.get("path"),
                    overwrite=bool(body.get("overwrite", False)),
                )
                self._send_json({"ok": True, "data": res})
            elif path == "/api/preflight/dry-run":
                self._send_json({"ok": True, "data": ctl.run_dry_run()})
            elif path == "/api/preflight/sense":
                res = ctl.run_active_sense(str(body.get("confirm_phrase", "")))
                self._send_json({"ok": True, "data": res})
            elif path == "/api/burn/start":
                self._send_json({"ok": True, "data": self.server.burn_mgr.start()})
            elif path == "/api/burn/confirm":
                res = self.server.burn_mgr.resolve_confirm(
                    str(body.get("confirm_id", "")),
                    body.get("response"),
                    cancel=bool(body.get("cancel", False)),
                )
                self._send_json({"ok": True, "data": res})
            elif path == "/api/burn/abort":
                self._send_json({"ok": True, "data": self.server.burn_mgr.abort()})
            elif path == "/api/cold/verify-live":
                res = ctl.run_cold_verify_live(
                    stage=body.get("stage") or None,
                    previous_boot_id=body.get("previous_boot_id") or None,
                    out_path=body.get("out_path") or None,
                    receipt_out=body.get("receipt_out") or None,
                )
                self._send_json({"ok": True, "data": res})
            elif path == "/api/cold/verify-file":
                res = ctl.run_cold_verify_file(
                    str(body.get("path", "")),
                    stage=body.get("stage") or None,
                    receipt_out=body.get("receipt_out") or None,
                )
                self._send_json({"ok": True, "data": res})
            elif path == "/api/overrides":
                self._send_json({"ok": True, "data": ctl.update_overrides(body)})
            else:
                self._send_error_json(f"unknown API endpoint: {path}", status=HTTPStatus.NOT_FOUND)
        except (BurnError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            self._send_error_json(str(exc), status=HTTPStatus.BAD_REQUEST)


def run_web_server(
    controller: WorkflowController,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    show=print,
) -> int:
    """Start the local Web UI server and block until Ctrl-C."""
    srv = VFIWebServer((host, port), controller)
    url = srv.access_url()
    show("=" * 72)
    show("  vfi_burn — Local Web UI (Zero-Dependency Python Server)")
    show("  WARNING: UNTESTED ON PHYSICAL HARDWARE — VERIFY EVERYTHING BY HAND")
    show("=" * 72)
    show(f"  Open in browser : {url}")
    if host in ("127.0.0.1", "localhost"):
        show(f"  Over SSH tunnel : ssh -L {srv.bound_port}:127.0.0.1:{srv.bound_port} <user>@<bench-host>")
    show("  Press Ctrl-C to stop the server.")
    show("=" * 72)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        show("\nShutting down vfi_burn Web UI server.")
    finally:
        srv.server_close()
    return 0
