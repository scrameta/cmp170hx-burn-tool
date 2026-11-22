"""Append-only, fsynced, exclusively locked JSONL audit journal and signal deferral."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import stat

from .model import BurnError

_CRITICAL_SIGNALS = tuple(
    s
    for s in (getattr(signal, n, None) for n in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"))
    if s is not None
)


@contextmanager
def deferred_signals():
    """Block SIGINT/SIGTERM/SIGHUP/SIGQUIT for the duration of an irreversible step.

    Ctrl-C or an SSH drop (SIGHUP) must not land between FUSEWDATA=word and the
    post-write journal entry. Pending signals are delivered when the block exits,
    after the outcome is fsynced. All hardware waits inside have bounded timeouts.
    """
    if not hasattr(signal, "pthread_sigmask") or not _CRITICAL_SIGNALS:
        yield
        return
    old = signal.pthread_sigmask(signal.SIG_BLOCK, _CRITICAL_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


class Journal:
    """Append-only, fsynced, exclusively locked JSONL with mode 0600 and no symlinks."""

    def __init__(self, file: str | Path) -> None:
        self.path = Path(file)
        if self.path.is_symlink():
            raise BurnError(f"refusing symlink journal: {self.path}")
        if self.path.parent.is_symlink():
            raise BurnError(f"refusing symlink journal parent directory: {self.path.parent}")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.parent.stat().st_mode & 0o077:
            raise BurnError(
                f"journal parent directory not private (must be mode 0700): {self.path.parent} "
                f"— run: chmod 700 {self.path.parent}"
            )
        self.fd: int | None = os.open(
            self.path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            if stat.S_IMODE(os.fstat(self.fd).st_mode) != 0o600:
                raise BurnError(f"journal is not mode 0600: {self.path}")
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BurnError(
                    f"journal {self.path} is locked by another vfi_burn session; "
                    "never run two sessions concurrently"
                ) from exc
        except Exception:
            os.close(self.fd)
            self.fd = None
            raise

    def append(self, **values) -> dict:
        if self.fd is None:
            raise BurnError("journal is closed")
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **values}
        line = (json.dumps(entry, sort_keys=True, default=str) + "\n").encode("utf-8")
        view = memoryview(line)
        try:
            while len(view):
                count = os.write(self.fd, view)
                if count <= 0:
                    raise BurnError("journal write failed; no further records allowed")
                view = view[count:]
        finally:
            view.release()
        os.fsync(self.fd)
        return entry

    def close(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *args) -> None:
        self.close()
