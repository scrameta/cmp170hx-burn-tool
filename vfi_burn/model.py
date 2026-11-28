"""Strict per-card plan decoding. Supplied FF words are opaque, never synthesized."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

SKU_CMP170HX_8GB = "10de:20c2"
SKU_CMP170HX_10GB = "10de:2082"
SUPPORTED_CMP170HX_SKUS = (SKU_CMP170HX_8GB, SKU_CMP170HX_10GB)
SKU_DISPLAY_NAMES: dict[str, str] = {
    SKU_CMP170HX_8GB: "CMP 170HX 8 GB",
    SKU_CMP170HX_10GB: "CMP 170HX 10 GB",
}


def sku_display_name(pci_id: str) -> str:
    """Return a human-readable model label for a supported CMP 170HX PCI ID."""
    return SKU_DISPLAY_NAMES.get(pci_id.strip().lower(), f"PCI {pci_id}")


DEBUGCTRL_ADDR = 0x00820060
POISON_PREFIXES = (0xBADF0000, 0xBAD00000)  # most common; is_poison() matches all 0xBADxxxxx

# Physical GA100 FF (fuseless-fuse) region. Rows 0..221 are control/regular/FPF,
# 487..511 are IFF. Defined here (not ffrecord) so Plan.load can enforce it.
FF_ROW_MIN = 222
FF_ROW_MAX = 486


class BurnError(Exception):
    """A fail-closed validation, controller, safety-gate or operator error."""


class PreTriggerRefusal(BurnError):
    """Refusal raised strictly BEFORE CMD_PROGRAM was issued: no fuse can have changed.

    Writers raise this (instead of a plain BurnError) for every failure that happens
    before the program command, so the journal can record ``write_issued=False``
    rather than a misleading ``HALT_UNVERIFIED`` / "may be partially programmed".
    """


def is_poison(value: int) -> bool:
    """Return True if a 32-bit word carries a GPU PRI-error / fuse-unit error prefix.

    Matches every 0xBADxxxxx value (0xBADF poison, 0xBAD0 fuse unit not answering,
    and the other 0xBAD1..0xBADE PRI error codes), the same test
    cmp170hx-8gb-vfi-tools uses (``(v >> 20) == 0xBAD``).
    """
    return ((value & 0xFFFFFFFF) >> 20) == 0xBAD


def is_debugctrl_target(target: str) -> bool:
    """Return True if a target refers to DEBUGCTRL by name or numeric address."""
    norm = target.strip().upper()
    if "DEBUGCTRL" in norm:
        return True
    try:
        return int(norm, 0) == DEBUGCTRL_ADDR
    except ValueError:
        return False


def u32(value, name: str = "value") -> int:
    if isinstance(value, bool):
        raise BurnError(f"{name}: boolean is not an integer")
    if isinstance(value, float):
        raise BurnError(f"{name}: float is not an integer")
    try:
        n = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as exc:
        raise BurnError(f"{name}: expected unsigned 32-bit integer or hex string") from exc
    if not 0 <= n <= 0xFFFFFFFF:
        raise BurnError(f"{name}: value {n} outside unsigned 32-bit range")
    return n


def _req(mapping: dict, key: str, context: str):
    if not isinstance(mapping, dict):
        raise BurnError(f"{context} must be a JSON object")
    if key not in mapping:
        raise BurnError(f"missing required field '{context}.{key}'")
    return mapping[key]


def require_private_file(path: str | Path, label: str = "private file") -> Path:
    """Ensure path is a non-symlink regular file with no group/other permissions (0600)."""
    p = Path(path)
    if p.is_symlink():
        raise BurnError(f"{label} must not be a symlink: {p}")
    try:
        st = p.stat()
    except FileNotFoundError as exc:
        raise BurnError(f"{label} not found: {p}") from exc
    if st.st_mode & 0o077:
        raise BurnError(
            f"{label} must have mode 0600 (no group/other access): {p} — run: chmod 600 {p}"
        )
    return p


def safe_num_mapping(obj, name: str) -> dict[str, int]:
    if not isinstance(obj, dict):
        raise BurnError(f"{name} must be a JSON object")
    return {str(k): u32(v, f"{name}.{k}") for k, v in obj.items()}


@dataclass(frozen=True)
class Record:
    name: str
    target: str
    word: int
    decode: str
    kind: str
    defect_sensitive: bool
    mask_field: str | None
    from_disable: int | None
    to_disable: int | None
    resulting_sm_count: int | None
    expected_cold: dict[str, int]
    link_cap2_before: int | None
    link_cap2_after: int | None

    @classmethod
    def parse(cls, r) -> "Record":
        if not isinstance(r, dict):
            raise BurnError("record must be a JSON object")
        kind = r.get("kind", "regular")
        if kind not in ("test", "regular", "pcie", "reconfig"):
            raise BurnError(f"unknown record kind {kind!r}; expected test, regular, pcie, or reconfig")
        defect_sensitive = r.get("defect_sensitive", False)
        if not isinstance(defect_sensitive, bool):
            raise BurnError("defect_sensitive must be a JSON boolean")
        mask = r.get("mask_update")
        if mask is not None and not isinstance(mask, dict):
            raise BurnError("mask_update must be a JSON object")
        if defect_sensitive and mask is None:
            raise BurnError("defect-sensitive record needs mask_update for reconciliation")
        if mask is not None and not defect_sensitive:
            raise BurnError("mask_update must be flagged defect_sensitive=true")
        for field in ("name", "target", "decode"):
            val = r.get(field)
            if not isinstance(val, str) or not val.strip():
                raise BurnError(f"record.{field} must be a nonempty string")
        word = u32(_req(r, "word", f"record[{r['name']}]"), f"record[{r['name']}].word")
        mask_field = None
        from_disable = None
        to_disable = None
        if mask is not None:
            raw_field = _req(mask, "field", f"record[{r['name']}].mask_update")
            if not isinstance(raw_field, str) or not raw_field.strip():
                raise BurnError(f"record[{r['name']}].mask_update.field must be a nonempty string")
            mask_field = raw_field.strip()
            from_disable = u32(_req(mask, "from", f"record[{r['name']}].mask_update"), "mask.from")
            to_disable = u32(_req(mask, "to", f"record[{r['name']}].mask_update"), "mask.to")
        sm_raw = r.get("resulting_sm_count")
        lc_before = r.get("link_cap2_before")
        lc_after = r.get("link_cap2_after")
        return cls(
            name=r["name"].strip(),
            target=r["target"].strip(),
            word=word,
            decode=r["decode"].strip(),
            kind=kind,
            defect_sensitive=defect_sensitive,
            mask_field=mask_field,
            from_disable=from_disable,
            to_disable=to_disable,
            resulting_sm_count=u32(sm_raw, "resulting_sm_count") if sm_raw is not None else None,
            expected_cold=safe_num_mapping(r.get("expected_cold", {}), "expected_cold"),
            link_cap2_before=u32(lc_before, "link_cap2_before") if lc_before is not None else None,
            link_cap2_after=u32(lc_after, "link_cap2_after") if lc_after is not None else None,
        )


def canonical_ff_prefix_sha256(rows: dict[int, int]) -> str:
    """Compute a deterministic SHA-256 digest over a canonical sorted (row, word) FF prefix."""
    payload = "".join(f"{row}:0x{rows[row] & 0xFFFFFFFF:08x}\n" for row in sorted(rows))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Plan:
    content_hash: str
    pci_id: str
    die_rows: dict[int, int]
    expected_masks: dict[str, int]
    defect_masks: dict[str, int]
    first_row: int
    last_row: int
    empty_word: int
    base_sm_ceiling: int
    override_sm_ceiling: int | None
    reconfig_semantics_verified: bool
    final_expected_cold: dict[str, int]
    records: tuple[Record, ...]
    baseline_ff_rows: dict[int, int] | None = None
    baseline_ff_sha256: str | None = None
    expected_next_row: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        # No YAML dependency or implicit integer conversion. Private plan files only.
        p = require_private_file(path, "private plan")
        raw = p.read_bytes()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BurnError(f"invalid JSON in plan {p}: {exc}") from exc
        if not isinstance(d, dict):
            raise BurnError("plan root must be a JSON object")
        if d.get("schema_version") != 1:
            raise BurnError(
                "only schema_version=1 is understood (use 'python3 -m vfi_burn configure' "
                "or menu [1] Setup -> [a] to build a plan directly from your card)"
            )

        card = _req(d, "card", "plan")
        fpf = _req(d, "fpf", "plan")
        ceiling = d.get("ceiling")
        if ceiling is None:
            ceiling = {"base_sm": 128}
        elif not isinstance(ceiling, dict):
            raise BurnError("plan.ceiling must be a JSON object")
        raw_records = _req(d, "records", "plan")
        if not isinstance(raw_records, list):
            raise BurnError("plan.records must be a JSON list")

        raw_die = _req(card, "die_rows", "card")
        if not isinstance(raw_die, dict):
            raise BurnError("card.die_rows must be a JSON object")
        die: dict[int, int] = {}
        for k, v in raw_die.items():
            try:
                idx = int(k)
            except (TypeError, ValueError) as exc:
                raise BurnError(f"card.die_rows key {k!r} is not an integer") from exc
            if idx in die:
                raise BurnError(f"duplicate die_rows key {idx}")
            die[idx] = u32(v, f"card.die_rows.{idx}")
        if set(die) != set(range(8, 16)):
            raise BurnError("must provide all eight cold die ID rows (8..15)")

        pci_id = str(_req(card, "pci_id", "card")).strip().lower()
        if pci_id not in SUPPORTED_CMP170HX_SKUS:
            raise BurnError(
                f"unsupported PCI ID {pci_id!r}; expected CMP 170HX 8 GB ({SKU_CMP170HX_8GB}) "
                f"or 10 GB ({SKU_CMP170HX_10GB})"
            )

        first = u32(_req(fpf, "first_row", "fpf"), "fpf.first_row")
        last = u32(_req(fpf, "last_row", "fpf"), "fpf.last_row")
        if not (FF_ROW_MIN <= first < last <= FF_ROW_MAX):
            raise BurnError(
                f"fpf.first_row/last_row ({first}..{last}) must lie inside the GA100 FF region "
                f"{FF_ROW_MIN}..{FF_ROW_MAX} (rows 0..221 control/FPF, 487..511 IFF)"
            )
        empty_word = u32(_req(fpf, "empty_word", "fpf"), "fpf.empty_word")

        baseline_ff_rows: dict[int, int] | None = None
        baseline_ff_sha256: str | None = None
        expected_next_row: int | None = None
        raw_baseline = fpf.get("baseline_ff_rows")
        raw_baseline_sha = fpf.get("baseline_ff_sha256")
        raw_next_row = fpf.get("expected_next_row")

        if raw_baseline_sha is not None and raw_baseline is None:
            raise BurnError("fpf.baseline_ff_sha256 requires fpf.baseline_ff_rows")
        if raw_baseline is not None:
            if not isinstance(raw_baseline, dict):
                raise BurnError("fpf.baseline_ff_rows must be a JSON object")
            parsed_base: dict[int, int] = {}
            for k, v in raw_baseline.items():
                try:
                    r_idx = int(k)
                except (TypeError, ValueError) as exc:
                    raise BurnError(f"fpf.baseline_ff_rows key {k!r} is not an integer") from exc
                if r_idx in parsed_base:
                    raise BurnError(f"duplicate fpf.baseline_ff_rows key {r_idx}")
                if not (first <= r_idx <= last):
                    raise BurnError(
                        f"fpf.baseline_ff_rows row {r_idx} outside declared window {first}..{last}"
                    )
                word = u32(v, f"fpf.baseline_ff_rows.{r_idx}")
                if word == empty_word:
                    raise BurnError(
                        f"fpf.baseline_ff_rows row {r_idx} contains empty_word (0x{empty_word:08x}); "
                        "store only the contiguous occupied prefix"
                    )
                # No poison test: baseline rows are OTP data from a cold dump, and a real FF
                # record with block data 0xBAD0..0xBADF reads 0xBADxxxxx.
                if (word >> 5) & 1:
                    raise BurnError(
                        f"fpf.baseline_ff_rows row {r_idx} word 0x{word:08x} has IFF bit 5 set"
                    )
                parsed_base[r_idx] = word
            expected_keys = set(range(first, first + len(parsed_base)))
            if set(parsed_base) != expected_keys:
                raise BurnError(
                    "fpf.baseline_ff_rows must form a contiguous prefix starting at "
                    f"fpf.first_row ({first})"
                )
            baseline_ff_rows = dict(sorted(parsed_base.items()))
            computed_sha = canonical_ff_prefix_sha256(baseline_ff_rows)
            if raw_baseline_sha is not None:
                if not isinstance(raw_baseline_sha, str) or raw_baseline_sha.strip().lower() != computed_sha:
                    raise BurnError(
                        "fpf.baseline_ff_sha256 does not match fpf.baseline_ff_rows "
                        f"(expected {computed_sha})"
                    )
            baseline_ff_sha256 = computed_sha
            expected_next_row = first + len(baseline_ff_rows)

        if raw_next_row is not None:
            parsed_next = u32(raw_next_row, "fpf.expected_next_row")
            if not (first <= parsed_next <= last):
                raise BurnError(
                    f"fpf.expected_next_row ({parsed_next}) outside declared window {first}..{last}"
                )
            if expected_next_row is not None and parsed_next != expected_next_row:
                raise BurnError(
                    f"fpf.expected_next_row ({parsed_next}) disagrees with "
                    f"baseline_ff_rows length (expected {expected_next_row})"
                )
            expected_next_row = parsed_next

        base_sm = u32(_req(ceiling, "base_sm", "ceiling"), "ceiling.base_sm")
        if base_sm == 0:
            raise BurnError("ceiling.base_sm must be positive")
        override_sm = (
            u32(ceiling["override_sm"], "ceiling.override_sm")
            if ceiling.get("override_sm") is not None
            else None
        )
        if override_sm is not None and override_sm < base_sm:
            raise BurnError(f"ceiling.override_sm ({override_sm}) cannot be less than base_sm ({base_sm})")
        reconfig_verified = ceiling.get("reconfig_semantics_verified", False)
        if not isinstance(reconfig_verified, bool):
            raise BurnError("ceiling.reconfig_semantics_verified must be a JSON boolean")

        records = tuple(Record.parse(v) for v in raw_records)
        if len(records) < 1 or records[0].kind != "test" or any(r.kind == "test" for r in records[1:]):
            raise BurnError("first record must be the sole throwaway 'test' record")
        if not records[0].expected_cold:
            raise BurnError("test record must define expected_cold values for cold verification")
        if len(records) > (last - first + 1):
            raise BurnError(
                f"plan has {len(records)} records but FPF window {first}..{last} only has {last - first + 1} rows"
            )
        if expected_next_row is not None and (expected_next_row + len(records) - 1) > last:
            raise BurnError(
                f"plan has {len(records)} records starting at row {expected_next_row}, "
                f"which exceeds fpf.last_row ({last})"
            )

        seen_names: set[str] = set()
        expected_masks = safe_num_mapping(_req(card, "expected_masks", "card"), "expected_masks")
        defects = safe_num_mapping(_req(card, "defect_masks", "card"), "defect_masks")
        for r in records:
            if r.name in seen_names:
                raise BurnError(f"duplicate record name {r.name!r} in plan")
            seen_names.add(r.name)
            if r.word == empty_word:
                raise BurnError(
                    f"record {r.name!r} word equals FPF empty_word (0x{empty_word:08x}); "
                    "would break free-slot detection"
                )
            if is_poison(r.word):
                raise BurnError(
                    f"record {r.name!r} word 0x{r.word:08x} has 0xBADx block data: deliberately "
                    "not burned (it would read back like a fuse-unit error at the moment of programming)"
                )
            if r.mask_field and (r.mask_field not in expected_masks or r.mask_field not in defects):
                raise BurnError(f"missing cold mask or genuine defect mask for {r.mask_field}")

        return cls(
            content_hash=hashlib.sha256(raw).hexdigest(),
            pci_id=pci_id,
            die_rows=die,
            expected_masks=expected_masks,
            defect_masks=defects,
            first_row=first,
            last_row=last,
            empty_word=empty_word,
            base_sm_ceiling=base_sm,
            override_sm_ceiling=override_sm,
            reconfig_semantics_verified=reconfig_verified,
            final_expected_cold=safe_num_mapping(d.get("final_expected_cold", {}), "final_expected_cold"),
            records=records,
            baseline_ff_rows=baseline_ff_rows,
            baseline_ff_sha256=baseline_ff_sha256,
            expected_next_row=expected_next_row,
        )
