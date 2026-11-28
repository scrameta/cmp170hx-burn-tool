"""GA100 Fuseless Fuse (FF) 32-bit record codec, replay, and option-field map.

Format verified across Kate's GA100VFI writeup and sayhellotojungle/cmp170hx-ffcalc
(live-burned on CMP 170HX 8 GB):
  - Physical FF region : rows 222..486 (rows 0..221 are regular/control/FPF; 487..511 are IFF)
  - Empty row marker   : 0x00000000
  - Bits [4:0]         : chain (0..31)
  - Bit  [5]           : 0 for FF record (1 = IFF; must NEVER be set for FF burns)
  - Bits [15:6]        : 16-bit block index within chain (0..1023)
  - Bits [31:16]       : replacement 16-bit block value (0x0000..0xFFFF)
  - Replay semantics   : rows 222..486 replayed in ascending order; later record wins per (chain, block).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .fuse_rules import evaluate_fuse_combinations, poweroff_domain_for_group
from .fuses import (
    A100_TARGET_DEVICE_IDS,
    DEVICE_ID_DENIED_BLOCKS,
    DIE_ID_ROWS,
    FFCALC_KEY_TO_FIELD,
    FF_EMPTY_WORD,
    FIELD_FUNCTIONAL_GROUP,
    FIELDS_BY_KEY,
    OPTION_FIELDS,
    OptionField,
    SM_PER_TPC,
    TOTAL_FBP,
    TOTAL_FUSE_ROWS,
    TOTAL_GPC,
    TPC_PER_GPC,
)
from .model import (
    FF_ROW_MAX,
    FF_ROW_MIN,
    BurnError,
    SKU_CMP170HX_8GB,
    SUPPORTED_CMP170HX_SKUS,
    canonical_ff_prefix_sha256,
    u32,
)


@dataclass(frozen=True)
class FFRecord:
    chain: int
    block: int
    data: int

    @property
    def word(self) -> int:
        return encode_ff(self.chain, self.block, self.data)


def encode_ff(chain: int, block: int, data: int) -> int:
    """Encode (chain, block, data) into a 32-bit GA100 FF record (IFF bit 5 = 0)."""
    if isinstance(chain, bool) or not (0 <= chain <= 31):
        raise BurnError(f"FF chain out of range 0..31: {chain!r}")
    if isinstance(block, bool) or not (0 <= block <= 1023):
        raise BurnError(f"FF block out of range 0..1023: {block!r}")
    if isinstance(data, bool) or not (0 <= data <= 0xFFFF):
        raise BurnError(f"FF data out of range 0..0xFFFF: {data!r}")
    return ((data & 0xFFFF) << 16) | ((block & 0x3FF) << 6) | (chain & 0x1F)


def decode_ff(word: int) -> FFRecord:
    """Decode a 32-bit word as an FF record. Rejects empty (0) and IFF (bit 5 = 1)."""
    w = u32(word, "ff_word")
    if w == FF_EMPTY_WORD:
        raise BurnError("word 0x00000000 is the empty FF marker, not a record")
    if (w >> 5) & 1:
        raise BurnError(f"word 0x{w:08x} has IFF bit 5 set; refusing non-FF record")
    return FFRecord(
        chain=w & 0x1F,
        block=(w >> 6) & 0x3FF,
        data=(w >> 16) & 0xFFFF,
    )


def fields_in_block(chain: int, block: int) -> list[OptionField]:
    """Return all known GA100 option fields that overlap (chain, block)."""
    return [f for f in OPTION_FIELDS if f.chain == chain and block in f.blocks]


def _field_bit_positions(f: OptionField, block: int) -> list[tuple[int, int]]:
    """[(field_relative_bit, block_local_bit)] for the bits of ``f`` that live in ``block``."""
    base = block * 16
    return [
        (base + i - f.bit, i)
        for i in range(16)
        if f.bit <= base + i < f.bit + f.width
    ]


def apply_ff_word_to_masks(word: int, masks: dict[str, int]) -> dict[str, int]:
    """Return the mask fields (canonical names) whose value is changed/fixed by ``word``.

    Uses FF replace semantics: after the record, every bit of (chain, block) equals the
    record's data. Only fields already present in ``masks`` are returned, with their
    bits in this block overwritten and all other bits kept.
    """
    rec = decode_ff(word)
    out: dict[str, int] = {}
    for f in fields_in_block(rec.chain, rec.block):
        name = FFCALC_KEY_TO_FIELD.get(f.key)
        if name is None or name not in masks:
            continue
        value = masks[name]
        for j, i in _field_bit_positions(f, rec.block):
            if (rec.data >> i) & 1:
                value |= 1 << j
            else:
                value &= ~(1 << j)
        out[name] = value & 0xFFFFFFFF
    return out


def ff_word_defect_audit(word: int, defect_masks: dict[str, int]) -> tuple[list[str], list[str]]:
    """Audit one FF word against genuine defect masks, independent of plan metadata.

    Returns ``(violations, unknown_fields)``. Under replace semantics the record fully
    determines its 16-bit block, so every genuinely defective unit whose disable bit
    lives in that block MUST stay disabled (bit = 1) in the new data. This holds no
    matter what the plan's ``mask_update``/``defect_sensitive`` metadata claims and
    regardless of the block's previous value.
    """
    rec = decode_ff(word)
    violations: list[str] = []
    unknown: list[str] = []
    for f in fields_in_block(rec.chain, rec.block):
        name = FFCALC_KEY_TO_FIELD.get(f.key)
        if name is None:
            continue
        if name not in defect_masks:
            unknown.append(name)
            continue
        defect = defect_masks[name]
        bad = [j for j, i in _field_bit_positions(f, rec.block) if (defect >> j) & 1 and not (rec.data >> i) & 1]
        if bad:
            violations.append(
                f"FF c{rec.chain}/b{rec.block}=0x{rec.data:04x} re-enables genuinely defective "
                f"{name} unit bit(s) {bad} (defect mask 0x{defect:x})"
            )
    return violations, unknown


def _field_local_mask(f: OptionField, block: int) -> int:
    base = block * 16
    mask = 0
    for i in range(16):
        bit = base + i
        if f.bit <= bit < f.bit + f.width:
            mask |= 1 << i
    return mask


def _compact_bit_list(bits: list[int]) -> str:
    if not bits:
        return "none"
    ranges: list[str] = []
    start = prev = bits[0]
    for b in bits[1:]:
        if b == prev + 1:
            prev = b
        else:
            ranges.append(f"{start}" if start == prev else f"{start}..{prev}")
            start = prev = b
    ranges.append(f"{start}" if start == prev else f"{start}..{prev}")
    return ", ".join(ranges)


def analyze_ff_block(
    chain: int,
    block: int,
    new_data: int,
    old_data: int | None = None,
) -> dict:
    """Inspect a 16-bit GA100 FF block write for known, unmapped, and poorly-understood bits."""
    base = block * 16
    new_data &= 0xFFFF
    fields = fields_in_block(chain, block)
    known_mask = 0
    for f in fields:
        known_mask |= _field_local_mask(f, block)
    unmapped_bits = [base + i for i in range(16) if not ((known_mask >> i) & 1)]

    # Promoted PCIe Gen4 unlock sub-bits inside opt-in magic_d (bits 19, 22, 25, 26, 28
    # -> c2/b46 bit 13 and c2/b47 bits 0, 3, 4, 6; clears MAGIC_BITS_D to A100 0x00200000).
    gen4_bit_mask = 0
    if chain == 2:
        for g4_key in (
            "pcie_magic_gen4_bit19",
            "pcie_magic_gen4_bit22",
            "pcie_magic_gen4_dis",
            "pcie_magic_gen4_bit26",
            "pcie_magic_gen4_bit28",
        ):
            if g4_key in FIELDS_BY_KEY:
                gen4_bit_mask |= _field_local_mask(FIELDS_BY_KEY[g4_key], block)

    warnings: list[str] = []
    changed_fields: list[OptionField] = []
    preserved_fields: list[OptionField] = []
    changed_unmapped_bits: list[int] = []
    changed_mask: int | None = None
    is_noop: bool | None = None

    if old_data is not None:
        old_data &= 0xFFFF
        changed_mask = (old_data ^ new_data) & 0xFFFF
        is_noop = changed_mask == 0
        changed_unmapped_bits = [
            base + i
            for i in range(16)
            if ((changed_mask >> i) & 1) and not ((known_mask >> i) & 1)
        ]
        for f in fields:
            lmask = _field_local_mask(f, block)
            eff_mask = lmask & ~gen4_bit_mask if f.key == "magic_d" else lmask
            if changed_mask & eff_mask:
                changed_fields.append(f)
            elif lmask:
                preserved_fields.append(f)

        if not is_noop:
            if not fields:
                warnings.append(
                    f"UNKNOWN FF BLOCK (chain={chain}, block={block}, 0x{old_data:04x}->0x{new_data:04x}): "
                    "not in known GA100 option map"
                )
            elif changed_unmapped_bits:
                warnings.append(
                    f"UNMAPPED / UNDOCUMENTED BITS CHANGING in chain={chain} block={block} "
                    f"(0x{old_data:04x}->0x{new_data:04x}): chain bits {_compact_bit_list(changed_unmapped_bits)}"
                )
            for f in changed_fields:
                if not f.burnable:
                    warnings.append(
                        f"READ-ONLY FIELD CHANGING: {f.name} ({f.note or 'not expected to be burned'})"
                    )
                elif f.opt_in:
                    warnings.append(
                        f"EXPERIMENTAL / POORLY UNDERSTOOD FIELD CHANGING: {f.name} — {f.note}"
                    )
                elif f.category in ("mig", "nvlink"):
                    warnings.append(
                        f"CAUTION / PARTIALLY UNDERSTOOD FIELD CHANGING: {f.name} — {f.note}"
                    )
    else:
        if not fields:
            warnings.append(
                f"UNKNOWN FF BLOCK (chain={chain}, block={block}, data=0x{new_data:04x}): "
                "not in known GA100 option map"
            )
        else:
            if unmapped_bits:
                warnings.append(
                    f"SHARED BLOCK HAS UNMAPPED BITS (chain={chain}, block={block}): "
                    f"chain bits {_compact_bit_list(unmapped_bits)} (connect target or dump to confirm preserved)"
                )
            for f in fields:
                if not f.burnable:
                    warnings.append(f"OVERLAPS READ-ONLY FIELD: {f.name} ({f.note})")
                elif f.opt_in:
                    warnings.append(f"OVERLAPS EXPERIMENTAL FIELD: {f.name} — {f.note}")
                elif f.category in ("mig", "nvlink"):
                    warnings.append(f"OVERLAPS CAUTION FIELD: {f.name} — {f.note}")

    return {
        "chain": chain,
        "block": block,
        "old_data": old_data,
        "new_data": new_data,
        "changed_mask": changed_mask,
        "is_noop": is_noop,
        "fields": fields,
        "changed_fields": changed_fields,
        "preserved_fields": preserved_fields,
        "unmapped_bits": unmapped_bits,
        "changed_unmapped_bits": changed_unmapped_bits,
        "warnings": warnings,
    }


def replay_ff_rows(rows: dict[int, int]) -> dict[int, dict[int, int]]:
    """Replay FF rows in [222..486] ascending order into {chain: {block: data16}}."""
    blocks: dict[int, dict[int, int]] = {}
    for row in sorted(rows):
        if not (FF_ROW_MIN <= row <= FF_ROW_MAX):
            continue
        word = rows[row]
        if word == FF_EMPTY_WORD:
            continue
        rec = decode_ff(word)
        blocks.setdefault(rec.chain, {})[rec.block] = rec.data
    return blocks


def get_bits(blocks_by_chain: dict[int, dict[int, int]], chain: int, bit: int, width: int) -> int:
    value = 0
    chain_blocks = blocks_by_chain.get(chain, {})
    for i in range(width):
        b = bit + i
        blk = chain_blocks.get(b // 16, 0)
        value |= ((blk >> (b % 16)) & 1) << i
    return value


def decode_all_fields(blocks_by_chain: dict[int, dict[int, int]]) -> dict[str, int]:
    return {f.key: get_bits(blocks_by_chain, f.chain, f.bit, f.width) for f in OPTION_FIELDS}


def decode_present_fields(blocks_by_chain: dict[int, dict[int, int]]) -> dict[str, int]:
    """Decode only option fields whose backing FF block(s) are all present in ``blocks_by_chain``."""
    out: dict[str, int] = {}
    for f in OPTION_FIELDS:
        chain_blocks = blocks_by_chain.get(f.chain, {})
        if all(b in chain_blocks for b in f.blocks):
            out[f.key] = get_bits(blocks_by_chain, f.chain, f.bit, f.width)
    return out


def simulate_ff_write(
    current_rows: dict[int, int],
    row: int,
    word: int,
) -> dict:
    """Simulate appending one FF record at `row` and return all affected option fields."""
    if not (FF_ROW_MIN <= row <= FF_ROW_MAX):
        raise BurnError(f"row {row} outside FF region {FF_ROW_MIN}..{FF_ROW_MAX}")
    rec = decode_ff(word)
    before_blocks = replay_ff_rows(current_rows)
    after_blocks = {c: dict(b) for c, b in before_blocks.items()}
    old_block = before_blocks.get(rec.chain, {}).get(rec.block, 0)
    after_blocks.setdefault(rec.chain, {})[rec.block] = rec.data

    before_fields = decode_all_fields(before_blocks)
    after_fields = decode_all_fields(after_blocks)
    changes = {
        key: (before_fields[key], after_fields[key])
        for key in before_fields
        if before_fields[key] != after_fields[key]
    }
    return {
        "row": row,
        "word": word,
        "chain": rec.chain,
        "block": rec.block,
        "old_block": old_block,
        "new_block": rec.data,
        "changes": changes,
    }


def compute_sm_count(gpc_disable: int, tpc_disables: list[int]) -> int:
    """Compute enabled SM count from GPC_DISABLE and 8x TPC_GPCi_DISABLE masks."""
    total = 0
    for i in range(TOTAL_GPC):
        if (gpc_disable >> i) & 1:
            continue
        tpc_mask = tpc_disables[i] & 0xFF
        enabled_tpcs = TPC_PER_GPC - bin(tpc_mask).count("1")
        total += enabled_tpcs * SM_PER_TPC
    return total


def decode_a100_devid_ff_word(word: int) -> tuple[str, int] | None:
    """If ``word`` is a validated A100 DEVIDA (c2/b37) or DEVIDB (c2/b38) FF record,
    return ``(which, devid)`` (e.g. ``("DEVIDA", 0x20F1)``); otherwise return ``None``.

    Enforces:
      - Valid non-IFF FF record targeting ``(2, 37)`` (DEVIDA) or ``(2, 38)`` (DEVIDB)
      - Bit 0 of block data is 0 (preserves chain2 bit 592 / ``DEVIDA[15]=0``)
      - ``data >> 1`` is in ``A100_TARGET_DEVICE_IDS`` (``0x20F1``, ``0x20B5``, ``0x20B0``, ``0x20B2``)
    """
    try:
        rec = decode_ff(word)
    except BurnError:
        return None
    if rec.chain != 2 or rec.block not in (37, 38):
        return None
    if (rec.data & 1) != 0:
        return None
    devid = rec.data >> 1
    if devid not in A100_TARGET_DEVICE_IDS:
        return None
    return ("DEVIDA" if rec.block == 37 else "DEVIDB", devid)


def encode_a100_devid_ff_words(target_devid: int) -> tuple[int, int]:
    """Encode paired ``(devida_word, devidb_word)`` FF records for a supported A100 Device ID."""
    if target_devid not in A100_TARGET_DEVICE_IDS:
        supported = ", ".join(f"0x{d:04x}" for d in sorted(A100_TARGET_DEVICE_IDS))
        raise BurnError(
            f"unsupported A100 target device ID 0x{target_devid:04x}; supported: {supported}"
        )
    block_data = (target_devid << 1) & 0xFFFF
    return encode_ff(2, 37, block_data), encode_ff(2, 38, block_data)


def ff_word_safety_audit(
    word: int,
    old_block: int | None = None,
    *,
    allow_a100_devid: bool = False,
) -> tuple[list[str], list[str]]:
    """Audit one 32-bit FF word for hard-denied and overridable safety violations.

    Returns ``(hard_denials, overridable_issues)``:
      - ``hard_denials`` are NEVER overridable (device-ID blocks c2/b37-38 unless
        ``allow_a100_devid=True`` AND ``word`` is a validated A100 DEVIDA/DEVIDB record;
        any changed ``burnable=False`` bit; or overlap with a ``burnable=False`` field
        when ``old_block`` is unknown so preservation of e.g. ``pcie_devidb[15]`` in
        c2/b39 cannot be proved).
      - ``overridable_issues`` are refused by default and require explicit
        ``--override-unmapped-ff`` + operator approval (unknown FF blocks, changed unmapped
        bits, unmapped bits with unknown baseline, or ``opt_in`` fields such as full ``magic_d``).
    """
    rec = decode_ff(word)
    hard_denials: list[str] = []
    overridable: list[str] = []

    if (rec.chain, rec.block) in DEVICE_ID_DENIED_BLOCKS:
        if allow_a100_devid and decode_a100_devid_ff_word(word) is not None:
            return hard_denials, overridable
        hard_denials.append(
            f"FF c{rec.chain}/b{rec.block}=0x{rec.data:04x} targets unconditionally denied "
            "read-only PCIe device-ID block (chain 2 blocks 37/38)"
        )
        return hard_denials, overridable

    analysis = analyze_ff_block(rec.chain, rec.block, rec.data, old_block)
    fields = analysis["fields"]

    if old_block is not None:
        for f in analysis["changed_fields"]:
            if not f.burnable:
                hard_denials.append(
                    f"FF c{rec.chain}/b{rec.block} (0x{old_block:04x}->0x{rec.data:04x}) "
                    f"modifies read-only field {f.name} ({f.note or 'burnable=False'})"
                )
            elif f.opt_in:
                overridable.append(
                    f"FF c{rec.chain}/b{rec.block} (0x{old_block:04x}->0x{rec.data:04x}) "
                    f"modifies experimental opt-in field {f.name} ({f.note})"
                )
        if not fields:
            overridable.append(
                f"FF c{rec.chain}/b{rec.block} (0x{old_block:04x}->0x{rec.data:04x}) "
                "targets an unknown FF block not in the GA100 option map"
            )
        elif analysis["changed_unmapped_bits"]:
            overridable.append(
                f"FF c{rec.chain}/b{rec.block} (0x{old_block:04x}->0x{rec.data:04x}) "
                f"changes unmapped/undocumented chain bits {_compact_bit_list(analysis['changed_unmapped_bits'])}"
            )
    else:
        for f in fields:
            if not f.burnable:
                hard_denials.append(
                    f"FF c{rec.chain}/b{rec.block}=0x{rec.data:04x} overlaps read-only field "
                    f"{f.name} and no verified baseline block is available to prove it is unchanged"
                )
            elif f.opt_in:
                overridable.append(
                    f"FF c{rec.chain}/b{rec.block}=0x{rec.data:04x} overlaps experimental opt-in "
                    f"field {f.name} without a verified baseline block"
                )
        if not fields:
            overridable.append(
                f"FF c{rec.chain}/b{rec.block}=0x{rec.data:04x} targets an unknown FF block "
                "not in the GA100 option map"
            )
        elif analysis["unmapped_bits"]:
            overridable.append(
                f"FF c{rec.chain}/b{rec.block}=0x{rec.data:04x} shares unmapped chain bits "
                f"{_compact_bit_list(analysis['unmapped_bits'])} and no verified baseline block "
                "is available to prove they are preserved"
            )
    return hard_denials, overridable


def ff_word_touches_pcie(word: int, old_block: int | None = None) -> bool:
    """Return True if ``word`` modifies (or, without baseline, overlaps) any PCIe option field."""
    try:
        rec = decode_ff(word)
    except BurnError:
        return False
    analysis = analyze_ff_block(rec.chain, rec.block, rec.data, old_block)
    fields = analysis["changed_fields"] if old_block is not None else analysis["fields"]
    return any(FIELD_FUNCTIONAL_GROUP.get(f.key, f.category) == "pcie" for f in fields)


def verify_noop_test_record(word: int, existing_blocks: dict[int, dict[int, int]]) -> None:
    """Prove that ``word`` is a demonstrable 0-bit-diff no-op against an existing FF block.

    Requires BOTH that ``(chain, block)`` already exists in ``existing_blocks`` (so we never
    assume an unmentioned block's underlying OTP default is 0) AND that ``data`` equals the
    currently effective 16-bit block value.
    """
    rec = decode_ff(word)
    if (rec.chain, rec.block) in DEVICE_ID_DENIED_BLOCKS:
        raise BurnError(
            f"test record c{rec.chain}/b{rec.block} targets denied PCIe device-ID block"
        )
    if not fields_in_block(rec.chain, rec.block):
        raise BurnError(
            f"test record c{rec.chain}/b{rec.block} targets an unknown FF block; "
            "use a known chain-1 block for the no-op test row"
        )
    chain_blocks = existing_blocks.get(rec.chain, {})
    if rec.block not in chain_blocks:
        raise BurnError(
            f"test record c{rec.chain}/b{rec.block}=0x{rec.data:04x} cannot be proved no-op: "
            "block is not present in the card's existing FF prefix (underlying OTP value unknown)"
        )
    current = chain_blocks[rec.block]
    if current != rec.data:
        diff = current ^ rec.data
        raise BurnError(
            f"test record c{rec.chain}/b{rec.block} is NOT a no-op: "
            f"effective block would change 0x{current:04x} -> 0x{rec.data:04x} (diff=0x{diff:04x})"
        )


def classify_ff_functional_groups(word: int, old_block: int | None = None) -> set[str]:
    """Classify which functional GPU subsystem(s) a 32-bit FF record modifies."""
    try:
        rec = decode_ff(word)
    except BurnError:
        return {"unknown"}
    analysis = analyze_ff_block(rec.chain, rec.block, rec.data, old_block)
    if old_block is not None:
        if analysis["is_noop"]:
            return {"noop"}
        groups = {FIELD_FUNCTIONAL_GROUP.get(f.key, f.category) for f in analysis["changed_fields"]}
        if not groups:
            return {"unknown"}
        return groups
    if not analysis["fields"]:
        return {"unknown"}
    return {FIELD_FUNCTIONAL_GROUP.get(f.key, f.category) for f in analysis["fields"]}


def evaluate_poweroff_sanity(
    live_rows: dict[int, int],
    plan,
    *,
    committed_records: list,
    remaining_records: list,
    baseline_blocks: dict[int, dict[int, int]] | None = None,
) -> dict:
    """Run a pre-power-off sanity check on the current FF state vs remaining planned records.

    Checks:
      1. Zero genuine defect bits enabled in the live replayed FF state.
      2. Replayed SM count does not exceed the plan ceiling (when compute masks are present).
      3. No functional subsystem (compute / memory / pcie / nvlink / mig) is split half-burned
         between already-committed records and remaining unburned records.
      4. Internal compute consistency (e.g. an enabled GPC with all 8 TPCs disabled).
    """
    errors: list[str] = []
    warnings: list[str] = []

    live_blocks = replay_ff_rows(live_rows)
    live_fields = decode_all_fields(live_blocks)

    # 1. Check all known genuine defect masks against the live replayed state.
    for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items():
        if vfi_name not in plan.defect_masks:
            continue
        defect = plan.defect_masks[vfi_name]
        cur_val = live_fields[ff_key]
        reenabled = defect & ~cur_val & 0xFFFFFFFF
        if reenabled:
            errors.append(
                f"LIVE REPLAY DEFECT VIOLATION: {vfi_name}=0x{cur_val:x} leaves genuinely "
                f"defective bits 0x{reenabled:x} enabled!"
            )

    # 2. Replayed SM count check (only when GPC/TPC blocks exist in replay or expected masks).
    sm_count: int | None = None
    tpc_keys = [f"TPC_GPC{i}_DISABLE" for i in range(TOTAL_GPC)]
    if "GPC_DISABLE" in plan.expected_masks and all(k in plan.expected_masks for k in tpc_keys):
        sm_count = compute_sm_count(
            live_fields["gpc_disable"],
            [live_fields[f"tpc_gpc{i}_disable"] for i in range(TOTAL_GPC)],
        )
        has_reconfig = any(r.kind == "reconfig" for r in committed_records)
        ceiling = plan.override_sm_ceiling if (has_reconfig and plan.override_sm_ceiling) else plan.base_sm_ceiling
        if ceiling is not None and sm_count > ceiling:
            errors.append(
                f"LIVE REPLAY SM COUNT {sm_count} exceeds plan ceiling {ceiling}!"
            )

    # 2b. Declarative cross-fuse combination & coherence rules (compute / memory / PCIe).
    present_fields = decode_present_fields(live_blocks)
    if "GPC_DISABLE" in plan.expected_masks and all(k in plan.expected_masks for k in tpc_keys):
        present_fields.setdefault("gpc_disable", live_fields["gpc_disable"])
        for i in range(TOTAL_GPC):
            present_fields.setdefault(f"tpc_gpc{i}_disable", live_fields[f"tpc_gpc{i}_disable"])
    for issue in evaluate_fuse_combinations(present_fields):
        if issue.severity == "error":
            errors.append(issue.message)
        else:
            warnings.append(issue.message)

    # 3. Functional-group / atomic power-off domain boundary check between committed and remaining records.
    if baseline_blocks is not None:
        walk_blocks = {c: dict(b) for c, b in baseline_blocks.items()}
    else:
        walk_blocks = {c: dict(b) for c, b in live_blocks.items()}

    committed_by_group: dict[str, list[str]] = {}
    committed_by_domain: dict[str, list[str]] = {}
    committed_groups_in_domain: dict[str, set[str]] = {}
    for rec in committed_records:
        try:
            ff = decode_ff(rec.word)
            old_b = walk_blocks.get(ff.chain, {}).get(ff.block)
            groups = classify_ff_functional_groups(rec.word, old_b)
            walk_blocks.setdefault(ff.chain, {})[ff.block] = ff.data
        except BurnError:
            groups = {"unknown"}
        for g in groups:
            if g != "noop":
                committed_by_group.setdefault(g, []).append(rec.name)
                dom = poweroff_domain_for_group(g)
                committed_groups_in_domain.setdefault(dom, set()).add(g)
                dom_recs = committed_by_domain.setdefault(dom, [])
                if rec.name not in dom_recs:
                    dom_recs.append(rec.name)

    remaining_by_group: dict[str, list[str]] = {}
    remaining_by_domain: dict[str, list[str]] = {}
    remaining_groups_in_domain: dict[str, set[str]] = {}
    for rec in remaining_records:
        try:
            ff = decode_ff(rec.word)
            old_b = walk_blocks.get(ff.chain, {}).get(ff.block)
            groups = classify_ff_functional_groups(rec.word, old_b)
            walk_blocks.setdefault(ff.chain, {})[ff.block] = ff.data
        except BurnError:
            groups = {"unknown"}
        for g in groups:
            if g != "noop":
                remaining_by_group.setdefault(g, []).append(rec.name)
                dom = poweroff_domain_for_group(g)
                remaining_groups_in_domain.setdefault(dom, set()).add(g)
                dom_recs = remaining_by_domain.setdefault(dom, [])
                if rec.name not in dom_recs:
                    dom_recs.append(rec.name)

    split_groups: list[str] = []
    for dom in sorted(set(committed_by_domain) & set(remaining_by_domain)):
        dom_groups = committed_groups_in_domain[dom] | remaining_groups_in_domain[dom]
        if len(dom_groups) == 1:
            grp = next(iter(dom_groups))
            split_groups.append(grp)
            warnings.append(
                f"PARTIAL '{grp.upper()}' FUNCTIONAL GROUP BURNED: committed "
                f"[{', '.join(committed_by_group[grp])}] while [{', '.join(remaining_by_group[grp])}] "
                f"are still pending! Powering off before finishing the '{grp}' group may leave "
                "an inconsistent configuration."
            )
        else:
            split_groups.append(dom)
            coupled = "+".join(sorted(dom_groups))
            warnings.append(
                f"PARTIAL '{dom.upper()}' ATOMIC GROUP ({coupled}) BURNED: committed "
                f"[{', '.join(committed_by_domain[dom])}] while [{', '.join(remaining_by_domain[dom])}] "
                "are still pending! Powering off with compute/memory (GPC/TPC/PES vs FBP/FBPA/FBIO/ROP_L2) "
                "split can leave SMs unlocked while PES/ROP_L2 remain masked and cause GSP boot failure."
            )

    safe_to_power_off = (not errors) and (not split_groups)

    lines = ["", "=== PRE-POWER-OFF FUSE SANITY CHECK ==="]
    committed_groups = sorted(committed_by_group)
    remaining_groups = sorted(remaining_by_group)
    lines.append(
        f"  Committed groups : {', '.join(committed_groups) if committed_groups else 'none (no-op test only)'}"
    )
    lines.append(
        f"  Remaining groups : {', '.join(remaining_groups) if remaining_groups else 'none (plan complete)'}"
    )
    if sm_count is not None:
        lines.append(f"  Replayed SM count: {sm_count} SMs (base ceiling={plan.base_sm_ceiling})")
    for err in errors:
        lines.append(f"  !!! ERROR   : {err}")
    for warn in warnings:
        lines.append(f"  !!! WARNING : {warn}")
    if safe_to_power_off and not warnings:
        lines.append(
            "  VERDICT: SAFE TO POWER OFF — burned FF rows form a clean functional boundary."
        )
    elif safe_to_power_off:
        lines.append(
            "  VERDICT: CAUTION — no split functional groups, but review warnings above before power-off."
        )
    else:
        lines.append(
            "  VERDICT: DO NOT POWER OFF YET (if glitch window is still healthy) — "
            "complete the split functional group or resolve errors above first."
        )
    return {
        "safe_to_power_off": safe_to_power_off,
        "errors": errors,
        "warnings": warnings,
        "committed_groups": committed_groups,
        "remaining_groups": remaining_groups,
        "split_groups": split_groups,
        "sm_count": sm_count,
        "lines": lines,
    }


def make_noop_test_record(ff_rows: dict[int, int]) -> dict:
    """Create a safe throwaway no-op FF record re-writing an existing chain-1 block unchanged.

    Mirrors kprinssu/ga100-ff-tools plan_0_noop.json: tests CMD_WRITE + CMD_READ + CRC 0x55
    + cold-boot row persistence while changing ZERO effective fuse bits.
    """
    blocks = replay_ff_rows(ff_rows)
    chain1 = blocks.get(1, {})
    candidates = [
        b for b in sorted(chain1)
        if fields_in_block(1, b) and (1, b) not in DEVICE_ID_DENIED_BLOCKS
        # 0xBADx block data is legal OTP but is never burned (Plan.load / program_row refuse
        # such payloads), so it cannot serve as the no-op test row.
        and (chain1[b] >> 4) != 0xBAD
    ]
    if not candidates:
        raise BurnError(
            "cannot derive no-op test record: no existing known chain-1 FF rows in dump "
            "(excluding blocks whose data is 0xBAD0..0xBADF)"
        )
    blk = 7 if 7 in candidates else candidates[0]
    data = chain1[blk]
    word = encode_ff(1, blk, data)
    return {
        "name": f"NOOP_TEST_C1_B{blk}",
        "kind": "test",
        "target": f"NOOP_CHAIN1_BLK{blk}",
        "word": f"0x{word:08x}",
        "decode": f"NO-OP test row: re-writes existing chain=1 blk={blk} with identical 0x{data:04x}",
        "defect_sensitive": False,
        "expected_cold": {},
    }


def build_vfi_plan_from_records(
    *,
    ff_rows: dict[int, int],
    die_rows_int: dict[int, int],
    expected_masks: dict[str, int],
    defect_masks: dict[str, int],
    planned_items: Sequence[dict],
    pci_id: str = SKU_CMP170HX_8GB,
    ff_last_row: int = FF_ROW_MAX,
    base_sm_ceiling: int | None = None,
    override_sm_ceiling: int | None = None,
    prepend_noop_test: bool = True,
    link_cap2_before: int = 0x06,
    link_cap2_after: int = 0x1E,
    extra_metadata: dict | None = None,
    source_name: str = "planner",
    on_record_replayed: Callable[[int, dict[int, int]], None] | None = None,
) -> dict:
    """Fail-closed builder converting derived FF burn records into a vfi_burn schema_version=1 plan dict.

    If ``base_sm_ceiling`` is omitted, automatically computes the exact non-defective SM count of
    this die from ``defect_masks`` (``GPC_DISABLE`` + ``TPC_GPC0..7_DISABLE``, max 128).
    """
    pci = str(pci_id).strip().lower()
    if pci not in SUPPORTED_CMP170HX_SKUS:
        raise BurnError(
            f"{source_name}: unsupported PCI ID {pci!r} (expected {' or '.join(SUPPORTED_CMP170HX_SKUS)})"
        )
    if isinstance(ff_last_row, bool) or not (FF_ROW_MIN < ff_last_row <= FF_ROW_MAX):
        raise BurnError(
            f"{source_name}: ff_last_row={ff_last_row!r} must lie in {FF_ROW_MIN + 1}..{FF_ROW_MAX}"
        )
    if base_sm_ceiling is None:
        gpc_def = defect_masks.get("GPC_DISABLE", 0)
        tpc_defs = [defect_masks.get(f"TPC_GPC{i}_DISABLE", 0) for i in range(TOTAL_GPC)]
        base_sm_ceiling = max(compute_sm_count(gpc_def, tpc_defs), 1)
    if isinstance(base_sm_ceiling, bool) or not isinstance(base_sm_ceiling, int) or base_sm_ceiling <= 0:
        raise BurnError("base_sm_ceiling must be a positive integer")
    missing_die = [r for r in DIE_ID_ROWS if r not in die_rows_int]
    if missing_die:
        raise BurnError(f"{source_name}: missing die ID rows {missing_die}")
    die_rows = {str(r): f"0x{die_rows_int[r]:08x}" for r in DIE_ID_ROWS}

    # Row occupancy must be fully known across FF_ROW_MIN..ff_last_row (below IFF_RECORD).
    total_slots = ff_last_row - FF_ROW_MIN + 1
    missing_ff = [r for r in range(FF_ROW_MIN, ff_last_row + 1) if r not in ff_rows]
    if missing_ff:
        raise BurnError(
            f"{source_name}: card dump covers only {total_slots - len(missing_ff)} of "
            f"{total_slots} FF rows ({FF_ROW_MIN}..{ff_last_row}, missing e.g. {missing_ff[:5]}); "
            "an unread row would look free"
        )
    region = [ff_rows[r] for r in range(FF_ROW_MIN, ff_last_row + 1)]
    first_free_idx = next((i for i, v in enumerate(region) if v == FF_EMPTY_WORD), None)
    if first_free_idx is None:
        raise BurnError(f"{source_name}: FF region ({FF_ROW_MIN}..{ff_last_row}) is full")
    if any(v != FF_EMPTY_WORD for v in region[first_free_idx:]):
        raise BurnError(
            f"{source_name}: FF rows in {FF_ROW_MIN}..{ff_last_row} are non-contiguous; "
            "vfi_burn's free-slot proof would refuse"
        )
    baseline_next_row = FF_ROW_MIN + first_free_idx
    next_row = baseline_next_row
    baseline_ff_prefix = {r: ff_rows[r] for r in range(FF_ROW_MIN, baseline_next_row)}
    for r_idx, w in baseline_ff_prefix.items():
        if (w >> 5) & 1:
            raise BurnError(
                f"{source_name}: row {r_idx} word 0x{w:08x} has IFF bit 5 set"
            )
    baseline_ff_sha = canonical_ff_prefix_sha256(baseline_ff_prefix)

    if not planned_items and not prepend_noop_test:
        raise BurnError(f"{source_name}: plan has no records")

    # Replay current FF rows and verify expected_masks against FF replay.
    running_rows = dict(ff_rows)
    running_masks = dict(expected_masks)
    vfi_records: list[dict] = []

    base_blocks = replay_ff_rows(ff_rows)
    replay_fields = decode_all_fields(base_blocks)
    for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items():
        if vfi_name in expected_masks and replay_fields[ff_key] != expected_masks[vfi_name]:
            raise BurnError(
                f"fused {vfi_name}=0x{expected_masks[vfi_name]:x} disagrees with FF replay "
                f"0x{replay_fields[ff_key]:x}; the FF-only option model does not hold for this card, "
                "so planned before/after masks cannot be trusted"
            )

    if prepend_noop_test:
        noop = make_noop_test_record(ff_rows)
        if "GPC_DISABLE" in expected_masks:
            noop["expected_cold"] = {"GPC_DISABLE": f"0x{expected_masks['GPC_DISABLE']:x}"}
        elif expected_masks:
            first_key = sorted(expected_masks)[0]
            noop["expected_cold"] = {first_key: f"0x{expected_masks[first_key]:x}"}
        vfi_records.append(noop)
        first_record_as_test = False
        next_row += 1
    else:
        first_word = planned_items[0]["word"]
        verify_noop_test_record(first_word, base_blocks)
        first_record_as_test = True

    if planned_items and (next_row + len(planned_items) - 1) > ff_last_row:
        raise BurnError(f"plan does not fit in the remaining FF rows (up to {ff_last_row})")

    for idx, item in enumerate(planned_items):
        replay_slot = baseline_next_row + len(vfi_records)
        if item.get("row") is not None and int(item["row"]) != replay_slot:
            raise BurnError(
                f"{source_name}: planned record[{idx}] targets row {int(item['row'])}, "
                f"not contiguous with expected row {replay_slot} (first free FF row is {baseline_next_row})"
            )
        row = replay_slot
        word = int(item["word"])
        chain = int(item["chain"])
        block = int(item["block"])
        new_data = int(item["new"])
        src_label = str(item.get("source_label", f"record[{idx}]"))
        if encode_ff(chain, block, new_data) != word:
            raise BurnError(
                f"{src_label} row {row}: record 0x{word:08x} disagrees with "
                f"chain={chain} block={block} new=0x{new_data:04x}"
            )
        replay_blocks_now = replay_ff_rows(running_rows)
        replay_old = replay_blocks_now.get(chain, {}).get(block)
        if item.get("old") is not None:
            expected_old = int(item["old"])
            if replay_old is None:
                if expected_old != 0:
                    raise BurnError(
                        f"{src_label} row {row}: planned old=0x{expected_old:04x} but "
                        f"c{chain}/b{block} is absent from replayed FF prefix"
                    )
            elif expected_old != replay_old:
                raise BurnError(
                    f"{src_label} row {row}: planned old=0x{expected_old:04x} != "
                    f"replayed current block 0x{replay_old:04x}"
                )
        devid_info = decode_a100_devid_ff_word(word)
        is_devid = (item.get("kind") == "devid") and (devid_info is not None)
        hard_denials, overridable_issues = ff_word_safety_audit(
            word, replay_old, allow_a100_devid=is_devid
        )
        if hard_denials:
            raise BurnError(f"{src_label} row {row}: REFUSED; " + "; ".join(hard_denials))
        non_optin_issues = [msg for msg in overridable_issues if "opt-in field" not in msg]
        if non_optin_issues:
            raise BurnError(f"{src_label} row {row}: REFUSED; " + "; ".join(non_optin_issues))

        sim = simulate_ff_write(running_rows, replay_slot, word)
        running_rows[replay_slot] = word

        fields = list(item.get("fields") or [])
        if not fields:
            fields = list(sim["changes"].keys()) or [f.key for f in fields_in_block(chain, block)]
        is_pcie = (not is_devid) and (
            ff_word_touches_pcie(word, replay_old)
            or any(
                FIELD_FUNCTIONAL_GROUP.get(k, FIELDS_BY_KEY[k].category) == "pcie"
                for k in fields
                if k in FIELDS_BY_KEY
            )
        )

        changed_masks = [
            (k, sim["changes"][k])
            for k in sim["changes"]
            if k in FFCALC_KEY_TO_FIELD and FFCALC_KEY_TO_FIELD[k] in running_masks
        ]

        # Enforce genuine defect mask check across ALL changed floorsweep masks.
        for ff_key, (before_v, after_v) in changed_masks:
            vfi_name = FFCALC_KEY_TO_FIELD[ff_key]
            cleared = before_v & ~after_v & 0xFFFFFFFF
            bad_bits = cleared & defect_masks[vfi_name]
            if bad_bits:
                raise BurnError(
                    f"{src_label} row {row} clears genuinely defective {vfi_name} bits 0x{bad_bits:x}"
                )

        if idx == 0 and first_record_as_test:
            kind = "test"
        elif is_devid:
            kind = "devid"
        elif is_pcie:
            kind = "pcie"
        else:
            kind = "regular"

        if is_devid and devid_info is not None:
            which_dev, target_dev = devid_info
            target_name = f"PCIE_{which_dev}"
            note = item.get("note") or (
                f"{A100_TARGET_DEVICE_IDS[target_dev]} — requires matching A100 VBIOS + 128 GB BAR1"
            )
            old_tag = f"0x{replay_old:04x}->" if replay_old is not None else "FPF->"
            decode_desc = (
                f"FF chain={chain} blk={block} {old_tag}0x{sim['new_block']:04x} "
                f"({which_dev}=0x{target_dev:04x}; {note})"
            )
        else:
            primary_field = fields[0] if fields else f"chain{chain}_blk{block}"
            target_name = FFCALC_KEY_TO_FIELD.get(primary_field, primary_field.upper())
            note = item.get("note")
            note_suffix = f" ({note})" if note else ""
            decode_desc = (
                f"FF chain={chain} blk={block} 0x{sim['old_block']:04x}->0x{sim['new_block']:04x} "
                f"fields={','.join(fields) or 'raw'}{note_suffix}"
            )

        rec_dict: dict = {
            "name": f"ROW_{row}_{target_name}",
            "kind": kind,
            "target": target_name,
            "word": f"0x{word:08x}",
            "decode": decode_desc,
            "defect_sensitive": False,
            "expected_cold": {},
        }

        for ff_key, (before_v, after_v) in changed_masks:
            vfi_name = FFCALC_KEY_TO_FIELD[ff_key]
            if running_masks[vfi_name] != before_v:
                raise BurnError(
                    f"{src_label} row {row}: tracked {vfi_name}=0x{running_masks[vfi_name]:x} "
                    f"!= replayed before-value 0x{before_v:x}"
                )
            running_masks[vfi_name] = after_v
        if len(changed_masks) == 1:
            ff_key, (before_v, after_v) = changed_masks[0]
            vfi_name = FFCALC_KEY_TO_FIELD[ff_key]
            rec_dict["defect_sensitive"] = True
            rec_dict["mask_update"] = {
                "field": vfi_name,
                "from": f"0x{before_v:x}",
                "to": f"0x{after_v:x}",
            }
        if any(
            any(tok in FFCALC_KEY_TO_FIELD[k] for tok in ("GPC", "TPC")) for k, _ in changed_masks
        ):
            gpc_d = running_masks.get("GPC_DISABLE", 0)
            tpc_ds = [running_masks.get(f"TPC_GPC{i}_DISABLE", 0) for i in range(TOTAL_GPC)]
            rec_dict["resulting_sm_count"] = compute_sm_count(gpc_d, tpc_ds)

        for ff_key, (_before_v, after_v) in changed_masks:
            rec_dict["expected_cold"][FFCALC_KEY_TO_FIELD[ff_key]] = f"0x{after_v:x}"
        if kind == "test" and not rec_dict["expected_cold"]:
            if "GPC_DISABLE" in expected_masks:
                rec_dict["expected_cold"]["GPC_DISABLE"] = f"0x{expected_masks['GPC_DISABLE']:x}"
            elif expected_masks:
                first_key = sorted(expected_masks)[0]
                rec_dict["expected_cold"][first_key] = f"0x{expected_masks[first_key]:x}"
            else:
                rec_dict["expected_cold"][target_name] = f"0x{sim['new_block']:x}"
        if kind == "pcie":
            rec_dict["link_cap2_before"] = f"0x{link_cap2_before:x}"
            rec_dict["link_cap2_after"] = f"0x{link_cap2_after:x}"

        vfi_records.append(rec_dict)
        if on_record_replayed is not None:
            on_record_replayed(idx, running_rows)

    # Compute final expected cold fields from replayed end state.
    final_blocks = replay_ff_rows(running_rows)
    final_decoded = decode_all_fields(final_blocks)
    final_gpc = final_decoded["gpc_disable"]
    final_tpcs = [final_decoded[f"tpc_gpc{i}_disable"] for i in range(TOTAL_GPC)]
    final_sm = compute_sm_count(final_gpc, final_tpcs)

    final_expected_cold = {
        "GPC_DISABLE": f"0x{final_gpc:x}",
        "FBP_DISABLE": f"0x{final_decoded['fbp_disable']:x}",
        "FBIO_LAYOUT": f"0x{final_decoded['fbio_disable']:x}",
        "ROP_L2_LAYOUT": f"0x{final_decoded['rop_l2_disable']:x}",
        "FBPA_LAYOUT": f"0x{final_decoded['fbpa_disable']:x}",
        "MAGIC_D": f"0x{final_decoded['magic_d']:x}",
        "BOOT_GEN3_DIS": f"0x{final_decoded['pcie_boot_gen3_dis']:x}",
        "DISABLE_GEN3_SPEED": f"0x{final_decoded['disable_gen3_speed_bit1'] | final_decoded['disable_gen3_speed_bit9']:x}",
        "LnkCap2": f"0x{(link_cap2_after if any(r.get('kind') == 'pcie' for r in vfi_records) else link_cap2_before):x}",
        "SM_COUNT": final_sm,
    }

    out: dict = {"schema_version": 1}
    if extra_metadata:
        out.update(extra_metadata)
    out.update(
        {
            "card": {
                "pci_id": pci,
                "die_rows": die_rows,
                "expected_masks": {k: f"0x{v:x}" for k, v in sorted(expected_masks.items())},
                "defect_masks": {k: f"0x{v:x}" for k, v in sorted(defect_masks.items())},
            },
            "fpf": {
                "first_row": FF_ROW_MIN,
                "last_row": ff_last_row,
                "empty_word": f"0x{FF_EMPTY_WORD:08x}",
                "expected_next_row": baseline_next_row,
                "baseline_ff_rows": {str(r): f"0x{w:08x}" for r, w in sorted(baseline_ff_prefix.items())},
                "baseline_ff_sha256": baseline_ff_sha,
            },
            "ceiling": {
                "base_sm": base_sm_ceiling,
                "override_sm": override_sm_ceiling,
                "reconfig_semantics_verified": False,
            },
            "final_expected_cold": final_expected_cold,
            "records": vfi_records,
        }
    )
    return out
