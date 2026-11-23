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
import hashlib
import json
from pathlib import Path

from .fuse_rules import evaluate_fuse_combinations, poweroff_domain_for_group
from .fuses import (
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
    canonical_ff_prefix_sha256,
    require_private_file,
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

    # Well-understood sub-bit inside opt-in magic_d (Gen4 disable flag, chain 2 bit 755).
    gen4_bit_mask = (
        _field_local_mask(FIELDS_BY_KEY["pcie_magic_gen4_dis"], block)
        if chain == 2 and "pcie_magic_gen4_dis" in FIELDS_BY_KEY
        else 0
    )

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


def ff_word_safety_audit(
    word: int,
    old_block: int | None = None,
) -> tuple[list[str], list[str]]:
    """Audit one 32-bit FF word for hard-denied and overridable safety violations.

    Returns ``(hard_denials, overridable_issues)``:
      - ``hard_denials`` are NEVER overridable (device-ID blocks c2/b37-38, any changed
        ``burnable=False`` bit, or overlap with a ``burnable=False`` field when ``old_block``
        is unknown so preservation of e.g. ``pcie_devidb[15]`` in c2/b39 cannot be proved).
      - ``overridable_issues`` are refused by default and require explicit
        ``--override-unmapped-ff`` + operator approval (unknown FF blocks, changed unmapped
        bits, unmapped bits with unknown baseline, or ``opt_in`` fields such as full ``magic_d``).
    """
    rec = decode_ff(word)
    hard_denials: list[str] = []
    overridable: list[str] = []

    if (rec.chain, rec.block) in DEVICE_ID_DENIED_BLOCKS:
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


GA100_FF_TOOLS_DEFECT_TO_FIELD: dict[str, str] = {
    "OPT_GPC_DEFECTIVE": "GPC_DISABLE",
    "OPT_FBP_DEFECTIVE": "FBP_DISABLE",
    "OPT_FBPA_DEFECTIVE": "FBPA_DISABLE",
    "OPT_FBIO_DEFECTIVE": "FBIO_DISABLE",
    "OPT_PES_DEFECTIVE": "PES_DISABLE",
    "OPT_ROP_L2_DEFECTIVE": "ROP_L2_DISABLE",
    "OPT_NVLINK_DEFECTIVE": "NVLINK_DISABLE",
    "OPT_SYS_PIPE_DEFECTIVE": "SYS_PIPE_DISABLE",
    **{f"OPT_TPC_GPC{i}_DEFECTIVE": f"TPC_GPC{i}_DISABLE" for i in range(TOTAL_GPC)},
}

# Standard 15 defect masks emitted by ga100-ff-tools/planner/read_defects.py.
GA100_FF_TOOLS_REQUIRED_DEFECTS: tuple[str, ...] = (
    "OPT_GPC_DEFECTIVE",
    "OPT_FBP_DEFECTIVE",
    "OPT_FBPA_DEFECTIVE",
    "OPT_FBIO_DEFECTIVE",
    "OPT_PES_DEFECTIVE",
    "OPT_ROP_L2_DEFECTIVE",
    "OPT_NVLINK_DEFECTIVE",
    *(f"OPT_TPC_GPC{i}_DEFECTIVE" for i in range(TOTAL_GPC)),
)

GA100_FF_TOOLS_FIELD_TO_KEY: dict[str, str] = {
    "OPT_GPC_DISABLE": "gpc_disable",
    "OPT_FBP_DISABLE": "fbp_disable",
    "OPT_FBPA_DISABLE": "fbpa_disable",
    "OPT_FBIO_DISABLE": "fbio_disable",
    "OPT_PES_DISABLE": "pes_disable",
    "OPT_ROP_L2_DISABLE": "rop_l2_disable",
    "OPT_NVLINK_DISABLE": "nvlink_disable",
    "OPT_SYS_PIPE_DISABLE": "sys_pipe_disable",
    "OPT_PCIE_DEVIDA": "pcie_devida",
    "OPT_PCIE_DEVIDB": "pcie_devidb",
    **{f"OPT_TPC_DISABLE_GPC{i}": f"tpc_gpc{i}_disable" for i in range(TOTAL_GPC)},
}


def ga100_ff_tools_fingerprint(ff_rows: dict[int, int], last_row: int) -> str:
    """Compute the 16-hex-char baseline fingerprint used by kprinssu/ga100-ff-tools."""
    if not (FF_ROW_MIN <= last_row <= FF_ROW_MAX):
        raise BurnError(f"ga100-ff-tools baseline.last_row {last_row} outside {FF_ROW_MIN}..{FF_ROW_MAX}")
    missing = [r for r in range(FF_ROW_MIN, last_row + 1) if r not in ff_rows]
    if missing:
        raise BurnError(f"cannot compute ga100-ff-tools fingerprint: missing rows {missing[:5]}")
    payload = ",".join(f"0x{ff_rows[r] & 0xFFFFFFFF:08X}" for r in range(FF_ROW_MIN, last_row + 1))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def _extract_die_rows_from_mapping(raw: dict, label: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for k, v in raw.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if idx in DIE_ID_ROWS:
            out[idx] = u32(v, f"{label}.{idx}")
    return out


def _extract_die_rows_from_ga100_rows(rows_list: list, label: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for entry in rows_list:
        if not isinstance(entry, dict) or "row" not in entry:
            continue
        idx = int(entry["row"])
        if idx in DIE_ID_ROWS:
            raw_val = entry.get("raw")
            if raw_val is None or entry.get("error"):
                raise BurnError(f"{label}: die ID row {idx} has read error / missing raw value")
            out[idx] = u32(raw_val, f"{label}.row_{idx}")
    return out


def load_die_rows_from_file(path: str | Path) -> dict[int, int]:
    """Load Die ID rows 8..15 from a mode-0600 JSON file.

    Accepts:
      - vfi_burn snapshot / plan JSON (``die_rows`` or ``card.die_rows``)
      - cmp170hx-ffcalc 512-row dump JSON (``ff_rows``)
      - ga100-ff-tools ``ff_dump.py --start 8`` JSON (``rows`` list)
      - plain mapping ``{"8": "0x...", ..., "15": "0x..."}``
    """
    p = require_private_file(path, "die ID rows file")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in die ID rows file {p}: {exc}") from exc
    if not isinstance(obj, dict):
        raise BurnError(f"die ID rows file {p} must contain a JSON object")

    found: dict[int, int] = {}
    if isinstance(obj.get("die_rows"), dict):
        found = _extract_die_rows_from_mapping(obj["die_rows"], "die_rows")
    elif isinstance(obj.get("card"), dict) and isinstance(obj["card"].get("die_rows"), dict):
        found = _extract_die_rows_from_mapping(obj["card"]["die_rows"], "card.die_rows")
    elif isinstance(obj.get("ff_rows"), dict):
        found = _extract_die_rows_from_mapping(obj["ff_rows"], "ff_rows")
    elif isinstance(obj.get("rows"), list):
        found = _extract_die_rows_from_ga100_rows(obj["rows"], "rows")
    else:
        found = _extract_die_rows_from_mapping(obj, "die_rows")

    missing = [r for r in DIE_ID_ROWS if r not in found]
    if missing:
        raise BurnError(
            f"die ID rows file {p} is missing die ID rows {missing}; "
            "all 8 rows (8..15) are required"
        )
    return {r: found[r] for r in DIE_ID_ROWS}


def _build_vfi_plan_from_records(
    *,
    ff_rows: dict[int, int],
    die_rows_int: dict[int, int],
    expected_masks: dict[str, int],
    defect_masks: dict[str, int],
    planned_items: Sequence[dict],
    base_sm_ceiling: int,
    override_sm_ceiling: int | None,
    prepend_noop_test: bool,
    allow_shifted_first_row: bool = False,
    link_cap2_before: int = 0x06,
    link_cap2_after: int = 0x1E,
    extra_metadata: dict | None = None,
    source_name: str = "ffcalc",
    on_record_replayed: Callable[[int, dict[int, int]], None] | None = None,
) -> dict:
    """Shared fail-closed builder converting normalized FF burn records into a vfi_burn plan dict."""
    if isinstance(base_sm_ceiling, bool) or not isinstance(base_sm_ceiling, int) or base_sm_ceiling <= 0:
        raise BurnError("base_sm_ceiling must be a positive integer supplied by the operator")
    missing_die = [r for r in DIE_ID_ROWS if r not in die_rows_int]
    if missing_die:
        raise BurnError(f"{source_name} missing die ID rows {missing_die}")
    die_rows = {str(r): f"0x{die_rows_int[r]:08x}" for r in DIE_ID_ROWS}

    # Row occupancy must be fully known across 222..486.
    missing_ff = [r for r in range(FF_ROW_MIN, FF_ROW_MAX + 1) if r not in ff_rows]
    if missing_ff:
        raise BurnError(
            f"{source_name} dump covers only {FF_ROW_MAX - FF_ROW_MIN + 1 - len(missing_ff)} of "
            f"{FF_ROW_MAX - FF_ROW_MIN + 1} FF rows (missing e.g. {missing_ff[:5]}); "
            "an unread row would look free"
        )
    region = [ff_rows[r] for r in range(FF_ROW_MIN, FF_ROW_MAX + 1)]
    first_free_idx = next((i for i, v in enumerate(region) if v == FF_EMPTY_WORD), None)
    if first_free_idx is None:
        raise BurnError(f"{source_name} dump shows the FF region full")
    if any(v != FF_EMPTY_WORD for v in region[first_free_idx:]):
        raise BurnError(
            f"{source_name} dump FF rows are non-contiguous; vfi_burn's free-slot proof would refuse"
        )
    baseline_next_row = FF_ROW_MIN + first_free_idx
    next_row = baseline_next_row
    baseline_ff_prefix = {r: ff_rows[r] for r in range(FF_ROW_MIN, baseline_next_row)}
    for r_idx, w in baseline_ff_prefix.items():
        if (w >> 5) & 1:
            raise BurnError(
                f"{source_name} dump row {r_idx} word 0x{w:08x} has IFF bit 5 set"
            )
    baseline_ff_sha = canonical_ff_prefix_sha256(baseline_ff_prefix)

    if not planned_items:
        raise BurnError(f"{source_name} burn plan has no records")

    # Replay current FF rows and verify expected_masks against FF replay.
    running_rows = dict(ff_rows)
    running_masks = dict(expected_masks)
    vfi_records: list[dict] = []

    base_blocks = replay_ff_rows(ff_rows)
    replay_fields = decode_all_fields(base_blocks)
    for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items():
        if vfi_name in expected_masks and replay_fields[ff_key] != expected_masks[vfi_name]:
            raise BurnError(
                f"dump fused {vfi_name}=0x{expected_masks[vfi_name]:x} disagrees with FF replay "
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
        next_row += 1  # the no-op occupies the first free row; planned rows shift by one if unshifted
    else:
        # Without prepending a no-op test row, record[0] MUST itself be a demonstrable no-op.
        first_word = planned_items[0]["word"]
        verify_noop_test_record(first_word, base_blocks)
        first_record_as_test = True

    planned_rows = [int(item["row"]) for item in planned_items]
    if len(set(planned_rows)) != len(planned_rows):
        raise BurnError(f"{source_name} burn plan uses the same row twice")
    expected_first = next_row - (1 if prepend_noop_test else 0)
    valid_starts = [expected_first]
    if allow_shifted_first_row and prepend_noop_test:
        valid_starts.append(expected_first + 1)
    if planned_rows[0] not in valid_starts or planned_rows != list(
        range(planned_rows[0], planned_rows[0] + len(planned_rows))
    ):
        raise BurnError(
            f"{source_name} planned rows {planned_rows[:3]}... are not contiguous from the first free "
            f"FF row {expected_first} of this dump (stale dump or plan?)"
        )
    if (next_row + len(planned_rows) - 1) > FF_ROW_MAX:
        raise BurnError("plan does not fit in the remaining FF rows")

    for idx, item in enumerate(planned_items):
        row = int(item["row"])
        word = int(item["word"])
        chain = int(item["chain"])
        block = int(item["block"])
        new_data = int(item["new"])
        src_label = str(item.get("source_label", f"burn[{idx}]"))
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
        hard_denials, overridable_issues = ff_word_safety_audit(word, replay_old)
        if hard_denials:
            raise BurnError(f"{src_label} row {row}: REFUSED; " + "; ".join(hard_denials))
        # Reject unknown blocks and changed unmapped bits at import time too.
        non_optin_issues = [msg for msg in overridable_issues if "opt-in field" not in msg]
        if non_optin_issues:
            raise BurnError(f"{src_label} row {row}: REFUSED; " + "; ".join(non_optin_issues))

        # Place replayed word at next sequential slot (identical replay order; row may be +1 when prepending no-op).
        replay_slot = baseline_next_row + len(vfi_records)
        sim = simulate_ff_write(running_rows, replay_slot, word)
        running_rows[replay_slot] = word

        fields = list(item.get("fields") or [])
        if not fields:
            fields = list(sim["changes"].keys()) or [f.key for f in fields_in_block(chain, block)]
        is_pcie = ff_word_touches_pcie(word, replay_old) or any(
            FIELD_FUNCTIONAL_GROUP.get(k, FIELDS_BY_KEY[k].category) == "pcie"
            for k in fields
            if k in FIELDS_BY_KEY
        )

        # Find primary floorsweep mask changed by this record (if any).
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
        elif is_pcie:
            kind = "pcie"
        else:
            kind = "regular"

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
                # Never silently drop reconciliation metadata: fail closed.
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
        # Multi-field shared blocks cannot carry one mask_update; the engine re-derives and
        # defect-checks every field from the word itself. Always attach the SM count.
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
        # Only expect the new link capability if the plan actually burns a PCIe record.
        "LnkCap2": f"0x{(link_cap2_after if any(r.get('kind') == 'pcie' for r in vfi_records) else link_cap2_before):x}",
        "SM_COUNT": final_sm,
    }

    out: dict = {"schema_version": 1}
    if extra_metadata:
        out.update(extra_metadata)
    out.update(
        {
            "card": {
                "pci_id": SKU_CMP170HX_8GB,
                "die_rows": die_rows,
                "expected_masks": {k: f"0x{v:x}" for k, v in sorted(expected_masks.items())},
                "defect_masks": {k: f"0x{v:x}" for k, v in sorted(defect_masks.items())},
            },
            "fpf": {
                "first_row": FF_ROW_MIN,
                "last_row": FF_ROW_MAX,
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


def import_ffcalc_plan(
    burn_plan_path: str | Path,
    dump_path: str | Path,
    *,
    base_sm_ceiling: int,
    override_sm_ceiling: int | None = None,
    first_record_as_test: bool = False,
    prepend_noop_test: bool = True,
    link_cap2_before: int = 0x06,
    link_cap2_after: int = 0x1E,
) -> dict:
    """Convert a cmp170hx-ffcalc burn plan + 512-row card dump into a vfi_burn plan dict.

    Validates:
      - All 8 Die ID rows (8..15) are present in the card dump (refuses incomplete dumps).
      - Every planned FF record decodes cleanly (IFF bit 5 == 0) and matches chain/block/new.
      - The dump covers every FF row 222..486, and its used rows are one contiguous prefix.
      - Preserves the canonical occupied FF prefix (``baseline_ff_rows`` + SHA-256 digest +
        ``expected_next_row``) so live preflight proves the on-card history matches this dump.
      - Planned rows are contiguous, start at the first free FF row and never collide.
      - Each planned ``old`` value equals the replayed current block.
      - The dump's fused mask registers agree with the FF replay (model check).
      - No planned record touches read-only device-ID blocks, unknown blocks, or unmapped bits.
      - No planned record leaves any *_defective unit enabled (word-derived check).
      - Ensures the first record (``kind="test"``) is a demonstrable 0-bit-diff no-op.
      - Computes exact before/after disable masks and SM counts from full FF replay.

    ``base_sm_ceiling`` is required: it is community/operator knowledge the card cannot
    supply, and defaulting it to the full-die 128 made the check vacuous.
    """
    if isinstance(base_sm_ceiling, bool) or not isinstance(base_sm_ceiling, int) or base_sm_ceiling <= 0:
        raise BurnError("base_sm_ceiling must be a positive integer supplied by the operator")
    bp = require_private_file(burn_plan_path, "ffcalc burn plan")
    dp = require_private_file(dump_path, "ffcalc card dump")
    try:
        burn_obj = json.loads(bp.read_text(encoding="utf-8"))
        dump_obj = json.loads(dp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in ffcalc input: {exc}") from exc

    if burn_obj.get("uuid") and dump_obj.get("uuid") and burn_obj["uuid"] != dump_obj["uuid"]:
        raise BurnError(
            f"UUID mismatch between burn plan ({burn_obj['uuid']}) and dump ({dump_obj['uuid']})"
        )

    raw_ff_rows = dump_obj.get("ff_rows")
    if not isinstance(raw_ff_rows, dict):
        raise BurnError("ffcalc dump is missing 'ff_rows' (must run full 512-row capture)")

    ff_rows = {int(k): u32(v, f"dump.ff_rows.{k}") for k, v in raw_ff_rows.items()}
    missing_die = [r for r in DIE_ID_ROWS if r not in ff_rows]
    if missing_die:
        raise BurnError(
            f"ffcalc dump missing die ID rows {missing_die}; re-capture all 512 rows with --read-ff-rows"
        )
    die_rows_int = {r: ff_rows[r] for r in DIE_ID_ROWS}

    opt_words = dump_obj.get("option_words", {})
    if not isinstance(opt_words, dict):
        raise BurnError("ffcalc dump missing 'option_words'")

    # Extract cold fused masks and genuine defective masks for every floorsweep field.
    expected_masks: dict[str, int] = {}
    defect_masks: dict[str, int] = {}
    for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items():
        field = FIELDS_BY_KEY[ff_key]
        # Prefer __fused when present because OEM firmware can mask status registers (e.g. tpc3/6).
        raw_cur = opt_words.get(ff_key + "__fused", opt_words.get(ff_key))
        raw_def = opt_words.get(field.defect_key) if field.defect_key else None
        if raw_cur is not None and raw_def is not None:
            expected_masks[vfi_name] = u32(raw_cur, f"option_words.{ff_key}")
            defect_masks[vfi_name] = u32(raw_def, f"option_words.{field.defect_key}")

    burn_list = burn_obj.get("burn")
    if not isinstance(burn_list, list) or not burn_list:
        raise BurnError("ffcalc burn plan has no records in 'burn'")

    planned_items = []
    for idx, item in enumerate(burn_list):
        if not isinstance(item, dict):
            raise BurnError(f"burn[{idx}] must be a JSON object")
        planned_items.append(
            {
                "row": int(item["row"]),
                "chain": int(item["chain"]),
                "block": int(item["block"]),
                "old": u32(item["old"], f"burn[{idx}].old") if "old" in item else None,
                "new": u32(item["new"], f"burn[{idx}].new"),
                "word": u32(item["record"], f"burn[{idx}].record"),
                "fields": list(item.get("fields", [])),
                "source_label": f"burn[{idx}]",
            }
        )

    return _build_vfi_plan_from_records(
        ff_rows=ff_rows,
        die_rows_int=die_rows_int,
        expected_masks=expected_masks,
        defect_masks=defect_masks,
        planned_items=planned_items,
        base_sm_ceiling=base_sm_ceiling,
        override_sm_ceiling=override_sm_ceiling,
        prepend_noop_test=prepend_noop_test,
        allow_shifted_first_row=False,
        link_cap2_before=link_cap2_before,
        link_cap2_after=link_cap2_after,
        extra_metadata={
            "source_ffcalc_uuid": burn_obj.get("uuid") or dump_obj.get("uuid"),
            "source_ffcalc_fingerprint": burn_obj.get("fingerprint") or dump_obj.get("fingerprint"),
        },
        source_name="ffcalc",
    )


def _verify_ga100_expect_registers(
    expect_registers: dict,
    decoded: dict[str, int],
    plan_label: str,
) -> None:
    """Cross-check ga100-ff-tools expect_registers against independent FF replay."""
    for raw_key, raw_want in expect_registers.items():
        want = u32(raw_want, f"{plan_label}.expect_registers[{raw_key}]")
        key_upper = str(raw_key).strip().upper()
        if key_upper.startswith("DISABLE_GEN3_SPEED"):
            got = decoded["disable_gen3_speed_bit1"] | decoded["disable_gen3_speed_bit9"]
        elif key_upper.startswith("BOOT_GEN23_DISABLE"):
            got = decoded["pcie_boot_gen23_dis"]
        elif key_upper.startswith("BOOT_GEN3_DISABLE"):
            got = decoded["pcie_boot_gen3_dis"]
        elif key_upper.startswith("PCIE_MAGIC_BITS_D"):
            got = decoded["magic_d"]
        else:
            continue
        if got != want:
            raise BurnError(
                f"{plan_label}: expect_registers[{raw_key!r}] wants 0x{want:x} "
                f"but FF replay produces 0x{got:x}"
            )


def import_ga100_ff_tools_plan(
    plan_paths: Sequence[str | Path],
    dump_path: str | Path,
    defects_path: str | Path,
    *,
    base_sm_ceiling: int,
    override_sm_ceiling: int | None = None,
    die_rows: dict[int, int] | None = None,
    die_rows_path: str | Path | None = None,
    prepend_noop_test: bool | None = None,
    link_cap2_before: int = 0x06,
    link_cap2_after: int = 0x1E,
) -> dict:
    """Convert one or more kprinssu/ga100-ff-tools plan JSONs + ff_dump.py + read_defects.py
    into a single strict mode-0600 vfi_burn plan dict (schema_version=1).

    Multiple plans from the same ``ffcard.py derive`` run (e.g. ``0_noop``, ``sm_hbm``,
    ``pcie``, ``nvlink``) can be passed together; they are validated to share the same
    baseline (``last_row`` + 16-hex ``fingerprint``) and sorted by starting row.

    Die ID rows 8..15 are required by vfi_burn and NEVER fabricated. Because stock
    ``ff_dump.py`` defaults to ``--start 222``, supply them via one of:
      - rows 8..15 inside ``dump_path`` (``ff_dump.py --start 8``), OR
      - ``die_rows_path`` (a vfi_burn snapshot, ffcalc dump, ``ff_dump.py --start 8`` dump,
        or plain ``{"8": ..., "15": ...}`` JSON), OR
      - ``die_rows`` dict read read-only from the live card (CLI ``--bdf``).
    """
    if isinstance(base_sm_ceiling, bool) or not isinstance(base_sm_ceiling, int) or base_sm_ceiling <= 0:
        raise BurnError("base_sm_ceiling must be a positive integer supplied by the operator")
    if isinstance(plan_paths, (str, Path)):
        plan_paths = [plan_paths]
    if not plan_paths:
        raise BurnError("at least one ga100-ff-tools plan JSON path is required")

    # 1. Load & validate all ga100-ff-tools plan files.
    loaded_plans: list[tuple[Path, dict]] = []
    for raw_p in plan_paths:
        pp = require_private_file(raw_p, "ga100-ff-tools plan")
        try:
            obj = json.loads(pp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BurnError(f"invalid JSON in ga100-ff-tools plan {pp}: {exc}") from exc
        if not isinstance(obj, dict):
            raise BurnError(f"ga100-ff-tools plan {pp} root must be a JSON object")
        recs = obj.get("records")
        if not isinstance(recs, list) or not recs:
            raise BurnError(f"ga100-ff-tools plan {pp} has no records in 'records'")
        base = obj.get("baseline")
        if not isinstance(base, dict) or "last_row" not in base or "fingerprint" not in base:
            raise BurnError(
                f"ga100-ff-tools plan {pp} is missing 'baseline.last_row' / 'baseline.fingerprint'"
            )
        loaded_plans.append((pp, obj))

    first_base = loaded_plans[0][1]["baseline"]
    base_last_row = u32(first_base["last_row"], "baseline.last_row")
    base_fp = str(first_base["fingerprint"]).strip().lower()
    if not base_fp:
        raise BurnError("ga100-ff-tools plan has empty baseline.fingerprint")
    card_names = {str(pl.get("card")) for _, pl in loaded_plans if pl.get("card") is not None}
    if len(card_names) > 1:
        raise BurnError(f"ga100-ff-tools plans disagree on card name: {sorted(card_names)}")
    for pp, pl in loaded_plans[1:]:
        b = pl["baseline"]
        if u32(b["last_row"], f"{pp.name}.baseline.last_row") != base_last_row or (
            str(b["fingerprint"]).strip().lower() != base_fp
        ):
            raise BurnError(
                f"ga100-ff-tools plan {pp.name} has a different baseline than {loaded_plans[0][0].name}; "
                "all plans must come from the same ffcard.py derive run"
            )
    loaded_plans.sort(key=lambda pair: int(pair[1]["records"][0]["row"]))

    # 2. Load & validate ff_dump.py dump JSON.
    dp = require_private_file(dump_path, "ga100-ff-tools dump")
    try:
        dump_obj = json.loads(dp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in ga100-ff-tools dump {dp}: {exc}") from exc
    if not isinstance(dump_obj, dict):
        raise BurnError("ga100-ff-tools dump root must be a JSON object")
    if dump_obj.get("shadow_only") is True:
        raise BurnError("ga100-ff-tools dump was captured with --shadow-only (no FF rows)")
    if dump_obj.get("read_errors"):
        raise BurnError(f"ga100-ff-tools dump has read_errors on rows {dump_obj['read_errors']}")
    if dump_obj.get("controller_idle") is False:
        raise BurnError("ga100-ff-tools dump reports controller_idle=false")
    if dump_obj.get("crc_status") is not None:
        crc_val = u32(dump_obj["crc_status"], "dump.crc_status")
        if crc_val != 0x55:
            raise BurnError(f"ga100-ff-tools dump crc_status=0x{crc_val:02x} (expected 0x55)")

    raw_rows = dump_obj.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise BurnError("ga100-ff-tools dump is missing 'rows' list (from ff_dump.py)")
    ff_rows: dict[int, int] = {}
    for entry in raw_rows:
        if not isinstance(entry, dict) or "row" not in entry:
            raise BurnError("ga100-ff-tools dump 'rows' entries must be JSON objects with 'row'")
        r_idx = int(entry["row"])
        if entry.get("error") or entry.get("raw") is None:
            raise BurnError(f"ga100-ff-tools dump row {r_idx} was not read cleanly")
        ff_rows[r_idx] = u32(entry["raw"], f"dump.rows[{r_idx}]")
    for b_row in dump_obj.get("blank_rows", []):
        ff_rows.setdefault(int(b_row), FF_EMPTY_WORD)

    missing_ff = [r for r in range(FF_ROW_MIN, FF_ROW_MAX + 1) if r not in ff_rows]
    if missing_ff:
        raise BurnError(
            f"ga100-ff-tools dump covers only {FF_ROW_MAX - FF_ROW_MIN + 1 - len(missing_ff)} of "
            f"{FF_ROW_MAX - FF_ROW_MIN + 1} FF rows (missing e.g. {missing_ff[:5]}); "
            "an unread row would look free"
        )
    # Verify baseline last_row & 16-hex fingerprint against dump rows.
    used_rows = [r for r in range(FF_ROW_MIN, FF_ROW_MAX + 1) if ff_rows[r] != FF_EMPTY_WORD]
    if not used_rows:
        raise BurnError("ga100-ff-tools dump has no occupied FF rows")
    if used_rows != list(range(FF_ROW_MIN, used_rows[-1] + 1)):
        raise BurnError("ga100-ff-tools dump FF rows are non-contiguous; vfi_burn's free-slot proof would refuse")
    if used_rows[-1] != base_last_row:
        raise BurnError(
            f"ga100-ff-tools dump last occupied row is {used_rows[-1]}, "
            f"but plan baseline.last_row is {base_last_row} (stale dump or plan?)"
        )
    computed_fp = ga100_ff_tools_fingerprint(ff_rows, base_last_row)
    if computed_fp != base_fp:
        raise BurnError(
            f"ga100-ff-tools dump rows {FF_ROW_MIN}..{base_last_row} fingerprint {computed_fp} "
            f"!= plan baseline.fingerprint {base_fp}: dump does not match this plan's card/state"
        )

    # 3. Resolve Die ID rows 8..15 (never fabricated).
    resolved_die: dict[int, int] = {}
    if all(r in ff_rows for r in DIE_ID_ROWS):
        resolved_die = {r: ff_rows[r] for r in DIE_ID_ROWS}
    if die_rows_path is not None:
        file_die = load_die_rows_from_file(die_rows_path)
        if resolved_die and resolved_die != file_die:
            raise BurnError("die ID rows in dump disagree with --die-rows-from file")
        resolved_die = file_die
    if die_rows is not None:
        supplied_die = {int(k): u32(v, f"die_rows.{k}") for k, v in die_rows.items() if int(k) in DIE_ID_ROWS}
        missing_supplied = [r for r in DIE_ID_ROWS if r not in supplied_die]
        if missing_supplied:
            raise BurnError(f"supplied die_rows missing rows {missing_supplied}")
        if resolved_die and resolved_die != supplied_die:
            raise BurnError("live/supplied die ID rows disagree with dump/--die-rows-from die ID rows")
        resolved_die = supplied_die
    if not resolved_die:
        raise BurnError(
            "ga100-ff-tools ff_dump.py defaults to rows 222..487 and does not include Die ID rows 8..15; "
            "vfi_burn requires all 8 Die ID rows for card identity verification. Supply them via "
            "--bdf <BDF> (read-only live read), --die-rows-from <snapshot-or-die-dump.json>, "
            "or re-run ff_dump.py with --start 8 --end 487"
        )

    # 4. Load & validate read_defects.py JSON.
    dfp = require_private_file(defects_path, "ga100-ff-tools defects")
    try:
        defects_obj = json.loads(dfp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in ga100-ff-tools defects {dfp}: {exc}") from exc
    if not isinstance(defects_obj, dict) or not isinstance(defects_obj.get("registers"), list):
        raise BurnError("ga100-ff-tools defects file must contain a 'registers' list (from read_defects.py)")
    defect_regs: dict[str, int] = {}
    for entry in defects_obj["registers"]:
        if not isinstance(entry, dict) or "name" not in entry or "value" not in entry:
            raise BurnError("ga100-ff-tools defects 'registers' entries must have 'name' and 'value'")
        rname = str(entry["name"]).strip().upper()
        defect_regs[rname] = u32(entry["value"], f"defects.{rname}")

    missing_def = [name for name in GA100_FF_TOOLS_REQUIRED_DEFECTS if name not in defect_regs]
    if missing_def:
        raise BurnError(
            f"ga100-ff-tools defects file is missing required defect register(s): {', '.join(missing_def)}"
        )
    defect_masks: dict[str, int] = {
        vfi_field: defect_regs[reg_name]
        for reg_name, vfi_field in GA100_FF_TOOLS_DEFECT_TO_FIELD.items()
        if reg_name in defect_regs
    }

    # 5. Replay baseline FF rows and cross-check dump shadow + defects OPT_* registers.
    base_blocks = replay_ff_rows(ff_rows)
    replay_fields = decode_all_fields(base_blocks)

    shadow = dump_obj.get("shadow")
    if isinstance(shadow, dict):
        for opt_name, ff_key in GA100_FF_TOOLS_FIELD_TO_KEY.items():
            if opt_name in shadow and shadow[opt_name] is not None:
                sh_val = u32(shadow[opt_name], f"dump.shadow.{opt_name}")
                if sh_val != replay_fields[ff_key]:
                    raise BurnError(
                        f"ga100-ff-tools dump shadow {opt_name}=0x{sh_val:x} disagrees with FF replay "
                        f"0x{replay_fields[ff_key]:x} (was ff_dump.py captured before SENSE_CTRL or on another card?)"
                    )
    for opt_name in ("OPT_PES_DISABLE", "OPT_ROP_L2_DISABLE"):
        if opt_name in defect_regs:
            ff_key = GA100_FF_TOOLS_FIELD_TO_KEY[opt_name]
            if defect_regs[opt_name] != replay_fields[ff_key]:
                raise BurnError(
                    f"ga100-ff-tools defects {opt_name}=0x{defect_regs[opt_name]:x} disagrees with FF replay "
                    f"0x{replay_fields[ff_key]:x} (defects file from a different card/state?)"
                )

    expected_masks: dict[str, int] = {
        vfi_name: replay_fields[ff_key]
        for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items()
        if vfi_name in defect_masks
    }

    # 6. Flatten records across the sorted plans and track per-subplan end indices for expectation checks.
    planned_items: list[dict] = []
    subplan_end_checks: dict[int, tuple[str, dict]] = {}
    modes: list[str] = []
    for pp, pl in loaded_plans:
        mode = str(pl.get("mode", pp.stem))
        modes.append(mode)
        for j, rec in enumerate(pl["records"]):
            if not isinstance(rec, dict):
                raise BurnError(f"{pp.name}.records[{j}] must be a JSON object")
            for req_k in ("chain", "block", "new_data", "encoded", "row"):
                if req_k not in rec:
                    raise BurnError(f"{pp.name}.records[{j}] missing required key {req_k!r}")
            planned_items.append(
                {
                    "row": int(rec["row"]),
                    "chain": int(rec["chain"]),
                    "block": int(rec["block"]),
                    "old": u32(rec["old_data"], f"{pp.name}.records[{j}].old_data")
                    if "old_data" in rec
                    else None,
                    "new": u32(rec["new_data"], f"{pp.name}.records[{j}].new_data"),
                    "word": u32(rec["encoded"], f"{pp.name}.records[{j}].encoded"),
                    "fields": [],
                    "note": str(rec["note"]).strip() if rec.get("note") else None,
                    "source_label": f"{pp.name}[{j}]",
                }
            )
        subplan_end_checks[len(planned_items) - 1] = (pp.name, pl)

    # 7. Auto-detect whether record 0 is already a verified 0-bit-diff no-op (e.g. *_plan_0_noop.json).
    first_is_noop = False
    try:
        verify_noop_test_record(planned_items[0]["word"], base_blocks)
        first_is_noop = True
    except BurnError:
        first_is_noop = False

    if prepend_noop_test is None:
        do_prepend = not first_is_noop
    elif prepend_noop_test is True:
        do_prepend = not first_is_noop
    else:
        do_prepend = False

    def _check_subplan_expectations(idx: int, cur_rows: dict[int, int]) -> None:
        if idx not in subplan_end_checks:
            return
        pname, pl = subplan_end_checks[idx]
        cur_decoded = decode_all_fields(replay_ff_rows(cur_rows))
        exp_fields = pl.get("expect_fields")
        if isinstance(exp_fields, dict):
            for opt_name, raw_want in exp_fields.items():
                ff_key = GA100_FF_TOOLS_FIELD_TO_KEY.get(str(opt_name).strip().upper())
                if ff_key is None:
                    continue
                want = u32(raw_want, f"{pname}.expect_fields.{opt_name}")
                got = cur_decoded[ff_key]
                if got != want:
                    raise BurnError(
                        f"{pname}: expect_fields.{opt_name}=0x{want:x} disagrees with "
                        f"independent FF replay 0x{got:x}"
                    )
        unch = pl.get("unchanged_fields")
        if isinstance(unch, dict):
            for opt_name, raw_want in unch.items():
                ff_key = GA100_FF_TOOLS_FIELD_TO_KEY.get(str(opt_name).strip().upper())
                if ff_key is None:
                    continue
                want = u32(raw_want, f"{pname}.unchanged_fields.{opt_name}")
                got = cur_decoded[ff_key]
                if got != want:
                    raise BurnError(
                        f"{pname}: unchanged_fields.{opt_name}=0x{want:x} changed to 0x{got:x} during FF replay"
                    )
        exp_regs = pl.get("expect_registers")
        if isinstance(exp_regs, dict):
            _verify_ga100_expect_registers(exp_regs, cur_decoded, pname)

    return _build_vfi_plan_from_records(
        ff_rows=ff_rows,
        die_rows_int=resolved_die,
        expected_masks=expected_masks,
        defect_masks=defect_masks,
        planned_items=planned_items,
        base_sm_ceiling=base_sm_ceiling,
        override_sm_ceiling=override_sm_ceiling,
        prepend_noop_test=do_prepend,
        allow_shifted_first_row=True,
        link_cap2_before=link_cap2_before,
        link_cap2_after=link_cap2_after,
        extra_metadata={
            "source_format": "ga100-ff-tools",
            "source_ga100_ff_tools_card": next(iter(card_names), None),
            "source_ga100_ff_tools_modes": modes,
            "source_ga100_ff_tools_baseline": {
                "last_row": base_last_row,
                "fingerprint": base_fp,
            },
        },
        source_name="ga100-ff-tools",
        on_record_replayed=_check_subplan_expectations,
    )
