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

from dataclasses import dataclass
import json
from pathlib import Path

from .model import FF_ROW_MAX, FF_ROW_MIN, BurnError, SKU_CMP170HX_8GB, require_private_file, u32

FF_EMPTY_WORD = 0x00000000
DIE_ID_ROWS: tuple[int, ...] = tuple(range(8, 16))
TOTAL_FUSE_ROWS = 512

TOTAL_GPC = 8
TPC_PER_GPC = 8
SM_PER_TPC = 2
TOTAL_FBP = 12


@dataclass(frozen=True)
class FFRecord:
    chain: int
    block: int
    data: int

    @property
    def word(self) -> int:
        return encode_ff(self.chain, self.block, self.data)


@dataclass(frozen=True)
class OptionField:
    key: str
    name: str
    chain: int
    bit: int
    width: int
    category: str
    defect_key: str = ""
    note: str = ""
    burnable: bool = True
    opt_in: bool = False

    @property
    def blocks(self) -> tuple[int, ...]:
        return tuple(sorted({(self.bit + i) // 16 for i in range(self.width)}))


def _tpc_fields() -> list[OptionField]:
    return [
        OptionField(
            f"tpc_gpc{i}_disable",
            f"NV_FUSE_OPT_TPC_GPC{i}_DISABLE",
            1,
            187 + 8 * i,
            8,
            "floorsweep",
            f"tpc_gpc{i}_defective",
        )
        for i in range(TOTAL_GPC)
    ]


OPTION_FIELDS: tuple[OptionField, ...] = tuple(
    [
        OptionField("fbio_disable", "NV_FUSE_OPT_FBIO_DISABLE", 1, 83, 24, "floorsweep", "fbio_defective"),
        OptionField("fbpa_disable", "NV_FUSE_OPT_FBPA_DISABLE", 1, 107, 24, "floorsweep", "fbpa_defective"),
        OptionField("gpc_disable", "NV_FUSE_OPT_GPC_DISABLE", 1, 131, 8, "floorsweep", "gpc_defective"),
    ]
    + _tpc_fields()
    + [
        OptionField("pes_disable", "NV_FUSE_OPT_PES_DISABLE", 1, 251, 24, "floorsweep", "pes_defective"),
        OptionField("fbp_disable", "NV_FUSE_OPT_FBP_DISABLE", 1, 307, 12, "floorsweep", "fbp_defective"),
        OptionField("rop_l2_disable", "NV_FUSE_OPT_ROP_L2_DISABLE", 1, 319, 24, "floorsweep", "rop_l2_defective"),
        OptionField(
            "nvlink_disable",
            "NV_FUSE_OPT_NVLINK_DISABLE",
            1,
            368,
            3,
            "nvlink",
            "nvlink_defective",
            "NVLink additionally needs U11 bridge mod + firmware check bypass",
        ),
        OptionField(
            "sys_pipe_disable",
            "NV_FUSE_OPT_SYS_PIPE_DISABLE",
            1,
            372,
            8,
            "mig",
            "sys_pipe_defective",
            "MIG/ECC/NVDEC unlock untested per Kate's writeup",
        ),
        OptionField("pcie_devida", "NV_FUSE_OPT_PCIE_DEVIDA", 2, 593, 16, "identity", "", "device ID (read-only)", burnable=False),
        OptionField("pcie_devidb", "NV_FUSE_OPT_PCIE_DEVIDB", 2, 609, 16, "identity", "", "device ID (read-only)", burnable=False),
        OptionField("disable_gen3_speed_bit1", "NV_FUSE_OPT_DISABLE_GEN3_SPEED", 2, 39 * 16 + 1, 1, "pcie", "", "PCIe Gen3 restriction (bit 1)"),
        OptionField("disable_gen3_speed_bit9", "NV_FUSE_OPT_DISABLE_GEN3_SPEED", 2, 39 * 16 + 9, 1, "pcie", "", "PCIe Gen3 restriction (bit 9)"),
        OptionField("pcie_magic_gen4_dis", "NV_FUSE_OPT_PCIE_MAGIC_BITS_D[25]", 2, 47 * 16 + 3, 1, "pcie", "", "Gen4 speed-disable flag"),
        OptionField("pcie_boot_gen23_dis", "NV_FUSE_OPT_PCIE_BOOT_GEN23_DISABLE", 2, 49 * 16 + 6, 1, "pcie", ""),
        OptionField("pcie_boot_gen3_dis", "NV_FUSE_OPT_PCIE_BOOT_GEN3_DISABLE", 2, 49 * 16 + 7, 1, "pcie", ""),
        OptionField(
            "magic_d",
            "NV_FUSE_OPT_PCIE_MAGIC_BITS_D",
            2,
            730,
            32,
            "identity",
            "",
            "whole MAGIC_BITS_D word (opt-in; A100 value 0x00200000 is experimental)",
            opt_in=True,
        ),
    ]
)

FIELDS_BY_KEY: dict[str, OptionField] = {f.key: f for f in OPTION_FIELDS}

# Maps ffcalc lowercase option keys to vfi_burn canonical uppercase mask/field names.
FFCALC_KEY_TO_FIELD: dict[str, str] = {
    "gpc_disable": "GPC_DISABLE",
    "fbp_disable": "FBP_DISABLE",
    "fbio_disable": "FBIO_DISABLE",
    "fbpa_disable": "FBPA_DISABLE",
    "rop_l2_disable": "ROP_L2_DISABLE",
    "pes_disable": "PES_DISABLE",
    "nvlink_disable": "NVLINK_DISABLE",
    "sys_pipe_disable": "SYS_PIPE_DISABLE",
    **{f"tpc_gpc{i}_disable": f"TPC_GPC{i}_DISABLE" for i in range(TOTAL_GPC)},
}


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


def make_noop_test_record(ff_rows: dict[int, int]) -> dict:
    """Create a safe throwaway no-op FF record re-writing an existing chain-1 block unchanged.

    Mirrors kprinssu/ga100-ff-tools plan_0_noop.json: tests CMD_WRITE + CMD_READ + CRC 0x55
    + cold-boot row persistence while changing ZERO effective fuse bits.
    """
    blocks = replay_ff_rows(ff_rows)
    chain1 = blocks.get(1, {})
    if not chain1:
        raise BurnError("cannot derive no-op test record: no existing chain-1 FF rows in dump")
    blk = 7 if 7 in chain1 else sorted(chain1)[0]
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


def import_ffcalc_plan(
    burn_plan_path: str | Path,
    dump_path: str | Path,
    *,
    base_sm_ceiling: int,
    stock_mv: float = 787.5,
    tolerance_mv: float = 25.0,
    override_sm_ceiling: int | None = None,
    first_record_as_test: bool = True,
    prepend_noop_test: bool = False,
    link_cap2_before: int = 0x06,
    link_cap2_after: int = 0x1E,
) -> dict:
    """Convert a cmp170hx-ffcalc burn plan + 512-row card dump into a vfi_burn plan dict.

    Validates:
      - All 8 Die ID rows (8..15) are present in the card dump (refuses incomplete dumps).
      - Every planned FF record decodes cleanly (IFF bit 5 == 0) and matches chain/block/new.
      - The dump covers every FF row 222..486, and its used rows are one contiguous prefix.
      - Planned rows are contiguous, start at the first free FF row and never collide.
      - Each planned ``old`` value equals the replayed current block.
      - The dump's fused mask registers agree with the FF replay (model check).
      - No planned record leaves any *_defective unit enabled (word-derived check).
      - Computes exact before/after disable masks and SM counts from full FF replay.

    ``base_sm_ceiling`` is required: it is community/operator knowledge the card cannot
    supply (BURN_SPEC §4), and defaulting it to the full-die 128 made the check vacuous.
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
    die_rows = {str(r): f"0x{ff_rows[r]:08x}" for r in DIE_ID_ROWS}

    # Row occupancy must be fully known (mirrors cmp170hx-ffcalc ffrows.require).
    missing_ff = [r for r in range(FF_ROW_MIN, FF_ROW_MAX + 1) if r not in ff_rows]
    if missing_ff:
        raise BurnError(
            f"ffcalc dump covers only {FF_ROW_MAX - FF_ROW_MIN + 1 - len(missing_ff)} of "
            f"{FF_ROW_MAX - FF_ROW_MIN + 1} FF rows (missing e.g. {missing_ff[:5]}); "
            "an unread row would look free"
        )
    region = [ff_rows[r] for r in range(FF_ROW_MIN, FF_ROW_MAX + 1)]
    first_free_idx = next((i for i, v in enumerate(region) if v == FF_EMPTY_WORD), None)
    if first_free_idx is None:
        raise BurnError("ffcalc dump shows the FF region full")
    if any(v != FF_EMPTY_WORD for v in region[first_free_idx:]):
        raise BurnError("ffcalc dump FF rows are non-contiguous; vfi_burn's free-slot proof would refuse")
    next_row = FF_ROW_MIN + first_free_idx

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

    # Replay current FF rows and walk through every planned record.
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
        else:
            first_key = sorted(expected_masks)[0]
            noop["expected_cold"] = {first_key: f"0x{expected_masks[first_key]:x}"}
        vfi_records.append(noop)
        first_record_as_test = False
        next_row += 1  # the no-op occupies the first free row; planned rows shift by one

    planned_rows = [int(item["row"]) for item in burn_list]
    if len(set(planned_rows)) != len(planned_rows):
        raise BurnError("ffcalc burn plan uses the same row twice")
    expected_first = next_row - (1 if prepend_noop_test else 0)
    if planned_rows != list(range(expected_first, expected_first + len(planned_rows))):
        raise BurnError(
            f"ffcalc planned rows {planned_rows[:3]}... are not contiguous from the first free "
            f"FF row {expected_first} of this dump (stale dump or plan?)"
        )
    if (next_row + len(planned_rows) - 1) > FF_ROW_MAX:
        raise BurnError("plan does not fit in the remaining FF rows")

    for idx, item in enumerate(burn_list):
        row = int(item["row"])
        word = u32(item["record"], f"burn[{idx}].record")
        chain = int(item["chain"])
        block = int(item["block"])
        new_data = u32(item["new"], f"burn[{idx}].new")
        if encode_ff(chain, block, new_data) != word:
            raise BurnError(
                f"burn[{idx}] row {row}: record 0x{word:08x} disagrees with "
                f"chain={chain} block={block} new=0x{new_data:04x}"
            )
        if "old" in item:
            replay_old = replay_ff_rows(running_rows).get(chain, {}).get(block, 0)
            if u32(item["old"], f"burn[{idx}].old") != replay_old:
                raise BurnError(
                    f"burn[{idx}] row {row}: planned old=0x{u32(item['old'], 'old'):04x} != "
                    f"replayed current block 0x{replay_old:04x}"
                )
        sim = simulate_ff_write(running_rows, row, word)
        running_rows[row] = word  # rows validated free + contiguous above; no-op test is replay-neutral

        fields = list(item.get("fields", []))
        is_pcie = any(
            FIELDS_BY_KEY[k].category == "pcie" for k in fields if k in FIELDS_BY_KEY
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
                    f"burn[{idx}] row {row} clears genuinely defective {vfi_name} bits 0x{bad_bits:x}"
                )

        if idx == 0 and first_record_as_test:
            kind = "test"
        elif is_pcie:
            kind = "pcie"
        else:
            kind = "regular"

        primary_field = fields[0] if fields else f"chain{chain}_blk{block}"
        target_name = FFCALC_KEY_TO_FIELD.get(primary_field, primary_field.upper())
        decode_desc = (
            f"FF chain={chain} blk={block} 0x{sim['old_block']:04x}->0x{sim['new_block']:04x} "
            f"fields={','.join(fields) or 'raw'}"
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
                    f"burn[{idx}] row {row}: tracked {vfi_name}=0x{running_masks[vfi_name]:x} "
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
            rec_dict["expected_cold"][target_name] = f"0x{sim['new_block']:x}"
        if kind == "pcie":
            rec_dict["link_cap2_before"] = f"0x{link_cap2_before:x}"
            rec_dict["link_cap2_after"] = f"0x{link_cap2_after:x}"

        vfi_records.append(rec_dict)

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

    return {
        "schema_version": 1,
        "source_ffcalc_uuid": burn_obj.get("uuid") or dump_obj.get("uuid"),
        "source_ffcalc_fingerprint": burn_obj.get("fingerprint") or dump_obj.get("fingerprint"),
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
        },
        "environment": {
            "stock_mv": stock_mv,
            "tolerance_mv": tolerance_mv,
        },
        "ceiling": {
            "base_sm": base_sm_ceiling,
            "override_sm": override_sm_ceiling,
            "reconfig_semantics_verified": False,
        },
        "final_expected_cold": final_expected_cold,
        "records": vfi_records,
    }
