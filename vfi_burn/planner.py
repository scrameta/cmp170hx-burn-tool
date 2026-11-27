"""Native GA100 (CMP 170HX 8 GB) fuse-unlock planner.

Reads current FF rows, Die IDs (rows 8..15), fused option shadows, and all 16 genuine
silicon defect masks directly from a live GPU (read-only) or simulated snapshot, lets the
operator choose which healthy subsystems / GPCs / TPCs / FBPs / PCIe speeds / NVLink to
unlock (or lock), and derives exact, coherence-checked GA100 FF records without any
external planner or import step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ffrecord import (
    FF_ROW_MAX,
    FF_ROW_MIN,
    build_vfi_plan_from_records,
    compute_sm_count,
    decode_all_fields,
    encode_ff,
    ff_word_safety_audit,
    fields_in_block,
    replay_ff_rows,
)
from .fuse_rules import CombinationIssue, evaluate_fuse_combinations
from .fuses import (
    DEVICE_ID_DENIED_BLOCKS,
    DIE_ID_ROWS,
    FFCALC_KEY_TO_FIELD,
    FF_EMPTY_WORD,
    FIELDS_BY_KEY,
    FUSES_BY_KEY,
    PES_PER_GPC,
    SM_PER_TPC,
    TOTAL_FBP,
    TOTAL_GPC,
    TPC_PER_GPC,
)
from .ga100 import read_ga100_card_state
from .model import BurnError, SUPPORTED_CMP170HX_SKUS

PCIE_MODES = ("keep", "gen3", "gen4", "lock_gen12")

# Known GA100 PCIe Gen4 MAGIC_BITS_D clear masks (proven on CMP 170HX -> 0x00200000):
#   c2/b47: bits 0, 3, 4, 6 (mask 0x0059 -> MAGIC_BITS_D bits 22, 25, 26, 28)
#   c2/b46: bit 13          (mask 0x2000 -> MAGIC_BITS_D bit 19)
_PCIE_GEN4_B47_CLEAR = 0x0059
_PCIE_GEN4_B46_CLEAR = 0x2000


@dataclass(frozen=True)
class CardSnapshot:
    """Verified read-only snapshot of a CMP 170HX 8 GB card used to derive unlock plans."""

    pci_id: str
    die_rows: dict[int, int]
    ff_rows: dict[int, int]
    current_masks: dict[str, int]
    defect_masks: dict[str, int]
    current_blocks: dict[int, dict[int, int]]
    current_fields: dict[str, int]
    next_free_row: int
    lnkcap2: int = 0x06

    @property
    def free_row_count(self) -> int:
        return FF_ROW_MAX - self.next_free_row + 1

    def gpc_defective(self, gpc: int) -> bool:
        return bool((self.defect_masks.get("GPC_DISABLE", 0) >> gpc) & 1)

    def tpc_defective(self, gpc: int, tpc: int) -> bool:
        if self.gpc_defective(gpc):
            return True
        return bool((self.defect_masks.get(f"TPC_GPC{gpc}_DISABLE", 0) >> tpc) & 1)

    def pes_defective_mask(self, gpc: int) -> int:
        return (self.defect_masks.get("PES_DISABLE", 0) >> (gpc * PES_PER_GPC)) & ((1 << PES_PER_GPC) - 1)

    def fbp_defective(self, fbp: int) -> bool:
        return bool((self.defect_masks.get("FBP_DISABLE", 0) >> fbp) & 1)

    def nvlink_all_defective(self) -> bool:
        return (self.defect_masks.get("NVLINK_DISABLE", 0) & 0x7) == 0x7

    def gpc_currently_enabled(self, gpc: int) -> bool:
        return not bool((self.current_fields["gpc_disable"] >> gpc) & 1)

    def tpc_currently_enabled(self, gpc: int, tpc: int) -> bool:
        if not self.gpc_currently_enabled(gpc):
            return False
        return not bool((self.current_fields[f"tpc_gpc{gpc}_disable"] >> tpc) & 1)

    def fbp_currently_enabled(self, fbp: int) -> bool:
        return not bool((self.current_fields["fbp_disable"] >> fbp) & 1)

    def nvlink_currently_enabled(self) -> bool:
        return (self.current_fields["nvlink_disable"] & 0x7) != 0x7

    def current_pcie_label(self) -> str:
        gen3_dis = (
            self.current_fields["disable_gen3_speed_bit1"]
            | self.current_fields["disable_gen3_speed_bit9"]
            | self.current_fields["pcie_boot_gen23_dis"]
            | self.current_fields["pcie_boot_gen3_dis"]
        )
        if gen3_dis:
            return "Gen1/Gen2 (restricted)"
        b47 = self.current_blocks.get(2, {}).get(47, 0)
        b46 = self.current_blocks.get(2, {}).get(46, 0)
        if (b47 & _PCIE_GEN4_B47_CLEAR) == 0 and (b46 & _PCIE_GEN4_B46_CLEAR) == 0:
            return "Gen4"
        return "Gen3"

    def current_sm_count(self) -> int:
        return compute_sm_count(
            self.current_fields["gpc_disable"],
            [self.current_fields[f"tpc_gpc{g}_disable"] for g in range(TOTAL_GPC)],
        )

    def max_healthy_sm_count(self) -> int:
        return compute_sm_count(
            self.defect_masks.get("GPC_DISABLE", 0),
            [self.defect_masks.get(f"TPC_GPC{g}_DISABLE", 0) for g in range(TOTAL_GPC)],
        )

    def current_fbp_count(self) -> int:
        return sum(1 for i in range(TOTAL_FBP) if self.fbp_currently_enabled(i))

    def max_healthy_fbp_count(self) -> int:
        return sum(1 for i in range(TOTAL_FBP) if not self.fbp_defective(i))


def snapshot_from_raw_state(
    *,
    pci_id: str,
    die_rows: dict[int, int],
    ff_rows: dict[int, int],
    defect_masks: dict[str, int],
    shadow_masks: dict[str, int] | None = None,
    lnkcap2: int = 0x06,
) -> CardSnapshot:
    """Validate raw card readout dicts and construct a ``CardSnapshot``."""
    pci = str(pci_id).strip().lower()
    if pci not in SUPPORTED_CMP170HX_SKUS:
        raise BurnError(
            f"unexpected PCI ID {pci} (expected {' or '.join(SUPPORTED_CMP170HX_SKUS)})"
        )
    missing_die = [r for r in DIE_ID_ROWS if r not in die_rows]
    if missing_die:
        raise BurnError(f"missing Die ID rows {missing_die}")
    missing_ff = [r for r in range(FF_ROW_MIN, FF_ROW_MAX + 1) if r not in ff_rows]
    if missing_ff:
        raise BurnError(
            f"FF row readout incomplete (missing {len(missing_ff)} rows, e.g. {missing_ff[:5]})"
        )
    region = [ff_rows[r] for r in range(FF_ROW_MIN, FF_ROW_MAX + 1)]
    first_free_idx = next((i for i, v in enumerate(region) if v == FF_EMPTY_WORD), None)
    if first_free_idx is None:
        raise BurnError("FF region (rows 222..486) is completely full")
    if any(v != FF_EMPTY_WORD for v in region[first_free_idx:]):
        raise BurnError("FF rows are non-contiguous; cannot safely derive an append plan")
    next_free_row = FF_ROW_MIN + first_free_idx
    for r in range(FF_ROW_MIN, next_free_row):
        if (ff_rows[r] >> 5) & 1:
            raise BurnError(f"FF row {r} (0x{ff_rows[r]:08x}) has IFF bit 5 set")

    blocks = replay_ff_rows(ff_rows)
    decoded = decode_all_fields(blocks)

    if shadow_masks is not None:
        for ff_key, sh_val in shadow_masks.items():
            if ff_key in FFCALC_KEY_TO_FIELD and ff_key in decoded and sh_val != decoded[ff_key]:
                raise BurnError(
                    f"fused shadow {ff_key}=0x{sh_val:x} disagrees with FF replay 0x{decoded[ff_key]:x} "
                    "(was the card modified without a cold boot, or is the FF-only model invalid here?)"
                )

    current_masks = {
        vfi_name: decoded[ff_key]
        for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items()
        if vfi_name in defect_masks
    }
    return CardSnapshot(
        pci_id=pci,
        die_rows={r: int(die_rows[r]) & 0xFFFFFFFF for r in DIE_ID_ROWS},
        ff_rows={r: int(ff_rows[r]) & 0xFFFFFFFF for r in range(FF_ROW_MIN, FF_ROW_MAX + 1)},
        current_masks=current_masks,
        defect_masks={k: int(v) & 0xFFFFFFFF for k, v in defect_masks.items()},
        current_blocks=blocks,
        current_fields=decoded,
        next_free_row=next_free_row,
        lnkcap2=int(lnkcap2) & 0xFFFFFFFF,
    )


def capture_card_snapshot(backend, *, label: str = "GPU") -> CardSnapshot:
    """Capture a verified ``CardSnapshot`` from a live or simulated ``FuseReader`` backend."""
    # Try full hardware readout first (works on GA100LiveAdapter and full BAR0 mocks).
    try:
        state = read_ga100_card_state(backend, label=label)
        return snapshot_from_raw_state(
            pci_id=state["pci_id"],
            die_rows=state["die_rows"],
            ff_rows=state["ff_rows"],
            defect_masks=state["defect_masks"],
            shadow_masks=state["shadow_masks"],
            lnkcap2=state["lnkcap2"],
        )
    except BurnError:
        # Let genuine hardware/unsensed/poison errors propagate immediately.
        raise
    except Exception as exc:
        raise BurnError(f"{label}: failed to capture card state ({exc})") from exc


@dataclass
class UnlockConfig:
    """Mutable operator selection of desired GPU unit states (supports both unlock and lock)."""

    card: CardSnapshot
    gpc_enabled: list[bool] = field(default_factory=list)
    tpc_enabled: list[list[bool]] = field(default_factory=list)
    fbp_enabled: list[bool] = field(default_factory=list)
    pcie_mode: str = "keep"  # "keep" | "gen3" | "gen4" | "lock_gen12"
    nvlink_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.gpc_enabled:
            self.reset_all()

    @classmethod
    def from_card(cls, card: CardSnapshot) -> "UnlockConfig":
        return cls(card=card)

    def reset_compute(self) -> None:
        self.gpc_enabled = [self.card.gpc_currently_enabled(g) for g in range(TOTAL_GPC)]
        self.tpc_enabled = [
            [self.card.tpc_currently_enabled(g, t) for t in range(TPC_PER_GPC)]
            for g in range(TOTAL_GPC)
        ]

    def reset_memory(self) -> None:
        self.fbp_enabled = [self.card.fbp_currently_enabled(i) for i in range(TOTAL_FBP)]

    def reset_all(self) -> None:
        self.reset_compute()
        self.reset_memory()
        self.pcie_mode = "keep"
        self.nvlink_enabled = self.card.nvlink_currently_enabled()

    # ------------------------------------------------------------------ Compute mutators
    def set_gpc(self, gpc: int, enabled: bool) -> None:
        if not (0 <= gpc < TOTAL_GPC):
            raise BurnError(f"GPC index out of range 0..{TOTAL_GPC - 1}: {gpc}")
        if enabled:
            if self.card.gpc_defective(gpc):
                raise BurnError(f"GPC{gpc} is genuinely defective in OPT_GPC_DEFECTIVE — cannot enable")
            healthy_tpcs = [t for t in range(TPC_PER_GPC) if not self.card.tpc_defective(gpc, t)]
            if not healthy_tpcs:
                raise BurnError(f"GPC{gpc} has zero non-defective TPCs — cannot enable")
            self.gpc_enabled[gpc] = True
            for t in range(TPC_PER_GPC):
                self.tpc_enabled[gpc][t] = not self.card.tpc_defective(gpc, t)
        else:
            self.gpc_enabled[gpc] = False
            for t in range(TPC_PER_GPC):
                self.tpc_enabled[gpc][t] = False

    def set_tpc(self, gpc: int, tpc: int, enabled: bool) -> None:
        if not (0 <= gpc < TOTAL_GPC) or not (0 <= tpc < TPC_PER_GPC):
            raise BurnError(f"TPC index out of range: GPC{gpc}/TPC{tpc}")
        if enabled:
            if self.card.tpc_defective(gpc, tpc):
                raise BurnError(
                    f"GPC{gpc}/TPC{tpc} is genuinely defective — cannot enable"
                )
            self.gpc_enabled[gpc] = True
            self.tpc_enabled[gpc][tpc] = True
        else:
            self.tpc_enabled[gpc][tpc] = False
            # Keep GPC coherent: disabling all TPCs disables the parent GPC too.
            if not any(self.tpc_enabled[gpc]):
                self.gpc_enabled[gpc] = False

    def unlock_all_healthy_compute(self) -> None:
        for g in range(TOTAL_GPC):
            if not self.card.gpc_defective(g) and any(
                not self.card.tpc_defective(g, t) for t in range(TPC_PER_GPC)
            ):
                self.set_gpc(g, True)
            else:
                self.set_gpc(g, False)

    def is_all_healthy_compute_unlocked(self) -> bool:
        for g in range(TOTAL_GPC):
            want_g = (not self.card.gpc_defective(g)) and any(
                not self.card.tpc_defective(g, t) for t in range(TPC_PER_GPC)
            )
            if self.gpc_enabled[g] != want_g:
                return False
            for t in range(TPC_PER_GPC):
                want_t = want_g and (not self.card.tpc_defective(g, t))
                if self.tpc_enabled[g][t] != want_t:
                    return False
        return True

    # ------------------------------------------------------------------ Memory mutators
    def set_fbp(self, fbp: int, enabled: bool) -> None:
        if not (0 <= fbp < TOTAL_FBP):
            raise BurnError(f"FBP index out of range 0..{TOTAL_FBP - 1}: {fbp}")
        if enabled and self.card.fbp_defective(fbp):
            raise BurnError(f"FBP{fbp} is genuinely defective in OPT_FBP_DEFECTIVE — cannot enable")
        self.fbp_enabled[fbp] = bool(enabled)

    def unlock_all_healthy_memory(self) -> None:
        for i in range(TOTAL_FBP):
            self.fbp_enabled[i] = not self.card.fbp_defective(i)

    def is_all_healthy_memory_unlocked(self) -> bool:
        return all(self.fbp_enabled[i] == (not self.card.fbp_defective(i)) for i in range(TOTAL_FBP))

    def unlock_all_healthy_sm_hbm(self) -> None:
        self.unlock_all_healthy_compute()
        self.unlock_all_healthy_memory()

    def is_all_healthy_sm_hbm_unlocked(self) -> bool:
        return self.is_all_healthy_compute_unlocked() and self.is_all_healthy_memory_unlocked()

    # ------------------------------------------------------------------ PCIe / NVLink
    def set_pcie_mode(self, mode: str) -> None:
        m = mode.strip().lower()
        if m not in PCIE_MODES:
            raise BurnError(f"unknown PCIe mode {mode!r}; expected one of {', '.join(PCIE_MODES)}")
        self.pcie_mode = m

    def set_nvlink(self, enabled: bool) -> None:
        if enabled and self.card.nvlink_all_defective():
            raise BurnError("all NVLink links are genuinely defective in OPT_NVLINK_DEFECTIVE — cannot enable")
        self.nvlink_enabled = bool(enabled)

    # ------------------------------------------------------------------ Target masks & readouts
    def target_sm_count(self) -> int:
        return sum(
            SM_PER_TPC
            for g in range(TOTAL_GPC)
            if self.gpc_enabled[g]
            for t in range(TPC_PER_GPC)
            if self.tpc_enabled[g][t]
        )

    def target_gpc_count(self) -> int:
        return sum(1 for en in self.gpc_enabled if en)

    def target_tpc_count(self) -> int:
        return sum(
            1
            for g in range(TOTAL_GPC)
            if self.gpc_enabled[g]
            for t in range(TPC_PER_GPC)
            if self.tpc_enabled[g][t]
        )

    def target_fbp_count(self) -> int:
        return sum(1 for en in self.fbp_enabled if en)

    def target_pcie_label(self) -> str:
        if self.pcie_mode == "keep":
            return f"keep ({self.card.current_pcie_label()})"
        if self.pcie_mode == "gen3":
            return "Gen3 (safe)"
        if self.pcie_mode == "gen4":
            return "Gen4"
        return "Lock Gen1/Gen2"

    def desired_floorsweep_masks(self) -> dict[str, int]:
        """Compute coherent target disable masks for all floorsweep fields."""
        c = self.card
        out: dict[str, int] = {}

        # 1. Compute: GPC, TPC0..7, PES (3 PES slices per GPC)
        gpc_dis = 0
        pes_dis = 0
        for g in range(TOTAL_GPC):
            if not self.gpc_enabled[g]:
                gpc_dis |= 1 << g
                out[f"tpc_gpc{g}_disable"] = 0xFF
                if c.gpc_currently_enabled(g):
                    # Explicitly locking an active GPC: mask all 3 of its PES slices too.
                    pes_bits = (1 << PES_PER_GPC) - 1
                else:
                    # Unchanged disabled GPC: preserve existing PES bits (or defect bits).
                    pes_bits = (c.current_fields["pes_disable"] >> (g * PES_PER_GPC)) & (
                        (1 << PES_PER_GPC) - 1
                    )
                pes_dis |= pes_bits << (g * PES_PER_GPC)
            else:
                tpc_mask = 0
                for t in range(TPC_PER_GPC):
                    if not self.tpc_enabled[g][t]:
                        tpc_mask |= 1 << t
                out[f"tpc_gpc{g}_disable"] = tpc_mask
                pes_dis |= c.pes_defective_mask(g) << (g * PES_PER_GPC)
        out["gpc_disable"] = gpc_dis
        out["pes_disable"] = pes_dis

        # 2. Memory: FBP + paired FBPA / FBIO / ROP_L2 slices (2 slices per FBP)
        fbp_dis = 0
        fbpa_dis = 0
        fbio_dis = 0
        rop_dis = 0
        for i in range(TOTAL_FBP):
            shift = 2 * i
            if self.fbp_enabled[i]:
                fbpa_pair = (c.defect_masks.get("FBPA_DISABLE", 0) >> shift) & 0x3
                fbio_pair = (c.defect_masks.get("FBIO_DISABLE", 0) >> shift) & 0x3
                rop_pair = (c.defect_masks.get("ROP_L2_DISABLE", 0) >> shift) & 0x3
            else:
                fbp_dis |= 1 << i
                if c.fbp_currently_enabled(i):
                    # Explicitly locking an active FBP: mask both paired slices coherently.
                    fbpa_pair = fbio_pair = rop_pair = 0x3
                else:
                    # Unchanged disabled FBP: preserve current slice bits to avoid spurious writes.
                    fbpa_pair = (c.current_fields["fbpa_disable"] >> shift) & 0x3
                    fbio_pair = (c.current_fields["fbio_disable"] >> shift) & 0x3
                    rop_pair = (c.current_fields["rop_l2_disable"] >> shift) & 0x3
            fbpa_dis |= fbpa_pair << shift
            fbio_dis |= fbio_pair << shift
            rop_dis |= rop_pair << shift
        out["fbp_disable"] = fbp_dis
        out["fbpa_disable"] = fbpa_dis
        out["fbio_disable"] = fbio_dis
        out["rop_l2_disable"] = rop_dis

        # 3. NVLink (3 bits in c1/b23)
        if self.nvlink_enabled:
            out["nvlink_disable"] = c.defect_masks.get("NVLINK_DISABLE", 0) & 0x7
        elif c.nvlink_currently_enabled():
            out["nvlink_disable"] = 0x7
        else:
            out["nvlink_disable"] = c.current_fields["nvlink_disable"] & 0x7

        return out

    def summary_line(self) -> str:
        cur_sm = self.card.current_sm_count()
        tgt_sm = self.target_sm_count()
        max_sm = self.card.max_healthy_sm_count()
        delta_sm = tgt_sm - cur_sm
        delta_s = f"+{delta_sm}" if delta_sm > 0 else str(delta_sm)
        cur_fbp = self.card.current_fbp_count()
        tgt_fbp = self.target_fbp_count()
        nvl = "ON" if self.nvlink_enabled else "OFF"
        return (
            f"SMs: {cur_sm}->{tgt_sm} ({delta_s}, healthy max {max_sm}) | "
            f"FBPs: {cur_fbp}->{tgt_fbp}/{TOTAL_FBP} | "
            f"PCIe: {self.target_pcie_label()} | NVLink: {nvl}"
        )


def _apply_field_to_blocks(
    current_blocks: dict[int, dict[int, int]],
    desired_blocks: dict[int, dict[int, int]],
    ff_key: str,
    target_val: int,
) -> None:
    """Overlay ``ff_key = target_val`` into ``desired_blocks``, hard-denying device-ID & absent blocks."""
    f = FIELDS_BY_KEY[ff_key]
    for offset in range(f.width):
        bit_pos = f.bit + offset
        blk, local_bit = divmod(bit_pos, 16)
        if (f.chain, blk) in DEVICE_ID_DENIED_BLOCKS:
            raise BurnError(f"REFUSED: {ff_key} overlaps read-only device-ID block c{f.chain}/b{blk}")
        if blk not in current_blocks.get(f.chain, {}):
            raise BurnError(
                f"REFUSED: cannot modify {ff_key} because c{f.chain}/b{blk} is absent from "
                "this card's FF prefix (neighboring bits in that 16-bit block are unknown)"
            )
        val = desired_blocks.setdefault(f.chain, {})[blk]
        if (target_val >> offset) & 1:
            val |= 1 << local_bit
        else:
            val &= ~(1 << local_bit)
        desired_blocks[f.chain][blk] = val & 0xFFFF


def derive_ff_records(card: CardSnapshot, config: UnlockConfig) -> tuple[list[dict], list[CombinationIssue]]:
    """Derive exact, ordered FF records and coherence evaluation for ``config`` on ``card``.

    Returns ``(planned_items, coherence_issues)`` where each item has
    ``chain``, ``block``, ``old``, ``new``, ``word``, ``fields``, and ``note``.
    Never mutates read-only device-ID blocks (c2/b37-38) or absent FF blocks.
    """
    desired_blocks: dict[int, dict[int, int]] = {c: dict(bm) for c, bm in card.current_blocks.items()}
    desired_masks = config.desired_floorsweep_masks()

    # 1. Apply floorsweep field changes (compute, memory, nvlink).
    sm_hbm_keys = (
        "fbio_disable",
        "fbpa_disable",
        "gpc_disable",
        *(f"tpc_gpc{i}_disable" for i in range(TOTAL_GPC)),
        "pes_disable",
        "fbp_disable",
        "rop_l2_disable",
    )
    for key in (*sm_hbm_keys, "nvlink_disable"):
        if desired_masks[key] != card.current_fields[key]:
            _apply_field_to_blocks(card.current_blocks, desired_blocks, key, desired_masks[key])

    # 2. Apply PCIe speed/boot changes (chain 2 blocks 39, 49, and optionally 47, 46 for Gen4).
    pcie_notes: dict[tuple[int, int], str] = {}
    if config.pcie_mode in ("gen3", "gen4", "lock_gen12"):
        for req_blk in (39, 49):
            if req_blk not in card.current_blocks.get(2, {}):
                raise BurnError(
                    f"REFUSED: cannot change PCIe mode to {config.pcie_mode} because c2/b{req_blk} is absent "
                    "from this card's FF prefix (neighboring bits like pcie_devidb[15] are unknown)"
                )
    if config.pcie_mode in ("gen3", "gen4"):
        for key in (
            "disable_gen3_speed_bit1",
            "disable_gen3_speed_bit9",
            "pcie_boot_gen23_dis",
            "pcie_boot_gen3_dis",
        ):
            if card.current_fields[key] != 0:
                _apply_field_to_blocks(card.current_blocks, desired_blocks, key, 0)
        pcie_notes[(2, 39)] = "DISABLE_GEN3_SPEED (bit1+bit9)"
        pcie_notes[(2, 49)] = "BOOT_GEN23 + BOOT_GEN3 disable"
        if config.pcie_mode == "gen4":
            for blk, clear_mask, note in (
                (47, _PCIE_GEN4_B47_CLEAR, "PCIE_MAGIC_BITS_D bits 25,22,26,28 (Gen4)"),
                (46, _PCIE_GEN4_B46_CLEAR, "PCIE_MAGIC_BITS_D bit19 (Gen4)"),
            ):
                if blk not in card.current_blocks.get(2, {}):
                    raise BurnError(
                        f"REFUSED: cannot enable PCIe Gen4 because c2/b{blk} is absent from "
                        "this card's FF prefix"
                    )
                cur = desired_blocks[2][blk]
                desired_blocks[2][blk] = cur & ~clear_mask
                pcie_notes[(2, blk)] = note
    elif config.pcie_mode == "lock_gen12":
        for key in (
            "disable_gen3_speed_bit1",
            "disable_gen3_speed_bit9",
            "pcie_boot_gen23_dis",
            "pcie_boot_gen3_dis",
        ):
            if card.current_fields[key] != 1:
                _apply_field_to_blocks(card.current_blocks, desired_blocks, key, 1)
        pcie_notes[(2, 39)] = "Lock DISABLE_GEN3_SPEED (Gen1/Gen2)"
        pcie_notes[(2, 49)] = "Lock BOOT_GEN23 + BOOT_GEN3 disable"

    # 3. Collect changed (chain, block) pairs in deterministic subsystem order:
    #    chain 1 SM/HBM (blocks < 23) -> chain 2 PCIe (39, 49, 47, 46, then any others) -> chain 1 NVLink (23) -> rest
    changed_pairs: list[tuple[int, int]] = []
    for chain, bm in desired_blocks.items():
        for blk, new_val in bm.items():
            old_val = card.current_blocks.get(chain, {}).get(blk)
            if old_val != new_val:
                changed_pairs.append((chain, blk))

    pcie_order = {39: 0, 49: 1, 47: 2, 46: 3}

    def _sort_key(cb: tuple[int, int]) -> tuple[int, int, int]:
        c, b = cb
        if c == 1 and b < 23:
            return (0, c, b)
        if c == 2:
            return (1, pcie_order.get(b, 100 + b), b)
        if c == 1 and b == 23:
            return (2, c, b)
        return (3, c, b)

    changed_pairs.sort(key=_sort_key)

    items: list[dict] = []
    for idx, (chain, blk) in enumerate(changed_pairs):
        old_val = card.current_blocks[chain][blk]
        new_val = desired_blocks[chain][blk]
        word = encode_ff(chain, blk, new_val)
        hard, over = ff_word_safety_audit(word, old_val)
        if hard:
            raise BurnError(f"c{chain}/b{blk}: REFUSED; " + "; ".join(hard))
        non_optin = [m for m in over if "opt-in field" not in m]
        if non_optin:
            raise BurnError(f"c{chain}/b{blk}: REFUSED; " + "; ".join(non_optin))
        items.append(
            {
                "row": card.next_free_row + 1 + idx,  # +1 reserves slot for automatic no-op test row
                "chain": chain,
                "block": blk,
                "old": old_val,
                "new": new_val,
                "word": word,
                "fields": [f.key for f in fields_in_block(chain, blk)],
                "note": pcie_notes.get((chain, blk)),
                "source_label": f"derived[{idx}]",
            }
        )

    final_decoded = decode_all_fields(desired_blocks)
    coherence_issues = evaluate_fuse_combinations(final_decoded)
    return items, coherence_issues


def build_plan_from_config(
    card: CardSnapshot,
    config: UnlockConfig,
    *,
    link_cap2_after: int | None = None,
) -> dict:
    """Build a complete, validated ``schema_version=1`` vfi_burn plan dict from ``config``.

    Automatically prepends a safe 0-bit-diff no-op test record (for flashing-time testing)
    and auto-computes ``ceiling.base_sm`` from this die's genuine defect masks.
    """
    items, issues = derive_ff_records(card, config)
    if not items:
        raise BurnError("no fuse changes selected — toggle at least one unit/feature to unlock or lock")
    errors = [iss.message for iss in issues if iss.severity == "error"]
    if errors:
        raise BurnError("selected configuration is incoherent: " + "; ".join(errors))

    if link_cap2_after is None:
        if config.pcie_mode == "gen4":
            link_cap2_after = 0x1E
        elif config.pcie_mode == "gen3":
            link_cap2_after = 0x0E
        elif config.pcie_mode == "lock_gen12":
            link_cap2_after = 0x06
        else:
            link_cap2_after = card.lnkcap2

    return build_vfi_plan_from_records(
        ff_rows=card.ff_rows,
        die_rows_int=card.die_rows,
        expected_masks=card.current_masks,
        defect_masks=card.defect_masks,
        planned_items=items,
        pci_id=card.pci_id,
        base_sm_ceiling=max(card.max_healthy_sm_count(), 1),
        prepend_noop_test=True,
        link_cap2_before=card.lnkcap2,
        link_cap2_after=link_cap2_after,
        extra_metadata={
            "source_format": "vfi_burn_native_planner",
            "configured_summary": config.summary_line(),
        },
        source_name="planner",
    )
