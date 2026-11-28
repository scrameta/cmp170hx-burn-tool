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
    decode_a100_devid_ff_word,
    decode_all_fields,
    encode_a100_devid_ff_words,
    encode_ff,
    fbp_defect_consistency_issues,
    ff_word_safety_audit,
    fields_in_block,
    replay_ff_rows,
)
from .fuse_rules import CombinationIssue, evaluate_fuse_combinations
from .fuses import (
    A100_SXM4_DEVICE_IDS,
    A100_TARGET_DEVICE_IDS,
    BURNABLE_TARGET_DEVICE_IDS,
    BUS_BITS_PER_FBP,
    CMP_VBIOS_GB_PER_FBP,
    CUDA_CORES_PER_SM,
    DEVICE_ID_DENIED_BLOCKS,
    DIE_ID_ROWS,
    FFCALC_KEY_TO_FIELD,
    FF_EMPTY_WORD,
    FIELDS_BY_KEY,
    FUSES_BY_KEY,
    MEDIA_ENGINE_EVIDENCE_REGS,
    PES_PER_GPC,
    SM_PER_TPC,
    TOTAL_FBP,
    TOTAL_GPC,
    TPC_PER_GPC,
    a100_gb_per_fbp_for_sku,
)
from .ga100 import read_ga100_card_state
from .model import (
    BurnError,
    Plan,
    SUPPORTED_CMP170HX_SKUS,
    sku_display_name,
)

PCIE_MODES = ("keep", "gen3", "gen4", "lock_gen12")

# Known GA100 PCIe Gen4 MAGIC_BITS_D clear masks (proven on CMP 170HX -> 0x00200000):
#   c2/b47: bits 0, 3, 4, 6 (mask 0x0059 -> MAGIC_BITS_D bits 22, 25, 26, 28)
#   c2/b46: bit 13          (mask 0x2000 -> MAGIC_BITS_D bit 19)
_PCIE_GEN4_B47_CLEAR = 0x0059
_PCIE_GEN4_B46_CLEAR = 0x2000


@dataclass(frozen=True)
class CardSnapshot:
    """Verified read-only snapshot of a CMP 170HX (8 GB / 10 GB) card used to derive unlock plans."""

    pci_id: str
    die_rows: dict[int, int]
    ff_rows: dict[int, int]
    current_masks: dict[str, int]
    defect_masks: dict[str, int]
    current_blocks: dict[int, dict[int, int]]
    current_fields: dict[str, int]
    next_free_row: int
    lnkcap2: int = 0x06
    ff_last_row: int = FF_ROW_MAX
    iff_record: int = 0
    # EXPERIMENTAL media-engine evidence {"NVDEC"|"NVJPG"|"OFA": (OPT, STATUS) or None}.
    # None (the whole dict) means the readout did not include these registers.
    engine_regs: dict[str, tuple[int, int] | None] | None = None

    @property
    def free_row_count(self) -> int:
        return self.ff_last_row - self.next_free_row + 1

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

    def sys_pipe_defective_mask(self) -> int:
        return self.defect_masks.get("SYS_PIPE_DISABLE", 0) & 0xFF

    def sys_pipe_current_disable(self) -> int:
        return self.current_fields["sys_pipe_disable"] & 0xFF

    def sys_pipe_unlockable_mask(self) -> int:
        return self.sys_pipe_current_disable() & ~self.sys_pipe_defective_mask() & 0xFF

    def current_sys_pipe_count(self) -> int:
        return 8 - bin(self.sys_pipe_current_disable()).count("1")

    def max_healthy_sys_pipe_count(self) -> int:
        return 8 - bin(self.sys_pipe_defective_mask()).count("1")

    # ------------------------------------------------------------------ EXPERIMENTAL media engines
    def media_engine_blockers(self, which: str) -> list[str]:
        """Reasons the EXPERIMENTAL ``which`` ("nvdec" | "nvjpg_ofa") unlock cannot be offered.

        These engines have no *_DEFECTIVE register, so nothing can prove a unit is
        physically present. The minimum evidence required here (ffcalc's rule) is that
        OPT_*_DISABLE and STATUS_OPT_* were both read and are equal, that OPT agrees with
        the FF replay, and that every FF block involved is already in the FF prefix.
        """
        if which == "nvdec":
            engines, keys, blocks = ("NVDEC",), ("nvdec_disable",), (8,)
        elif which == "nvjpg_ofa":
            engines = ("NVJPG", "OFA")
            keys = ("nvjpg_ofa_disable_bit146", "nvjpg_ofa_disable_bit371")
            blocks = (9, 23)
        else:
            raise BurnError(f"unknown media engine group {which!r}")
        reasons: list[str] = []
        regs = self.engine_regs or {}
        readings: dict[str, tuple[int, int]] = {}
        for eng in engines:
            pair = regs.get(eng)
            if pair is None:
                opt_a, st_a = MEDIA_ENGINE_EVIDENCE_REGS[eng]
                reasons.append(
                    f"{eng} OPT/STATUS (0x{opt_a:06x}/0x{st_a:06x}) not read; no evidence the unit exists"
                )
                continue
            opt, status = pair
            readings[eng] = pair
            if opt != status:
                reasons.append(
                    f"STATUS_OPT_{eng}=0x{status:x} != OPT_{eng}_DISABLE=0x{opt:x} "
                    "(unit(s) reported absent or defective)"
                )
        for blk in blocks:
            if blk not in self.current_blocks.get(1, {}):
                reasons.append(
                    f"c1/b{blk} is absent from this card's FF prefix (neighbouring bits unknown)"
                )
        if reasons:
            return reasons
        if which == "nvdec":
            if (readings["NVDEC"][0] & 0x1F) != self.current_fields["nvdec_disable"]:
                reasons.append(
                    f"OPT_NVDEC_DISABLE=0x{readings['NVDEC'][0]:x} disagrees with FF replay "
                    f"0x{self.current_fields['nvdec_disable']:x}"
                )
        else:
            opt_bits = sorted((readings["NVJPG"][0] & 1, readings["OFA"][0] & 1))
            ff_bits = sorted(self.current_fields[k] & 1 for k in keys)
            if opt_bits != ff_bits:
                reasons.append(
                    f"OPT_NVJPG/OFA_DISABLE {opt_bits} disagree with FF bits 146/371 {ff_bits}"
                )
        return reasons

    def media_engine_currently_disabled(self, which: str) -> bool:
        if which == "nvdec":
            return self.current_fields.get("nvdec_disable", 0) != 0
        if which == "nvjpg_ofa":
            return any(
                self.current_fields.get(k, 0)
                for k in ("nvjpg_ofa_disable_bit146", "nvjpg_ofa_disable_bit371")
            )
        raise BurnError(f"unknown media engine group {which!r}")

    def media_engine_evidence_line(self, which: str) -> str:
        engines = ("NVDEC",) if which == "nvdec" else ("NVJPG", "OFA")
        parts = []
        for eng in engines:
            pair = (self.engine_regs or {}).get(eng)
            parts.append(
                f"{eng} OPT/STATUS=" + ("not read" if pair is None else f"0x{pair[0]:x}/0x{pair[1]:x}")
            )
        return ", ".join(parts)

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

    def current_gpc_count(self) -> int:
        return sum(1 for g in range(TOTAL_GPC) if self.gpc_currently_enabled(g))

    def current_cuda_cores(self) -> int:
        return self.current_sm_count() * CUDA_CORES_PER_SM

    def max_healthy_cuda_cores(self) -> int:
        return self.max_healthy_sm_count() * CUDA_CORES_PER_SM

    def current_fbp_count(self) -> int:
        return sum(1 for i in range(TOTAL_FBP) if self.fbp_currently_enabled(i))

    def max_healthy_fbp_count(self) -> int:
        return sum(1 for i in range(TOTAL_FBP) if not self.fbp_defective(i))

    def a100_gb_per_fbp(self) -> int:
        return a100_gb_per_fbp_for_sku(self.pci_id) or 4

    def est_vram_cmp_gb(self, fbp_count: int) -> int:
        return fbp_count * CMP_VBIOS_GB_PER_FBP

    def est_vram_a100_gb(self, fbp_count: int) -> int:
        return fbp_count * self.a100_gb_per_fbp()

    def bus_width_bits(self, fbp_count: int) -> int:
        return fbp_count * BUS_BITS_PER_FBP


@dataclass(frozen=True)
class BeforeAfterSummary:
    """Structured Before/After hardware capability summary for UIs, Plan Review, and Cold Verify."""

    pci_id_before: str
    pci_id_after: str
    sku_before: str
    sku_after: str
    sm_before: int | None
    sm_after: int | None
    sm_max_healthy: int | None
    gpc_before: int | None
    gpc_after: int | None
    fbp_before: int | None
    fbp_after: int | None
    fbp_max_healthy: int | None
    a100_gb_per_fbp: int
    targeting_a100_vbios: bool
    pcie_before: str
    pcie_after: str
    nvlink_before: bool | None
    nvlink_after: bool | None
    sys_pipe_before: int | None
    sys_pipe_after: int | None
    sys_pipe_max: int | None
    nvdec_before: bool | None
    nvdec_after: bool | None
    nvjpg_ofa_before: bool | None
    nvjpg_ofa_after: bool | None

    @property
    def cuda_cores_before(self) -> int | None:
        return self.sm_before * CUDA_CORES_PER_SM if self.sm_before is not None else None

    @property
    def cuda_cores_after(self) -> int | None:
        return self.sm_after * CUDA_CORES_PER_SM if self.sm_after is not None else None

    @property
    def cuda_cores_max_healthy(self) -> int | None:
        return self.sm_max_healthy * CUDA_CORES_PER_SM if self.sm_max_healthy is not None else None

    @property
    def bus_width_before(self) -> int | None:
        return self.fbp_before * BUS_BITS_PER_FBP if self.fbp_before is not None else None

    @property
    def bus_width_after(self) -> int | None:
        return self.fbp_after * BUS_BITS_PER_FBP if self.fbp_after is not None else None

    @property
    def vram_cmp_before_gb(self) -> int | None:
        return self.fbp_before * CMP_VBIOS_GB_PER_FBP if self.fbp_before is not None else None

    @property
    def vram_cmp_after_gb(self) -> int | None:
        return self.fbp_after * CMP_VBIOS_GB_PER_FBP if self.fbp_after is not None else None

    @property
    def vram_a100_before_gb(self) -> int | None:
        return self.fbp_before * self.a100_gb_per_fbp if self.fbp_before is not None else None

    @property
    def vram_a100_after_gb(self) -> int | None:
        return self.fbp_after * self.a100_gb_per_fbp if self.fbp_after is not None else None

    def vram_label(self) -> str:
        if self.fbp_before is None or self.fbp_after is None:
            return "unknown"
        cmp_b, cmp_a = self.vram_cmp_before_gb, self.vram_cmp_after_gb
        a100_b, a100_a = self.vram_a100_before_gb, self.vram_a100_after_gb
        if self.targeting_a100_vbios:
            return f"{cmp_b} GB (CMP) -> {a100_a} GB (A100 VBIOS, {self.a100_gb_per_fbp} GB/FBP)"
        cmp_s = f"{cmp_b}->{cmp_a} GB" if cmp_b != cmp_a else f"{cmp_a} GB"
        a100_s = f"{a100_b}->{a100_a} GB" if a100_b != a100_a else f"{a100_a} GB"
        return f"{cmp_s} (CMP VBIOS) / {a100_s} (w/ A100 VBIOS)"

    def features_list(self) -> list[dict]:
        rows: list[dict] = []
        if self.nvlink_before is not None and self.nvlink_after is not None:
            rows.append({
                "name": "NVLink",
                "before": "ON" if self.nvlink_before else "OFF",
                "after": "ON" if self.nvlink_after else "OFF",
                "changed": self.nvlink_before != self.nvlink_after,
            })
        if self.sys_pipe_before is not None and self.sys_pipe_after is not None:
            mx = f"/{self.sys_pipe_max}" if self.sys_pipe_max is not None else "/8"
            rows.append({
                "name": "MIG / SYS_PIPE",
                "before": f"{self.sys_pipe_before}{mx} pipes",
                "after": f"{self.sys_pipe_after}{mx} pipes",
                "changed": self.sys_pipe_before != self.sys_pipe_after,
            })
        if self.nvdec_before is not None and self.nvdec_after is not None:
            rows.append({
                "name": "NVDEC x5 (exp)",
                "before": "ON" if self.nvdec_before else "OFF",
                "after": "ON" if self.nvdec_after else "OFF",
                "changed": self.nvdec_before != self.nvdec_after,
            })
        if self.nvjpg_ofa_before is not None and self.nvjpg_ofa_after is not None:
            rows.append({
                "name": "NVJPG+OFA (exp)",
                "before": "ON" if self.nvjpg_ofa_before else "OFF",
                "after": "ON" if self.nvjpg_ofa_after else "OFF",
                "changed": self.nvjpg_ofa_before != self.nvjpg_ofa_after,
            })
        rows.append({
            "name": "PCIe Device ID",
            "before": f"{self.pci_id_before} ({self.sku_before})",
            "after": f"{self.pci_id_after} ({self.sku_after})",
            "changed": self.pci_id_before != self.pci_id_after,
        })
        return rows

    def to_dict(self) -> dict:
        return {
            "pci_id_before": self.pci_id_before,
            "pci_id_after": self.pci_id_after,
            "sku_before": self.sku_before,
            "sku_after": self.sku_after,
            "sm_before": self.sm_before,
            "sm_after": self.sm_after,
            "sm_max_healthy": self.sm_max_healthy,
            "cuda_cores_before": self.cuda_cores_before,
            "cuda_cores_after": self.cuda_cores_after,
            "cuda_cores_max_healthy": self.cuda_cores_max_healthy,
            "gpc_before": self.gpc_before,
            "gpc_after": self.gpc_after,
            "fbp_before": self.fbp_before,
            "fbp_after": self.fbp_after,
            "fbp_max_healthy": self.fbp_max_healthy,
            "bus_width_before": self.bus_width_before,
            "bus_width_after": self.bus_width_after,
            "vram_cmp_before_gb": self.vram_cmp_before_gb,
            "vram_cmp_after_gb": self.vram_cmp_after_gb,
            "vram_a100_before_gb": self.vram_a100_before_gb,
            "vram_a100_after_gb": self.vram_a100_after_gb,
            "a100_gb_per_fbp": self.a100_gb_per_fbp,
            "targeting_a100_vbios": self.targeting_a100_vbios,
            "vram_label": self.vram_label(),
            "pcie_before": self.pcie_before,
            "pcie_after": self.pcie_after,
            "features": self.features_list(),
        }

    def format_lines(self) -> list[str]:
        lines = ["=== BEFORE / AFTER CAPABILITY SUMMARY ==="]
        if self.sm_before is not None and self.sm_after is not None:
            d_sm = self.sm_after - self.sm_before
            d_cc = (self.cuda_cores_after or 0) - (self.cuda_cores_before or 0)
            sign = "+" if d_sm >= 0 else ""
            max_s = (
                f", healthy max {self.sm_max_healthy} SMs / {self.cuda_cores_max_healthy:,} CUDA"
                if self.sm_max_healthy is not None
                else ""
            )
            lines.append(
                f"  CUDA Cores / SMs : {self.cuda_cores_before:,} -> {self.cuda_cores_after:,} CUDA cores "
                f"({sign}{d_cc:,}) | {self.sm_before} -> {self.sm_after} SMs ({sign}{d_sm}{max_s})"
            )
        if self.fbp_before is not None and self.fbp_after is not None:
            d_fbp = self.fbp_after - self.fbp_before
            sign_f = "+" if d_fbp >= 0 else ""
            lines.append(
                f"  Est. VRAM        : {self.vram_label()}"
            )
            lines.append(
                f"  Memory Bus Width : {self.bus_width_before}-bit -> {self.bus_width_after}-bit "
                f"({self.fbp_before} -> {self.fbp_after}/{TOTAL_FBP} FBPs, {sign_f}{d_fbp})"
            )
        lines.append(f"  PCIe Link Speed  : {self.pcie_before} -> {self.pcie_after}")
        feat_parts = []
        for f in self.features_list():
            if f["changed"]:
                feat_parts.append(f"{f['name']}: {f['before']} -> {f['after']}")
            elif f["name"] in ("NVLink", "PCIe Device ID"):
                feat_parts.append(f"{f['name']}: {f['after']}")
        if feat_parts:
            lines.append("  Features         : " + " | ".join(feat_parts))
        return lines


def snapshot_from_raw_state(
    *,
    pci_id: str,
    die_rows: dict[int, int],
    ff_rows: dict[int, int],
    defect_masks: dict[str, int],
    shadow_masks: dict[str, int] | None = None,
    lnkcap2: int = 0x06,
    ff_last_row: int | None = None,
    iff_record: int = 0,
    engine_regs: dict[str, tuple[int, int] | None] | None = None,
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

    eff_last_row = FF_ROW_MAX if ff_last_row is None else int(ff_last_row)
    iff_start = int(iff_record) & 0x1FF  # 9-bit start row field (factory 0x01FB01E8 -> 488)
    if ff_last_row is None and iff_record != 0 and FF_ROW_MIN < iff_start <= 512:
        eff_last_row = min(FF_ROW_MAX, iff_start - 1)
    if not (FF_ROW_MIN < eff_last_row <= FF_ROW_MAX):
        raise BurnError(f"invalid ff_last_row {eff_last_row} (must be in {FF_ROW_MIN + 1}..{FF_ROW_MAX})")

    missing_ff = [r for r in range(FF_ROW_MIN, eff_last_row + 1) if r not in ff_rows]
    if missing_ff:
        raise BurnError(
            f"FF row readout incomplete across {FF_ROW_MIN}..{eff_last_row} "
            f"(missing {len(missing_ff)} rows, e.g. {missing_ff[:5]})"
        )
    region = [ff_rows[r] for r in range(FF_ROW_MIN, eff_last_row + 1)]
    first_free_idx = next((i for i, v in enumerate(region) if v == FF_EMPTY_WORD), None)
    if first_free_idx is None:
        raise BurnError(f"FF region (rows {FF_ROW_MIN}..{eff_last_row}) is completely full")
    if any(v != FF_EMPTY_WORD for v in region[first_free_idx:]):
        runs: list[list[int]] = []
        for off, val in enumerate(region):
            if val != FF_EMPTY_WORD:
                r_idx = FF_ROW_MIN + off
                if runs and r_idx == runs[-1][1] + 1:
                    runs[-1][1] = r_idx
                else:
                    runs.append([r_idx, r_idx])
        run_str = ", ".join(f"{a}..{b}" if a != b else str(a) for a, b in runs)
        iff_note = f" (IFF_RECORD=0x{iff_record:08x})" if iff_record else ""
        raise BurnError(
            f"FF rows in {FF_ROW_MIN}..{eff_last_row} are non-contiguous "
            f"(non-zero runs: {run_str}{iff_note}); cannot safely derive an append plan"
        )
    next_free_row = FF_ROW_MIN + first_free_idx
    for r in range(FF_ROW_MIN, next_free_row):
        if (ff_rows[r] >> 5) & 1:
            raise BurnError(f"FF row {r} (0x{ff_rows[r]:08x}) has IFF bit 5 set")

    scoped_ff_rows = {r: int(ff_rows[r]) & 0xFFFFFFFF for r in range(FF_ROW_MIN, eff_last_row + 1)}
    blocks = replay_ff_rows(scoped_ff_rows)
    decoded = decode_all_fields(blocks)

    if shadow_masks is not None:
        for ff_key, sh_val in shadow_masks.items():
            if ff_key in FFCALC_KEY_TO_FIELD and ff_key in decoded and sh_val != decoded[ff_key]:
                iff_hint = (
                    f" — note: this card has persistent IFF records (IFF_RECORD=0x{iff_record:08x}, "
                    f"rows {iff_start}..{(iff_record >> 16) & 0x1FF}) which can override fuse shadows "
                    "at boot after FF replay"
                    if iff_record and iff_start <= FF_ROW_MAX
                    else " (was the card modified without a cold boot, or is the FF-only model invalid here?)"
                )
                raise BurnError(
                    f"fused shadow {ff_key}=0x{sh_val:x} disagrees with FF replay 0x{decoded[ff_key]:x}"
                    f"{iff_hint}"
                )

    fbp_issues = fbp_defect_consistency_issues(defect_masks)
    if fbp_issues:
        raise BurnError(
            "REFUSED: genuine memory defect masks are inconsistent ("
            + "; ".join(fbp_issues)
            + "). The four registers should carry one dead-FBP judgement; re-read the card "
            "and check the register map before planning anything."
        )

    current_masks = {
        vfi_name: decoded[ff_key]
        for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items()
        if vfi_name in defect_masks
    }
    return CardSnapshot(
        pci_id=pci,
        die_rows={r: int(die_rows[r]) & 0xFFFFFFFF for r in DIE_ID_ROWS},
        ff_rows=scoped_ff_rows,
        current_masks=current_masks,
        defect_masks={k: int(v) & 0xFFFFFFFF for k, v in defect_masks.items()},
        current_blocks=blocks,
        current_fields=decoded,
        next_free_row=next_free_row,
        lnkcap2=int(lnkcap2) & 0xFFFFFFFF,
        ff_last_row=eff_last_row,
        iff_record=int(iff_record) & 0xFFFFFFFF,
        engine_regs=(
            None
            if engine_regs is None
            else {
                k: (None if v is None else (int(v[0]) & 0xFFFFFFFF, int(v[1]) & 0xFFFFFFFF))
                for k, v in engine_regs.items()
            }
        ),
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
            ff_last_row=state.get("ff_last_row"),
            iff_record=state.get("iff_record", 0),
            engine_regs=state.get("engine_regs"),
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
    a100_devid: int | None = None  # None = keep current, or key in A100_TARGET_DEVICE_IDS
    sys_pipe_unlock: bool = False  # EXPERIMENTAL: clear non-defective SYS_PIPE_DISABLE bits (MIG)
    nvdec_unlock: bool = False  # EXPERIMENTAL: clear NVDEC_DISABLE (c1/b8 bits 11..15)
    nvjpg_ofa_unlock: bool = False  # EXPERIMENTAL: clear NVJPG/OFA pair (c1 bits 146 + 371)

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
        self.a100_devid = None
        self.sys_pipe_unlock = False
        self.nvdec_unlock = False
        self.nvjpg_ofa_unlock = False

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

    # ------------------------------------------------------------------ PCIe / NVLink / A100 Identity / MIG
    def set_pcie_mode(self, mode: str) -> None:
        m = mode.strip().lower()
        if m not in PCIE_MODES:
            raise BurnError(f"unknown PCIe mode {mode!r}; expected one of {', '.join(PCIE_MODES)}")
        self.pcie_mode = m

    def set_nvlink(self, enabled: bool) -> None:
        if enabled and self.card.nvlink_all_defective():
            raise BurnError("all NVLink links are genuinely defective in OPT_NVLINK_DEFECTIVE — cannot enable")
        self.nvlink_enabled = bool(enabled)

    def set_a100_devid(self, target_devid: int | None) -> None:
        offered = {d: n for d, n in BURNABLE_TARGET_DEVICE_IDS.items() if d not in A100_SXM4_DEVICE_IDS}
        if target_devid is not None and target_devid not in offered:
            supported = ", ".join(f"0x{d:04x}" for d in sorted(offered))
            extra = (
                " (A100 SXM4 IDs are not offered: the CMP 170HX is a PCIe board)"
                if target_devid in A100_SXM4_DEVICE_IDS
                else ""
            )
            raise BurnError(
                f"unsupported target Device ID 0x{target_devid:04x}; supported: {supported}{extra}"
            )
        self.a100_devid = target_devid

    def set_media_engine_unlock(self, which: str, enabled: bool) -> None:
        """EXPERIMENTAL: select clearing the NVDEC or the NVJPG/OFA disable bits."""
        if which not in ("nvdec", "nvjpg_ofa"):
            raise BurnError(f"unknown media engine group {which!r}")
        if enabled:
            if not self.card.media_engine_currently_disabled(which):
                raise BurnError(f"{which.upper()} is already enabled on this card")
            blockers = self.card.media_engine_blockers(which)
            if blockers:
                raise BurnError(f"{which.upper()} unlock not offered: " + "; ".join(blockers))
        setattr(self, f"{which}_unlock", bool(enabled))

    def set_sys_pipe_unlock(self, enabled: bool) -> None:
        if enabled and self.card.sys_pipe_unlockable_mask() == 0:
            raise BurnError(
                "all non-defective SYS_PIPE units are already enabled on this card"
            )
        self.sys_pipe_unlock = bool(enabled)

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

    def target_devid_label(self) -> str:
        if self.a100_devid is None:
            return f"keep ({self.card.pci_id})"
        return f"10de:{self.a100_devid:04x} ({BURNABLE_TARGET_DEVICE_IDS[self.a100_devid]})"

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

        # 4. MIG / SYS_PIPE (8 bits in c1/b23, experimental; never lock active pipes)
        if self.sys_pipe_unlock:
            out["sys_pipe_disable"] = c.sys_pipe_defective_mask()
        else:
            out["sys_pipe_disable"] = c.sys_pipe_current_disable()

        return out

    def before_after_summary(self) -> BeforeAfterSummary:
        c = self.card
        tgt_pci = f"10de:{self.a100_devid:04x}" if self.a100_devid is not None else c.pci_id
        targeting_a100 = (
            (self.a100_devid in A100_TARGET_DEVICE_IDS)
            if self.a100_devid is not None
            else (c.pci_id in ("10de:20f1", "10de:20b5", "10de:20b0", "10de:20b2"))
        )
        pcie_after = (
            c.current_pcie_label()
            if self.pcie_mode == "keep"
            else (
                "Gen3"
                if self.pcie_mode == "gen3"
                else ("Gen4" if self.pcie_mode == "gen4" else "Gen1/Gen2 (restricted)")
            )
        )
        nvdec_cur = not c.media_engine_currently_disabled("nvdec")
        nvjpg_cur = not c.media_engine_currently_disabled("nvjpg_ofa")
        return BeforeAfterSummary(
            pci_id_before=c.pci_id,
            pci_id_after=tgt_pci,
            sku_before=sku_display_name(c.pci_id),
            sku_after=sku_display_name(tgt_pci),
            sm_before=c.current_sm_count(),
            sm_after=self.target_sm_count(),
            sm_max_healthy=c.max_healthy_sm_count(),
            gpc_before=c.current_gpc_count(),
            gpc_after=self.target_gpc_count(),
            fbp_before=c.current_fbp_count(),
            fbp_after=self.target_fbp_count(),
            fbp_max_healthy=c.max_healthy_fbp_count(),
            a100_gb_per_fbp=c.a100_gb_per_fbp(),
            targeting_a100_vbios=targeting_a100,
            pcie_before=c.current_pcie_label(),
            pcie_after=pcie_after,
            nvlink_before=c.nvlink_currently_enabled(),
            nvlink_after=self.nvlink_enabled,
            sys_pipe_before=c.current_sys_pipe_count(),
            sys_pipe_after=(
                c.max_healthy_sys_pipe_count()
                if self.sys_pipe_unlock
                else c.current_sys_pipe_count()
            ),
            sys_pipe_max=c.max_healthy_sys_pipe_count(),
            nvdec_before=nvdec_cur,
            nvdec_after=(True if self.nvdec_unlock else nvdec_cur),
            nvjpg_ofa_before=nvjpg_cur,
            nvjpg_ofa_after=(True if self.nvjpg_ofa_unlock else nvjpg_cur),
        )

    def to_dict(self) -> dict:
        """Return a complete JSON-serializable view of card silicon + current unlock config."""
        c = self.card
        gpcs = []
        for g in range(TOTAL_GPC):
            tpcs = []
            for t in range(TPC_PER_GPC):
                tpcs.append({
                    "index": t,
                    "defective": c.tpc_defective(g, t),
                    "current": c.tpc_currently_enabled(g, t),
                    "target": bool(self.gpc_enabled[g] and self.tpc_enabled[g][t]),
                })
            gpcs.append({
                "index": g,
                "defective": c.gpc_defective(g),
                "current": c.gpc_currently_enabled(g),
                "target": bool(self.gpc_enabled[g]),
                "tpcs": tpcs,
            })
        fbps = []
        for i in range(TOTAL_FBP):
            fbps.append({
                "index": i,
                "stack": i // 2,
                "slices": [2 * i, 2 * i + 1],
                "defective": c.fbp_defective(i),
                "current": c.fbp_currently_enabled(i),
                "target": bool(self.fbp_enabled[i]),
            })
        media_info = {}
        for which in ("nvdec", "nvjpg_ofa"):
            cur_dis = c.media_engine_currently_disabled(which)
            blockers = c.media_engine_blockers(which) if cur_dis else []
            media_info[which] = {
                "currently_disabled": cur_dis,
                "selected": bool(getattr(self, f"{which}_unlock")),
                "available": bool(cur_dis and not blockers),
                "blockers": blockers,
                "evidence": c.media_engine_evidence_line(which),
            }
        try:
            items, issues = derive_ff_records(c, self)
            derive_error = None
        except BurnError as exc:
            items, issues = [], []
            derive_error = str(exc)
        return {
            "pci_id": c.pci_id,
            "sku_name": sku_display_name(c.pci_id),
            "next_free_row": c.next_free_row,
            "free_row_count": c.free_row_count,
            "all_healthy_compute": self.is_all_healthy_compute_unlocked(),
            "all_healthy_memory": self.is_all_healthy_memory_unlocked(),
            "gpcs": gpcs,
            "fbps": fbps,
            "pcie_mode": self.pcie_mode,
            "pcie_current": c.current_pcie_label(),
            "nvlink_enabled": self.nvlink_enabled,
            "nvlink_current": c.nvlink_currently_enabled(),
            "nvlink_defective": c.nvlink_all_defective(),
            "a100_devid": f"0x{self.a100_devid:04x}" if self.a100_devid is not None else None,
            "sys_pipe_unlock": self.sys_pipe_unlock,
            "sys_pipe_unlockable": c.sys_pipe_unlockable_mask() != 0,
            "sys_pipe_current": c.current_sys_pipe_count(),
            "sys_pipe_max": c.max_healthy_sys_pipe_count(),
            "media": media_info,
            "summary": self.before_after_summary().to_dict(),
            "summary_line": self.summary_line(),
            "planned_records": [
                {
                    "row": it["row"],
                    "chain": it["chain"],
                    "block": it["block"],
                    "old": f"0x{it['old']:04x}" if it.get("old") is not None else None,
                    "new": f"0x{it['new']:04x}",
                    "word": f"0x{it['word']:08x}",
                    "fields": it["fields"],
                    "kind": it.get("kind", "regular"),
                    "note": it.get("note") or "",
                }
                for it in items
            ],
            "coherence_issues": [
                {"rule_id": iss.rule_id, "severity": iss.severity, "message": iss.message}
                for iss in issues
            ],
            "derive_error": derive_error,
        }

    def apply_patch(self, patch: dict) -> str:
        """Apply a single structured UI mutation action and return a human-readable status message."""
        action = str(patch.get("action", "")).strip()
        if action == "toggle_all_compute":
            if self.is_all_healthy_compute_unlocked():
                self.reset_compute()
                return f"Reset Compute to current card state ({self.target_sm_count()} SMs)."
            self.unlock_all_healthy_compute()
            return f"Unlocked all healthy Compute ({self.target_sm_count()} SMs / {self.target_sm_count() * CUDA_CORES_PER_SM:,} CUDA cores)."
        if action == "unlock_all_compute":
            self.unlock_all_healthy_compute()
            return f"Unlocked all healthy Compute ({self.target_sm_count()} SMs)."
        if action == "reset_compute":
            self.reset_compute()
            return f"Reset Compute to current card state ({self.target_sm_count()} SMs)."
        if action == "toggle_all_memory":
            if self.is_all_healthy_memory_unlocked():
                self.reset_memory()
                return f"Reset Memory to current card state ({self.target_fbp_count()}/{TOTAL_FBP} FBPs)."
            self.unlock_all_healthy_memory()
            return f"Unlocked all healthy Memory ({self.target_fbp_count()}/{TOTAL_FBP} FBPs, {self.card.bus_width_bits(self.target_fbp_count())}-bit)."
        if action == "unlock_all_memory":
            self.unlock_all_healthy_memory()
            return f"Unlocked all healthy Memory ({self.target_fbp_count()}/{TOTAL_FBP} FBPs)."
        if action == "reset_memory":
            self.reset_memory()
            return f"Reset Memory to current card state ({self.target_fbp_count()}/{TOTAL_FBP} FBPs)."
        if action == "reset_all":
            self.reset_all()
            return "Reset all selections back to current card state."
        if action == "set_gpc":
            g = int(patch["gpc"])
            en = bool(patch["enabled"])
            self.set_gpc(g, en)
            return f"GPC{g} {'enabled' if en else 'disabled'} ({self.target_sm_count()} SMs)."
        if action == "set_tpc":
            g = int(patch["gpc"])
            t = int(patch["tpc"])
            en = bool(patch["enabled"])
            self.set_tpc(g, t, en)
            return f"GPC{g}/TPC{t} {'enabled' if en else 'disabled'} ({self.target_sm_count()} SMs)."
        if action == "set_fbp":
            i = int(patch["fbp"])
            en = bool(patch["enabled"])
            self.set_fbp(i, en)
            return f"FBP{i} {'enabled' if en else 'disabled'} ({self.target_fbp_count()}/{TOTAL_FBP} FBPs)."
        if action == "set_pcie_mode":
            mode = str(patch["mode"])
            self.set_pcie_mode(mode)
            return f"PCIe speed set to {self.target_pcie_label()}."
        if action == "set_nvlink":
            en = bool(patch["enabled"])
            self.set_nvlink(en)
            return f"NVLink set to {'ON' if en else 'OFF'}."
        if action == "set_devid":
            raw = patch.get("devid")
            if raw in (None, "", "keep"):
                self.set_a100_devid(None)
                return f"PCIe Device ID reset to keep ({self.card.pci_id})."
            tid = int(str(raw), 16) if isinstance(raw, str) and not str(raw).lower().startswith("0x") else int(str(raw), 0)
            self.set_a100_devid(tid)
            return f"PCIe Device ID set to 10de:{tid:04x} ({BURNABLE_TARGET_DEVICE_IDS[tid]})."
        if action == "set_sys_pipe":
            en = bool(patch["enabled"])
            self.set_sys_pipe_unlock(en)
            return f"MIG / SYS_PIPE unlock set to {'ON' if en else 'OFF'}."
        if action == "set_media":
            which = str(patch["which"])
            en = bool(patch["enabled"])
            self.set_media_engine_unlock(which, en)
            return f"EXPERIMENTAL {which.upper()} unlock set to {'ON' if en else 'OFF'}."
        raise BurnError(f"unknown configuration action {action!r}")

    def summary_line(self) -> str:
        cur_sm = self.card.current_sm_count()
        tgt_sm = self.target_sm_count()
        max_sm = self.card.max_healthy_sm_count()
        delta_sm = tgt_sm - cur_sm
        delta_s = f"+{delta_sm}" if delta_sm > 0 else str(delta_sm)
        cur_fbp = self.card.current_fbp_count()
        tgt_fbp = self.target_fbp_count()
        tgt_bus = self.card.bus_width_bits(tgt_fbp)
        tgt_cmp_gb = self.card.est_vram_cmp_gb(tgt_fbp)
        tgt_a100_gb = self.card.est_vram_a100_gb(tgt_fbp)
        nvl = "ON" if self.nvlink_enabled else "OFF"
        vram_s = (
            f"{tgt_a100_gb}GB A100"
            if (self.a100_devid in A100_TARGET_DEVICE_IDS)
            else f"{tgt_cmp_gb}GB CMP/{tgt_a100_gb}GB A100"
        )
        parts = [
            f"SMs: {cur_sm}->{tgt_sm} ({delta_s}, {tgt_sm * CUDA_CORES_PER_SM:,} CUDA, max {max_sm})",
            f"FBPs: {cur_fbp}->{tgt_fbp}/{TOTAL_FBP} ({tgt_bus}b, ~{vram_s})",
            f"PCIe: {self.target_pcie_label()}",
            f"NVLink: {nvl}",
        ]
        if self.a100_devid is not None:
            tag = "A100 VBIOS+BAR1!" if self.a100_devid in A100_TARGET_DEVICE_IDS else "CMP VBIOS!"
            parts.append(f"DevID: 0x{self.a100_devid:04x} [{tag}]")
        if self.sys_pipe_unlock:
            parts.append(f"MIG/SYS_PIPE: {self.card.current_sys_pipe_count()}->{self.card.max_healthy_sys_pipe_count()}/8")
        media = [n for n, on in (("NVDEC", self.nvdec_unlock), ("NVJPG+OFA", self.nvjpg_ofa_unlock)) if on]
        if media:
            parts.append(f"Media (EXPERIMENTAL): {'+'.join(media)}")
        return " | ".join(parts)


def _apply_field_to_blocks(
    current_blocks: dict[int, dict[int, int]],
    desired_blocks: dict[int, dict[int, int]],
    ff_key: str,
    current_val: int,
    target_val: int,
) -> None:
    """Overlay changed bits of ``ff_key`` into ``desired_blocks``, hard-denying device-ID & absent blocks.

    Unchanged bit positions (``cur_bit == want_bit``) are skipped so multi-block fields
    (e.g. ``pes_disable`` spanning c1/b15..b17 or ``rop_l2_disable`` spanning c1/b19..b21)
    do not fail when a trailing block was never written in factory FF because all of its
    disable bits are already 0 and remain 0.
    """
    f = FIELDS_BY_KEY[ff_key]
    for offset in range(f.width):
        cur_bit = (current_val >> offset) & 1
        want_bit = (target_val >> offset) & 1
        if cur_bit == want_bit:
            continue
        bit_pos = f.bit + offset
        blk, local_bit = divmod(bit_pos, 16)
        if (f.chain, blk) in DEVICE_ID_DENIED_BLOCKS:
            raise BurnError(f"REFUSED: {ff_key} overlaps read-only device-ID block c{f.chain}/b{blk}")
        if blk not in current_blocks.get(f.chain, {}):
            raise BurnError(
                f"REFUSED: cannot modify {ff_key} bit {offset} because c{f.chain}/b{blk} is absent "
                "from this card's FF prefix (neighboring bits in that 16-bit block are unknown)"
            )
        val = desired_blocks.setdefault(f.chain, {})[blk]
        if want_bit:
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
    for key in (*sm_hbm_keys, "nvlink_disable", "sys_pipe_disable"):
        if desired_masks[key] != card.current_fields[key]:
            _apply_field_to_blocks(
                card.current_blocks,
                desired_blocks,
                key,
                card.current_fields[key],
                desired_masks[key],
            )

    # 1b. EXPERIMENTAL media engines (no defect register; evidence re-checked here so a
    #     config built before the card state changed cannot slip through).
    pcie_notes: dict[tuple[int, int], str] = {}
    engine_selection = (
        ("nvdec", config.nvdec_unlock, ("nvdec_disable",), "NVDEC x5"),
        (
            "nvjpg_ofa",
            config.nvjpg_ofa_unlock,
            ("nvjpg_ofa_disable_bit146", "nvjpg_ofa_disable_bit371"),
            "NVJPG0 + OFA0 pair",
        ),
    )
    for which, wanted, keys, label in engine_selection:
        if not wanted:
            continue
        blockers = card.media_engine_blockers(which)
        if blockers:
            raise BurnError(f"REFUSED: {label} unlock: " + "; ".join(blockers))
        for key in keys:
            if card.current_fields[key] != 0:
                _apply_field_to_blocks(card.current_blocks, desired_blocks, key, card.current_fields[key], 0)
            blk = FIELDS_BY_KEY[key].bit // 16
            prev = pcie_notes.get((1, blk))
            tag = f"EXPERIMENTAL media engine: {label}"
            pcie_notes[(1, blk)] = tag if prev is None else f"{prev}; {tag}"

    # 2. Apply PCIe speed/boot changes (chain 2 blocks 39, 49, and optionally 47, 46 for Gen4).
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
                _apply_field_to_blocks(
                    card.current_blocks, desired_blocks, key, card.current_fields[key], 0
                )
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
                _apply_field_to_blocks(
                    card.current_blocks, desired_blocks, key, card.current_fields[key], 1
                )
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

    # 4. Optional PCIe Device ID conversion / revert (paired DEVIDA c2/b37 + DEVIDB c2/b38).
    #    Encoded as ``target_devid << 1``. c2/b37 bit 0 (chain-2 bit 592) is not part of
    #    DEVIDA: it is preserved from the FF prefix when c2/b37 is present there, else
    #    written as 0 (the community-observed value). Always emitted as an atomic pair.
    if config.a100_devid is not None:
        target_dev = config.a100_devid
        if target_dev in A100_SXM4_DEVICE_IDS:
            raise BurnError(
                f"REFUSED: 10de:{target_dev:04x} is an A100 SXM4 ID; the CMP 170HX is a PCIe board "
                "and has no matching VBIOS for it (use 0x20F1 or 0x20B5)"
            )
        old_b37 = card.current_blocks.get(2, {}).get(37)
        b37_bit0 = (old_b37 & 1) if old_b37 is not None else 0
        devida_word, devidb_word = encode_a100_devid_ff_words(target_dev, b37_bit0=b37_bit0)
        dev_desc = BURNABLE_TARGET_DEVICE_IDS[target_dev]
        vbios_note = (
            "REQUIRES A100 VBIOS + 128 GB BAR1"
            if target_dev in A100_TARGET_DEVICE_IDS
            else "REVERT; REQUIRES CMP 170HX VBIOS"
        )
        for blk, word, which_field, which_label in (
            (37, devida_word, "pcie_devida", "DEVIDA"),
            (38, devidb_word, "pcie_devidb", "DEVIDB"),
        ):
            old_val = card.current_blocks.get(2, {}).get(blk)
            hard, over = ff_word_safety_audit(word, old_val, allow_a100_devid=True)
            if hard or over:
                raise BurnError(f"c2/b{blk}: REFUSED; " + "; ".join(hard + over))
            next_idx = len(items)
            new_data = (word >> 16) & 0xFFFF
            note = f"{dev_desc} ({which_label}) — {vbios_note}"
            if blk == 37 and old_b37 is None:
                note += "; c2/b37 bit 0 (chain-2 bit 592) not in FF prefix, written as 0"
            items.append(
                {
                    "row": card.next_free_row + 1 + next_idx,
                    "chain": 2,
                    "block": blk,
                    "old": old_val,
                    "new": new_data,
                    "word": word,
                    "kind": "devid",
                    "fields": [which_field],
                    "note": note,
                    "source_label": f"derived_devid[{which_label}]",
                }
            )
            desired_blocks.setdefault(2, {})[blk] = new_data

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
        ff_last_row=card.ff_last_row,
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


def _pcie_label_from_blocks(blocks: dict[int, dict[int, int]], fields: dict[str, int]) -> str:
    gen3_dis = (
        fields.get("disable_gen3_speed_bit1", 0)
        | fields.get("disable_gen3_speed_bit9", 0)
        | fields.get("pcie_boot_gen23_dis", 0)
        | fields.get("pcie_boot_gen3_dis", 0)
    )
    if gen3_dis:
        return "Gen1/Gen2 (restricted)"
    b47 = blocks.get(2, {}).get(47, 0)
    b46 = blocks.get(2, {}).get(46, 0)
    if (b47 & _PCIE_GEN4_B47_CLEAR) == 0 and (b46 & _PCIE_GEN4_B46_CLEAR) == 0:
        return "Gen4"
    return "Gen3"


def _pcie_label_from_lnkcap2(val: int | None) -> str:
    if val is None:
        return "unknown"
    speeds = val & 0xFE
    if speeds & 0x10:
        return "Gen4"
    if speeds & 0x08:
        return "Gen3"
    return "Gen1/Gen2 (restricted)"


def summarize_plan_before_after(plan: Plan) -> BeforeAfterSummary:
    """Derive a structured Before/After capability summary directly from a loaded ``Plan``."""
    before_blocks: dict[int, dict[int, int]] | None = None
    after_blocks: dict[int, dict[int, int]] | None = None
    before_fields: dict[str, int] | None = None
    after_fields: dict[str, int] | None = None

    if plan.baseline_ff_rows is not None:
        before_blocks = replay_ff_rows(plan.baseline_ff_rows)
        before_fields = decode_all_fields(before_blocks)
        after_blocks = {c: dict(bm) for c, bm in before_blocks.items()}
        for rec in plan.records:
            try:
                from .ffrecord import decode_ff

                ff = decode_ff(rec.word)
                after_blocks.setdefault(ff.chain, {})[ff.block] = ff.data
            except BurnError:
                pass
        after_fields = decode_all_fields(after_blocks)

    tpc_keys = [f"TPC_GPC{i}_DISABLE" for i in range(TOTAL_GPC)]
    sm_before: int | None = None
    sm_after: int | None = None
    gpc_before: int | None = None
    gpc_after: int | None = None
    fbp_before: int | None = None
    fbp_after: int | None = None

    if before_fields is not None and after_fields is not None:
        sm_before = compute_sm_count(
            before_fields["gpc_disable"],
            [before_fields[f"tpc_gpc{g}_disable"] for g in range(TOTAL_GPC)],
        )
        sm_after = compute_sm_count(
            after_fields["gpc_disable"],
            [after_fields[f"tpc_gpc{g}_disable"] for g in range(TOTAL_GPC)],
        )
        gpc_before = TOTAL_GPC - bin(before_fields["gpc_disable"] & 0xFF).count("1")
        gpc_after = TOTAL_GPC - bin(after_fields["gpc_disable"] & 0xFF).count("1")
        fbp_before = TOTAL_FBP - bin(before_fields["fbp_disable"] & 0xFFF).count("1")
        fbp_after = TOTAL_FBP - bin(after_fields["fbp_disable"] & 0xFFF).count("1")
    else:
        if "GPC_DISABLE" in plan.expected_masks and all(k in plan.expected_masks for k in tpc_keys):
            sm_before = compute_sm_count(
                plan.expected_masks["GPC_DISABLE"], [plan.expected_masks[k] for k in tpc_keys]
            )
            gpc_before = TOTAL_GPC - bin(plan.expected_masks["GPC_DISABLE"] & 0xFF).count("1")
        if "SM_COUNT" in plan.final_expected_cold:
            sm_after = plan.final_expected_cold["SM_COUNT"]
        else:
            decl = [r.resulting_sm_count for r in plan.records if r.resulting_sm_count is not None]
            sm_after = decl[-1] if decl else sm_before
        if "GPC_DISABLE" in plan.final_expected_cold:
            gpc_after = TOTAL_GPC - bin(plan.final_expected_cold["GPC_DISABLE"] & 0xFF).count("1")
        else:
            gpc_after = gpc_before
        if "FBP_DISABLE" in plan.expected_masks:
            fbp_before = TOTAL_FBP - bin(plan.expected_masks["FBP_DISABLE"] & 0xFFF).count("1")
        if "FBP_DISABLE" in plan.final_expected_cold:
            fbp_after = TOTAL_FBP - bin(plan.final_expected_cold["FBP_DISABLE"] & 0xFFF).count("1")
        else:
            fbp_after = fbp_before

    sm_max: int | None = None
    if "GPC_DISABLE" in plan.defect_masks and all(k in plan.defect_masks for k in tpc_keys):
        sm_max = compute_sm_count(
            plan.defect_masks["GPC_DISABLE"], [plan.defect_masks[k] for k in tpc_keys]
        )
    else:
        sm_max = plan.base_sm_ceiling

    fbp_max: int | None = None
    if "FBP_DISABLE" in plan.defect_masks:
        fbp_max = TOTAL_FBP - bin(plan.defect_masks["FBP_DISABLE"] & 0xFFF).count("1")

    target_devid: int | None = None
    for rec in plan.records:
        if rec.kind == "devid":
            info = decode_a100_devid_ff_word(rec.word)
            if info is not None:
                target_devid = info[1]
    pci_after = f"10de:{target_devid:04x}" if target_devid is not None else plan.pci_id
    targeting_a100 = (
        (target_devid in A100_TARGET_DEVICE_IDS)
        if target_devid is not None
        else (plan.pci_id in ("10de:20f1", "10de:20b5", "10de:20b0", "10de:20b2"))
    )

    if before_blocks is not None and before_fields is not None and after_blocks is not None and after_fields is not None:
        pcie_before = _pcie_label_from_blocks(before_blocks, before_fields)
        pcie_after = _pcie_label_from_blocks(after_blocks, after_fields)
    else:
        pcie_recs = [r for r in plan.records if r.link_cap2_before is not None and r.link_cap2_after is not None]
        if pcie_recs:
            pcie_before = _pcie_label_from_lnkcap2(pcie_recs[0].link_cap2_before)
            pcie_after = _pcie_label_from_lnkcap2(pcie_recs[-1].link_cap2_after)
        elif "LnkCap2" in plan.final_expected_cold:
            lbl = _pcie_label_from_lnkcap2(plan.final_expected_cold["LnkCap2"])
            pcie_before = lbl
            pcie_after = lbl
        else:
            pcie_before = "unchanged"
            pcie_after = "unchanged"

    nvlink_before = (before_fields["nvlink_disable"] & 0x7) != 0x7 if before_fields else None
    nvlink_after = (after_fields["nvlink_disable"] & 0x7) != 0x7 if after_fields else None
    sys_pipe_before = (
        8 - bin(before_fields["sys_pipe_disable"] & 0xFF).count("1") if before_fields else None
    )
    sys_pipe_after = (
        8 - bin(after_fields["sys_pipe_disable"] & 0xFF).count("1") if after_fields else None
    )
    sys_pipe_max = (
        8 - bin(plan.defect_masks["SYS_PIPE_DISABLE"] & 0xFF).count("1")
        if "SYS_PIPE_DISABLE" in plan.defect_masks
        else None
    )
    nvdec_before = (before_fields.get("nvdec_disable", 0) == 0) if before_fields else None
    nvdec_after = (after_fields.get("nvdec_disable", 0) == 0) if after_fields else None
    nvjpg_before = (
        (
            before_fields.get("nvjpg_ofa_disable_bit146", 0) == 0
            and before_fields.get("nvjpg_ofa_disable_bit371", 0) == 0
        )
        if before_fields
        else None
    )
    nvjpg_after = (
        (
            after_fields.get("nvjpg_ofa_disable_bit146", 0) == 0
            and after_fields.get("nvjpg_ofa_disable_bit371", 0) == 0
        )
        if after_fields
        else None
    )

    return BeforeAfterSummary(
        pci_id_before=plan.pci_id,
        pci_id_after=pci_after,
        sku_before=sku_display_name(plan.pci_id),
        sku_after=sku_display_name(pci_after),
        sm_before=sm_before,
        sm_after=sm_after,
        sm_max_healthy=sm_max,
        gpc_before=gpc_before,
        gpc_after=gpc_after,
        fbp_before=fbp_before,
        fbp_after=fbp_after,
        fbp_max_healthy=fbp_max,
        a100_gb_per_fbp=a100_gb_per_fbp_for_sku(plan.pci_id) or 4,
        targeting_a100_vbios=targeting_a100,
        pcie_before=pcie_before,
        pcie_after=pcie_after,
        nvlink_before=nvlink_before,
        nvlink_after=nvlink_after,
        sys_pipe_before=sys_pipe_before,
        sys_pipe_after=sys_pipe_after,
        sys_pipe_max=sys_pipe_max,
        nvdec_before=nvdec_before,
        nvdec_after=nvdec_after,
        nvjpg_ofa_before=nvjpg_before,
        nvjpg_ofa_after=nvjpg_after,
    )
