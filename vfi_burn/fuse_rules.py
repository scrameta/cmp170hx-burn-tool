"""Declarative GA100 fuse combination groups and cross-fuse coherence rules.

Models which fuses must work together (compute hierarchy, memory slice pairing,
and PCIe speed/boot fuse combinations) and evaluates a decoded fuse state for
inconsistent combinations before burning or powering off.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping

from .fuses import (
    FUSE_CATALOG,
    PES_PER_GPC,
    TOTAL_FBP,
    TOTAL_GPC,
)


@dataclass(frozen=True)
class FuseGroup:
    """A set of fuses belonging to one coupled GPU subsystem."""

    name: str
    description: str
    member_keys: tuple[str, ...]
    atomic_poweroff: bool = True
    poweroff_domain: str = ""


def _keys_for_group(group: str) -> tuple[str, ...]:
    return tuple(
        f.key for f in FUSE_CATALOG if f.ff_overridable and f.functional_group == group
    )


FUSE_GROUPS: tuple[FuseGroup, ...] = (
    FuseGroup(
        name="compute",
        description="Compute floorsweep hierarchy (GPC_DISABLE, TPC_GPC0..7_DISABLE, PES_DISABLE)",
        member_keys=_keys_for_group("compute"),
        atomic_poweroff=True,
        poweroff_domain="sm_hbm",
    ),
    FuseGroup(
        name="memory",
        description="Memory partition hierarchy (FBP_DISABLE, FBPA_DISABLE, FBIO_DISABLE, ROP_L2_DISABLE)",
        member_keys=_keys_for_group("memory"),
        atomic_poweroff=True,
        poweroff_domain="sm_hbm",
    ),
    FuseGroup(
        name="pcie",
        description="PCIe Gen3/Gen4 speed & boot-speed fuses (chain 2 blocks 39, 45-47, 49)",
        member_keys=_keys_for_group("pcie"),
        atomic_poweroff=True,
        poweroff_domain="pcie",
    ),
    FuseGroup(
        name="nvlink",
        description="NVLink disable mask (requires U11 bridge mod + firmware bypass)",
        member_keys=_keys_for_group("nvlink"),
        atomic_poweroff=True,
        poweroff_domain="nvlink",
    ),
    FuseGroup(
        name="mig",
        description="SYS_PIPE / MIG disable mask (experimental)",
        member_keys=_keys_for_group("mig"),
        atomic_poweroff=True,
        poweroff_domain="mig",
    ),
    FuseGroup(
        name="engine",
        description="EXPERIMENTAL media-engine floorsweep bits (NVDEC c1/b8, NVJPG/OFA pair c1/b9 + c1/b23)",
        member_keys=_keys_for_group("engine"),
        atomic_poweroff=True,
        poweroff_domain="engine",
    ),
)

FUSE_GROUPS_BY_NAME: dict[str, FuseGroup] = {g.name: g for g in FUSE_GROUPS}


def poweroff_domain_for_group(group: str) -> str:
    """Return the atomic power-off domain for functional group ``group``.

    ``compute`` (GPC/TPC/PES) and ``memory`` (FBP/FBPA/FBIO/ROP_L2) share the
    ``sm_hbm`` atomic power-off domain because power-cycling with SMs unlocked
    while PES/ROP_L2 remain masked causes GSP boot failure.
    """
    fg = FUSE_GROUPS_BY_NAME.get(group)
    if fg is not None and fg.poweroff_domain:
        return fg.poweroff_domain
    return group


def poweroff_domains_for_groups(groups: Iterable[str]) -> set[str]:
    """Map functional group names to their non-noop atomic power-off domains."""
    return {poweroff_domain_for_group(g) for g in groups if g != "noop"}


@dataclass(frozen=True)
class CombinationIssue:
    """A single cross-fuse coherence finding."""

    rule_id: str
    severity: str  # "warning" | "error"
    group: str
    message: str
    fields: tuple[str, ...]


class FuseCombinationRule(ABC):
    """Abstract base class for declarative cross-fuse combination rules."""

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Stable machine-readable identifier for this rule."""

    @property
    @abstractmethod
    def group(self) -> str:
        """Functional group name (e.g. 'compute', 'memory', 'pcie')."""

    @property
    @abstractmethod
    def required_fields(self) -> tuple[str, ...]:
        """Fuse keys that must be present in ``fields`` for this rule to apply."""

    def applies_to(self, fields: Mapping[str, int]) -> bool:
        return all(k in fields for k in self.required_fields)

    @abstractmethod
    def evaluate(self, fields: Mapping[str, int]) -> list[CombinationIssue]:
        """Inspect ``fields`` and return any combination coherence issues."""


class GpcTpcCoherenceRule(FuseCombinationRule):
    """An enabled GPC must have at least one enabled TPC."""

    @property
    def rule_id(self) -> str:
        return "compute.gpc_tpc_coherence"

    @property
    def group(self) -> str:
        return "compute"

    @property
    def required_fields(self) -> tuple[str, ...]:
        return ("gpc_disable",) + tuple(f"tpc_gpc{i}_disable" for i in range(TOTAL_GPC))

    def evaluate(self, fields: Mapping[str, int]) -> list[CombinationIssue]:
        issues: list[CombinationIssue] = []
        gpc_dis = fields["gpc_disable"]
        for i in range(TOTAL_GPC):
            tpc_key = f"tpc_gpc{i}_disable"
            if not ((gpc_dis >> i) & 1) and (fields[tpc_key] & 0xFF) == 0xFF:
                issues.append(
                    CombinationIssue(
                        rule_id=self.rule_id,
                        severity="warning",
                        group=self.group,
                        message=(
                            f"GPC{i} is enabled in GPC_DISABLE (0x{gpc_dis:02x}) but all 8 TPCs "
                            f"in TPC_GPC{i}_DISABLE are disabled (0xff)."
                        ),
                        fields=("gpc_disable", tpc_key),
                    )
                )
        return issues


class GpcPesCoherenceRule(FuseCombinationRule):
    """An enabled GPC must have at least one enabled PES (3 PES slices per GPC)."""

    @property
    def rule_id(self) -> str:
        return "compute.gpc_pes_coherence"

    @property
    def group(self) -> str:
        return "compute"

    @property
    def required_fields(self) -> tuple[str, ...]:
        return ("gpc_disable", "pes_disable")

    def evaluate(self, fields: Mapping[str, int]) -> list[CombinationIssue]:
        issues: list[CombinationIssue] = []
        gpc_dis = fields["gpc_disable"]
        pes_dis = fields["pes_disable"]
        pes_all_mask = (1 << PES_PER_GPC) - 1
        for i in range(TOTAL_GPC):
            if not ((gpc_dis >> i) & 1):
                gpc_pes = (pes_dis >> (i * PES_PER_GPC)) & pes_all_mask
                if gpc_pes == pes_all_mask:
                    issues.append(
                        CombinationIssue(
                            rule_id=self.rule_id,
                            severity="warning",
                            group=self.group,
                            message=(
                                f"GPC{i} is enabled in GPC_DISABLE (0x{gpc_dis:02x}) but all "
                                f"{PES_PER_GPC} PES slices (bits {i * PES_PER_GPC}..{(i + 1) * PES_PER_GPC - 1}) "
                                f"in PES_DISABLE (0x{pes_dis:06x}) are disabled."
                            ),
                            fields=("gpc_disable", "pes_disable"),
                        )
                    )
        return issues


class MemorySlicePairingRule(FuseCombinationRule):
    """Every enabled FBP must have matching active FBPA/FBIO/ROP_L2 slice pairs."""

    @property
    def rule_id(self) -> str:
        return "memory.fbp_slice_pairing"

    @property
    def group(self) -> str:
        return "memory"

    @property
    def required_fields(self) -> tuple[str, ...]:
        return ("fbp_disable",)

    def evaluate(self, fields: Mapping[str, int]) -> list[CombinationIssue]:
        issues: list[CombinationIssue] = []
        fbp_dis = fields["fbp_disable"]

        for slice_key, label in (
            ("fbpa_disable", "FBPA_DISABLE"),
            ("fbio_disable", "FBIO_DISABLE"),
            ("rop_l2_disable", "ROP_L2_DISABLE"),
        ):
            if slice_key not in fields:
                continue
            slice_mask = fields[slice_key]
            for i in range(TOTAL_FBP):
                if not ((fbp_dis >> i) & 1):
                    pair = (slice_mask >> (2 * i)) & 0x3
                    if pair == 0x3:
                        issues.append(
                            CombinationIssue(
                                rule_id=self.rule_id,
                                severity="warning",
                                group=self.group,
                                message=(
                                    f"FBP{i} is enabled in FBP_DISABLE (0x{fbp_dis:03x}) but both "
                                    f"paired slices (bits {2 * i},{2 * i + 1}) in {label} "
                                    f"(0x{slice_mask:06x}) are disabled."
                                ),
                                fields=("fbp_disable", slice_key),
                            )
                        )

        if "fbpa_disable" in fields and "fbio_disable" in fields:
            fbpa = fields["fbpa_disable"]
            fbio = fields["fbio_disable"]
            for i in range(TOTAL_FBP):
                if not ((fbp_dis >> i) & 1):
                    fbpa_pair = (fbpa >> (2 * i)) & 0x3
                    fbio_pair = (fbio >> (2 * i)) & 0x3
                    if fbpa_pair != fbio_pair:
                        issues.append(
                            CombinationIssue(
                                rule_id="memory.fbpa_fbio_mismatch",
                                severity="warning",
                                group=self.group,
                                message=(
                                    f"FBP{i} has mismatched FBPA/FBIO slice disables: "
                                    f"FBPA bits[{2 * i + 1}:{2 * i}]=0b{fbpa_pair:02b} != "
                                    f"FBIO bits[{2 * i + 1}:{2 * i}]=0b{fbio_pair:02b}."
                                ),
                                fields=("fbp_disable", "fbpa_disable", "fbio_disable"),
                            )
                        )
        return issues


class PcieSpeedCoherenceRule(FuseCombinationRule):
    """PCIe Gen3, Gen4, and boot speed restriction fuses must form a coherent speed unlock."""

    @property
    def rule_id(self) -> str:
        return "pcie.speed_coherence"

    @property
    def group(self) -> str:
        return "pcie"

    @property
    def required_fields(self) -> tuple[str, ...]:
        return ()

    def applies_to(self, fields: Mapping[str, int]) -> bool:
        pcie_keys = (
            "disable_gen3_speed_bit1",
            "disable_gen3_speed_bit9",
            "pcie_magic_gen4_dis",
            "pcie_boot_gen23_dis",
            "pcie_boot_gen3_dis",
        )
        return any(k in fields for k in pcie_keys)

    def evaluate(self, fields: Mapping[str, int]) -> list[CombinationIssue]:
        issues: list[CombinationIssue] = []

        if "disable_gen3_speed_bit1" in fields and "disable_gen3_speed_bit9" in fields:
            b1 = fields["disable_gen3_speed_bit1"] & 1
            b9 = fields["disable_gen3_speed_bit9"] & 1
            if b1 != b9:
                issues.append(
                    CombinationIssue(
                        rule_id="pcie.gen3_split_bits",
                        severity="warning",
                        group=self.group,
                        message=(
                            f"DISABLE_GEN3_SPEED bits disagree: bit1={b1}, bit9={b9} "
                            "(both bits in chain 2 block 39 should move together)."
                        ),
                        fields=("disable_gen3_speed_bit1", "disable_gen3_speed_bit9"),
                    )
                )

        if "pcie_boot_gen23_dis" in fields and "pcie_boot_gen3_dis" in fields:
            g23 = fields["pcie_boot_gen23_dis"] & 1
            g3 = fields["pcie_boot_gen3_dis"] & 1
            if g3 == 0 and g23 == 1:
                issues.append(
                    CombinationIssue(
                        rule_id="pcie.boot_gen23_conflict",
                        severity="warning",
                        group=self.group,
                        message=(
                            "PCIE_BOOT_GEN3_DISABLE is cleared (0) while PCIE_BOOT_GEN23_DISABLE "
                            "remains set (1) in chain 2 block 49."
                        ),
                        fields=("pcie_boot_gen23_dis", "pcie_boot_gen3_dis"),
                    )
                )

        if "pcie_magic_gen4_dis" in fields and (fields["pcie_magic_gen4_dis"] & 1) == 0:
            still_restricted = [
                k
                for k in ("disable_gen3_speed_bit1", "disable_gen3_speed_bit9", "pcie_boot_gen3_dis")
                if k in fields and (fields[k] & 1) == 1
            ]
            if still_restricted:
                issues.append(
                    CombinationIssue(
                        rule_id="pcie.gen4_with_gen3_disabled",
                        severity="warning",
                        group=self.group,
                        message=(
                            "PCIE_MAGIC_BITS_D[25] (Gen4 disable) is cleared (0) while Gen3 "
                            f"restriction fuse(s) remain set: {', '.join(still_restricted)}."
                        ),
                        fields=("pcie_magic_gen4_dis", *still_restricted),
                    )
                )
        return issues


COMBINATION_RULES: tuple[FuseCombinationRule, ...] = (
    GpcTpcCoherenceRule(),
    GpcPesCoherenceRule(),
    MemorySlicePairingRule(),
    PcieSpeedCoherenceRule(),
)


def evaluate_fuse_combinations(
    fields: Mapping[str, int],
    *,
    rules: tuple[FuseCombinationRule, ...] = COMBINATION_RULES,
) -> list[CombinationIssue]:
    """Run all applicable cross-fuse combination rules against ``fields``."""
    issues: list[CombinationIssue] = []
    for rule in rules:
        if rule.applies_to(fields):
            issues.extend(rule.evaluate(fields))
    return issues
