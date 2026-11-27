"""Private CMP 170HX 8 GB FF fuse-burn safety controller with verified GA100 protocol."""
from .backend import FakeController, Health, Host, MissingWriter, has_verified_writer
from .engine import (
    Journal,
    Overrides,
    check_health,
    check_plan,
    preflight,
    process_records,
    resolve_fpf_pointer,
    scan_fpf_rows,
    scan_slots,
)
from .ffrecord import FFRecord, build_vfi_plan_from_records, decode_ff, encode_ff, replay_ff_rows, simulate_ff_write
from .fuse_rules import (
    COMBINATION_RULES,
    FUSE_GROUPS,
    CombinationIssue,
    FuseCombinationRule,
    FuseGroup,
    evaluate_fuse_combinations,
    poweroff_domain_for_group,
    poweroff_domains_for_groups,
)
from .fuses import FFLocation, FUSE_CATALOG, FuseDefinition, FuseType
from .ga100 import (
    DiscoveredGPU,
    GA100FuseController,
    GA100LiveAdapter,
    discover_pci_gpus,
    format_gpu_entry,
    prompt_select_gpu_bdf,
)
from .interfaces import FuseControllerHAL, FuseReader, FuseWriter, MMIOBus, MMIOReadBus
from .menu import BurnMenu, MenuSession
from .menuconfig import run_menuconfig
from .mmio import AlignedMMIO, ReadOnlyMMIO
from .model import BurnError, Plan, Record
from .planner import CardSnapshot, UnlockConfig, build_plan_from_config, capture_card_snapshot, derive_ff_records
from .sim_pcie import (
    PCIeTransaction,
    SimulatedGA100Adapter,
    SimulatedGA100PCIe,
    TracingMMIOBus,
    trace_hal_program_sequence,
)

__version__ = "0.4.0"

__all__ = [
    "AlignedMMIO",
    "BurnError",
    "BurnMenu",
    "COMBINATION_RULES",
    "CardSnapshot",
    "CombinationIssue",
    "DiscoveredGPU",
    "FFLocation",
    "FFRecord",
    "FUSE_CATALOG",
    "FUSE_GROUPS",
    "FakeController",
    "FuseCombinationRule",
    "FuseControllerHAL",
    "FuseDefinition",
    "FuseGroup",
    "FuseReader",
    "FuseType",
    "FuseWriter",
    "GA100FuseController",
    "GA100LiveAdapter",
    "Health",
    "Host",
    "Journal",
    "MMIOBus",
    "MMIOReadBus",
    "MenuSession",
    "MissingWriter",
    "Overrides",
    "PCIeTransaction",
    "Plan",
    "ReadOnlyMMIO",
    "Record",
    "SimulatedGA100Adapter",
    "SimulatedGA100PCIe",
    "TracingMMIOBus",
    "UnlockConfig",
    "build_plan_from_config",
    "build_vfi_plan_from_records",
    "capture_card_snapshot",
    "check_health",
    "check_plan",
    "decode_ff",
    "derive_ff_records",
    "discover_pci_gpus",
    "encode_ff",
    "evaluate_fuse_combinations",
    "format_gpu_entry",
    "has_verified_writer",
    "poweroff_domain_for_group",
    "poweroff_domains_for_groups",
    "preflight",
    "process_records",
    "prompt_select_gpu_bdf",
    "replay_ff_rows",
    "resolve_fpf_pointer",
    "run_menuconfig",
    "scan_fpf_rows",
    "scan_slots",
    "simulate_ff_write",
    "trace_hal_program_sequence",
]
