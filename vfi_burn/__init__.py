"""Private CMP 170HX 8 GB FF fuse-burn safety controller with verified GA100 protocol."""
from .backend import FakeController, Health, Host, MissingWriter, has_verified_writer
from .engine import Journal, Overrides, check_health, check_plan, preflight, process_records, scan_slots
from .ffrecord import FFRecord, decode_ff, encode_ff, import_ffcalc_plan, replay_ff_rows, simulate_ff_write
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
from .ga100 import GA100FuseController, GA100LiveAdapter
from .interfaces import FuseControllerHAL, FuseReader, FuseWriter, MMIOBus, MMIOReadBus
from .menu import BurnMenu, MenuSession
from .mmio import AlignedMMIO, ReadOnlyMMIO
from .model import BurnError, Plan, Record
from .sim_pcie import (
    PCIeTransaction,
    SimulatedGA100Adapter,
    SimulatedGA100PCIe,
    TracingMMIOBus,
    trace_hal_program_sequence,
)

__version__ = "0.3.0"

__all__ = [
    "AlignedMMIO",
    "BurnError",
    "BurnMenu",
    "COMBINATION_RULES",
    "CombinationIssue",
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
    "check_health",
    "check_plan",
    "decode_ff",
    "encode_ff",
    "evaluate_fuse_combinations",
    "has_verified_writer",
    "import_ffcalc_plan",
    "poweroff_domain_for_group",
    "poweroff_domains_for_groups",
    "preflight",
    "process_records",
    "replay_ff_rows",
    "scan_slots",
    "simulate_ff_write",
    "trace_hal_program_sequence",
]
