"""Private CMP 170HX 8 GB FF fuse-burn safety controller with verified GA100 protocol."""
from .backend import FakeController, Health, Host, MissingWriter, has_verified_writer
from .engine import Journal, Overrides, check_health, check_plan, preflight, process_records, scan_slots
from .ffrecord import FFRecord, decode_ff, encode_ff, import_ffcalc_plan, replay_ff_rows, simulate_ff_write
from .ga100 import GA100FuseController, GA100LiveAdapter
from .mmio import AlignedMMIO, ReadOnlyMMIO
from .model import BurnError, Plan, Record

__version__ = "0.3.0"

__all__ = [
    "AlignedMMIO",
    "BurnError",
    "FFRecord",
    "FakeController",
    "GA100FuseController",
    "GA100LiveAdapter",
    "Health",
    "Host",
    "Journal",
    "MissingWriter",
    "Overrides",
    "Plan",
    "ReadOnlyMMIO",
    "Record",
    "check_health",
    "check_plan",
    "decode_ff",
    "encode_ff",
    "has_verified_writer",
    "import_ffcalc_plan",
    "preflight",
    "process_records",
    "replay_ff_rows",
    "scan_slots",
    "simulate_ff_write",
]
