# vfi_burn — CMP 170HX 8 GB FF Fuse-Burn Safety Controller

> [!CAUTION]
> **UNTESTED ON PHYSICAL HARDWARE — THIS TOOL HAS NEVER BEEN USED TO BURN A REAL GPU YET.**
>
> **Do not treat this as battle-tested software.** Every test in this repository runs offline against synthetic in-memory mocks (`FakeController` / `FakeGA100MMIO`). Passing the test suite proves control flow and safety-gate logic only — **it does NOT prove real silicon behavior**.
>
> Fuse programming is **physically irreversible**, and the worst-case failure is a **permanently bricked card**. Treat your first real run as untested code:
> - Inspect everything read-only first (`plan`, `inspect`, `manual status/read-rows`).
> - Cross-check every decoded FF word, row number, and defect mask by hand against independent tools (`cmp170hx-ffcalc` / `ga100-ff-tools`).
> - Always burn a single **no-op test row** (automatically prepended as record 0 by `configure` / `[1] Setup -> [a]`) first and verify read-back + `CRC=0x55` before burning real unlock records (you can either continue in the **same Goldilocks window** with `--stage all` so you don't lose a hard-won glitch, or split across a cold boot with `--stage test` / `--stage real`).
> - Use entirely at your own risk.

---

## Overview

`vfi_burn` is a standalone Python 3.10+ (standard library only, zero third-party dependencies) fail-closed safety controller for programming GA100 Fuseless Fuse (FF) records on the **NVIDIA CMP 170HX 8 GB (`10de:20c2`)**.

It sits downstream of your **voltage-glitch / hold setup** (e.g. [`dalzyu/pico-ga100vfi`](https://github.com/dalzyu/pico-ga100vfi), [`darkmaniac7/cmp170hx-8gb-vfi-tools`](https://github.com/darkmaniac7/cmp170hx-8gb-vfi-tools)) — `vfi_burn` never glitches, dips NVVDD, or pulses `PEX_RST`. It includes a built-in **kernel `menuconfig`-style unlock/lock planner** (`configure` / `[1] Setup -> [a]`) that reads the card's Die IDs, FF rows, fused shadows, and all 16 genuine silicon defect masks read-only and derives exact, coherence-checked FF records directly (with an automatic 0-bit-diff no-op test row prepended for flashing-time verification).

Design philosophy:
- **Passive dry-run by default**: `vfi_burn burn` performs passive preflight and slot proving without firing active sense or programming fuses unless you explicitly opt in (`--sense` / `--execute`).
- **One record at a time, mandatory read-back**: no batching, no automatic retries, and the row pointer never advances until both read-back and the full post-write health/host gate pass.
- **Full audit trail**: every read, preflight/sense pass or failure, override, confirmation, write attempt (`fsync`ed *before* touching hardware), halt, and cold verification is appended to an exclusively locked (`flock`), mode-`0600` JSONL journal.
- **Granular manual overrides, never a blanket `--force`**: bench edge cases have explicit, loud, double-confirmed, journaled overrides while hard brick-prevention invariants remain unbreakable.

---

## GA100 Hardware Protocol & Sources

The register map, FF binary format, and Goldilocks arming/programming sequences implemented in [`vfi_burn/ffrecord.py`](vfi_burn/ffrecord.py), [`vfi_burn/ga100.py`](vfi_burn/ga100.py), and [`vfi_burn/mmio.py`](vfi_burn/mmio.py) were cross-checked across public GA100 VFI research (again, **not yet run on real hardware by this codebase**):

- [Kate Land — *Using voltage glitching to unleash the A100 in my mining card*](https://blog.kate.land/posts/using-voltage-glitching-to-unlock-the-a100/)
- [`sayhellotojungle/cmp170hx-ffcalc`](https://github.com/sayhellotojungle/cmp170hx-ffcalc)
- [`kprinssu/ga100-ff-tools`](https://github.com/kprinssu/ga100-ff-tools)
- [`darkmaniac7/cmp170hx-8gb-vfi-tools`](https://github.com/darkmaniac7/cmp170hx-8gb-vfi-tools)
- [`dalzyu/pico-ga100vfi`](https://github.com/dalzyu/pico-ga100vfi)

Key protocol details implemented:
- **FF Record Encoding**: `word = (data16 << 16) | (block10 << 6) | chain5`, with **bit 5 (`IFF`) strictly `0`**. Later records for the same `(chain, block)` override earlier records.
- **FF Row Bounds & Empty Marker**: rows **`222..486`** only (rows `0..221` control/regular/FPF and `487..511` IFF are hard-refused). Blank rows read **`0x00000000`** via `CMD_READ`.
- **Single Aligned 32-Bit MMIO Stores**: [`AlignedMMIO`](vfi_burn/mmio.py) uses `memoryview(mm)[off:off+4].cast("I")[0] = value` (patch 0012 — never `struct.pack_into`, which zeroes the register first). `DEBUGCTRL (0x00820060)` is never written (must read `0`, `RWL=0`).
- **Goldilocks Arming (`SENSE_CTRL`)**:
  1. `EN_SW_OVERRIDE (0x820040) = 1` (sticks only while `DSOV = 0`) and `WDATA_WR_SECURE (0x82048C) = 0` (`PLM_f4` opens to `0xFFFFFFFF`).
  2. Verify `RESENSE_DISABLE (0x821058) == 0`, clear `IFF_RECORD (0x820050)` if nonzero, and latch **no-burn safety guard** `FUSEWDATA (0x82000C) = 0` (readback verified).
  3. Write `FUSECTRL = 0 (IDLE)` → `FUSECTRL = 3 (SENSE_CTRL)`, poll `IDLE` (state=4). Verify `EFP (0x820078) == 1`, `DSOV == 0`, `CRC == 0x55`, `DFP (0x82007C) == 0`, `PLM_f8` bit 0 open. Idempotent when `EFP == 1` already (only one `SENSE_CTRL` per glitch window because `RESENSE_DISABLE` becomes `1`).
  4. Shadow-clear `SBFE_SHADOW (0x821108) = 0` and assert all 12 armed invariants (`FUSECTRL` idle, `DEBUGCTRL == 0`, `CRC == 0x55`, `DSOV == 0`, `EFP == 1`, `DFP == 0`, `EN_SW_OVERRIDE == 1`, `SBFE == 0`, `WDATA_WR_SECURE == 0`, `PLM_f4 == PLM_f8 == PLM_fc == 0xFFFFFFFF`).
- **Single-Record Program (`CMD_PROGRAM = 2`)**:
  - Assert all 12 armed invariants + target row reads `0x00000000` → save `FUSEADDR` → `FUSECTRL = 0` → `FUSEADDR = row` (verify latch) → `FUSEWDATA = word` (verify readback; never `FUSEDATA` `0x820008`) → `FUSECTRL = 2` → poll `IDLE` (immediate halt without further BAR0 reads on `0xBADxxxxx` poison) → restore `FUSEADDR` & `FUSEWDATA = 0` → `CMD_READ` verify `== word` → full post-write health/host gate → second `CMD_READ`.

For a full walkthrough of the fuse controller (register map, `FUSECTRL` command state machine, the locked → Goldilocks → sensed → armed security state machine, and sequence diagrams for read / sense / arm / program), see [`docs/FUSE_CONTROLLER.md`](docs/FUSE_CONTROLLER.md).

---

## Repository Layout

```text
vfi_burn/
  model.py       # Strict mode-0600 Plan & Record schema validation
  fuses.py       # Unified GA100 fuse & register catalog (FF location, BAR0 addresses, types, FF-overridable flags)
  fuse_rules.py  # Declarative fuse combination groups & cross-fuse coherence rules (compute/memory/PCIe)
  interfaces.py  # Explicit abc.ABC contracts (MMIOReadBus, MMIOBus, FuseControllerHAL, FuseReader, FuseWriter)
  mmio.py        # ReadOnlyMMIO & AlignedMMIO (single 32-bit aligned stores + DEBUGCTRL guard)
  ffrecord.py    # GA100 FF codec, replay simulator, SM math, no-op test builder, plan builder
  planner.py     # Native GA100 unlock/lock planner (live/simulated card readout -> exact coherent FF records)
  menuconfig.py  # Kernel menuconfig-style interactive TUI (stdlib curses + plain-text fallback)
  ga100.py       # GA100FuseController HAL (fuse request -> PCIe BAR0) & Linux sysfs BAR0 GA100LiveAdapter
  sim_pcie.py    # Simulated PCIe BAR0 MMIO bus, HAL transaction tracer, and HAL-backed SimulatedGA100Adapter
  backend.py     # MissingWriter, FakeController, adapter loader, atomic 0600 JSON I/O
  journal.py     # Append-only, fsynced, flock-locked mode-0600 JSONL audit journal & signal deferral
  engine.py      # Fail-closed state machine, Overrides, preflight, burn loop, verify_cold
  console.py     # Interactive manual console, plan/status reports, and HAL-traced PCIe write previews
  menu.py        # High-level numbered interactive menu (SSH/tmux-friendly, read-only by default)
  commands.py    # Decoupled command execution layer (zero argparse dependency; injectable CommandIO)
  cli.py         # Pure argparse CLI grammar and dispatcher
tests/
  test_fuses.py              # Unified fuse catalog invariants & BAR0 alignment
  test_fuse_rules.py         # Declarative compute/memory/PCIe fuse combination coherence rules
  test_interfaces.py         # abc.ABC contract enforcement & subclass conformance
  test_sim_pcie.py           # Simulated PCIe execution, HAL BAR0 tracing, & HAL-backed end-to-end engine tests
  test_commands.py           # Decoupled command execution (zero argparse) & CLI --sim-pcie tests
  test_engine.py             # Core state machine, gates, overrides, pre-trigger check, journal locking
  test_ga100.py              # Hardware-accurate GA100 fuse-controller FSM & end-to-end Goldilocks tests
  test_ffrecord.py           # FF codec, Kate/ffcalc vectors, replay, SM math, plan builder tests
  test_planner_menuconfig.py # Native unlock/lock planner, menuconfig TUI, CLI configure & end-to-end simulate tests
  test_mmio.py               # AlignedMMIO, ReadOnlyMMIO, DEBUGCTRL guard, mmap close tests
  test_cli_console.py        # CLI, faulted-card read diagnostics, dry-run, interactive console tests
  test_menu.py               # High-level interactive menu workflow, Live Execute guard & CLI entry tests
  test_review_fixes.py / test_safety_review_fixes.py  # Safety regression suites (zero-CMD_PROGRAM invariants)
examples/              # Synthetic-only example plan/fake/cold files (NEVER put real die IDs here)
docs/
  menu_driven.md       # Interactive menu guide & offline screenshot walkthrough
  cmd_line.md          # Command-line reference, offline simulation & live bench workflow
  FUSE_CONTROLLER.md   # How the GA100 fuse controller works (state machines + programming sequence)
  screenshots/         # Terminal screenshots captured in offline simulation mode
adapter_template.py    # ABC-backed template for custom hardware adapters
```

---

## Usage & Documentation

`vfi_burn` provides two interfaces over the same fail-closed engine, safety gates, and `fsync`ed JSONL journal:

- **[Interactive Menu Guide (`docs/menu_driven.md`)](docs/menu_driven.md)** *(recommended for bench use)* — SSH/serial/tmux-friendly numbered menu (`python3 -m vfi_burn` or `python3 -m vfi_burn menu`), read-only by default on live hardware, covering native `menuconfig` unlock/lock planning (`[1] Setup -> [a]`), plan review, live status, active sense, dry-run preview, unified burn/resume (`[6] Burn Plan`), overrides, the staged `preview` → `apply` manual console, FF word decoding, cold-boot verification, and experimental register inspection.
- **[Command-Line Interface (`docs/cmd_line.md`)](docs/cmd_line.md)** — scriptable subcommands (`configure`, `plan`, `decode`, `snapshot`, `inspect`, `burn`, `simulate`, `manual`, `verify-cold`), offline simulation examples, step-by-step live hardware bench workflow, and low-level `manual` diagnostics/console commands.
- **[GA100 Fuse Controller Reference (`docs/FUSE_CONTROLLER.md`)](docs/FUSE_CONTROLLER.md)** — register map, `FUSECTRL` command FSM, locked → Goldilocks → sensed → armed security state machine, and read/sense/arm/program sequence diagrams.

---

## Safety Model

### Hard Invariants (NEVER overrideable)
- Single aligned 32-bit MMIO stores (`AlignedMMIO`); `DEBUGCTRL` never written implicitly
- No `nvidia` or `nouveau` driver loaded or bound to the GPU
- SKU `10de:20c2` and all 8 Die ID rows (`8..15`) match the plan
- FF row bounds `222..486`, IFF bit 5 == `0`, blank target slot (`0x00000000`), no poison payload
- Genuine `defect_masks` bits are strictly refused — derived from the **FF word itself** (replace semantics over its 16-bit block), so missing or wrong `mask_update` metadata, shared multi-field blocks, and console one-liners cannot bypass it; a word touching a field with no known defect mask is refused unless `--override-undetermined-ceiling`
- Declared `mask_update.to` / `resulting_sm_count` must match what the word actually writes; GPC/TPC changes are checked against the SM ceiling from the word
- Instantaneous pre-trigger check *after* confirmation and *immediately before* `CMD_PROGRAM`
- Mandatory `CMD_READ` read-back + post-write health/host gate + second `CMD_READ`; immediate `HALT_UNVERIFIED` with no retry and no pointer advance on any error
- Failures strictly before `CMD_PROGRAM` are journaled as `aborted_pre_trigger` (`write_issued=false`), not as a possibly-partial `HALT_UNVERIFIED`
- `SIGINT`/`SIGTERM`/`SIGHUP`/`SIGQUIT` are held back from the `attempt` journal entry until the record's outcome is fsynced (Ctrl-C or an SSH drop cannot land mid-program); `FUSEWDATA` is reset to the `0` no-burn latch after every program
- Any `0xBADxxxxx` register read is treated as poison (same test as `cmp170hx-8gb-vfi-tools`). A fuse *row* reading `0xBADxxxxx` is real OTP data (an FF record with block data `0xBAD0..0xBADF`) unless the controller is unhealthy straight after the read or a second read differs; such records are never burned
- Real-stage `verify-cold` requires `fpf_rows` and fails if any journaled row reads `0` after the cold boot

### Granular Manual Overrides (No blanket `--force`)

Every override is off by default, prints a loud banner, requires its own distinct typed confirmation phrase, and is logged in the journal — while the hard invariants above remain unbreakable. Overrides can be passed as CLI flags (see [`docs/cmd_line.md`](docs/cmd_line.md#granular-manual-overrides-no-blanket---force)) or toggled interactively via menu option `[8]` (see [`docs/menu_driven.md`](docs/menu_driven.md#5-8-granular-manual-overrides--10-offline-32-bit-ff-word-decoder)).
