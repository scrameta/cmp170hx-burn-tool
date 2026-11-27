# vfi_burn — Command-Line Interface

> [!CAUTION]
> **UNTESTED ON PHYSICAL HARDWARE.** Fuse programming is physically irreversible and the worst-case failure is a permanently bricked card. Read the safety notes in the [README](../README.md) before running any live command.

This page documents the scriptable `python3 -m vfi_burn <command>` interface. For the guided, numbered interactive menu (recommended for bench use), see [`menu_driven.md`](menu_driven.md). Both interfaces share the exact same safety gates, `fsync`ed JSONL journal, and typed confirmations.

---

## Command Reference

| Command | Purpose | Touches hardware? |
|---|---|---|
| `menu` | High-level interactive menu (also launched with no arguments) — see [`menu_driven.md`](menu_driven.md) | Read-only unless Live Execute is enabled |
| `plan` | Pure offline plan validation and review | Never |
| `decode <word>` | Decode a 32-bit GA100 FF record word into chain/block/data/fields | Never |
| `configure` | Read card state (live `--bdf` or simulated `--sim-pcie`/`--fake`) and build a mode-0600 plan via interactive kernel `menuconfig` TUI or non-interactive flags (`--sm-hbm`, `--pcie`, `--nvlink`) | Read-only (when using `--bdf`) |
| `snapshot` | Capture a mode-0600 cold-boot snapshot from a live CMP 170HX for `verify-cold` | Read-only |
| `inspect` | Read-only preflight, health report, and free-slot proof (`--sense` explicitly invokes `SENSE_CTRL`) | Read-only (unless `--sense`) |
| `burn` | Live hardware burn workflow — passive **dry-run by default**; requires `--execute` + TTY to program fuses | Only with `--execute` |
| `simulate` | Exercise the exact write loop on **fake** data only | Never |
| `manual` | Read-only diagnostics, interactive console, and gated single-record write | Read-only unless `--execute` |
| `verify-cold` | Compare a user-supplied cold snapshot with expected fields (optionally emit a PASS receipt) | Never |

`manual` sub-actions: `status`, `experimental`, `read-row`, `read-rows`, `read-reg`, `read-field`, `console`, `write-record`, `simulate-write-record`.

Run `python3 -m vfi_burn <command> --help` for the full list of flags.

---

## Offline Verification & Simulation (Safe — No GPU Touched)

```bash
# Run the full offline test suite:
python3 -m unittest discover -s tests -v

# Set up a private mode-0700 working directory:
mkdir -m 700 -p private
cp examples/plan.example.json private/plan.json
cp examples/fake.example.json private/fake.json
cp examples/cold.example.json private/cold.json
chmod 600 private/*.json

# Decode any 32-bit GA100 FF record word:
python3 -m vfi_burn decode 0x0fe805c1

# Review a plan offline:
python3 -m vfi_burn plan --plan private/plan.json

# Inspect and simulate Stage 1 (test row) -> cold verify -> Stage 2 (real):
python3 -m vfi_burn inspect --plan private/plan.json --fake private/fake.json --sense --journal private/session.jsonl
python3 -m vfi_burn simulate --plan private/plan.json --fake private/fake.json --stage test --journal private/session.jsonl --save-fake private/after-test.json
python3 -m vfi_burn verify-cold --plan private/plan.json --cold private/cold.json --stage test --journal private/session.jsonl --receipt-out private/test.receipt
python3 -m vfi_burn simulate --plan private/plan.json --fake private/after-test.json --stage real --receipt private/test.receipt --journal private/session.jsonl --save-fake private/after-real.json
```

---

## Bench Workflow (Live Hardware — Untested, Proceed With Extreme Caution)

```bash
# 1A. Configure unlocks/locks interactively with the kernel menuconfig-style TUI
#     (run on a cold card before glitching, or after SENSE_CTRL when EFP=1):
sudo python3 -m vfi_burn configure \
  --bdf 0000:05:00.0 \
  --out private/plan.json
#   Reads Die ID rows 8..15, FF rows 222..486, fused option shadows, and all 16 genuine
#   OPT_*_DEFECTIVE registers read-only. Lets you unlock or lock all healthy SM+HBM,
#   individual GPCs/TPCs/FBPs (keeping PES and FBPA/FBIO/ROP_L2 coherent), PCIe Gen3/Gen4/lock,
#   and NVLink, while blocking genuinely defective units and automatically prepending a
#   0-bit-diff no-op test row for flashing-time verification. Pass --no-curses for plain text.

# 1B. Or generate a plan non-interactively with explicit flags:
sudo python3 -m vfi_burn configure \
  --bdf 0000:05:00.0 \
  --sm-hbm --pcie gen3 --nvlink \
  --out private/plan.json
#   Use --pcie gen4 for experimental PCIE_MAGIC_BITS_D Gen4 unlock (burning Gen4 records
#   requires --override-unmapped-ff / menu [7] -> [8]).

# 2. Review every decoded record, die ID row (8..15), and defect mask offline:
python3 -m vfi_burn plan --plan private/plan.json

# 3. After holding a Goldilocks frame with your glitcher, inspect (passive first):
sudo python3 -m vfi_burn inspect --plan private/plan.json \
  --bdf 0000:05:00.0 --journal private/session.jsonl
#    Before SENSE_CTRL the option shadows are UNSENSED and read 0 (the glitch mis-senses the
#    row 0/1 control fuses: DSOV=0, EFP=0). Passive inspect therefore prints
#    "Option shadows not sensed yet ..." and defers the identity-mask comparison; identity
#    still rests on die rows 8..15 and the FF prefix (CMD_READ of rows is unaffected).
#    `inspect --sense` (or the burn itself) compares every mask AFTER SENSE_CTRL.

# 4. Passive dry-run preview (does NOT run SENSE_CTRL or write fuses). In a pre-SENSE
#    frame the mask comparison is deferred the same way and journaled as
#    `mask_check_deferred`; add --sense to compare masks in the dry-run:
sudo python3 -m vfi_burn burn --plan private/plan.json \
  --bdf 0000:05:00.0 \
  --stage all --journal private/session.jsonl

# 5A. SINGLE-WINDOW BURN (recommended when re-glitching is slow/hard — keeps your glitch!):
#     - Runs SENSE_CTRL once (prompts 'SENSE'), burns the no-op test row first and verifies
#       read-back + CRC=0x55 + PLM health.
#     - Then pauses at an explicit checkpoint: type 'CONTINUE SAME WINDOW' to burn the real
#       records in the same Goldilocks window (confirming each record with 'YES <row> <0xword>'),
#       OR press Enter to stop cleanly and cold-boot verify the test row first.
#     IMPORTANT: Burn all SM/HBM records (GPC/TPC/PES/FBP/FBPA/FBIO/ROP_L2) in ONE window;
#     do NOT power-cycle with SMs unlocked while PES/ROP_L2 remain masked (GSP boot will fail).
sudo python3 -m vfi_burn burn --plan private/plan.json \
  --bdf 0000:05:00.0 \
  --stage all --journal private/session.jsonl --execute

# 5B. OPTIONAL SPLIT-WINDOW BURN (if your card re-glitches easily and you want a cold boot
#     after the test row before touching real unlock records):
#   sudo python3 -m vfi_burn burn --plan private/plan.json --bdf 0000:05:00.0 \
#     --stage test --journal private/session.jsonl --execute
#   # Cold power-cycle (~30 s off), then snapshot + verify-cold --stage test --receipt-out private/test.receipt
#   # Re-glitch to Goldilocks, then run:
#   sudo python3 -m vfi_burn burn --plan private/plan.json --bdf 0000:05:00.0 \
#     --stage real --receipt private/test.receipt \
#     --journal private/session.jsonl --execute

# 6. Cold power-cycle (~30 s off), capture final cold snapshot, and verify:
sudo setpci -s 0000:05:00.0 COMMAND=0x0002:0x0002   # only if BAR0 reads 0xFFFFFFFF
sudo python3 -m vfi_burn snapshot --bdf 0000:05:00.0 \
  --previous-boot-id <BOOT_ID_FROM_JOURNAL> --out private/cold-final.json
python3 -m vfi_burn verify-cold --plan private/plan.json \
  --cold private/cold-final.json --stage real --journal private/session.jsonl
```

---

## Manual Console

Read-only manual commands (`status`, `read-row`, `read-rows`, `read-reg`, `read-field`, `console`) work even on cold, poisoned, or latched cards (refusing only if `nvidia`/`nouveau` is loaded/bound):

```bash
sudo python3 -m vfi_burn manual --plan private/plan.json \
  --bdf 0000:05:00.0 \
  --journal private/session.jsonl console
```

Inside the console, run `preview <word>` (or `preview <target> <word> <description>`) to inspect the complete decoded GA100 fuse controller state and the exact ordered 32-bit aligned PCIe BAR0 MMIO writes, which are **staged** so you simply type `apply` (still gated by row/word confirmation & read-back) or `clear` to discard. See the [menu walkthrough](menu_driven.md#6-9-manual-low-level-console--fuse-controller-state-exact-pcie-bar0-writes--staged-preview--apply) for a screenshot.

---

## Granular Manual Overrides (No blanket `--force`)

Every override is off by default, prints a loud banner, requires its own distinct typed confirmation phrase, and is logged in the journal (accepted by `menu`, `burn`, `simulate`, `manual console`, `manual write-record`, and `manual simulate-write-record`):

| Flag | Purpose | Typed confirmation |
|---|---|---|
| `--override-slot ROW` | Target a specific **empty** FF row when contiguous scan is ambiguous | `OVERRIDE SLOT <ROW>` |
| `--override-skip-sense` | Skip `SENSE_CTRL` (passive goldilocks checks remain) | `OVERRIDE SKIP SENSE` |
| `--override-no-test-receipt` | Run `real` stage when test row was verified outside `vfi_burn` | `OVERRIDE NO TEST RECEIPT` |
| `--override-mask-mismatch` | Skip initial live mask equality check | `OVERRIDE MASK MISMATCH` |
| `--override-undetermined-ceiling` | Override undetermined/exceeded SM ceiling (genuine defects remain refused) | `OVERRIDE CEILING` |
| `--override-baseline-mismatch` | Allow proceeding when live FF prefix differs from (or plan lacks) `baseline_ff_rows` | `OVERRIDE BASELINE MISMATCH` |
| `--override-unmapped-ff` | Allow unknown FF blocks, changed unmapped/opt-in bits, or fields missing defect masks | `OVERRIDE UNMAPPED FF` |
| `--override-allow-debugctrl` | **Manual mode only**: permit `DEBUGCTRL` (`0x820060`) target | `REVIEW DEBUGCTRL` + `CONFIRM DEBUGCTRL` |
