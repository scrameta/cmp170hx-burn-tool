# vfi_burn — Interactive Menu Interface (Recommended for Bench Use)

> [!CAUTION]
> **UNTESTED ON PHYSICAL HARDWARE.** Fuse programming is physically irreversible and the worst-case failure is a permanently bricked card. Read the safety notes in the [README](../README.md) before running any live session.

This page documents the guided, numbered interactive menu. For the scriptable subcommand interface, see [`cmd_line.md`](cmd_line.md).

---

## Launching the Menu

Running `vfi_burn` with **no arguments** (or with `menu` plus optional pre-filled flags) launches a plain numbered menu designed for SSH/serial/tmux sessions (no fragile `curses`, full scrollback preserved):

```bash
# Launch empty menu and configure everything interactively:
python3 -m vfi_burn

# Launch pre-loaded for offline simulation (no GPU touched):
python3 -m vfi_burn menu \
  --plan private/plan.json \
  --fake private/fake.json \
  --journal private/session.jsonl

# Launch pre-loaded for a live bench session (starts in LIVE READ-ONLY / DRY-RUN mode):
sudo python3 -m vfi_burn menu \
  --plan private/plan.json \
  --bdf 0000:05:00.0 \
  --port /dev/ttyACM0 \
  --journal private/session.jsonl
```

---

## Safety Behaviour & Menu Options

The menu wraps every CLI workflow with the exact same safety gates, `fsync`ed JSONL journal, and typed confirmations:
- **Safe by default on live hardware**: opens in `LIVE READ-ONLY / DRY-RUN` mode. Physical burns (`[6]` Single-Window `--stage all` or `[7]` Split-Window) are refused before opening BAR0 until you explicitly enable Live Execute in `[1] Setup -> [g]` by typing `ENABLE LIVE WRITES` (or pass `--execute`), and switching targets automatically resets Live Execute back to `OFF`.
- **Full end-to-end bench flow in one place**:

| Option | Action |
|---|---|
| `[1]` | Setup / import `cmp170hx-ffcalc` plan |
| `[2]` | Review Plan & Expected Outcome (FF rows taken/remaining, enabled/disabled breakdown, unknown/experimental fuse audit) |
| `[3]` | Live Status & Passive Inspection |
| `[4]` | Active Sense (`SENSE_CTRL`) |
| `[5]` | Passive Dry-Run Preview |
| `[6]` | Single-Window Burn (no-op test row → `CONTINUE SAME WINDOW` → real records) |
| `[7]` | Split-Window Burn |
| `[8]` | Granular Manual Overrides |
| `[9]` | Manual Low-Level Console |
| `[10]` | Offline FF Word Decoder |
| `[11]` | Cold-Boot Snapshot & Verify |

---

## Walkthrough Screenshots (Offline Simulation Mode)

> All screenshots below were captured offline against synthetic fixtures (no physical GPU contacted).

### 1. Main Menu & Live Status Header
![vfi_burn Main Interactive Menu](screenshots/01-main-menu.png)

### 2. `[2]` Review Plan & Expected Outcome (Enabled/Disabled, Unknown Fuses, FF Rows Taken & Remaining)
Shows exact FF rows used/taken/remaining, what compute/PCIe/feature units are being enabled or disabled (with defect masks preserved), and flags any unknown FF blocks, unmapped bits, or experimental/untested fields before you ever touch a burn command.
![vfi_burn Review Plan and Expected Outcome](screenshots/02-review-plan.png)

### 3. `[3]` Live Status, Passive Health Gates & Free-Slot Proof
![vfi_burn Live Status and Passive Inspection](screenshots/03-status-inspect.png)

### 4. `[6]` Single-Window Burn (`--stage all`) With `CONTINUE SAME WINDOW` Checkpoint
On live hardware (`--execute`), each record additionally requires typing the exact row and hex word (`YES <row> <0xword>`), plus `SENSE` before `SENSE_CTRL`.
![vfi_burn Single-Window Burn Walkthrough](screenshots/04-single-window-burn.png)

### 5. `[8]` Granular Manual Overrides & `[10]` Offline 32-Bit FF Word Decoder
See the [README's Safety Model](../README.md#granular-manual-overrides-no-blanket---force) for the full list of overrides and their typed confirmation phrases.
![vfi_burn Manual Overrides and FF Word Decoder](screenshots/05-overrides-and-decode.png)

### 6. `[9]` Manual Low-Level Console — Fuse Controller State, Exact PCIe BAR0 Writes & Staged `preview` → `apply`
Run `preview <word>` (or `preview <target> <word> <description>`) to inspect the complete decoded GA100 fuse controller state (`FUSECTRL`, `DEBUGCTRL`, `CRC`, `DSOV`, `EFP/DFP/RESENSE_DIS`, `EN_SW/SBFE/WDATA_WR_SEC`, `PLM_f4/f8/fc`, VRM rail, host hold) and the exact ordered 32-bit aligned PCIe BAR0 MMIO writes — and **stage** that exact write so you simply type `apply` (no arguments, zero risk of re-typing typos; still gated by row/word confirmation & read-back) or `clear` to discard.
![vfi_burn Manual Console Fuse State and Exact PCIe BAR0 Writes](screenshots/06-manual-console-writes.png)
