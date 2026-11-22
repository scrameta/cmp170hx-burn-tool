# vfi_burn: fuse-burn tool spec (first draft)

A design spec for a human-operated FF-record burn tool for the CMP 170HX 8 GB
(`10de:20c2`), to sit downstream of the glitch-and-read tools in
`cmp170hx-8gb-vfi-tools` and consume a held goldilocks frame. This is the step the
public repo leaves out on purpose. This document is a design draft, not a recipe:
the irreversible specifics (record encoding, write opcodes, voltage) are called out
as owner-supplied and are not guessed at here. See "Known gaps" at the end.

Status: draft for review. Nothing here has burned a card.

## 0. The one rule that shapes everything

A burn is permanent. A glitch is not. The whole of the existing repo is safe to
share because its worst failure is a power cycle. This tool's worst failure is a
dead card, and that asymmetry, not liability, is why it is a separate, private,
human-driven tool and not a `cimp` subcommand or an agent phase.

Two consequences, stated once and assumed everywhere below:

1. **You cannot integration-test an irreversible operation.** Offline tests against
   a fake controller prove the control flow, not the burn. Every brick is a test
   case you cannot re-run. "Passes the suite" means much less here than it does for
   the hunter. Treat the first real burn on any card as untested code.
2. **Read-back is the only thing standing between a bug and a brick.** Every record
   is written, read back, and verified before the free-slot pointer advances.
   Nothing is batched. A mismatch halts.

## 1. Scope

**In scope.** Take a held goldilocks frame (DSOV 0, CRC_STATUS `0x55`, PLM
`0xFFFFFFFF`, FUSECTRL idle), confirm host write access is actually open, then write
a reviewed per-card list of FF (field-programmable fuse) records one at a time, each
behind a read-back and a human yes, with a low-level manual console as a fallback.

**Out of scope, by design.**

- No glitching. The tool never dips the rail or pulls `PEX_RST` to *create* a frame.
  It is handed a frame that the existing tools are already holding, or it refuses.
- Not wired into `cimp hunt`, `bootrun.sh`, the planner or any autonomous loop. It is
  invoked by a person, at a prompt, after a hold. An unattended burn is the one
  combination to avoid: the irreversible step with nobody watching.
- Not distributed with any card's dumps, die IDs or record lists. The tool is
  generic; the payload is private and per-card (`.gitignore` the plans).
- No burn-planning intelligence. It does not decide *what* to burn. It executes a
  record list a human reviewed, and refuses records it cannot reconcile (see 4).

## 2. Safety invariants (never-rules)

Carried from `AGENTS.md`, plus the burn-specific ones:

1. Never glitch, sense, arm or write a fuse unless a held, write-capable goldilocks
   frame is verified present this instant (re-checked, not assumed from a log).
2. Never load a GPU driver; card bound to `pci-stub` or nothing; lockdown `[none]`.
3. Every MMIO write is one aligned 32-bit store (`memoryview(mm)[o:o+4].cast("I")`),
   never `struct.pack_into` (patch 0012: it zeroes the register first).
4. Never advance the record pointer without a verified read-back of the record just
   written. Never write two records before verifying the first. No batching.
5. On any poison (`0xBADFxxxx`), fuse-unit error (`0xBAD0xxxx`), latch
   (STATUS_WORD `0x2042`), PLM collapse, or read-back mismatch: stop writing, hold
   state, report, and hand to the person. Do not "recover and continue" a burn.
6. Never write `DEBUGCTRL` (`0x820060`) implicitly. If a write path legitimately
   needs it (unknown, see gaps), it is an explicit, separately-confirmed manual step
   with its own warning, never part of the automatic loop.
7. Die ID (`fuse rows 8..15`) must match the plan the record list was made for, read
   this boot, before the first write. Mismatch stops the run (exit as in repo).
8. A test row first (the lab burned a throwaway record before the real set). A burn
   is not verified by read-back; only a cold boot verifies it. Report what the cold
   boot shows, not what the write returned.
9. Keep every dump, die ID and record list at mode 0600 on the owner's machine.
   Never into a chat, issue, wiki or commit.

## 3. Entry state: is this frame actually write-capable

The lesson from card C: a frame can read as a passive goldilocks and still not
survive the active sense. The sense on card C threw `0xBADF510E`, and PLM_f8
(`0x8200F8`) had gone `0xFFFFFFFE` at the same moment: bit 0, host read access, had
closed. The controller was fine (the same sense passed on a cold card). The frame
was too deep.

So before any write, the tool runs a preflight on the held frame and stops unless all
of it holds:

| Check | Pass condition | On fail |
|---|---|---|
| Frame still held | DSOV `0`, CRC `0x55`, PLM `0xFFFFFFFF`, FUSECTRL idle `0xE0040000` | abort: frame lost, re-hunt |
| Host read access | PLM_f8 (`0x8200F8`) bit 0 set (`0xFFFFFFFF`, not `0xFFFFFFFE`) | abort: open window collapsed, too deep; re-hunt one step shallower or slower slew |
| Rail parked | READ_VOUT at stock (about 787.5 mV) under PMBus, not the ~0.3 V floor | park first; if the GPU is asking for the floor the link is dead, power cycle |
| Identity + die ID | GPC/SM masks and fuse rows 8..15 match the plan | stop: wrong card / wrong plan |
| Host state | no driver, `pci-stub`, lockdown off, VRM profile not tainted | stop with the specific line |
| Active-sense dry run | an owner-reviewed, non-committing sense reads clean and does NOT poison or drop PLM_f8 bit 0 | abort as "too deep": shallower / slower, do not go deeper |

The active-sense step is the gate card C teaches: a passive goldilocks whose sense
poisons is too deep. The tool never responds to a sense-poison by trying a deeper
frame.

## 4. The record model and the free slot

Inputs the tool consumes (none invented by the tool):

- **A reviewed record list**, per card, each entry: fuse address / record target, the
  value, a human-readable decode ("GPC_DISABLE -> 0x00", "LnkCap2 -> 16 GT/s"), and a
  flag for whether it changes a floorsweep/defect field.
- **The card's genuine defect mask**, from its cold snapshot, so re-enables can be
  reconciled.
- **The non-defective ceiling** for this SKU (community-derived, off-card: e.g. 122,
  126 with the RECONFIG override). The card cannot tell you this; the operator
  supplies it.

**Free-slot discovery.** Walk the FPF region, find the next empty record slot, and
*prove* the pointer before using it: re-read, confirm the slots below it are written
and the slot and everything above are empty, and show the operator the index and the
surrounding rows. A mis-located pointer writes into the wrong place, which is
unrecoverable. Ambiguity here halts and asks; it never guesses.

**Defect-mask reconciliation.** For every record that clears a disable bit, check it
against the card's real defect mask. Re-enabling a unit the mask says is genuinely
dead gives a card that posts the higher SM count and faults under load. The tool
classifies each change as: matches a known-good ceiling (ok), touches a genuine
defect bit (refuse), or cannot be determined from on-card data (halt and ask, because
the answer is external knowledge the operator must supply).

## 5. The write loop

For each record in the reviewed list, in order:

1. **Show.** Print the record: slot index, target, value, decode, defect flag, and the
   current read-back of that location.
2. **Reconcile.** Run the 4 check. Refuse or halt as above.
3. **Confirm.** Require an explicit yes. Dry-run mode shows the whole plan and writes
   nothing.
4. **Write.** One record, through the owner-supplied write primitive (see gaps),
   single aligned stores only.
5. **Read back.** Re-read the slot. It must equal what was intended, exactly.
6. **Poison/latch gate.** Re-read FUSECTRL first, then DSOV, CRC, PLM, PLM_f8,
   STATUS_WORD, READ_VOUT. Any `0xBADF`/`0xBAD0`, any PLM/PLM_f8 drop, any latch:
   stop (see 6).
7. **Advance** the pointer only now. Log the record (6). Next.

The pointer never moves on a write whose read-back did not verify. If step 5 or 6
fails, the tool stops with the slot state captured and hands over: the operator
decides whether a partial or wrong record landed and what to do, because that is the
one place "retry" can make it worse.

## 6. Decision points (what the tool shows, and who decides)

This is the enumeration the tool is built around: known situations resolve
automatically to the safe side; anything outside them halts and asks.

| Situation | What it shows | Auto or ask | Default-safe action |
|---|---|---|---|
| Frame still held, access open | all green | auto | proceed to write |
| PLM_f8 bit 0 cleared before any write | `0xFFFFFFFE` | auto-abort | stop: too deep, re-hunt shallower/slower, never deeper |
| Sense poisons (`0xBADF...`) pre-write | FUSECTRL/EN_SW `0xBADFxxxx` | auto-abort | stop: too deep; power cycle; re-hunt |
| Defect reconcile: known-good | within ceiling | auto | proceed |
| Defect reconcile: touches real defect | bit in the defect mask | auto-refuse | skip record, report |
| Defect reconcile: undetermined | not derivable on-card | **ask** | halt, operator supplies ceiling/intent |
| Free slot ambiguous | pointer not provable | **ask** | halt, show rows, operator confirms slot |
| Read-back mismatch after write | intended != read | auto-halt | stop, capture state, hand over (did it partially take?) |
| Poison during write | `0xBADFxxxx` mid-sequence | auto-halt | stop; the serious case: assess whether a record committed before any further action |
| Fuse-unit error | `0xBAD0xxxx` | auto-halt | stop; power cycle clears the unit, but assess first |
| Controller latch | STATUS_WORD `0x2042`, READ_VOUT ~41 mV | auto-halt | power cycle; tell the person |
| RECONFIG override record | enables beyond base ceiling | **ask** | off unless operator opts in, with the ceiling shown |
| PCIe gen record (GEN23/GEN3) | link-speed fuse | **ask** | show current LnkCap2, confirm target |
| Test row vs real record | first write of a session | **ask** | burn a throwaway test row first, verify, then the real set |

The rule behind the table: the tool automates only toward *not writing*. Every
automatic decision that is not "obviously safe to proceed" resolves to stop. Every
decision that needs knowledge the card does not contain is an ask, not a default.

## 7. Manual mode (the steering wheel)

A low-level fuse-controller console for when the plan does not fit the automatic path,
kept behind the same discipline. Manual means the operator chooses the target; it does
not mean the gates come off.

- **Read** any fuse row / register freely (read is safe; this is also how you prove a
  slot and diff before and after).
- **Write a single record** only: show, confirm, write one aligned store, read back,
  poison-gate, log. Identical to the loop body in 5, just with an operator-chosen
  target instead of a list entry.
- **Allowlist / denylist.** `DEBUGCTRL` (`0x820060`) is never written without its own
  separate, explicit, second confirmation and a printed warning. Reads outside the
  fuse/status set and any write are logged. The glitch command and the controller
  unlock stay owned by the firmware, as in the repo; manual mode does not reach past
  the documented interface.
- **Loud by default.** Every manual write prints what it targets, what it will become,
  and that it is permanent, before the confirm. A "poke anything, no read-back" mode
  is explicitly not provided: that is a brick machine, and removing the read-back is
  the exact footgun this whole tool exists to replace.

The analogy holds: automatic path for the known road, manual for when you need the
wheel, and the car still will not let you drive it into a wall without telling you
that is a wall.

## 8. Fault handling, in burn context

The repo's outcome ladder, re-read for a tool that may have already written:

- **Poison in the sense / before any write:** benign in the sense that nothing is
  permanent yet. Means too deep. Power cycle, re-hunt shallower or on a slower slew.
- **Poison after a write has started:** the serious one. Stop. The question is whether
  the last record committed. Capture the slot, do not write again, do not advance,
  hand to the person. Per-record read-back is what makes this answerable at all.
- **`0xBAD0xxxx`:** fuse unit did not answer (seen ~1-2% at large delays on one card).
  Treat as poison for safety: stop, assess, power cycle clears the unit.
- **Latch (`0x2042`):** controller output off. Power cycle. Nothing was being written
  through the controller's config; assess the last fuse record state on the cold boot.
- **PLM / PLM_f8 drop:** host access revoked mid-session. Stop; the window is gone.
- **`POISON_RESET`** (100 ms `PEX_RST` once GFW ready, from the runner) is a *hunt*
  recovery, for clearing poisons cheaply between shots. It is not part of the burn
  loop and must not be used to "clear and continue" a burn.

## 9. Verification

- Read-back verifies the *write*, not the *unlock*. Only a cold boot verifies the
  card. After the planned records, power cycle cold and re-snapshot.
- Diff the cold snapshot against the intended end state: GPC_DISABLE, FBP, the
  FBIO/ROP_L2/FBPA layout, MAGIC_D, BOOT_GEN3_DIS / DISABLE_GEN3_SPEED, LnkCap2, and
  the SM line (and the count under a driver that takes the card's geometry).
- Test row first, always. Burn one throwaway record, verify its read-back and its
  cold-boot effect, then the real set.

## 10. Logging and provenance

- One append-only log line per record: timestamp, card, slot index, target, intended
  value, read-back value, verify result, the full post-write gate read, and the
  operator's confirmation. Private, 0600, on the owner's machine, may hold identities.
- A session journal mirroring the repo's: card, held-frame source (which run/shot),
  records attempted, outcomes, the cold-boot verify. Every claim cites a log line.

## 11. Known gaps (must be settled on your card before any burn)

These are the parts I did not fill in because getting them wrong is the brick, and I
do not hold them reliably. They are inputs you verify from your own card's dumps and
the community sources, not things this spec supplies:

- **FF record binary encoding.** How a record encodes "set field X to value Y":
  layout, width, any checksum/parity per record. Owner-supplied from known-good
  dumps.
- **The fuse-controller WRITE path.** The exact command/opcode sequence to program a
  record via FUSECTRL (the read path is FUSEADDR + READ, FUSEADDR restored; the write
  path is not documented in the repo and is not assumed here), including whether
  `DEBUGCTRL` or any enable bit is required, and in what order.
- **Burn voltage.** Whether programming needs an elevated fuse-burn voltage applied
  externally, or the on-die mechanism suffices once the write path is enabled. Unknown
  to me; confirm before wiring anything.
- **Free-record terminator.** How "next empty slot" is marked, so the pointer logic is
  provably correct rather than heuristic.
- **RECONFIG override semantics.** How the override record interacts with the base
  ceiling and the defect masks.

Until each of these is pinned from your own hardware and verified read-only, the tool
should run in dry-run and manual-read-only mode only. The write primitives stay
unimplemented stubs that refuse until the operator wires in a verified, reviewed
implementation.

## 12. Testing

- Offline, against a fake fuse controller in the style of `tests/fake`, covering: the
  check gates, free-slot proving, defect reconciliation branches, the decision table,
  the read-back-mismatch halt, every poison/latch/PLM path, and that the pointer never
  advances without a verified read-back. These prove control flow only.
- The integration test does not exist and cannot: see 0. Start on a spare or an
  already-high card you can afford to lose, test row first, one record at a time,
  cold-boot verify between records for the first real session.
