# Testing

**Verified count: 424 passed, 114 skipped** (538 collected) on a clean clone at the released commit. The suite's job during the campaign was specific: make *frozen* behavior untouchable while experiments churned around it. Run it:

```bash
python -m pytest tests -q
```

`tests/conftest.py` adds the relocated entry-point directories (`analysis/`, `archive/dcl-hardware/`, `archive/rl-training/`) to `sys.path` — archived-era integration tests are still exercised rather than deleted.

## What the suite actually verifies

**Controller logic (synthetic-input unit tests).** The commit latch family is the deepest: arming preconditions (previous-gate residual must clear), the alignment band, the derivative rule, real-time dwell accumulation, and the pinned fact that the frame-count rule is *inert* when the seconds rule is active (`test_stable_s_supersedes_the_frame_count`) — the test that later killed a wrong causal theory during the forensics. Similar coverage for the terminal descent (arm/release hysteresis), the blind hold (arm on valid→invalid with descent active; release on timer/reacquisition/leg advance), the exit-level latch, entry arrest, and the vertical floor.

**Default-off / bit-for-bit regression checks.** Every experimental mechanism must, with its flag absent, leave the controller's outputs *byte-identical* to the pre-mechanism baseline on recorded scenarios — and legs below the mechanism's scope (`ag < 2` for Gate-3 logic) must be untouched even with the flag on. These are the tests that let a 3,900-line file absorb ~150 experiments without silent drift.

**CSV schema alignment.** The ~94-column telemetry header and row emission are AST-checked for column-count agreement — a mismatch corrupts every downstream analysis, so it fails loudly at test time instead.

**Batch automation (`test_qualifier_batch.py`, ~1,100 lines).** Simulator lifecycle self-tests (PID-set differencing, orphan detection, port-release verification against real processes), keystroke sequencing (exact Esc→launch→settle→Enter order; the extended-key flag on Down), classification ordering (FULL_COURSE only stop; purge-after-record; uncertainty resolves to keep), and stray-flier cleanup against a real spawned process.

**Protocol & contracts.** MAVLink compliance (message framing, arm/GO sequencing), the sim's observation contract, and detector/rail regression fixtures.

**Replay testing as method.** Beyond pytest, the campaign's control-rule changes were validated by *offline replay* against logged tick data before ever flying — the commit-rule sweep in [findings.md](findings.md) (256 rule variants × 302 runs) is the pattern at full scale.

## Skips, honestly

Every one of the 114 skips is gated on a data file this repository cannot ship, and each states its real reason at the skip site:

| skips | reason | why it can't ship |
|---|---|---|
| 108 | a named flight log is absent (`filt4/5/6/9/29.csv`, `flt9`/`flt11`) | the tuning-run corpus is 489 MB across 264 files — permanently out of scope for a git repository |
| 6 | `aigp_distill_final.zip` absent at the canonical path | the distilled policy weights from the archived RL era, likewise too large |

These are *replay* tests: they assert controller behavior against real recorded ticks, so without the recording there is nothing to assert. Run with the corpus present and they execute; run on a fresh clone and they say exactly what is missing. A skip with a true reason beats a green lie or a deleted test.

Two consequences worth stating plainly. First, **the 424 that pass on a clean clone are the reproducible core** — synthetic-input logic, schema, protocol, automation — and the badge reports that number, not the larger figure available on a machine holding the corpus. Second, the detector tests are *not* skipped: `tests/fixtures/vision_frame.png` is tracked precisely so those 25 run everywhere, having previously been skipping silently on every clone because the fixture lived in gitignored bulk data.

Note that the batch-automation tests spawn real PowerShell processes with generous timeouts. Under heavy parallel load one of them can exceed its budget; it is timing-sensitive, not a behavioral failure.
