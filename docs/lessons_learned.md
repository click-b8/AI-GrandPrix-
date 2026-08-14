# Lessons learned

Written for the next person who attempts this class of problem — including the falsified hypotheses, because those cost the most and taught the most.

## Engineering process

1. **Replay before you fly.** Every threshold swept offline against logged runs (commit rate/dwell, hold duration, the final latch rule) cost minutes; every threshold tuned by live iteration cost hours of flights. The final investigation answered ten questions and produced a validated rule change from 187k logged ticks and zero flights.
2. **Audit the metric before trusting the trend.** The "max-size frame" crossing metric was silently measuring post-plane spurious detections; it inverted a conclusion ("vertical is variance-limited") that survived days. The corrected true-crossing metric overturned the tuning direction.
3. **Real-time units or rate-dependent behavior.** Any rule counted in frames means different things at 20 vs 100 Hz. The one frame-count rule that shipped was later proven inert only because a seconds-based rule superseded it — by luck as much as design.
4. **Freeze-and-guard beats tune-and-hope.** Validated behavior became untouchable (bit-for-bit default-off tests, 538 total). Every regression this campaign suffered came from something *not yet* frozen.
5. **One variable per test — including the environment.** Battery state, power plans, and background load were flight variables. The lifecycle fix (stray sim processes merging UDP streams) retired a month of "mystery variance"; until then, half the tuning signal was noise.
6. **Correlation at n=302 is still not mechanism.** "Rate < 88 Hz selects the miss mode" fit at 77.8% and felt causal; per-leg analysis dissolved it in an afternoon. The discipline that saved us: before acting on a split, ask what *more precise* variable predicts better, and check the counterexamples (the best runs were the counterexamples).
7. **Name the failure honestly and it becomes an asset.** 0-for-546 with the whole error space sampled is not "bad luck" — it is proof of a systematic cause and an upper bound (p < 0.5%/attempt) that redirects effort from volume to mechanism.

## Automation

8. **Automate the loop, not the login.** The brittle GUI step (online account login) stayed human, once per night; everything repeatable (reset, launch, classify, purge) was automated. The result ran 546 attempts with zero interventions.
9. **Games eat synthetic input silently.** `SendKeys` does nothing to a fullscreen Unreal title; `SendInput` with scancodes works — and `Down` needs the extended-key flag or some builds read numpad-2. Every keystroke's effect must be verified by an observable (the flier's own state prints), not assumed.
10. **Two silent killers of child processes:** PowerShell's `Start-Process -ArgumentList` does not quote (a space in a path split an argument and the child died at argparse, in a way that *looked like* a timing bug for days), and Python block-buffers redirected stdout (`-u` or your readiness signal arrives after the timeout).
11. **Ordering around a countdown:** anything with a race-start gate demands arm-first sequencing — pause the world, attach the listener, then fire the start.
12. **When deleting data automatically, write the record first and resolve every uncertainty to "keep."** The purge deleted 3.5 GB overnight and lost nothing that mattered because summary rows and JSONs always preceded deletion.

## Strategy

13. **The last constraint is the real problem.** Gates 1–2 fell to tuning; Gate 3 was a different problem class (terminal blindness + a trajectory bifurcation upstream of every tunable). Recognizing a problem-class boundary earlier would have redirected weeks.
14. **A deadline is a forcing function for honesty.** The overnight batch was designed so that either outcome produced value: qualification, or proof of why not plus the dataset to find the mechanism. Design experiments so failure is informative.
