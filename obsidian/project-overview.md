#project-overview

# Project overview

## What this is

Scuba Lab's entry for the [Anduril AI Grand Prix](https://www.aigrandprix.com/),
an autonomous-drone racing competition run in partnership with the Drone
Champions League (DCL), Neros Technologies, and JobsOhio. The competition
runs a virtual qualifier series in mid-2026, followed by a physical
qualifier in the fall and a championship in November.

We build a vision-based racing policy in simulation, train it with PPO,
and plan to submit it to the DCL simulator once the simulator ships (May
2026).

## Competition timeline (all dates 2026)

| Stage | Month | What's required |
|---|---|---|
| Virtual Qualifier 1 (VQ1) | May | Code submission to DCL sim; desaturated / simplified gates |
| Virtual Qualifier 2 (VQ2) | June | Code submission; complex, low-SNR imagery, no visual aids |
| Physical Qualifier | September | In-person in Southern California |
| Final Championship | November | Ohio |

VQ1 is the next forcing function. Today is 2026-04-21. See
[[submission-readiness]] for what would break if we had to submit right
now — which is **most of the deployment layer**.

## What "done" looks like

**For the project overall:** a trained vision policy that finishes all 8
gates reliably on the DCL simulator, emitting spec-compliant MAVLink v2
commands via UDP at 50–120 Hz.

**For VQ1 specifically:**
1. A model that drives to 8/8 gate completion (have it — see [[models#aigp_distill_final]]).
2. An adapter that loads the trained weights end-to-end. **Do not have it.**
   The current `dcl_adapter.py` silently loads zero policy-head weights
   and runs on random init. See [[fragilities#The policy head is untrained at deployment]].
3. A MAVLink layer that emits spec-compliant `SET_ATTITUDE_TARGET` or
   `SET_POSITION_TARGET_LOCAL_NED` frames. **Do not have it.** The existing
   `dcl_mavlink_adapter.py` has a zeroed CRC, no UDP transport, a
   malformed payload, and a semantic bug that treats body-rate commands
   as attitude setpoints. See [[fragilities]].
4. Real vision frames flowing from the DCL sim, not zero placeholders.
   See [[deployment#Vision stream is a placeholder]].

## What Scuba Lab has built

- A MuJoCo-based racing simulator with curriculum learning, domain
  randomization, VIO noise, motion blur, and a simulated event camera
  (EGM model from Zou et al. 2026). See [[simulation]] and [[perception]].
- A coarse-to-fine CNN + state-MLP feature extractor feeding a PPO
  policy. See [[vision-model]].
- Three trained artifacts: `aigp_8gates_final`, `aigp_racer_final`
  (state-based, privileged observation) and `aigp_distill_final`
  (vision, distilled from the state expert). See [[models]].
- An adapter framework intended to bridge the trained model to the
  DCL competition API. **The framework exists; it does not work
  end-to-end.** See [[deployment]].

## What Scuba Lab has NOT done

- Measured real inference time on competition-class hardware (we only
  have a number on an M4 Mac — see [[fragilities#Inference time]]).
- Validated the adapter loads weights correctly. (It doesn't; see
  [[fragilities#The policy head is untrained at deployment]]).
- Validated MAVLink frames against a real parser (they wouldn't parse;
  CRC is zeroed).
- Tested on any actual drone.
- Received the DCL simulator (expected May 2026).

## Honest state

The headline claims in the top-level docs ("✓ Training Complete",
"✓ Adapter Ready", "Status: READY FOR DCL INTEGRATION", "✅ READY FOR
COMPETITION") describe an aspiration, not a validated state. The
training is in fact complete. The adapter and the MAVLink layer
contain silent, compounding failures that would make the submission
fly terribly, not fail loudly. See [[fragilities#Silent failure chain]]
for the category-level picture, and [[submission-readiness]] for
the operational consequence.

## See also

- [[fragilities]] — what's broken
- [[submission-readiness]] — what to do about it before VQ1
- [[competition]] — the official specs
- [[models]] — the model registry
