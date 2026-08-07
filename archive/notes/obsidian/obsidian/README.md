#wiki-index

# AI Grand Prix / Scuba Lab — project wiki

This wiki is the reference for things that aren't obvious from reading the
code: decisions, trade-offs, silent load-bearing constants, drift between
documents, and the measurement trail. It was written after an audit that
exposed several serious deployment-layer problems. **Read [[fragilities]]
and [[submission-readiness]] before touching any deployment code.**

The top-level project READMEs (`README.md`, `README_SCUBA_LAB.md`,
`DCL_INTEGRATION.md`, `COMPETITION_STRATEGY.md`, `COMPETITION_CHECKLIST.md`,
`README_COMPETITION.md`) project confidence that the audit did not
substantiate. Where those documents and this wiki disagree, trust the wiki.

## Orientation by question

| If you're asking... | Start at |
|---|---|
| What is this project, who are we, what phase are we in? | [[project-overview]] |
| How does the whole system fit together? | [[architecture]] |
| What's actually broken, and what's just drift? | [[fragilities]] |
| Can we submit for Virtual Qualifier 1 today? | [[submission-readiness]] |
| Which model should I deploy, and where does it live? | [[models]] |
| How does the simulator work, what are the rewards? | [[simulation]] |
| How is the FPV / event camera modelled? | [[perception]] |
| What's the policy network, and why is it shaped this way? | [[vision-model]] |
| How were the models trained, what survived? | [[training]], [[experiments-log]] |
| How does deployment work, and what doesn't? | [[deployment]] |
| What does the competition spec say? | [[competition]] |
| Why did we choose X over Y? | [[decisions-log]] |
| What don't we know yet? | [[open-questions]] |
| What would we do if we had more time? | [[future-work]] |

## Suggested reading order for a newcomer

1. [[project-overview]] — 10 min, enough to know what the project is.
2. [[fragilities]] — 15 min. Do this before anything else touching deployment.
3. [[submission-readiness]] — 5 min. What works, what doesn't, right now.
4. [[architecture]] — 15 min. How the pieces fit.
5. [[models]] — 5 min. The registry; which file to load.
6. Whatever you actually need for your task.

## How the audit was conducted

This wiki was produced from a bottom-up scan of every `.py`, `.md`, and
`.sh` file in the project (excluding `venv*/`, `__pycache__/`, and
`event-sharp-nerf-drones/`), cross-referenced against MODELS.json, the
actual `.zip` artifacts on disk, git history in both the root repo and
the nested `drone-race-sim/` repo, the raw PyTorch checkpoint
(`drone-race-sim/trained_distilled/policy.pth`), and one on-device
inference timing measurement. Where the source documents disagreed, the
wiki resolved to a canonical answer and noted the historical claim.
Where the source documents claimed something that a physical check
contradicted, the wiki records the measurement and flags the prior
claim as wrong.

See [[fragilities#Measurement status]] for which drifts were resolved
by measurement vs by choice vs still open.

## See also

- [[fragilities]] — the most important file
- [[submission-readiness]] — operational checklist
- [[decisions-log]] — why we chose what we chose
- [[open-questions]] — what we don't know
