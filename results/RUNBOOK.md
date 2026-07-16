# RUNBOOK

One line per launched run, per `experiment_ideas/training_run_plan.md` R5. Append-only, newest at
top. `status` is one of: `running`, `pass`, `fail`, `aborted`. Full milestone runs (M0-M7) must
additionally archive the exact `.cfg`, checkpoints, and `summary_<seed>.npy` files under
`results/<experiment_name>/<config>/` (git-ignored / stored off-repo — see note below); this table
is the index into that archive, not a replacement for it.

Config paths are relative to `configuration_files/`. Wall-clock is wall time for the run as
launched (not per-episode; see the run's own `summary_<seed>.npy` `time` field for per-episode
detail).

| Date | Tier/Milestone | Config | Commit | Seeds | Status | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|
| 2026-07-16 | S0 | `smoke/BEH26q_TNbond2_s0_smoke.cfg` (generated via `scripts/make_smoke_cfg.py --tier s0` from `TensorRL_fixed/BEH26q_TNbond2.cfg`) | 8dc5d70 | 1 (seed 0) | pass | ~2s (5 episodes) | T2 verification: generator produces a config that parses identically to the source and runs cleanly end-to-end; `summary_0.npy` schema matched `Saver`. Results deleted after verification (not archived). |
| 2026-07-16 | S0 (ad hoc, pre-generator) | hand-made temp copy of `TensorRL_fixed/BEH26q_TNbond2.cfg` (episodes=5) | 8dc5d70 | 1 (seed 0) | pass | ~2s (5 episodes) | T1 environment sanity: packages present, `.qpy` warm-start loads, no crashes, `summary_0.npy` schema correct. Temp cfg + results deleted afterward. |

---

**Note on archiving full runs**: this repo's `.gitignore` does not currently exclude `results/`;
before M0 launches (5 seeds x 4 configs + 2 extra arms), decide whether run artifacts
(checkpoints are tens of MB each) are committed, git-ignored and kept only on the run machine, or
pushed to external storage. Not yet decided -- flag before T3.
