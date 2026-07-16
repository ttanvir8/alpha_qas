# TensorZero-QAS — Training-Run Plan (smoke tests, full-run milestones, configs & benchmarks)

Companion to `implementation_plan_v1.md` (phases P0–P6) and `implementation_plan_v1_review.md`.
This document answers: **after which phase do we launch a full training run, what smoke test must
pass first, and which configs/benchmarks feed the A\* conference tables.**

All configs referenced under `configuration_files/` are the real shipped files
(`TensorRL_fixed/BEH26q_TNbond2.cfg`, `H2O8q_TNbond2.cfg`, `LIH12q_TNbond2.cfg`,
`H2O8q_noise.cfg`, …); new ones live under `configuration_files/TensorZero_fixed/` and
`configuration_files/smoke/`.

---

## 1. Global rules (apply to every milestone)

- **R1 — No full run without a passing smoke run** of the *exact same config*, with only
  `episodes`, `global_iters`, and seed count reduced. Smoke configs live in
  `configuration_files/smoke/` and are generated from the full config by a script
  (`scripts/make_smoke_cfg.py`), never hand-edited, so they cannot drift.
- **R2 — Flags-off parity gate before every milestone that changed `environments/` or `agents/`**:
  run the *unmodified baseline config* (e.g. `TensorRL_fixed/BEH26q_TNbond2.cfg`) through the new
  code with all new flags at their defaults, 1 seed, ~100 episodes, and diff the
  `summary_<seed>.npy` trajectory against a stored pre-change reference (same machine, same seed —
  qulacs/COBYLA/torch are deterministic under fixed seeds). Any drift = the change is not
  flag-gated correctly; fix before proceeding.
- **R3 — Seeds**: smoke = 1–2 seeds; standard full run = **5 seeds** (matches the paper's
  protocol); headline-table rows and success-probability claims = **10 seeds**.
- **R4 — Full training runs happen only at milestones M0–M7 below**, not after every commit.
  Between milestones, only S0/S1 smoke tiers run in CI/dev.
- **R5 — Every full run archives**: the exact `.cfg`, git commit hash, `summary_<seed>.npy`,
  checkpoints, and wall-clock — wall-clock is a first-class metric (ablation 1 is
  matched-wall-clock, and the reviewer promoted baseline re-running to a decision, see M0).

## 2. Smoke-test tiers

| Tier | Duration | What it is | Pass criteria |
|---|---|---|---|
| **S0 — mechanical** | minutes | `pytest` of the phase's new units (gadget predicates, Rotosolve index map, snapshot round-trip, attention forward shapes) **plus** a 5-episode run on `smoke/BEH26q_smoke.cfg` | No crash; `summary_<seed>.npy` written with **all** expected keys; checkpoint saves and *reloads/resumes* cleanly; rewards finite |
| **S1 — learning smoke** | 1–3 h CPU | 6-BeH₂ smoke config (`episodes = 300`, `global_iters = 200`), 2 seeds | Reward trend non-decreasing; ε (or MCTS policy entropy) decays; ≥1 success episode *or* monotone error improvement; **phase-specific signal** (see per-milestone rows) |
| **S2 — dress rehearsal** | overnight | The *exact* full-run config, 1 seed, full episode budget, smallest molecule of the milestone | Completes within the projected wall-clock ×1.5; checkpointing/restart verified mid-run; memory flat (no leak over thousands of episodes) |

S0 runs on every PR touching `environments/`/`agents/`. S1 gates every milestone. S2 gates only
the expensive milestones (M4–M7).

---

## 3. Milestones: when to do full training runs

### M0 — Baseline reproduction (BEFORE any new code lands; can run in parallel with P0 dev)

The reviewer's verdict made this a **decision, not an open question** (plan §6 Q6): ablation 1
(matched wall-clock) is invalid unless TensorRL-fixed baselines are re-run on **our** hardware.
M0 also produces the frozen `summary_<seed>.npy` references that rule R2's parity gate diffs
against.

- **Smoke**: S0 only (code is untouched; this validates the *environment setup*: `env` alias,
  requirements, qulacs build, `.qpy` warm-starts load).
- **Full run**: unmodified code, 5 seeds each on
  `TensorRL_fixed/BEH26q_TNbond2.cfg`, `TensorRL_fixed/H2O8q_TNbond2.cfg`,
  `TensorRL_fixed/H2O10q_TNbond2.cfg`, `TensorRL_fixed/LIH12q_TNbond2.cfg`,
  plus one `TensorRL_trainable` and one `StructureRL` arm on 8-H₂O (for ablation 6's
  warm-start-off comparisons later).
- **Pass / go-gate**: reproduce the NeurIPS Table 1 rows within seed variance —
  6-BeH₂: err ~6.8e-5, depth 7, CNOT 5, ROT 12; 8-H₂O: err ~8.9e-4, depth 6, CNOT 9, ROT 15;
  10-H₂O: err ~4.1e-4, CNOT 15; 12-LiH (fixed): err ~2.4e-2, CNOT 30. Success probability 100%
  of seeds at ≤10q. Record per-molecule wall-clock/episode → this is the matched-clock budget
  for M4.
- **If it fails**: stop. Nothing downstream is publishable until the baseline reproduces.

### M1 — after P0 (gadget rewards) [+ merged with M2 if P1 lands in the same window]

- **Smoke**: S0 (gadget predicate unit tests — each family fires on its canonical window, does
  not fire on shuffled windows, `actions_in_gadgets` mask prevents double-count) + S1 on
  `smoke/BEH26q_gadget_smoke.cfg`. Phase-specific S1 signal: `'gadgets'` log shows >0 completions;
  **anti-farming check** — fraction of episodes with gadget bonus but no energy improvement <10%
  (reviewer concern d.2).
- **Full run**: 6-BeH₂ + 8-H₂O, `TensorZero_fixed/{BEH26q,H2O8q}_TNbond2_gadget.cfg`,
  gadget-on (5 seeds) vs gadget-off (reuse M0, don't re-run).
- **Pass**: ≥15% median CNOT reduction on 6-BeH₂ at unchanged success probability (plan P0
  criterion). Secondary: no degradation on 8-H₂O.
- **Feeds paper**: ablation 2 (gadgets on/off) preliminary numbers; bonus-calibration sweep
  (one 3-point sweep on 6-BeH₂ only — `gadget_bonus_* ∈ {0.5×, 1×, 2×}` of defaults).

### M2 — after P1 (Rotosolve)

- **Smoke**: S0 (the `(layer, axis, qubit) → qulacs parameter index` unit test vs
  `construct_ansatz` layer-major ordering — reviewer E5, *silent-failure trap*; Rotosolve-vs-COBYLA
  agreement test on `LIH_4q` toy) + S1 on 6-BeH₂ with `method = rotosolve`.
  Phase-specific signal: summed `nfev` per episode drops ≥3× already in smoke.
- **Full run**: 8-H₂O, 5 seeds, `method = rotosolve, full_opt_every ∈ {1, 5, 20}` vs M0's
  `scipy_each_step`.
- **Pass**: ≥5× total-nfev reduction and ≤10% success-probability loss at `full_opt_every = 5`.
- **Feeds paper**: the nfev/wall-clock efficiency figure (analog of paper Fig. 3); picks the
  `full_opt_every` default used by all later milestones.

### M3 — after P2 (attention network)

- **Smoke**: S0 (forward-pass shape test for every state-tensor size in the benchmark set: 6, 8,
  10, 12 qubits × their `num_layers`; gradient-flow check — loss decreases on an overfit batch)
  + S1 on 6-BeH₂ with `network_type = axial_attention`. Phase signal: no NaN, GPU/CPU memory
  within 2× of MLP.
- **Full run**: 8-H₂O + 12-LiH, `network_type ∈ {mlp, axial_attention}`, 5 seeds, everything else
  fixed at M1/M2 winners.
- **Pass**: attention ≥ MLP success probability at 8q; strictly better episodes-to-threshold
  curve on 12-LiH. (If attention only wins at 12q, that is still the paper's scaling story —
  reviewer d.5 fallback.)
- **Feeds paper**: ablation 4 (network) at 8q/12q; the 20q/30q cells are filled at M6.

### M4 — after P3 (MCTS agent) — **the headline milestone**

- **Smoke**: S0 (snapshot/restore round-trip test over the *exhaustive* field list incl.
  `current_action, previous_action, energy, error, halting_step, current_number_of_cnots` —
  reviewer revision 5; tree invariants: visit counts sum, PUCT selects argmax) +
  S1 on 6-BeH₂ with `num_simulations = 25, surrogate_level = 1` (cheapest tree) +
  **S2 overnight** on 8-H₂O, 1 seed, `num_simulations = 100, surrogate_level = 0`, to measure
  per-move wall-clock and calibrate the matched-clock budget against M0's recorded baseline clock.
- **Full run**: 8-H₂O, 5 seeds per arm:
  - DDQN baseline = M0 (given equal wall-clock budget, i.e. allowed to run more episodes),
  - MCTS `num_simulations ∈ {50, 100}` × `surrogate_level ∈ {0, 1}` (4 arms).
- **Pass**: ≥1 MCTS arm beats DDQN success probability at matched wall-clock, or reaches equal
  accuracy with fewer *committed* env steps. This is ablation 1 and headline claim territory —
  if all arms fail, the paper pivots to P0–P2 improvements on DDQN (plan §P3 fallback) and M5
  is re-scoped to pretrain the DDQN's policy net instead.
- **Feeds paper**: ablation 1; ablation 5 (surrogate level) partial.

### M5 — after P4 (demonstrations, re-scoped per review E4)

- **Smoke**: S0 (demo generator: legality of every sampled action, gadget-splice correctness;
  pretraining loss decreases on 1k demos) + S1: pretrained vs scratch MCTS agent on 6-BeH₂,
  300 episodes — pretrained must not be *worse*.
- **Full run**: 8-H₂O, 5 seeds per arm: {no pretrain, random-demo policy-only pretrain,
  DMRG-Δχ pretrain, mixed} (4 arms; value-head supervision only from Δχ/greedy-rollout demos,
  per reviewer E4 re-scope).
- **Pass**: best pretrained arm reaches first success in ≤50% of the episodes of no-pretrain.
- **Feeds paper**: ablation 3 (demos on/off + source ablation).

### M6 — after P5 (MPS surrogate, scale-up)

- **Smoke**: S0 (MPS-vs-statevector energy agreement on 6-BeH₂ to 1e-6 at χ=64; the **rank-
  correlation experiment** on 10-H₂O — 1k tree nodes, MPS-χ16 vs statevector, ρ ≥ 0.9 — is itself
  cheap enough to be the smoke gate) + S1: TFIM-20q, 50 episodes — verifies the pauli-only npz
  path, that **no dense-matrix code path is hit in `__init__` *or* `reset()`** (reviewer E2), and
  flat memory.
- **Full run**: (a) 10-H₂O with `surrogate_level = 2` in-tree vs level 0, 5 seeds (fills the rest
  of ablation 5); (b) TFIM-20q, 5 seeds, vs the paper's 20q TFIM appendix result and the DMRG
  warm-start energy; (c) 30q attempt, 3 seeds, reported vs `fake_min_energy` = DMRG-χmax
  reference. **Every final circuit re-evaluated at high χ (χ ≥ 4× commit_bond_dim)** before any
  energy is quoted (reviewer d.4).
- **Pass**: (a) no accuracy loss at ≥5× leaf-eval speedup; (b) beats warm-start energy at 20q.
  30q is stretch — a negative result caps the claim at 20q (plan fallback) without blocking M7.
- **Feeds paper**: headline claim 2 (largest RL-QAS), ablation 5 complete.

### M7 — P6: the publication grid (final full-training campaign)

- **Smoke**: R2 parity gate re-run for *all five* env variants after the noise-file replication
  (`environment_qulacs_TN_notin_agent_noise.py` etc.); S2 on one noisy config
  (`H2O8q_noise`-derived) because noise sims are the slowest per episode.
- **Full run — the A\* tables** (see §4 for the exact matrix): every benchmark × the best
  TensorZero config × ablation arms, **5 seeds everywhere, 10 seeds on headline rows**
  (8-H₂O, 10-H₂O noiseless; 8-H₂O noisy).
- **Pass**: headline ≥2× CNOT/depth reduction over TensorRL-fixed at equal accuracy on ≥2
  molecules; noisy success probability ≥ TensorRL-fixed's 100%/depolarizing row.
- **Also at M7**: `analysis/motif_stats.py` n-gram mining over all successful episodes (the
  "emergent motif" section), and reproduction-of-baseline check one final time on the frozen
  release commit.

---

## 4. Config × benchmark matrix for the publication

Benchmarks and the baseline numbers each cell competes against (from the NeurIPS 2025 paper,
reproduced at M0 on our hardware):

| System | Config family | Baseline to beat (TensorRL-fixed) | Milestones that fill it |
|---|---|---|---|
| 6-BeH₂ | `BEH26q_TNbond2*` | err 6.8e-5, D7, CNOT 5, ROT 12 | M1, M7 |
| 8-H₂O | `H2O8q_TNbond2*` | err 8.9e-4, D6, CNOT 9, ROT 15 | M1–M5, M7 (10 seeds) |
| 10-H₂O | `H2O10q_TNbond2*` | err 4.1e-4, D17, CNOT 15, ROT 17 | M6a, M7 (10 seeds) |
| 12-LiH | `LIH12q_TNbond2*` | err 2.4e-2 (fixed) / 1.0e-2 (trainable), CNOT 30/37 | M3, M7 |
| 8/10-CH₂ | `CH2_{8,10}q*` (new cfgs) | appendix-level comparison | M7 only |
| TFIM 20q | new `tfim_20q*` (pauli-only npz) | paper's 20q TFIM appendix + DMRG warm-start energy | M6b |
| TFIM/Heis. 30q | new `*_30q*` | DMRG-χmax reference (`fake_min_energy`) | M6c (stretch) |
| 8-H₂O noisy | `H2O8q_noise*` (depolarizing 1%/5%, 10⁴ shots) | err 9.0e-4, D7, CNOT 5, 100% success | M7 (10 seeds) |

Ablation arms (idea doc §4.3) and where they run — all on 8-H₂O unless noted:

| Ablation | Arms | Milestone |
|---|---|---|
| 1. MCTS vs DDQN (matched wall-clock) | DDQN(M0) vs MCTS×{50,100 sims} | M4 |
| 2. Gadgets on/off (+ per-family) | off / all / leave-one-family-out (4 arms, M7) | M1, M7 |
| 3. Demos on/off + source | none / random / Δχ / mixed | M5 |
| 4. Network MLP vs attention | 8q, 12q (+20q at M6) | M3, M6 |
| 5. Surrogate level 0/1/2 | in-tree, 8–10q | M4, M6a |
| 6. Warm-start on/off under new stack | TN-init vs empty-init, best config | M7 |

## 5. Compute budgeting & scheduling notes

- M0–M3 are CPU-feasible (the paper trains ≤8q on CPU; 6-BeH₂ full run ≈ 1–2 h/seed, 8-H₂O
  ≈ overnight/seed on the M0-measured clock). Run seeds in parallel — everything is
  embarrassingly parallel over (config, seed).
- M4's grid is 4 arms × 5 seeds × matched-clock budget: reserve HPC; the S2 rehearsal's measured
  per-move cost decides between {50,100} sims *before* committing the grid.
- M5 demo generation (100k random + ~200 Δχ demos) is a one-off CPU batch job, cacheable in
  `demos/data/`.
- M6/M7 are the only GPU-relevant stages (attention net at 20–30q; noise sims). Noisy episodes
  are ~10× noiseless cost — budget M7's noisy rows first.
- Keep a running `results/RUNBOOK.md`: one line per launched run (config, commit, seeds, status,
  wall-clock) — this becomes the reproducibility appendix.

## 6. Decision points recap (go/no-go tree)

```
M0 pass? ──no──► stop, fix environment/repro
   │yes
M1/M2/M3 (independent) ──each fail──► drop that ingredient's arm (paper survives; ablation reports it)
   │ ≥1 pass
M4 pass? ──no──► pivot: paper = P0–P2 on DDQN (still ≥3 ablations + efficiency story); skip M5-as-planned
   │yes
M5, M6a ──fail──► report as negative ablation results (still publishable content)
M6b pass? ──no──► cap scale claim at 12q, drop headline 2
M7 = final tables regardless of M5/M6 outcomes
```
