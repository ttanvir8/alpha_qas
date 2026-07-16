# TensorZero-QAS — Training & Code Track Log

**Created:** 2026-07-16
**Purpose:** A single systematic record of (1) how the current task list was derived, (2) which
files were read/used to reason about it, (3) the decisions made along the way (with their
justification and source), and (4) the resulting ordered task list. Future sessions should append
to Section 6 (Run/Change Log) rather than rewriting history.

---

## 1. Document provenance — how we arrived at this task list

The task list was derived by reading the four planning documents in `experiment_ideas/` in the
order they were produced. Each document builds on and corrects the previous one:

| # | Document | Role | Status |
|---|---|---|---|
| 1 | `experiment_ideas/a_star_paper_idea1.md` | The paper idea: **TensorZero-QAS** — transplant five ingredients from AlphaTensor-Quantum (MCTS planning, gadget-shaped rewards, symmetrized axial attention, synthetic demonstrations, symmetry augmentation) into this repo's TensorRL-QAS pipeline. Defines phases **P0–P6**, benchmarks, ablations 1–6, and headline claims. | Idea doc; contains known small errors corrected by doc #2 §5 (e.g. CH₂O → CH₂) |
| 2 | `experiment_ideas/implementation_plan_v1.md` (rev **v1.1**) | File-by-file engineering plan mapping P0–P6 onto this repo: keep-table (§1), AlphaTensor port table (§2), per-phase touchpoints/configs/validation (§3), dependency graph (§4), corrections to the idea doc (§5), open questions (§6), accepted review amendments (§7), phase→milestone map (§8). | **Current plan of record** (already amended to v1.1 after review) |
| 3 | `experiment_ideas/implementation_plan_v1_review.md` | Independent code-verified review of the plan against both ground-truth codebases. Verdict: "needs revision — but fundamentally sound", feasibility grade B. Found errors E1–E6 and design concerns d.1–d.6; issued a 9-item revision list. | **All 9 revision items accepted** into plan v1.1 (see review §(g) addendum) |
| 4 | `experiment_ideas/training_run_plan.md` | Operational run schedule: global rules R1–R5, smoke tiers S0/S1/S2, full-run milestones **M0–M7** mapped to phases, publication config×benchmark matrix, compute budgeting, go/no-go decision tree. | Governs when training runs happen |

**Chain of reasoning:** idea (#1) → engineering plan (#2, v1) → review (#3) → amendments folded
back into #2 (v1.1) → run schedule (#4). The task list in Section 4 below is the entry point that
plan §8 and the run plan's M0 milestone jointly prescribe.

---

## 2. Code file paths used to reason (ground truth)

The plan and review verified their claims against two codebases. These are the files the task
list's touchpoints refer to.

### 2.1 This repo (TensorRL-QAS) — `/mnt/c/Users/LEGION/Work/quantum_root/versions_of_tensor_rl/quary_original_tensor_rl/`

| Path | Why it matters to the task list |
|---|---|
| `TensorRL_fixed_noiseless.py` | Baseline entry script (`Saver`, `one_episode`, `train`, `agent_test`, `modify_state`). Stays **untouched**; new entry scripts are clones. Known blocker: interactive `input("Enter 0 or 1")` device prompt at line ~206 blocks batch jobs. `init_net` reload block at lines 239–252; `agent.saver` injection idiom at line 237. |
| `TensorRL_fixed_noise.py`, `TensorRL_training_and_structureRL_noiseless.py`, `Tensor_training_and_structureRL_noise.py`, `*_restricted.py` | Baseline variants for M0/M7; untouched. |
| `environments/environment_qulacs_TN_notin_agent.py` (`CircuitEnv`) | **Main integration point.** Action decode at lines ~255–258; `rwd = self.reward_fn(energy)` at ~306; optimizer dispatch `scipy_each_step` at ~283; `scipy_optim` method at ~452; dense `Statevector`/`Operator` loads in `__init__` (127–131, 158, 162) **and** `reset()` (372–377); hardcoded `['heisenberg','tfim_j1_h0.05']` at 78/126/372; `fake_min_energy` at 60; unused `cnot_rwd_weight` at 63. Latent bugs to fix in passing: `self.TN_state = Statevector(self.tenor_circ)` (158) runs even when `tn_bond = 0` (NameError); `nfev`/`opt_ang_save` undefined when `optim_method != "scipy_each_step"`. |
| `environments/environment_qulacs.py`, `*_noise*.py` variants | Modified only in P6 (flag-gated diff replication). |
| `environments/VQAs/VQE_qulacs_TN_notin_RL.py` | Cost functions: `Parametric_Circuit.construct_ansatz` (layer-major parameter enumeration, lines 13–45 — the E5 index-mapping trap), `get_energy_qulacs(..., which_angles=[])` at 48–50 (already supports per-index angle setting — enables Rotosolve). |
| `environments/utils/utils.py` | `get_config` lines 19–34: new float/list config keys **must** be registered in the `floats`/`lists` lists or they parse as strings. `dictionary_of_actions`, `dict_of_actions_revert_q` reused by gadgets/augmentation/demos. |
| `environments/utils/curricula.py` | `VanillaCurriculum`, `MovingThreshold`, `SuccesCountThreshold` — untouched, orthogonal. |
| `agents/DeepQ.py` | `DQN.unpack_network` at line 147 (P2 branch point); `__init__` already handles `angles=0` / `en_state` / `threshold_in_state` sizing (lines 43–46). |
| `agents/DeepQNstep.py` | `DQN_Nstep` + `N_step_ReplayMemory` — the shipped baseline agent (`agent_type = DeepQNstep`); untouched. |
| `agents/__init__.py` | Imports only `DeepQ_restricted, DeepQNstep, DeepQNstep_restricted`; entry scripts resolve agents via `agents.__dict__[agent_type].__dict__[agent_class]` — every new agent module must be registered here. |
| `configuration_files/TensorRL_fixed/*.cfg` | Shipped baseline configs: `BEH26q_TNbond2.cfg` (`num_layers=47`, `accept_err=1.6e-3`, `agent_type=DeepQNstep`, `method=scipy_each_step`, `optim_alg=COBYLA`, `global_iters=1000`), `H2O8q_TNbond2.cfg`, `H2O10q_TNbond2.cfg`, `LIH12q_TNbond2.cfg`, `H2O8q_noise.cfg`. Must keep reproducing old numbers. |
| `dmrg-to-qc/` (`dmrg_to_qc.py`, `mps2qc.py`, `stiefel_opt.py`, `tnqc_ansatze.py`) | Phase-A warm-start generation — hard keep, untouched. P4 calls `mps2qc.gs_dmrg` (returns `(mps, dmrg)` tuple) / `mps_to_qc` (random-initializes every gate at `mps2qc.py:300` — hence the frozen-prefix fitting requirement) / `mpo_from_paulis` (takes `dict[str, complex]`, needs a zip-adapter) as a library. |
| `dmrg-to-qc/heisenberg_model.py` | Builds **dense 2ⁿ×2ⁿ kron matrices, Heisenberg only** (lines 7–60) — cannot emit 20–30q data (review E2); a new sparse/MPO generator is required for P5. |
| `dmrg-to-qc/mol_data/*.npz` | Benchmark Hamiltonians: `CH2_8q/CH2_10q` exist (no CH₂O); `LIH_4q_..._parity.npz` has no `paulis` key; only `heisenberg_5q.npz` and `tfim_j1_h0.001_6q.npz` for spin models. Untouched; new >12q npz files are additive. |
| `dmrg-to-qc/init_state_circ/*` | `.qpy` warm-start artifacts (bond-2/3 for H₂O exist). Untouched. |

### 2.2 AlphaTensor-Quantum (port source) — `/mnt/c/Users/LEGION/Work/quantum_root/projects/rl-cut/current_works/alphatensor_quantum/`

| Path | What is ported |
|---|---|
| `src/factors.py` | Gadget mechanism: sliding-window predicates + fixed reward savings (`TOFFOLI_REWARD_SAVING=5.0`, `CS_REWARD_SAVING=1.0`) → logic-port to P0's `gadgets.py`. |
| `src/environment.py` (lines 141–185) | Gadget check ordering, `factors_in_gadgets` double-count guard, reward composition; bonus granted **on the completing step** (no stale-reward problem in replay buffers). |
| `src/networks.py` | `Symmetrization`, `_SelfAttention`, `_TransformerDecoderBlock`, `_SymmetrizedAxialAttention`, `TorsoNetwork` → 1:1 JAX/Haiku→PyTorch port in P2. |
| `src/demo/agent.py` | `_recurrent_fn` (deterministic env ⇒ real env states in tree), `_loss_fn` (KL policy + value loss, `(1−w)·acting + w·demos` mixing), `GameStats` best-ever tracking → P3/P4. **Sampled MCTS is NOT in the open code** (it uses full-width `mctx.muzero_policy`) — K-sampled PUCT must be written fresh. |
| `src/demonstrations.py` | `Demonstration` container, gadget injection (`prob_include_gadget=0.9`) → P4 (heavily re-scoped, see decision D6). |
| `src/change_of_basis.py`, `src/config.py`, `src/demo/demo_config.py` | Reference-only: augmentation concept and knob defaults (`demonstrations_boundaries_and_scales = {60:0.99, 200:0.5, 5000:0.2, 10000:0.1}`). |

License note: AlphaTensor-Quantum is Apache-2.0 — logic ports with attribution are fine.

---

## 3. Decisions made (with source and rationale)

### 3.1 Resolved decisions

| ID | Decision | Rationale | Source |
|---|---|---|---|
| D1 | **Start at M0 (baseline reproduction) before any new code lands.** Re-run TensorRL-fixed baselines on project hardware; promoted from open question to decision. | Ablation 1 (matched wall-clock MCTS-vs-DDQN) is invalid without same-hardware baselines; M0 outputs double as R2 parity references. | Review §(d).3 + §(f).4; plan §6 Q6 (RESOLVED); run plan M0 |
| D2 | **Pipeline-parity ground rule:** all env/agent changes flag-gated with defaults = old behavior; baseline entry scripts/configs untouched; new entry scripts are clones; `summary_<seed>.npy` schema stays byte-compatible. | Every run must stay directly comparable to the NeurIPS 2025 tables. | Plan header + §1 |
| D3 | **Gadget logging lives in the cloned entry script** `TensorZero_fixed_noiseless.py` (created early, in P0, not P3); `TensorRL_fixed_noiseless.py` stays truly untouched. The clone also replaces the interactive device `input()` with a `--device` flag. | Resolves the plan's internal contradiction (review E1) and the HPC batch blocker. | Plan §7.1; review E1 |
| D4 | **Anti-farming guard on gadget bonuses:** bonus paid only when the completing step does not increase energy + per-episode cap `max_gadgets_per_episode` (default 5). Paper must present bonuses as *shaping*, not objective correction; motif predicates are structural, not unitary-level identities (no particle-conservation claims for bare `CNOT·RY·CNOT`). | Unlike AlphaTensor (where savings equal true cost), here an agent could farm ≈ +3–4/episode from tiled motifs with zero energy progress, competing with the +5 success reward. | Plan §7.2; review d.2 |
| D5 | **Rotosolve needs an explicit `(layer, axis, qubit) → qulacs parameter index` helper** with a unit test against `construct_ansatz`'s layer-major ordering. | The newest gate is not in general the last parameter (insertion at `moments[rot_qubit]` can be an earlier layer); getting it wrong silently optimizes the wrong angle. | Plan §7.3; review E5 |
| D6 | **P4 re-scoped:** random-circuit demos are *policy-head-only* pretraining (legality + motif priors) — they are NOT solutions (no target-conditioning in TensorRL's observation, unlike AlphaTensor). Value-head supervision only from (a) greedy/Rotosolve descent rollouts and (b) DMRG-Δχ demos, which require **new frozen-prefix fitting code** (`mps_to_qc` random-initializes every gate, so two independent fits share no common prefix). DQN behavior-cloning aside dropped. | Random demos here are behavior-cloning noise for the value head; the property that made AlphaTensor's demos work (target defined by the solution) does not transfer. | Plan §7.6; review E4 |
| D7 | **P5 data path rewritten:** pauli-strings-only npz schema for >12q (no dense `hamiltonian`/`eigvals`; DMRG reference energy via `fake_min_energy`), a new sparse/MPO generator (dense-kron `heisenberg_model.py` cannot scale and has no TFIM), the `leaf_evaluator = mps` branch must bypass dense loads in **both `__init__` and `reset()`**, and final 20–30q circuits are re-evaluated at high χ (≥4× `commit_bond_dim`) before any energy is quoted. | Dense path is a 17 TB array at 20q; `reset()` was missing from the original mod list; χ-truncated ⟨H⟩ is the energy of a different state. | Plan §7.7; review E2 + d.4 |
| D8 | **P3 interface decisions:** env injected into `MCTSAgent` post-construction (`agent.env = env`, mirroring the `agent.saver` idiom); value targets `z` backfilled at episode end; adapted checkpoint format (no `target_net`, no `Transition` buffer); snapshot/restore covers the exhaustive field list (`current_action`, `previous_action`, `energy`, `error`, `halting_step`, `current_number_of_cnots`, plus state/moments/illegal_actions/step_counter/prev_energy/action_history/actions_in_gadgets/threshold) with a round-trip unit test. | The plan's original "two-line delta" understated the surgery; tree search needs the live env. | Plan §7.4–7.5; review E3 |
| D9 | **P2 paper framing:** the torso is "axial attention with an auxiliary symmetrized qubit×qubit branch", not "symmetrized axial attention" wholesale (depth×qubit grid is not transpose-symmetric; symmetrization applies to the N×N CNOT co-occurrence branch only). | Honest framing of the one deliberate deviation from the source network. | Plan §7.9; review d.5 |
| D10 | **Small interface fixes adopted:** dict-adapter for `mpo_from_paulis`; `gs_dmrg` returns a `(mps, dmrg)` tuple; noiseless trainable script is `TensorRL_training_and_structureRL_noiseless.py`; `environment_qulacs_noise.py` added to P6's replication list (or trainable-noise arm explicitly descoped). | Review-verified interface mismatches. | Plan §7.8; review E6 |
| D11 | **Run governance rules R1–R5:** no full run without a passing generated (never hand-edited) smoke config; flags-off parity gate (R2) diffs `summary_<seed>.npy` against M0 references after every env/agent change; 5 seeds standard / 10 on headline rows; full runs only at milestones M0–M7; every run archives cfg + commit hash + summaries + checkpoints + wall-clock. | Reproducibility and comparability discipline. | Run plan §1 |
| D12 | **Benchmark naming corrected:** 8/10-CH₂ (methylene), not CH₂O — repo ships `CH2_8q/CH2_10q` npz only. Action-space formula `3N + 2·C(N,2)` ≡ code's `num_qubits*(num_qubits+2)` — consistent. | Verified against `mol_data/`. | Plan §5.1, §5.4 |

### 3.2 Still-open decisions (need a human call; none block M0/P0)

| ID | Question | Options | Blocks |
|---|---|---|---|
| Q2 | Risk-seeking value head form | Quantile-regression head acting on 0.75-quantile (closer to AlphaTensor's claim, extra knob) vs simple max-backup in the tree | P3 (M4) |
| Q3 | 30q reference-energy convention | Error vs DMRG-χmax via `fake_min_energy` vs extrapolated DMRG | P5 (M6c) headline table |
| Q4 | Transfer experiments in v1 scope? | In (requires multi-Hamiltonian `CircuitEnv` — nontrivial) vs deferred | Paper scope only |
| Q5 | Per-run wall-clock cap for matched-clock protocol (e.g. 48 h/seed HPC) | Decide after M4's S2 dress rehearsal measures real per-move cost | M4 grid commit |

---

## 4. The resulting ordered task list (the "start here" answer)

Dependency graph (plan §4): P0, P1, P2 are mutually independent → P3 consumes P1+P2 → P4 and P5
hang off P3 → P6 last. Go/no-go tree is in run plan §6.

### Immediate tasks (this week)

- [ ] **T1 — S0 environment sanity.** Confirm the `env` python environment runs the repo untouched:
      requirements installed (qulacs, quimb, torch, qiskit), `.qpy` warm-starts load, 5-episode
      smoke on `TensorRL_fixed/BEH26q_TNbond2.cfg` completes and writes `summary_<seed>.npy` with
      all expected keys. (Watch for the interactive device `input()` at
      `TensorRL_fixed_noiseless.py:206` — answer it manually for now; the clone fixes it.)
- [ ] **T2 — Tooling.** Write `scripts/make_smoke_cfg.py` (rule R1: smoke configs generated, never
      hand-edited) and start `results/RUNBOOK.md` (one line per launched run: config, commit,
      seeds, status, wall-clock).
- [ ] **T3 — Launch M0.** Unmodified code, 5 seeds each on `BEH26q_TNbond2.cfg`,
      `H2O8q_TNbond2.cfg`, `H2O10q_TNbond2.cfg`, `LIH12q_TNbond2.cfg` + one TensorRL_trainable and
      one StructureRL arm on 8-H₂O. Record wall-clock/episode. CPU-feasible; embarrassingly
      parallel over (config, seed).
      **Pass gate:** NeurIPS Table-1 within seed variance — 6-BeH₂ err ~6.8e-5 / D7 / CNOT 5 /
      ROT 12; 8-H₂O err ~8.9e-4 / D6 / CNOT 9 / ROT 15; 10-H₂O err ~4.1e-4 / CNOT 15; 12-LiH err
      ~2.4e-2 / CNOT 30; 100% seed success at ≤10q. **If M0 fails: stop, fix environment/repro.**
- [ ] **T4 — P0 development (parallel with M0 compute).**
      - New `environments/utils/gadgets.py`: `GadgetLibrary` + predicates `is_givens_block`,
        `is_fsim_block`, `is_brickwork_cell`, `is_symmetry_block` (windows of decoded actions;
        at most one family completes per step; `actions_in_gadgets` double-count mask).
      - Flag-gated `CircuitEnv` edits: parse `use_gadgets`/bonuses/`cnot_cost`/`rot_cost` (defaults
        off ⇒ bit-identical), `action_history`/`actions_in_gadgets` in `reset()`, gadget/cost terms
        after `reward_fn` in `step()`, new `fn_type = "gadget_shaped"` branch.
      - Anti-farming guard per D4 (energy gate + `max_gadgets_per_episode`).
      - Register new keys in `get_config`'s `floats`/`lists` (D12 trap).
      - Clone `TensorZero_fixed_noiseless.py` per D3 (gadget logging + `--device` flag).
      - New config `configuration_files/TensorZero_fixed/BEH26q_TNbond2_gadget.cfg`.
      - Unit tests: each predicate fires on its canonical window, not on shuffled windows; mask
        prevents double-count.

### Milestone sequence after that

- [ ] **M1** (after P0): R2 parity gate → S1 gadget smoke (anti-farming check: <10% of episodes
      with bonus but no energy improvement) → full run 6-BeH₂ + 8-H₂O gadget-on (5 seeds) vs M0.
      Pass: ≥15% median CNOT reduction on 6-BeH₂ at unchanged success probability. Bonus sweep
      {0.5×, 1×, 2×} on BeH₂ only. → Feeds ablation 2.
- [ ] **P1 + M2** (mergeable with M1): `environments/utils/rotosolve.py` incl. the
      `action_to_param_index` helper + unit test (D5); `rotosolve` dispatch branch in `step()`
      (also fix the `nfev`/`opt_ang_save` latent bug); full run 8-H₂O
      `full_opt_every ∈ {1,5,20}` vs M0. Pass: ≥5× nfev reduction, ≤10% success loss at
      `full_opt_every = 5`. → Picks the default optimizer for all later milestones.
- [ ] **P2 + M3**: `agents/attention_torso.py` (PyTorch port per D9) + `agents/augment.py`;
      `network_type` branch in `DQN.unpack_network`; full run 8-H₂O + 12-LiH mlp vs attention.
      Pass: ≥ MLP at 8q, strictly better curve at 12-LiH. → Ablation 4.
- [ ] **P3 + M4 (headline)**: `agents/MCTSZero.py` (K-sampled PUCT written fresh, D8 interfaces);
      `snapshot()/restore()/simulate_step(level)` env API; S2 overnight rehearsal measures
      per-move cost (resolves Q5) → full run 8-H₂O, DDQN(M0) vs MCTS {50,100 sims}×{level 0,1} at
      matched wall-clock. Pass: ≥1 arm beats DDQN. **Fail ⇒ pivot: paper = P0–P2 on DDQN; M5
      re-scoped to pretrain the DDQN policy net.** → Ablations 1, 5.
- [ ] **P4 + M5**: `demos/` package per D6 (random = policy-head-only; frozen-prefix Δχ fitting is
      new code — budget it); mixed-loss learner with the demo-weight schedule. Full run 8-H₂O,
      4 pretraining arms. Pass: best arm reaches first success in ≤50% episodes of no-pretrain.
      → Ablation 3.
- [ ] **P5 + M6**: `environments/VQAs/VQE_mps_surrogate.py` + pauli-only npz generator per D7;
      rank-correlation smoke (ρ ≥ 0.9, MPS-χ16 vs statevector on 10-H₂O) → full runs: 10-H₂O
      level-2-in-tree; TFIM-20q vs paper appendix + warm-start energy; 30q stretch (3 seeds).
      High-χ re-evaluation of every final circuit. 30q negative result caps claim at 20q without
      blocking M7. → Headline 2, ablation 5 complete.
- [ ] **P6 + M7**: replicate flag-gated diffs into the noise/trainable env variants (diff/patch
      script; five-way divergence risk), R2 parity on all variants, full publication grid (§4 of
      run plan; 10 seeds on 8-H₂O/10-H₂O noiseless + 8-H₂O noisy), `analysis/motif_stats.py`
      n-gram mining, final baseline reproduction on the frozen release commit.

---

## 5. Key numbers to keep at hand (M0 pass gates / baselines to beat)

| System | Config | TensorRL-fixed baseline (NeurIPS 2025) |
|---|---|---|
| 6-BeH₂ | `BEH26q_TNbond2` | err 6.8e-5, depth 7, CNOT 5, ROT 12 |
| 8-H₂O | `H2O8q_TNbond2` | err 8.9e-4, depth 6, CNOT 9, ROT 15 |
| 10-H₂O | `H2O10q_TNbond2` | err 4.1e-4, depth 17, CNOT 15, ROT 17 |
| 12-LiH | `LIH12q_TNbond2` | err 2.4e-2 (fixed) / 1.0e-2 (trainable), CNOT 30/37 |
| 8-H₂O noisy | `H2O8q_noise` (depol 1%/5%, 10⁴ shots) | err 9.0e-4, depth 7, CNOT 5, 100% success |

Paper headline targets: ≥2× CNOT/depth reduction over TensorRL-fixed at equal accuracy on ≥2
molecules; first RL-QAS at 30q; zero-shot transfer (scope pending Q4); ≥1 emergent motif outside
the gadget library.

---

## 6. Run / change log (append-only; newest at top)

| Date | Entry | Type | Notes |
|---|---|---|---|
| 2026-07-16 | **T2 complete — smoke-config tooling + RUNBOOK.** Wrote `scripts/make_smoke_cfg.py`: reads a full production `.cfg` with `configparser`, overrides only `episodes` (+ `global_iters` for tier s1) per rule R1, writes to `configuration_files/smoke/<name>_<tier>_smoke.cfg` with a provenance header comment (source path, tier, generated values, timestamp) — never hand-edited, regenerated on demand. Tier defaults match `training_run_plan.md` §2 exactly: s0 = 5 episodes/global_iters unchanged/1 seed; s1 = 300 episodes/global_iters=200/2 seeds. Verified: (a) generated s0 config from `TensorRL_fixed/BEH26q_TNbond2.cfg` parses identically to the source via `get_config` (including the embedded-semicolon `geometry` string surviving the configparser round-trip), (b) a live 5-episode run through `TensorRL_fixed_noiseless.py` against the generated config completed cleanly with correct `summary_0.npy` schema, (c) s1 tier generates and parses with the right overrides (300/200) — not run live, since 300-episode learning-smoke runs are a per-milestone (M1+) concern, not T2's. Committed the two generated cfgs (`BEH26q_TNbond2_{s0,s1}_smoke.cfg`) as durable script artifacts, analogous to a lockfile. Started `results/RUNBOOK.md` (R5 format: date/tier-milestone/config/commit/seeds/status/wall-clock/notes, newest-first, append-only) and logged the T1 ad hoc run + this task's own s0 verification run as its first two entries. **Open flag for T3**: repo has no `.gitignore`, and whether full-run artifacts (checkpoints are tens of MB/seed) get committed, git-ignored, or pushed to external storage is undecided — noted in RUNBOOK, needs a human call before M0's 5-seeds x 4-configs x 2-extra-arms archive lands. | tooling | Repo at commit `8dc5d70`. New tracked files: `scripts/make_smoke_cfg.py`, `configuration_files/smoke/BEH26q_TNbond2_{s0,s1}_smoke.cfg`, `results/RUNBOOK.md`. Next action: T3 (launch M0) — but resolve the archiving-strategy flag first. |
| 2026-07-16 | **T1 complete — S0 environment sanity passed.** Verified in the `env` venv (`/home/tanvir/fun/tts/voxtral/.venv`, Python 3.12.3): `qulacs 0.6.13`, `torch 2.10.0+cu128` (CUDA available), `qiskit 2.0.0`, `quimb 1.8.4` all present — no installs needed. Confirmed the BeH₂ 6q TNbond2 `.qpy` warm-start (`dmrg-to-qc/init_state_circ/init_BEH2_6q_..._TNbond2.qpy`) loads via `qpy.load` (6 qubits, depth 27, 108 gates). Ran a 5-episode smoke test of `TensorRL_fixed_noiseless.py` against a **hand-made temporary copy** of `configuration_files/TensorRL_fixed/BEH26q_TNbond2.cfg` with `episodes=5` (piped `"0"` into the interactive device `input()` at line 206 via stdin instead of editing the file, keeping the baseline script untouched per D2/D3) — completed with no crashes; `summary_0.npy` schema matched `Saver.get_new_episode` exactly (`actions/errors/errors_noiseless/done_threshold/bond_distance/nfev/opt_ang/time/save_circ/reward`, 20 timesteps/episode, `test` dict empty since `agent_test` isn't called from this script's `train()`). Temp cfg and results dir deleted afterward — **not** a R1-compliant generated smoke config, just an ad hoc sanity check; T2's `make_smoke_cfg.py` still needed before any milestone run. | verification | Repo at commit `8dc5d70`. No permanent files added. Next action: T2 (smoke-config tooling + RUNBOOK) then T3 (launch M0). |
