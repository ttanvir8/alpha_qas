# Review of `implementation_plan_v1.md` (TensorZero-QAS)

Reviewer pass: every file/class/function/line reference in the plan was checked against the two
ground-truth codebases:

- TensorRL-QAS: `/mnt/c/Users/LEGION/Work/quantum_root/versions_of_tensor_rl/quary_original_tensor_rl/`
- AlphaTensor-Quantum: `/mnt/c/Users/LEGION/Work/quantum_root/projects/rl-cut/current_works/alphatensor_quantum/`

---

## (a) Verdict

**Needs revision — but fundamentally sound.** The plan's factual grounding is unusually good:
of the ~40 concrete code touchpoints it names (files, classes, functions, config keys, *and line
numbers*), essentially all check out, including the three "latent env bugs" it claims to have
found and all eight items in its own corrections section. The pipeline-parity strategy
(flag-gated env diffs + cloned entry scripts + new config dirs) is genuinely comparability-safe.
The revisions needed are concentrated in three places: (1) an internal contradiction in P0's
logging touchpoint, (2) P5's data-generation claim is infeasible as written (dense-Hamiltonian
paths cannot reach 20–30 qubits and the plan's env-mod list misses `reset()`), and (3) P4's
random-demonstration premise silently drops the property that made AlphaTensor's synthetic demos
work, which threatens that phase's scientific payoff. **Overall feasibility grade: B** — P0–P2
are near-certain, P3 is heavy but doable, P4/P5 need targeted redesign of specific components
before implementation starts.

---

## (b) Verified-correct claims (spot-check results)

**TensorRL-QAS side — all confirmed:**

- `CircuitEnv.step()` decodes `ctrl/targ/rot_qubit/rot_axis` at lines 255–258; `rwd = self.reward_fn(energy)` at line 306; optimizer dispatch `if self.optim_method in ["scipy_each_step"]` at line 283 (`environments/environment_qulacs_TN_notin_agent.py`).
- `scipy_optim` is a `CircuitEnv` method at line 452 (plan correction #2 ✓); `get_energy_qulacs(..., which_angles=[])` at `environments/VQAs/VQE_qulacs_TN_notin_RL.py:48–50` and it does support per-index angle setting ✓.
- All three "latent bugs" are real: `self.TN_state = Statevector(self.tenor_circ)` at line 158 sits *outside* the `if self.TN_bond:` block (NameError when `tn_bond = 0`); `self.opt_ang_save = opt_ang` (line 288) and `self.nfev = nfev` (line 316) run unconditionally → NameError whenever `optim_method != "scipy_each_step"`; the interactive `input("Enter 0 or 1: ")` device prompt at `TensorRL_fixed_noiseless.py:206`.
- Hardcoded `['heisenberg', 'tfim_j1_h0.05']` at lines 78, 126, 372 while the shipped file is `tfim_j1_h0.001_6q.npz` (plan correction #3 ✓ — TFIM is currently *unloadable*, understated if anything).
- `fake_min_energy` handled at line 60 and registered in `get_config`'s `floats` list; new float/list keys indeed parse as strings unless registered (`environments/utils/utils.py:19–34`) — plan correction #8 ✓.
- `cnot_rwd_weight` is parsed (line 63) and never used in any reward computation ✓.
- `agents/__init__.py` imports only `DeepQ_restricted, DeepQNstep, DeepQNstep_restricted`; entry scripts use `agents.__dict__[agent_type].__dict__[agent_class]` (`TensorRL_fixed_noiseless.py:236`) — plan correction #6 ✓.
- `DQN.unpack_network` at `agents/DeepQ.py:147`; `DQN.__init__` already adjusts `state_size` for `angles=0` and appends +1 for `en_state`/`threshold_in_state` (lines 43–46), matching P2's interface claim exactly ✓.
- `init_net` reload block at `TensorRL_fixed_noiseless.py:239–252` ✓ (but see Error E3).
- `action_size = num_qubits*(num_qubits+2)` (line 192) = idea doc's `3N + 2·C(N,2)` ✓ (correction #4).
- Shipped configs: `TensorRL_fixed/BEH26q_TNbond2.cfg` has `num_layers = 47`, `accept_err = 1.6e-3`, `agent_type = DeepQNstep`, `method = scipy_each_step`, `optim_alg = COBYLA`, `global_iters = 1000` — all P0/P1 premises ✓. `H2O8q`, `LIH12q`, `H2O8q_noise` configs exist ✓.
- `mol_data/` ships `CH2_8q/CH2_10q` (no CH₂O — plan correction #1 ✓), `LIH_4q_..._parity.npz` ✓; molecule npz files contain `paulis`/`weights` arrays ✓ (exception noted in E6); bond-2/3 `.qpy` warm starts for H2O exist ✓; quimb is in `requirements.txt` ✓; `mps2qc.gs_dmrg`, `mps_to_qc`, `mpo_from_paulis`, `tnqc_ansatze.qiskit_circ_from_tn_params` all exist ✓; curricula classes `VanillaCurriculum/MovingThreshold/SuccesCountThreshold` ✓.
- `summary_<seed>.npy` schema in `Saver.get_new_episode` matches the plan's key list, and it is a pickled dict indexed by key, so an *additive* key is indeed format-safe ✓.

**AlphaTensor-Quantum side — all confirmed:**

- `factors.py`: `factors_form_toffoli_gadget`, `factors_form_cs_gadget`, `TOFFOLI_REWARD_SAVING = 5.0`, `CS_REWARD_SAVING = 1.0`, calibrated so net gadget reward = compiled cost (−2.0) ✓.
- `environment.py` gadget check ordering, `factors_in_gadgets[-6:]`/`[-2:]` double-count guards, and reward composition sit at **exactly lines 141–185** as cited ✓; `EnvState.past_factors`, `Observation.sqrt_played_fraction` ✓. Importantly, the bonus is granted **on the completing step**, not by rewriting past rewards — so it ports cleanly into the DDQN/N-step replay buffer with no stale-reward problem (the plan's mechanism is correct on this point).
- `networks.py`: `Symmetrization` (`A∘X + (1−A)∘Xᵀ`, sigmoid of zero-init logits), `_SelfAttention`, `_TransformerDecoderBlock`, `_SymmetrizedAxialAttention`, `TorsoNetwork` with `LinearProjectScalars` on `sqrt_played_fraction` — all as described; the modules are small and a 1:1 PyTorch port is realistic ✓. The plan correctly self-flags that `Symmetrization` requires a square S×S grid, which (num_layers×num_qubits) is not.
- `demo/agent.py`: `_recurrent_fn` returns real env states as "embeddings" (deterministic env) ✓; `_loss_fn` is KL-vs-visit-weights policy loss + squared value loss with `(1−w)·acting + w·demos` mixing via `optax.piecewise_constant_schedule` ✓; `mctx.muzero_policy` with `qtransform_by_parent_and_siblings` ✓; 800 sims paper-scale / 80 demo-scale ✓ — confirming plan correction #5 that *sampled* MCTS is not in the open code.
- `demonstrations.py`: `Demonstration` NamedTuple, `generate_synthetic_demonstrations`, `_overwrite_factors_with_gadgets`, `get_action_and_value` (reward-to-go including gadget savings) ✓; `prob_include_gadget = 0.9`, `max_num_gadgets`, `prob_toffoli_gadget` ✓.
- `demo/demo_config.py`: `demonstrations_boundaries_and_scales = {60: 0.99, 200: 0.5, 5000: 0.2, 10000: 0.1}`, `init_demonstrations_weight = 1.0` — exact values as quoted ✓.

**P1 math:** the Rotosolve closed form quoted in the plan is the standard Ostaszewski formula and
is *exact* here: every parametrized gate is a single-qubit RX/RY/RZ (generator eigenvalues ±1/2 →
E(θ) = a + b·cosθ + c·sinθ), and the TN warm-start enters only as the initial state loaded by
`get_exp_val` (`state.load(TN_state)`), which preserves linearity — the warm start does **not**
break the sinusoid. 3 evaluations per angle is correct.

---

## (c) Errors found

**E1 — Internal contradiction + missing writer for the gadget log (P0 / Section 1).**
Section 1's keep-table declares `TensorRL_fixed_noiseless.py` "**Untouched**; new entry scripts
are clones", but P0 then modifies that very file (`Saver.get_new_episode`: add `'gadgets': []`).
Worse, nothing is specified to *append* to that key — `one_episode` writes every other stat
(lines 122–143), so it too would need editing, which the plan never says. Evidence:
`TensorRL_fixed_noiseless.py:21–44, 122–143`. **Fix:** either (i) do P0 logging entirely inside a
cloned `TensorZero_fixed_noiseless.py` even for the DDQN arm, or (ii) explicitly amend the
keep-table to "additive-only edit" and specify the `one_episode` append
(`env.gadget_events` → `stats_file[...]['gadgets']`).

**E2 — P5's 20–30q data-generation and env paths are infeasible as written.**
The plan says `dmrg-to-qc/heisenberg_model.py` / `making_molecules.py` are "**used, not
modified**" to emit `tfim_*_20q.npz`…`30q.npz`. But `heisenberg_model.construct_hamiltonian`
builds a **dense 2ⁿ×2ⁿ matrix via `np.kron`** (heisenberg_model.py:15–27) — at 20q that is a
17 TB array; and it implements the Heisenberg model only (no TFIM/field terms; the file contains a
single function). Moreover the npz schema stores the dense `hamiltonian` array, and `CircuitEnv`
loads it and calls `Operator(...).reverse_qargs().to_matrix()` in **both** `__init__` (lines
127–131, 162) **and `reset()` (lines 372–377)** — the plan's P5 env-mod list covers `__init__`
but not `reset()`, so a 20q run would still crash on the first `reset()`. Evidence:
`environments/environment_qulacs_TN_notin_agent.py:372–377`; `dmrg-to-qc/heisenberg_model.py:7–60`.
**Fix:** (i) a new pauli-strings-only npz schema for >12q systems (no `hamiltonian`/`eigvals`
arrays; DMRG reference energy in place of eigvals), emitted by a *new* sparse/MPO generator (the
`mpo_from_paulis` route, not dense kron); (ii) the `leaf_evaluator = mps` branch must bypass the
dense load in `reset()` as well as `__init__`; (iii) drop the "used, not modified" claim —
`heisenberg_model.py` needs a TFIM variant regardless.

**E3 — P3's "two-line delta" and "same ctor signature" understate the entry-script surgery.**
`MCTSAgent.act(self, state, ill_action)` cannot run tree search: the search needs the live
`CircuitEnv` to call `snapshot()/simulate_step()`, and neither the ctor
`(conf, action_size, state_size, device)` nor `act`'s signature carries an env handle. Also:
(i) the AlphaZero value target `z` is only known at episode end, so `remember(state, policy, z)`
requires end-of-episode backfill, not the per-step `remember` the loop does now
(`TensorRL_fixed_noiseless.py:127–131`); (ii) the `init_net` reload block the plan relies on for
P4 (`lines 239–252`) *always* loads `target_net` (line 242) and the replay buffer
(lines 247–249) — `MCTSAgent` has neither a `target_net` nor `Transition`-shaped memory, so the
clone's reload block must be adapted, and `demos/pretrain.py` must emit compatible artifacts.
**Fix:** inject the env after construction (mirroring the existing `agent.saver = Saver(...)`
idiom at line 237), buffer the episode and backfill `z` on `done`, and specify the adapted
checkpoint format. None of this is hard; all of it must be in the plan.

**E4 — P4's random demonstrations lose the property that made AlphaTensor's demos work.**
In AlphaTensor, a synthetic demo is a *solved instance by construction*: random factors are drawn
first and the **target tensor is defined as their sum**, and the target (residual tensor) is part
of the network's observation (`demonstrations.py:473–560`; `TorsoNetwork` consumes
`observations.tensor`). The trajectory is therefore near-optimal *for its own target*, and the
network can generalize across targets. In TensorRL the Hamiltonian is fixed per run and is **not**
a network input — a random gate sequence is not a solution to anything; its "reward-to-go" is the
return of a random policy (typically ending in the −5 timeout), and cross-entropy on its actions
is behavior-cloning noise. The plan's phrase "a 'solved' demo means a circuit and its true reward
trace" papers over this: a *true reward trace of a failing episode* is not a demonstration in the
AlphaZero sense. What survives the port: legal-action structure, gadget-motif priors (via the
0.9 splicing), and the DMRG-Δχ demos (which are genuinely on-task). Additionally, the Δχ recipe
"takes the transpiled gate-difference" between two *independent* `mps_to_qc` fits — but
`mps_to_qc` random-initializes every gate (`mps2qc.py:300` `rand_uni(4)`) and has no support for
freezing a prefix, so the χ_lo and χ_hi circuits share no common prefix and the "difference" is
ill-defined. **Fix:** (i) demote random demos to a legality/motif-prior pretraining role (policy
head only — the plan's own fallback) and say so up front; (ii) for Δχ demos, add a frozen-prefix
fit (initialize the χ_hi ansatz with the χ_lo solution's gates frozen or as warm init and only
optimize the added layers), which is new code in `mps2qc.py`/a wrapper, not a library call;
(iii) consider replacing "random circuit" demos with cheap greedy/Rotosolve rollouts that
actually descend in energy, so value targets are meaningful.

**E5 — P1: mapping "the newly placed angle" to a qulacs parameter index is glossed over.**
The new rotation is inserted at layer `moments[rot_qubit]`, which can be an *earlier* layer than
gates already present on other qubits; `Parametric_Circuit.construct_ansatz` enumerates
parameters layer-major (`VQE_qulacs_TN_notin_RL.py:13–45`), so **the newest gate is not in
general the last parameter**. `rotosolve_optim` must replicate the layer-major enumeration to find
the right `which_angles` index; getting this wrong optimizes a *different* angle silently (no
crash, just degraded results). **Fix:** add an explicit
`(layer, axis, qubit) → parameter-index` helper with a unit test asserting agreement with
`circuit.get_parameter_count()` ordering.

**E6 — Small interface mismatches.**
(i) `mps2qc.mpo_from_paulis(hamiltonian: dict[str, complex])` takes a dict, not the npz
`paulis`/`weights` arrays — a (trivial) zip-adapter is needed (`mps2qc.py:24–26`).
(ii) `LIH_4q_...parity.npz` has **no `paulis` key** (only hamiltonian/weights/eigvals/energy_shift)
— irrelevant for its P1 unit-test role but worth knowing.
(iii) `gs_dmrg` returns a `(mps, dmrg)` tuple despite its annotation (`mps2qc.py:154`).
(iv) Section-1 filename glob "`Tensor_training_and_structureRL_*.py`" matches only the noise
script; the noiseless one is `TensorRL_training_and_structureRL_noiseless.py`.
(v) P6's replication list omits `environment_qulacs_noise.py` (used by
`Tensor_training_and_structureRL_noise.py`) — either add it or state that the trainable-noise arm
is out of scope.

---

## (d) Design concerns, ranked by severity

1. **P4 premise (high; see E4).** Without target-conditioning, most of the demo corpus is noise;
   the phase's success criterion (≤50% episodes-to-first-success) may fail for reasons unrelated
   to the mixing machinery, muddying ablation 3.
2. **P0 reward hacking (medium-high).** In AlphaTensor the gadget saving *corrects the reward to
   equal the true objective* (the gadget genuinely costs 2 T gates, so −7+5 = truth). Here the
   bonus is pure shaping on top of an energy objective — a completed "Givens block" on irrelevant
   qubits still pays +0.30. With num_layers = 47 an agent can farm ≈ +3–4 per episode from tiled
   motifs with zero energy progress, competing with the +5 success reward. The plan's fallback
   (bonuses→0) exists, but prevention is cheap: gate the bonus on `energy ≤ prev_energy` (or make
   it potential-based), and cap rewarded gadgets per episode. Also note the motif predicates are
   *structural*, not semantic: `CNOT·RY·CNOT` implements `exp(−iθ ZY/2)`, which is Givens-like
   only up to single-qubit frames and is not particle-conserving by itself — fine as a motif
   library, but the paper should not claim unitary-level identities.
3. **P3 cost accounting depends on open question 6 resolving to "re-run baselines" (medium).**
   The matched-wall-clock claim is only meaningful if TensorRL-fixed baselines are re-run on the
   same hardware; the plan lists this as an open question, but ablation 1 is invalid without it —
   it should be a decision, not a question. The per-move budget estimate (~100 sims × ~3–4
   statevector evals) is honest at ≤12q; snapshot/restore of `CircuitEnv` is feasible (all mutable
   state is small tensors/lists), though the snapshot list should be made exhaustive:
   add `current_action`, `previous_action`, `energy`, `error`, `halting_step`, and
   `current_number_of_cnots` to the plan's list.
4. **P5 scientific validity at 20–30q (medium-high, beyond E2).** ⟨H⟩ of a χ-truncated CircuitMPS
   is the energy of a *different* state than the true circuit; a "beats the TN warm-start" headline
   must re-evaluate the final circuit at (much) higher χ or with DMRG-quality contraction, and the
   plan's `commit_bond_dim = 64` should be justified by the planned χ-convergence check, not just
   the rank-correlation experiment.
5. **P2 branding (low-medium).** The plan's own deviation (symmetrization moved to an auxiliary
   N×N CNOT-co-occurrence branch because depth×qubit is not transpose-symmetric) means the main
   torso is plain axial attention. Honest as flagged, but the paper should not headline
   "symmetrized axial attention" without the caveat. Data-hunger without P4 is acknowledged;
   the fallback (attention as the ≥12q scaling story only) is credible.
6. **Gadget contiguity (low).** Requiring the motif's actions to be played consecutively is
   faithful to AlphaTensor (its 7 factors must also be consecutive) but restricts discovery in a
   setting where actions on disjoint qubits commute; the P6 n-gram mining partially compensates.
   Worth one sentence in the paper's limitations.

---

## (e) Feasibility per phase

| Phase | Grade | Why |
|---|---|---|
| P0 gadget reward | **Green** | All touchpoints verified; mechanism ports cleanly into per-step rewards (no stale-reward issue in the N-step buffer); needs E1 fixed and the anti-farming gate from concern #2. Bonus calibration is the only real tuning burden. |
| P1 Rotosolve | **Green** | Math is exact for this gate set and unaffected by the TN warm start; `which_angles` plumbing already exists. One silent-failure trap (E5) — unit-test the index mapping. Cheapest phase, biggest guaranteed payoff (nfev is already logged, so the metric is free). |
| P2 attention torso | **Yellow** | Port scope is realistic (5 small modules, demo-scale hyperparams); state-tensor split matches the code exactly. Risks: data hunger before P4 lands, and the symmetrization deviation weakens the transplant story. Fallback (MLP) preserves the paper. |
| P3 MCTS | **Yellow** | Feasible: deterministic env + cheap snapshots + Level-0 leaves make the tree affordable at ≤12q, and the plan's fallback ladder is credible. But it is the largest engineering item (hand-rolled sampled PUCT, new agent, new entry script), E3's gaps must be closed, and the headline ablation hinges on re-running baselines. |
| P4 demonstrations | **Yellow/Red** | Machinery (mixed loss, schedule, container) ports verbatim and is verified. The *content* is the problem: random demos are not solutions here (E4) and the Δχ source needs new frozen-prefix fitting code. Re-scope before building, or the phase risks a null/negative result on its own success criterion. |
| P5 MPS surrogate | **Yellow at ≤12q, Red at 20–30q as written** | The surrogate-in-tree at 10–12q (rank-correlation validation) is workable with quimb and the existing `paulis` arrays. The 20–30q scale-up is blocked by E2 (dense npz schema, dense env paths in `__init__` *and* `reset()`, no scalable generator) plus the energy-validity issue (#4). Fix E2 and the phase is Yellow; the plan's own "cap the claim at 20q" fallback is the right instinct. |
| P6 grid + noise | **Green** | Compute-bound and embarrassingly parallel; the "reproduce three baseline rows with flags off first" gate is exactly right. Five-way env-file replication is tedious and error-prone — do it with a diff/patch script, and add `environment_qulacs_noise.py` or descope it explicitly. |

Dependency ordering (Section 4) is correct: P0/P1/P2 are genuinely independent; P4 correctly
requires P3's policy targets (the DQN behavior-cloning aside should be cut — cross-entropy
pretraining of a Q-network's argmax is not behavior cloning and would distort Q-values); P5
correctly hangs off P3's `simulate_step` hook.

---

## (f) Concrete revision list

1. **P0:** Resolve the keep-table contradiction — move `'gadgets'` logging into the cloned entry
   script (or mark `TensorRL_fixed_noiseless.py` "additive edit") and specify the `one_episode`
   append that writes it. (E1)
2. **P0:** Add an anti-farming guard: bonus paid only when the completing step does not increase
   energy (or potential-based shaping), plus a per-episode cap on rewarded gadgets. Reword the
   "net cost = compiled cost" calibration to acknowledge it is shaping, not objective correction.
3. **P1:** Add the `(layer, axis, qubit) → qulacs parameter index` helper with a unit test
   against `construct_ansatz`'s layer-major ordering. (E5)
4. **P3:** Specify env injection into `MCTSAgent` (post-construction setter, like `agent.saver`),
   end-of-episode backfill of value targets `z`, and the adapted checkpoint/reload format
   (no `target_net`, no `Transition` replay buffer). Promote open question 6 to a decision:
   baselines *will* be re-run on project hardware for ablation 1. (E3)
5. **P3:** Extend the snapshot field list to `current_action`, `previous_action`, `energy`,
   `error`, `halting_step`, `current_number_of_cnots` (exhaustive, with a restore round-trip test).
6. **P4:** Re-scope — random demos: policy-head-only pretraining for legality/motif priors
   (state the E4 limitation explicitly); replace or augment with greedy/Rotosolve descent rollouts
   for meaningful value targets; Δχ demos: add frozen-prefix fitting to the `mps_to_qc` pipeline
   (new code, budget it). Drop the DQN behavior-cloning aside. (E4)
7. **P5:** Rewrite the data-prep paragraph: new pauli-only npz schema for >12q (no dense
   `hamiltonian`/`eigvals`), a new sparse/MPO generator (dense-kron `heisenberg_model.py` cannot
   scale and has no TFIM), and add `reset()` (lines 372–377) to the env-mod list alongside
   `__init__`. Add a final-circuit re-evaluation at high χ to the validation protocol. (E2)
8. **P5/P6 small fixes:** dict adapter for `mpo_from_paulis`; note `gs_dmrg` returns a tuple;
   correct the `Tensor_training_and_structureRL_*` filename; add or descope
   `environment_qulacs_noise.py`. (E6)
9. **P2:** In the paper-facing text, present the torso as "axial attention with an auxiliary
   symmetrized qubit×qubit branch" rather than "symmetrized axial attention" wholesale.

---

## (g) Post-review addendum — status of the revision list

The plan was amended to **v1.1** (see `implementation_plan_v1.md` Section 7): all nine items in
the revision list above were accepted and folded in — E1 (gadget logging moves to the cloned
entry script, created in P0), the P0 anti-farming guard + per-episode gadget cap, the E5
parameter-index helper with unit test, the E3 env-injection/value-backfill/checkpoint-format
specification, the exhaustive snapshot field list, the E4 re-scope of P4 (random demos =
policy-head-only; value supervision from greedy/Rotosolve rollouts + Δχ demos with new
frozen-prefix fitting code), the E2 rewrite of P5's data path (pauli-only npz schema, sparse/MPO
generator, `reset()` added to the env-mod list, high-χ re-evaluation of final circuits), the E6
small fixes, and the P2 paper-framing change.

**Open question 6 is resolved as this review recommended**: baselines will be re-run on project
hardware, formalized as milestone **M0** in the new `training_run_plan.md`. That document also
operationalizes the review's process concerns: a flags-off parity gate (rule R2) that diffs
`summary_<seed>.npy` against M0 references after every env/agent change, mandatory smoke tiers
(S0 mechanical / S1 learning / S2 dress-rehearsal) before every full training run, full-run
milestones M0–M7 mapped to phases with go/no-go criteria against the NeurIPS 2025 Table-1/Table-2
numbers, and the publication config × benchmark matrix (5 seeds standard, 10 seeds on headline
rows: 8-H₂O, 10-H₂O noiseless and 8-H₂O noisy). Remaining truly open decisions: value-head form
(quantile vs max-backup, plan §6 Q2), 30q reference-energy convention (Q3), and transfer-
experiment scope (Q4).
