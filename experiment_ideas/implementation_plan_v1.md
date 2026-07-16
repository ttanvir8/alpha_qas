# TensorZero-QAS — Implementation Plan v1 (rev. v1.1)

Companion to `experiment_ideas/a_star_paper_idea1.md` (phases P0–P6, Section 5).
**v1.1 update:** reviewed in `implementation_plan_v1_review.md` (verdict: sound, needs targeted
revision — accepted amendments in Section 7 below); training runs, smoke tests, and the
publication config/benchmark matrix are specified in `training_run_plan.md` (milestones M0–M7,
mapped to phases in Section 8 below).
Ground rule: **the TensorRL-QAS pipeline is kept intact** — Phase A (`dmrg-to-qc/`) untouched,
Phase B keeps entry-script → `.cfg` → `CircuitEnv` → agent → episode loop → `Saver`/`summary_<seed>.npy`,
so every run stays directly comparable to the NeurIPS 2025 tables.

All file paths below are relative to the repo root
(`/mnt/c/Users/LEGION/Work/quantum_root/versions_of_tensor_rl/quary_original_tensor_rl/`) unless
they start with `alphatensor_quantum/`, which refers to
`/mnt/c/Users/LEGION/Work/quantum_root/projects/rl-cut/current_works/alphatensor_quantum/`.

---

## 1. What we keep (TensorRL-QAS side)

| File / module | Keep as-is or Modified | Justification |
|---|---|---|
| `dmrg-to-qc/dmrg_to_qc.py`, `mps2qc.py`, `stiefel_opt.py`, `tnqc_ansatze.py` | **Untouched** (P4 additionally *calls* `mps2qc.gs_dmrg`/`mps_to_qc` as a library) | Phase A warm-start generation is a hard keep; identical `.qpy` artifacts in `dmrg-to-qc/init_state_circ/` |
| `dmrg-to-qc/mol_data/*.npz`, `dmrg-to-qc/init_state_circ/*` | **Untouched** (P5 adds *new* npz files for >12q, never edits existing) | Benchmark comparability |
| `TensorRL_fixed_noiseless.py` (`Saver`, `one_episode`, `train`, `agent_test`, `modify_state`) | **Untouched**; new entry scripts are clones | `summary_<seed>.npy` schema (`actions/errors/errors_noiseless/nfev/opt_ang/time/reward/...`) must stay byte-compatible for the paper's analysis scripts |
| `TensorRL_fixed_noise.py`, `Tensor_training_and_structureRL_*.py`, `*_restricted.py` | Untouched | Baseline reproduction |
| `environments/environment_qulacs_TN_notin_agent.py` (`CircuitEnv`) | **Modified** (P0: reward; P1: optimizer dispatch; P3: clone/simulate API; P5: leaf evaluator switch) — all changes behind config flags defaulting to old behavior | This is the main integration point; flags off ⇒ bit-identical baseline |
| `environments/environment_qulacs.py`, `*_noise*.py` variants | Modified only in P6 (same flag-gated diffs replicated) | Noisy + trainable-TN benchmarks |
| `environments/VQAs/VQE_qulacs_TN_notin_RL.py` (`Parametric_Circuit.construct_ansatz`, `get_energy_qulacs`, `get_exp_val`) | **Modified** (P1 adds Rotosolve helpers; existing functions untouched) | The COBYLA cost function stays the ground-truth evaluator |
| `environments/utils/utils.py` (`get_config`, `dictionary_of_actions`, `dict_of_actions_revert_q`) | **Modified** (register new float/list config keys in `get_config`) | New keys must be added to the `floats`/`lists` lists or they parse as strings |
| `environments/utils/curricula.py` (`VanillaCurriculum`, `MovingThreshold`, `SuccesCountThreshold`) | Untouched | Curriculum logic orthogonal to all new ideas |
| `agents/DeepQ.py` (`DQN`: `act`, `replay`, `unpack_network`, replay memories) | **Modified** (P2: `unpack_network` gains a `network_type` branch) | ε-greedy DDQN remains the P0/P1 learner and the matched-wall-clock baseline for ablation 1 |
| `agents/DeepQNstep.py` (`DQN_Nstep`, `N_step_ReplayMemory`) | Untouched | Shipped configs use `agent_type = DeepQNstep`; stays the baseline agent |
| `agents/utils.py`, `agents/utils_topology_restrict.py`, `agents/__init__.py` | `__init__.py` **modified** (register new agent modules — required by the `agents.__dict__[agent_type].__dict__[agent_class]` lookup in the entry scripts) | Action dictionary is reused *everywhere* (gadget detector, permutation augment, demos) |
| `configuration_files/{StructureRL,TensorRL_fixed,TensorRL_trainable}/*.cfg` | Untouched; new configs added under new dirs (`configuration_files/TensorZero_*/`) | Old configs must keep reproducing old numbers |

## 2. What we port from AlphaTensor-Quantum

| `alphatensor_quantum/src/` file | Concept carried | Port mode |
|---|---|---|
| `factors.py` (`factors_form_toffoli_gadget`, `factors_form_cs_gadget`, `TOFFOLI_REWARD_SAVING`/`CS_REWARD_SAVING`) | Gadget = pattern over the last k actions + fixed reward saving so the motif's net cost equals its compiled cost | **Logic-port** (P0). The GF(2) linear-dependency checks don't transfer; what transfers is the *mechanism*: sliding-window pattern match on action history + retroactive bonus + a `factors_in_gadgets`-style mask preventing double counting |
| `environment.py` (`Environment.step`: gadget check ordering, `factors_in_gadgets` update, reward composition; `EnvState.past_factors`; `Observation.sqrt_played_fraction`) | Where in `step()` the gadget check sits, masking already-gadgetized actions, progress scalar as network input | **Logic-port** (P0, P2). Env stays `CircuitEnv`; we add `self.action_history` + `self.actions_in_gadgets` mirroring `past_factors`/`factors_in_gadgets` |
| `networks.py` (`Symmetrization`, `_SelfAttention`, `_TransformerDecoderBlock`, `_SymmetrizedAxialAttention`, `TorsoNetwork`) | Symmetrized axial attention torso | **Near-verbatim logic, JAX/Haiku → PyTorch rewrite** (P2). Every module has a 1:1 `nn.Module` translation (`hk.Linear`→`nn.Linear`, `hk.LayerNorm`→`nn.LayerNorm`, `hk.get_parameter`→`nn.Parameter`, einshape→`einops`/`view`). Axes change: (size×size) grid → (num_layers×num_qubits) grid |
| `demo/agent.py` (`NeuralNetwork` policy+value heads, `_recurrent_fn`, `_loss_fn` mixed acting+demonstration loss, `_run_iteration_agent_env_interaction`, `GameStats` best-ever tracking) | AlphaZero acting/learning loop; deterministic env ⇒ tree nodes hold real env states; KL policy loss + squared value loss; demonstrations weight via `piecewise_constant_schedule` | **Logic-port** (P3, P4). Note: the released demo uses `mctx.muzero_policy` with `qtransform_by_parent_and_siblings`; the paper's *sampled* MCTS is **not** in the open code — our K-sampled PUCT must be written fresh in PyTorch/NumPy following mctx's interface shape |
| `demonstrations.py` (`generate_synthetic_demonstrations`, `_overwrite_factors_with_gadgets`, `get_action_and_value`, `Demonstration` NamedTuple) | Random solutions with gadgets injected at prob 0.9; per-move (action, value) supervision extracted by index | **Logic-port** (P4). Replace "random factors → tensor" by "random gate sequence → statevector/energy"; keep the gadget-injection scheme and the `Demonstration`-style container almost verbatim |
| `change_of_basis.py` (`generate_change_of_basis`, `apply_change_of_basis`) | Symmetry/data augmentation at episode start + train time | **Reference-only** (P2). GF(2) basis changes have no VQE analog; the transferred idea is qubit-permutation augmentation, implemented from scratch against `dictionary_of_actions` |
| `config.py`, `demo/demo_config.py` (`EnvironmentParams.use_gadgets`, `DemonstrationsParams`, `LossParams.demonstrations_boundaries_and_scales`, `ExperimentParams.num_mcts_simulations`) | Which knobs exist and their paper defaults | **Reference-only** → mapped onto new `.cfg` keys (Section 3, per-phase) |
| `tensors.py`, `demo/run_demo.py`, `demo/checkpointing.py` | Target-tensor zoo, training-loop shell | **Reference-only** — TensorRL's entry scripts and Saver replace them |

License note: AlphaTensor-Quantum is Apache-2.0 — logic ports with attribution are fine.

---

## 3. Per-phase plan

### P0 — Gadget detector + reshaped reward (in the existing DDQN stack)

**Goal.** Reward becomes
`r_t = base_incremental − c_cnot·1[CNOT] − c_rot·1[rotation] + gadget_bonus(a_{t−k:t})` with the
existing ±5 terminal ends kept. **Success criterion:** on `TensorRL_fixed/BEH26q_TNbond2.cfg`
(6-BeH₂), gadget-on runs reach `accept_err = 1.6e-3`-class accuracy with **≥15% fewer CNOTs**
(median over ≥5 seeds) than gadget-off, at unchanged success probability.

**Existing files modified.**
- `environments/environment_qulacs_TN_notin_agent.py`:
  - `CircuitEnv.__init__`: parse `[env] use_gadgets`, per-gadget bonuses, `cnot_cost`, `rot_cost`
    (all defaulting to off/0 ⇒ old behavior); build `self.gadget_lib = gadgets.GadgetLibrary(conf)`.
  - `CircuitEnv.reset()`: init `self.action_history = []`, `self.actions_in_gadgets = []`
    (the analog of `EnvState.past_factors` / `factors_in_gadgets` in
    `alphatensor_quantum/src/environment.py`).
  - `CircuitEnv.step()`: after decoding `ctrl/targ/rot_qubit/rot_axis` (lines ~255–260), append the
    decoded action to `self.action_history`; after `rwd = self.reward_fn(energy)` (line ~306) add
    the gadget/cost terms (or add a new `fn_type` branch, see next bullet).
  - `CircuitEnv.reward_fn()`: add branch `fn_type == "gadget_shaped"` that wraps the existing
    `incremental_with_fixed_ends` body and adds
    `self.gadget_lib.bonus(self.action_history, self.actions_in_gadgets) - gate_costs(action)`.
    Old `fn_type` string keeps old code path untouched.
- `environments/utils/utils.py` `get_config`: append `cnot_cost`, `rot_cost`,
  `gadget_bonus_givens`, `gadget_bonus_fsim`, `gadget_bonus_brickwork`, `gadget_bonus_symm`
  to the `floats` list; `gadget_types` to `lists`.
- `TensorRL_fixed_noiseless.py` `Saver.get_new_episode`: add a `'gadgets': []` key (additive —
  `summary_<seed>.npy` is a pickled dict, downstream readers index by key, so this is safe).

**New file.** `environments/utils/gadgets.py`:
```python
class GadgetLibrary:
    def __init__(self, conf): ...          # enabled families + bonuses
    def check(self, history, in_gadget_mask) -> tuple[str|None, int]  # (family, window_len)
    def bonus(self, history, in_gadget_mask) -> float
# pattern predicates, one per family (mirror factors.factors_form_*_gadget):
def is_givens_block(window, num_qubits) -> bool      # CNOT(q,p)·RY(p)·CNOT(q,p) cores & 4-CNOT variant
def is_fsim_block(window, num_qubits) -> bool        # RZ/RY-sandwiched double CNOT on same pair
def is_brickwork_cell(window, num_qubits, parity) -> bool  # CNOT ladder matching Phase-A layer parity
def is_symmetry_block(window, num_qubits) -> bool    # particle-number-conserving pattern (RZ,CNOT,RY,CNOT)
```
Windows are lists of decoded actions `[ctrl, x, rot_qubit, rot_axis]` (same encoding as
`agents/utils.dictionary_of_actions` values). Like AlphaTensor's ordering trick, patterns are
defined so at most one family completes per step; completed windows are flagged in
`actions_in_gadgets` so no action is counted twice (port of the `factors_in_gadgets[-6:]` guard).

**AlphaTensor reference.** `factors.py` (predicates + reward savings),
`environment.py:step` lines 141–185 (check ordering, mask update, reward composition).
What changes: patterns are over hardware gates, not GF(2) factors; bonuses are calibrated so a
completed motif's net cost equals its "compiled" cost (e.g. a Givens block scored as 1 native op).

**Config additions** (`configuration_files/TensorZero_fixed/BEH26q_TNbond2_gadget.cfg`):
```ini
[env]
fn_type = gadget_shaped
use_gadgets = 1
gadget_types = ["givens","fsim","brickwork","symm"]
gadget_bonus_givens = 0.30
gadget_bonus_fsim = 0.20
gadget_bonus_brickwork = 0.15
gadget_bonus_symm = 0.20
cnot_cost = 0.05
rot_cost = 0.01
```

**Validation experiment.** 5 seeds × {gadget-on, gadget-off} on 6-BeH₂ (cheapest molecule,
`num_layers = 47`-class budget); metric: episodes-to-first-success, CNOT count of best circuit
(recoverable from `summary_<seed>.npy['train'][ep]['actions']` via `dictionary_of_actions`),
vs. TensorRL-fixed paper numbers.

**Risk & fallback.** Low. Worst case bonuses distort learning ⇒ set all bonuses to 0 (exact
baseline) and keep only `cnot_cost` (already precedented by the unused `cnot_rwd_weight` key in
`CircuitEnv.__init__`). Bonus magnitudes are the main tuning burden.

### P1 — Rotosolve single-angle analytic evaluation

**Goal.** Replace the per-step full COBYLA (`scipy_optim` with `global_iters = 1000`) by:
analytic Rotosolve update of the newly placed angle (+ optional Rotosolve sweep), full COBYLA only
every `full_opt_every` steps and on episode termination. **Success criterion:** on
`TensorRL_fixed/H2O8q_TNbond2.cfg`, ≥5× reduction in summed `nfev` (already logged per step in
`summary_<seed>.npy`) at ≤10% degradation in success probability.

**Existing files modified.**
- `environments/environment_qulacs_TN_notin_agent.py`:
  - `CircuitEnv.step()`: the dispatch `if self.optim_method in ["scipy_each_step"]` (line 283)
    gains `elif self.optim_method == "rotosolve"` calling `self.rotosolve_optim(...)`. Must still
    set `self.nfev` and `self.opt_ang_save` (currently only defined on the scipy path — a latent
    crash if `optim_method` is None; fix by initializing both in `reset()`).
  - New method `CircuitEnv.rotosolve_optim(self, sweep: bool) -> (thetas, nfev, opt_ang)`:
    identifies the last-placed rotation from `self.current_action`, evaluates the qulacs cost at
    θ ∈ {φ, φ±π/2} via the existing `vc.get_energy_qulacs` with its `which_angles` argument
    (already supported, `VQE_qulacs_TN_notin_RL.py:48`), applies
    `θ* = φ − π/2 − atan2(2E(φ) − E(φ+π/2) − E(φ−π/2), E(φ+π/2) − E(φ−π/2))`.
    3 evaluations per rotation instead of ~1000 (CNOT actions: 0 evaluations).
- `environments/utils/utils.py`: nothing (`method`, `optim_alg` already parsed as strings).

**New file.** `environments/utils/rotosolve.py` — the pure math
(`rotosolve_step(cost_fn, x0, idx) -> float`, `rotosolve_sweep(cost_fn, x0, n_passes) -> ndarray`),
unit-testable against COBYLA on 4q LiH (`mol_data/LIH_4q_...parity.npz`).

**AlphaTensor reference.** None (this ingredient is TensorRL-specific; it is what makes P3's tree
affordable — the "Level 0" surrogate of the idea doc §3.4).

**Config additions.** In `[non_local_opt]`: `method = rotosolve`, `full_opt_every = 5`,
`rotosolve_sweeps = 1` (ints parse natively in `get_config`).

**Validation.** H2O-8q, 3 seeds, `method = scipy_each_step` vs `rotosolve`: compare wall-clock/step
(`summary['train'][ep]['time']`), total nfev, and final error vs the paper's TensorRL-fixed row.

**Risk & fallback.** Medium: single-angle updates can stall in plateaus. Fallback ladder is built
into the config: `full_opt_every = 1` degenerates to the baseline; intermediate values trade cost
for accuracy. Note Rotosolve's closed form is exact here because every parametrized gate is a
single-qubit RX/RY/RZ (generator with eigenvalues ±1/2).

### P2 — Symmetrized axial attention network + permutation augmentation

**Goal.** Config-selectable replacement of the MLP built by `DQN.unpack_network`
(`agents/DeepQ.py:147`). **Success criterion:** ≥ MLP success probability on 8q at equal episode
budget, and a strictly better learning curve on 12-LiH (`LIH12q_TNbond2.cfg`), where the MLP input
is largest.

**Existing files modified.**
- `agents/DeepQ.py` `DQN.__init__`/`unpack_network`: read `conf['agent'].get('network_type','mlp')`;
  branch `axial_attention` builds `AxialAttentionQNet` from the new module. The net keeps the MLP's
  exact interface — input: flat state of length `state_size` (+1 if `en_state`, +1 if
  `threshold_in_state`, appended by `modify_state` in the entry script), output:
  `(batch, action_size)` — so `act`, `replay`, checkpointing (`policy_net.state_dict()`) all work
  unchanged.
- `agents/__init__.py`: no change needed for P2 (same `DQN_Nstep` class is used).

**New files.**
- `agents/attention_torso.py` (PyTorch ports, 1:1 with `alphatensor_quantum/src/networks.py`):
  ```python
  class Symmetrization(nn.Module):          # A∘X + (1−A)∘X^T, A = sigmoid(logits), logits zero-init
  class SelfAttention(nn.Module): ...
  class TransformerDecoderBlock(nn.Module): # pre-LN, GELU MLP, skip connections
  class SymmetrizedAxialAttention(nn.Module):  # alternate depth-axis / qubit-axis attention + Symmetrization
  class AxialAttentionQNet(nn.Module):
      def __init__(self, num_layers, num_qubits, action_size, num_scalars, cfg): ...
      def forward(self, flat_state) -> Tensor  # (batch, action_size)
  ```
  Input handling: split the flat vector back into the `(num_layers, num_qubits+3, num_qubits)`
  binary tensor (angles are excluded when `conf['agent']['angles'] = 0`, matching
  `CircuitEnv.step`'s `next_state[:, :self.num_qubits+3]` truncation) plus trailing scalars
  (energy/threshold). Grid = (num_layers × num_qubits); per-cell channels = that layer's column
  slice (CNOT row block + 3 rotation one-hots). Scalars enter as a broadcast plane via a linear
  projection — the port of `TorsoNetwork`'s `LinearProjectScalars` on `sqrt_played_fraction`.
  Symmetrization applies to the qubit×qubit CNOT co-occurrence embedding (an extra N×N branch
  pooled over layers), because depth×qubit is not transpose-symmetric — this is the one deliberate
  deviation from the source, flagged for the paper's Methods.
- `agents/augment.py`:
  ```python
  def permute_state(flat_state, perm, num_layers, num_qubits, with_angles) -> Tensor
  def permute_action_index(action_idx, perm, num_qubits) -> int   # via dictionary_of_actions
  def commutation_reorder(episode_actions, num_qubits, rng) -> list[int]  # disjoint-support swaps
  def hamiltonian_symmetry_perms(ham_type, geometry, num_qubits) -> list[tuple]
  ```
  Applied inside `DQN_Nstep.replay` sampling (transition-level relabeling), gated by
  `[agent] augment_permutations = 1`.

**AlphaTensor reference.** `networks.py` for the torso (near-verbatim);
`change_of_basis.py` reference-only (permutations replace GF(2) basis changes; molecule symmetry
groups are small — spin-up/down orbital swap under Jordan–Wigner — while TFIM/Heisenberg chains get
reflection + translation).

**Config additions.** `[agent] network_type = axial_attention`, `attn_heads = 8`,
`attn_head_depth = 8`, `attn_layers = 4`, `attn_mlp_widening = 2` (demo-scale defaults from
`demo_config.py`, not the paper-scale 16×32×4), `augment_permutations = 1`.

**Validation.** `H2O8q_TNbond2.cfg` and `LIH12q_TNbond2.cfg`, 5 seeds, `network_type` mlp vs
axial_attention, all else fixed. Metric: success probability + episodes-to-threshold vs the
NeurIPS table row.

**Risk & fallback.** Medium: attention is data-hungrier than the MLP (AlphaTensor needed
demonstrations for the same reason — P4 mitigates). Fallback: keep `network_type = mlp` for the
paper's ablation 4 and present attention as the scaling story (12q+) only.

### P3 — Sample-based MCTS actor (AlphaZero-style)

**Goal.** Planner replaces ε-greedy action selection; DDQN Bellman learner replaced by
policy/value learner from visit counts. **Success criterion:** at *matched wall-clock* on H2O-8q,
MCTS agent beats the DDQN baseline's success probability, or reaches equal accuracy with fewer
committed env steps (ablation 1 of the idea doc).

**Existing files modified.**
- `environments/environment_qulacs_TN_notin_agent.py`: add lightweight planning API —
  `CircuitEnv.snapshot() -> dict` / `CircuitEnv.restore(snap)` (copies of `state`, `moments`,
  `illegal_actions`, `step_counter`, `prev_energy`, `action_history`, `actions_in_gadgets`,
  curriculum threshold) and `CircuitEnv.simulate_step(action, level) -> (reward, done, energy)`
  which mirrors `step()` but (a) never touches `self.curriculum`/`Saver` state, (b) evaluates the
  leaf per surrogate `level`: 0 = frozen angles + Rotosolve on the new angle (P1 machinery),
  1 = no evaluation (value-head bootstrap only), 2 = MPS evaluator (P5).
- `agents/__init__.py`: `from . import MCTSZero`.
- New entry script `TensorZero_fixed_noiseless.py` (clone of `TensorRL_fixed_noiseless.py`):
  identical `Saver`/loop; two-line delta — `agent.act` unchanged signature, and after `env.step`
  the agent's `remember` stores `(state, mcts_policy, z)` instead of Q-transitions. Cloning (not
  editing) preserves the baseline script byte-for-byte. Also replace the interactive
  `input("Enter 0 or 1")` device prompt with a `--device` flag (HPC batch requirement).

**New file.** `agents/MCTSZero.py`:
```python
class PolicyValueNet(nn.Module):     # AxialAttention torso (P2) + policy head (action_size logits,
                                     # masked by illegal actions) + distributional value head
                                     # (N_q quantiles; act on a risk-seeking quantile, train on all)
class MCTSAgent:
    def __init__(self, conf, action_size, state_size, device)   # same ctor signature as DQN
    def act(self, state, ill_action) -> (int, bool)             # runs K-sampled PUCT search
    def remember(self, state, policy_target, value_target, ...) 
    def replay(self, batch_size) -> float                       # KL(policy) + quantile(value) loss
class SearchNode: ...   # holds env snapshot, prior, N/W/Q per sampled child
```
Search: at the root, sample `k_sampled_actions` (16–32) legal actions from the softmax prior
(sampled-AlphaZero style, needed because `action_size = num_qubits*(num_qubits+2)` — 120 at 10q —
and each expansion may cost a physics evaluation); PUCT selection with
`qtransform_by_parent_and_siblings`-style normalized Q (port the normalization trick from mctx as
used in `demo/agent.py:_run_iteration_agent_env_interaction`); `num_simulations` 50–400.
Deterministic env ⇒ nodes carry real `CircuitEnv.snapshot()`s, exactly like
`_recurrent_fn` returning env states as "embeddings".

**AlphaTensor reference.** `demo/agent.py`: `_recurrent_fn` (deterministic-env tree),
`_loss_fn` (KL policy loss against visit-count weights + squared value loss — we keep KL, swap
squared for quantile loss for risk-seeking), action sampling from `policy_output.action_weights`,
`GameStats` best-ever tracking (we mirror it as `best_circuit` tracking in the Saver dict).
What must change: JAX/mctx → hand-rolled PyTorch tree (no vmap over batch of games — TensorRL runs
one env), per-move budget is precious ⇒ K-sampled actions + surrogate levels.

**Config additions.** `[agent] agent_type = MCTSZero`, `agent_class = MCTSAgent`,
`num_simulations = 100`, `k_sampled_actions = 16`, `c_puct = 1.25`, `dirichlet_alpha = 0.3`,
`root_exploration_frac = 0.25`, `value_quantiles = 8`, `risk_quantile = 0.75`,
`surrogate_level = 0`. Add `c_puct`, `dirichlet_alpha`, `root_exploration_frac`, `risk_quantile`
to `get_config`'s `floats` list.

**Validation.** H2O-8q: DDQN (paper config) vs MCTS at equal wall-clock budget; report both
env-query count and time-to-solution (the "honest accounting" of idea doc §4.4).

**Risk & fallback.** **High** (cost blow-up is the top risk in idea doc §6). Fallbacks, in order:
surrogate level 1 (zero leaf evaluations — pure AlphaZero), `num_simulations = 50`, and finally
keep P0–P2 improvements on the DDQN stack (each independently publishable per the idea doc).

### P4 — Synthetic demonstrations + mixed-loss pretraining

**Goal.** Pretrain `PolicyValueNet` (and optionally the DQN, as behavior cloning) before touching
the expensive VQE env. **Success criterion:** demo-pretrained P3 agent reaches first success on
H2O-8q in ≤50% of the episodes of the non-pretrained P3 agent.

**New files** (new top-level `demos/` package, as the idea doc's phase table specifies):
- `demos/generate_demos.py`:
  ```python
  class CircuitDemonstration(NamedTuple):   # mirror of demonstrations.Demonstration
      states: np.ndarray      # (T, L, N+6, N) binary tensors along the trajectory
      actions: np.ndarray     # (T,) indices into dictionary_of_actions(N)
      values: np.ndarray      # (T,) reward-to-go incl. gadget savings
      gadget_mask: np.ndarray
  def generate_random_demo(num_qubits, num_layers, dem_conf, rng) -> CircuitDemonstration
      # random legal action sequences; with prob prob_include_gadget, splice in whole
      # gadget windows from environments/utils/gadgets.py (port of
      # demonstrations._overwrite_factors_with_gadgets, incl. the insert-index scheme)
  def generate_dmrg_delta_demo(mol_npz, chi_lo, chi_hi, num_layers) -> CircuitDemonstration
      # runs mps2qc.gs_dmrg at chi_lo and chi_hi, fits both with mps2qc.mps_to_qc,
      # takes the transpiled gate-difference (via tnqc_ansatze.qiskit_circ_from_tn_params),
      # re-encodes gates into the RL state-tensor layout (reuse the DAG depth-wise decoding
      # already in CircuitEnv.__init__: circuit_to_dag(...).layers())
  def main(): ...  # writes demos/data/<system>_<source>.npz
  ```
- `demos/pretrain.py`: supervised loop — cross-entropy on actions + quantile/MSE on values
  (`get_action_and_value` analog is trivial here: values are stored reward-to-go); saves a
  checkpoint loadable through the existing `init_net`/`init_path` config mechanism
  (`TensorRL_fixed_noiseless.py` lines 239–252 already reload `policy_net`/`optim`).

**Existing files modified.**
- `agents/MCTSZero.py` `MCTSAgent.replay`: mixed loss
  `L = (1−w(t))·L_acting + w(t)·L_demo` with `w(t)` a step-indexed piecewise-constant schedule —
  direct port of `_loss_fn` + `LossParams.demonstrations_boundaries_and_scales`
  (`demo_config.py`: `{60: 0.99, 200: 0.5, 5000: 0.2, 10000: 0.1}`, `init = 1.0`).
- `environments/environment_qulacs_TN_notin_agent.py`: none (demos are offline data).

**AlphaTensor reference.** `demonstrations.py` (generation scheme, gadget injection probabilities:
`prob_include_gadget = 0.9`, `max_num_gadgets`, `prob_toffoli_gadget` → per-family probs) and
`demo/agent.py:_loss_fn` / `_update_demonstrations_and_states` (mixing). What changes: a "solved"
demo means *a circuit and its true reward trace*, computed by replaying the action sequence through
`CircuitEnv` in `train_flag=False` mode at small N (qulacs statevector, ≤12q) — physics ground
truth instead of algebraic factorizations. The DMRG-Δχ source has no AlphaTensor analog (novelty
claim #4 of the idea doc).

**Config additions.** `[demos] enabled = 1`, `demo_path = demos/data/H2O_8q_mixed.npz`,
`init_demo_weight = 1.0`, `demo_boundaries = [60,200,5000]`, `demo_scales = [0.99,0.5,0.2]`
(register `init_demo_weight` in `floats`, `demo_boundaries`/`demo_scales` in `lists`).

**Validation.** Generate 100k random-circuit demos + ~200 DMRG-Δχ demos (χ 2→3, 2→4) for H2O-8q
(bond-2/3 `.qpy` artifacts already exist in `init_state_circ/`); pretrain; compare
episodes-to-first-success and ablate demo source (random vs Δχ vs mixed).

**Risk & fallback.** Low-medium. Random circuits may be too off-distribution — mitigated by gadget
biasing and by the Δχ source being exactly on-task. Fallback: use demos only for policy-head
pretraining (skip value head), which cannot hurt the planner.

### P5 — MPS surrogate inside the tree; 20–30 qubit scale-up

**Goal.** Tree-node evaluation at truncated bond dimension via quimb (already a Phase-A
dependency), unlocking N > statevector limits. **Success criterion:** (a) at 10–12q, MPS-surrogate
(χ_s = 8–16) tree decisions agree with statevector decisions ≥90% of the time at ≥5× leaf-eval
speedup; (b) first RL-QAS result at 20–30q TFIM/Heisenberg beating the TN warm-start energy.

**New file.** `environments/VQAs/VQE_mps_surrogate.py`:
```python
class MPS_Parametric_Circuit:            # same construct_ansatz(state) contract as
    def __init__(self, n_qubits, chi, init_mps): ...   # VQE_qulacs_TN_notin_RL.Parametric_Circuit
    def construct_ansatz(self, state): ...             # gates applied via quimb CircuitMPS with
                                                       # max_bond=chi, cutoff config
def get_exp_val_mps(n_qubits, circ_mps, ham_mpo, chi) -> float
def ham_mpo_from_npz(npz_path) -> qtn.MatrixProductOperator   # reuses mps2qc.mpo_from_paulis
                                                               # on the npz 'paulis'/'weights' arrays
```
The warm-start enters as the χ-bond MPS itself (from `mps2qc.gs_dmrg`) instead of the dense
`self.TN_state = Statevector(...).data` — at 30q the dense statevector (line 158 of the env) is
impossible, so this branch bypasses it.

**Existing files modified.**
- `environments/environment_qulacs_TN_notin_agent.py`:
  - `__init__`: `leaf_evaluator = conf['env'].get('leaf_evaluator','statevector')`; when `mps`,
    skip the `Statevector`/dense-`Operator` construction (also fixes the latent bug that
    `self.tenor_circ` line 158 runs even when `tn_bond = 0`) and load the MPO/MPS instead.
  - `simulate_step(..., level=2)`: route to `get_exp_val_mps`.
  - `get_energy`/`scipy_optim`: gain an `mps` branch for ≥20q where even *committed* steps use the
    MPS evaluator (exact eigvals unavailable ⇒ `fake_min_energy` config key, already supported at
    line 60, carries the DMRG reference energy).
- `dmrg-to-qc/heisenberg_model.py` / `making_molecules.py`: **used, not modified**, to emit
  `mol_data/tfim_*_20q.npz` … `30q.npz` (with `eigvals` replaced by DMRG references; note the env's
  non-molecule branch currently hardcodes `['heisenberg','tfim_j1_h0.05']` — generalize that check
  to `startswith(('heisenberg','tfim'))`, flagged as a correction below).

**AlphaTensor reference.** None — this is the repo-native contribution (idea doc §3.4 Level 2,
novelty claim #3).

**Config additions.** `[env] leaf_evaluator = mps`, `surrogate_bond_dim = 16`,
`commit_bond_dim = 64`, `fake_min_energy = <DMRG ref>` (already a recognized float key).

**Validation.** (a) Fidelity check on H2O-10q: rank correlation between MPS-χ16 and statevector
leaf energies over 1k random tree nodes. (b) TFIM-20q run vs the TensorRL-QAS paper's 20q TFIM
result; then 30q attempt.

**Risk & fallback.** **Highest** (per idea doc §6 this phase may underdeliver without sinking the
paper). Fallbacks: keep χ_s small and use MPS only at *tree leaves* with statevector commits
(≤12q); cap the scale-up claim at 20q noiseless.

### P6 — Full benchmark grid, noise, writing

**Goal.** The paper's tables: molecules {6-BeH₂, 8-H₂O, 10-H₂O, 12-LiH, 8/10-CH₂} + TFIM/Heisenberg,
ablations 1–6 of idea doc §4.3, noisy runs. **Success criterion:** every table cell has ≥5 seeds;
headline ≥2× CNOT reduction over TensorRL-fixed at equal accuracy on ≥2 molecules.

- **Existing files modified:** replicate the P0/P1/P3 flag-gated diffs into
  `environment_qulacs_TN_notin_agent_noise.py` (+`_restricted`) and `environment_qulacs.py`
  (trainable variant) — mechanical, do last to avoid five-way divergence during development;
  create `TensorZero_fixed_noise.py` entry clone.
- **New files:** `configuration_files/TensorZero_fixed/*.cfg` grid (one per molecule × ablation
  arm), a `scripts/run_grid.sh` seed sweep, `analysis/motif_stats.py` (reads
  `summary_<seed>.npy['train'][ep]['actions']` + gadget log, mines recurring action n-grams
  *outside* the library — the "Karatsuba moment" analysis, mirroring `GameStats.best_factors`
  inspection).
- **Config:** noise cells reuse `noise_values`/`n_shots` exactly as `H2O8q_TNbond2_noise.cfg`.
- **Validation:** reproduce three baseline table rows with flags off before running any new arm.
- **Risk:** compute volume only; the grid is embarrassingly parallel over (config, seed).

---

## 4. Integration order & dependencies

```
P0 (gadgets/reward)────────────┐
P1 (rotosolve)──────┐          ├──► P3 (MCTS agent) ──► P4 (demos, mixed loss) ─┐
P2 (attention net)──┴──────────┘         │                                      ├──► P6 (grid)
                                         └──► P5 (MPS surrogate, 20–30q) ───────┘
```
- P0, P1, P2 are mutually independent; each lands as a standalone ablation on the unchanged DDQN
  stack (P0/P1 touch only env+config; P2 touches only agent+config).
- P3 consumes P2's torso (policy/value heads) and P1's Level-0 leaf evaluator; can start with a
  plain-MLP torso if P2 slips.
- P4 requires P3's loss structure (policy targets); demo *generation* can start any time (depends
  only on `dictionary_of_actions` + `gadgets.py` from P0).
- P5 requires P3's `simulate_step(level=...)` hook; its data prep (large npz + MPS warm-starts) is
  independent and can run in parallel.
- P6 last; noisy-env replication deliberately deferred until P0–P3 diffs are frozen.

## 5. Corrections to `a_star_paper_idea1.md` (vs. the real code)

1. **CH₂O → CH₂.** `mol_data/` ships `CH2_8q_...npz` and `CH2_10q_...npz` (methylene). There is no
   CH₂O/formaldehyde data. All plan text uses 8/10-CH₂.
2. **P1 touchpoint:** `scipy_optim` is a method of `CircuitEnv` in
   `environments/environment_qulacs_TN_notin_agent.py` (line 452), not of
   `environments/VQAs/VQE_qulacs_TN_notin_RL.py`; the VQAs file holds the cost functions
   (`get_energy_qulacs`, `get_exp_val`).
3. **TFIM/Heisenberg assets:** repo has only `heisenberg_5q.npz` and `tfim_j1_h0.001_6q.npz`, while
   the env's special-case list hardcodes `'tfim_j1_h0.05'` (`CircuitEnv.__init__` line 78). 20–30q
   spin data must be generated (P5) and the hardcoded name check generalized.
4. **Action-space formula:** idea doc's `3N + 2·C(N,2)` equals the code's
   `action_size = num_qubits*(num_qubits+2)` — consistent, kept.
5. **Sampled MCTS is not in the open AlphaTensor code:** `demo/agent.py` uses full-width
   `mctx.muzero_policy` (800 sims paper / 80 demo). The K-sampled search must be implemented fresh.
6. **`agents/__init__.py` only imports `DeepQ_restricted`, `DeepQNstep`, `DeepQNstep_restricted`**;
   any new agent module must be registered there for the entry scripts' reflection lookup.
7. **Latent env bugs to fix in passing:** `self.TN_state = Statevector(self.tenor_circ)` (line 158)
   executes even when `tn_bond = 0` (NameError); `nfev`/`opt_ang` are undefined in `step()` when
   `optim_method` ≠ `scipy_each_step`; the entry script's interactive device `input()` blocks batch
   jobs.
8. **New float-valued config keys must be registered** in the `floats` list inside
   `environments/utils/utils.get_config`, else they arrive as strings.

## 6. Open questions (need a human decision)

1. **Gadget bonus calibration** (P0): fixed hand-set values (AlphaTensor style: net motif cost =
   compiled cost) vs. normalized to the energy-reward scale? Proposal: fixed, with one sweep on
   6-BeH₂ (scheduled as part of milestone M1, see `training_run_plan.md`).
2. **Risk-seeking value head** (P3): quantile-regression head acting on the 0.75-quantile, or
   simple max-backup in the tree? Quantile head is closer to the paper's claim but adds a knob.
3. **Reference energy at 30q** (P5): report error vs. DMRG-χ_max energy via `fake_min_energy`, or
   vs. extrapolated DMRG? Affects the headline "first 30q RL-QAS" table.
4. **Transfer experiments** (idea doc §2 row 7): in scope for v1 of the paper, or deferred? They
   require a multi-Hamiltonian `CircuitEnv` (the `curriculum_dict` keyed by `ham_type` suggests
   this was once planned) — nontrivial extra work not covered by P0–P6.
5. **Compute budget:** MCTS at 100 sims × Level-0 leaves ≈ 3 rotation-evals per sim per move —
   is a per-run wall-clock cap (e.g. 48h/seed on HPC) acceptable for the matched-wall-clock
   ablation protocol? (M4's S2 dress rehearsal measures the real per-move cost before the grid
   is committed.)
6. ~~**Baseline re-runs**~~ — **RESOLVED (v1.1, per review):** TensorRL-fixed baselines WILL be
   re-run on project hardware as milestone **M0** in `training_run_plan.md`; ablation 1
   (matched wall-clock) is invalid otherwise, and M0's outputs double as the flags-off parity
   references.

## 7. Amendments accepted after review (v1.1)

Full evidence in `implementation_plan_v1_review.md`; these supersede the corresponding text above.

1. **P0 logging contradiction (E1):** `TensorRL_fixed_noiseless.py` stays truly untouched. The
   `'gadgets'` key and its `one_episode` append move into the cloned entry script
   `TensorZero_fixed_noiseless.py` (created early, in P0 rather than P3); the DDQN gadget arm
   runs through the clone.
2. **P0 anti-farming guard (review d.2):** gadget bonus is paid only when the completing step does
   not increase energy, plus a per-episode cap on rewarded gadgets (`max_gadgets_per_episode`,
   default 5, registered in `get_config`). The bonuses are *shaping*, not objective correction
   (unlike AlphaTensor's true-cost savings) — paper text must say so. Motif predicates are
   structural, not unitary-level identities; no particle-conservation claims for the bare
   `CNOT·RY·CNOT` pattern.
3. **P1 parameter-index mapping (E5):** add `environments/utils/rotosolve.py::action_to_param_index
   (layer, axis, qubit) -> int` replicating `construct_ansatz`'s layer-major enumeration, with a
   unit test against `circuit.get_parameter_count()` ordering. Silent-failure trap otherwise.
4. **P3 interface gaps (E3):** `MCTSAgent` receives the live env via a post-construction setter
   (`agent.env = env`, mirroring the `agent.saver` idiom at `TensorRL_fixed_noiseless.py:237`);
   value targets `z` are backfilled at episode end (per-episode buffer, flushed on `done`);
   the clone's `init_net` reload block is adapted (no `target_net`, no `Transition` replay
   buffer) and `demos/pretrain.py` emits that adapted checkpoint format.
5. **P3 snapshot completeness (review rev. 5):** `snapshot()`/`restore()` cover the exhaustive
   field list — add `current_action`, `previous_action`, `energy`, `error`, `halting_step`,
   `current_number_of_cnots` — with a round-trip unit test.
6. **P4 re-scope (E4):** random-circuit demos are *policy-head-only* pretraining (legality +
   motif priors); they are NOT solutions (no target-conditioning in TensorRL's observation, unlike
   AlphaTensor where the target tensor is constructed from the demo and observed). Value-head
   supervision comes only from (a) greedy/Rotosolve descent rollouts that actually reduce energy
   and (b) DMRG-Δχ demos. The Δχ recipe requires **new frozen-prefix fitting code** (a
   `mps_to_qc` wrapper initializing the χ_hi ansatz from the χ_lo solution with the prefix
   frozen) — `mps2qc.py:300` random-initializes every gate, so two independent fits share no
   common prefix. The DQN behavior-cloning aside is dropped.
7. **P5 data path rewrite (E2):** `heisenberg_model.py` builds dense 2ⁿ×2ⁿ matrices via `np.kron`
   (Heisenberg only, no TFIM) — it cannot emit 20–30q data. New: pauli-strings-only npz schema
   for >12q (no dense `hamiltonian`/`eigvals`; DMRG reference energy instead) + a new sparse/MPO
   generator script. The `leaf_evaluator = mps` branch must bypass the dense
   `Operator(...).to_matrix()` load in **`reset()` (lines 372–377) as well as `__init__`**.
   Final circuits at 20–30q are re-evaluated at high χ before any energy is quoted.
8. **Small fixes (E6):** dict-adapter for `mps2qc.mpo_from_paulis` (takes `dict[str, complex]`,
   not the npz arrays); `gs_dmrg` returns a `(mps, dmrg)` tuple; the noiseless trainable script is
   `TensorRL_training_and_structureRL_noiseless.py` (Section 1 glob corrected);
   `environment_qulacs_noise.py` added to P6's replication list (or the trainable-noise arm is
   explicitly descoped).
9. **P2 paper framing (review d.5):** the torso is presented as "axial attention with an auxiliary
   symmetrized qubit×qubit branch", not "symmetrized axial attention" wholesale.

## 8. Phase → training-milestone map (see `training_run_plan.md` for full specs)

| After phase | Milestone | Smoke gate | Full training run | Feeds |
|---|---|---|---|---|
| — (before P0) | **M0** baseline reproduction | S0 env-setup | 5 seeds × {BeH₂, H₂O-8q, H₂O-10q, LiH-12q} + trainable/StructureRL on 8q, unmodified code; record wall-clock | Parity references, matched-clock budget, ablation 6 |
| P0 | **M1** | S0 predicates + S1 gadget smoke (anti-farming check) | 6-BeH₂ + 8-H₂O gadget-on vs M0, 5 seeds; bonus sweep on BeH₂ | Ablation 2 |
| P1 | **M2** (mergeable with M1) | S0 index-map test + S1 rotosolve | 8-H₂O `full_opt_every ∈ {1,5,20}` vs M0, 5 seeds | Efficiency figure; default optimizer |
| P2 | **M3** | S0 shapes/gradients + S1 attention | 8-H₂O + 12-LiH, mlp vs attention, 5 seeds | Ablation 4 |
| P3 | **M4** (headline) | S0 snapshot/tree tests + S1 cheap-tree + **S2 rehearsal** | 8-H₂O, DDQN(M0) vs MCTS {50,100 sims}×{level 0,1}, 5 seeds, matched wall-clock | Ablations 1, 5 |
| P4 | **M5** | S0 demo gen + S1 pretrained-vs-scratch | 8-H₂O, 4 pretraining arms, 5 seeds | Ablation 3 |
| P5 | **M6** | S0 MPS agreement + rank-correlation + S1 TFIM-20q | 10-H₂O level-2-in-tree; TFIM-20q; 30q stretch | Ablation 5, headline 2 |
| P6 | **M7** | R2 parity on all env variants + S2 noisy | Full publication grid, 5–10 seeds, noise rows, motif mining | All tables |
