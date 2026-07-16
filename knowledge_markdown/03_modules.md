# Components, Module by Module

## `dmrg-to-qc/` — Phase A: Hamiltonian → MPS → PQC

| File | Role |
|---|---|
| `dmrg_to_qc.py` | Entry point / orchestrator. Interactive CLI: pick molecule, bond dimension `χ`, number of brickwork layers. Calls `mps2qc` + `tnqc_ansatze`, writes `init_state_circ/*.{qasm,qpy}`. Also runs a sanity check comparing statevector energy before/after qasm round-trip. |
| `mps2qc.py` | The DMRG + MPS→circuit machinery: `mpo_from_paulis` (Pauli-string Hamiltonian → quimb MPO), `gs_dmrg` (runs quimb's DMRG to get the ground-state MPS), `mps_to_qc` (fits a TN circuit to the MPS by maximizing overlap using the supplied optimizer), `compute_energy`. |
| `stiefel_opt.py` | `StiefelAdam`: Riemannian Adam optimizer operating on the complex Stiefel manifold (unitary matrices) via Cayley-transform retraction — implements the algorithm in the paper's Appendix B, used to keep each 2-qubit gate unitary during gradient-based fitting. |
| `tnqc_ansatze.py` | Defines the brickwork ansatz structure and `qiskit_circ_from_tn_params`, which turns fitted TN parameters into an actual `qiskit.QuantumCircuit`, transpiled to the target basis gate set (`{cx,rx,ry,rz}` or the `su4`-decomposed `{rxx,ryy,rzz,...}` variant). |
| `making_molecules.py`, `heisenberg_model.py` | Build the problem Hamiltonians ahead of time (molecular electronic-structure Hamiltonians / Heisenberg & TFIM spin models) and serialize them to `mol_data/*.npz` (`hamiltonian`, `eigvals`, `weights`, `paulis`). |
| `mol_data/` | Precomputed Hamiltonians for each benchmark system (6-BeH₂, 8/10/12-qubit H₂O, 12-LiH, CH₂, Heisenberg/TFIM models). |
| `init_state_circ/` | Output of Phase A: the TN warm-start circuits (one per molecule × bond-dimension combo) that Phase B loads. |

## `agents/` — the RL agent (Double DQN)

| File | Role |
|---|---|
| `DeepQ.py` | Base `DQN` class: builds the MLP `policy_net`/`target_net` (`unpack_network`, configurable hidden layer sizes/dropout from `conf['agent']['neurons']`), holds the (optionally prioritized) replay memory, implements `act` (ε-greedy action selection, masking illegal actions with `-inf`), `replay` (Double-DQN Bellman update — action selected by `policy_net`, evaluated by `target_net`; `SmoothL1Loss`; periodic target-net sync; ε geometric decay). |
| `DeepQNstep.py` | `DQN_Nstep(DQN)` — swaps in `N_step_ReplayMemory`, which collapses `n_step` (default 5) consecutive transitions into a single n-step-return transition before pushing to the main buffer, then does the same Double-DQN update as the base class. This is the agent class actually referenced by the shipped `.cfg` files (`agent_type = DeepQNstep`, `agent_class = DQN_Nstep`). |
| `DeepQ_restricted.py`, `DeepQNstep_restricted.py` | Same as above but action-decoding uses a restricted topology's action dictionary (see `utils_topology_restrict.py`) for hardware-connectivity-constrained experiments. |
| `utils.py` | Re-exports/local copies of `dictionary_of_actions` / `dict_of_actions_revert_q` used to translate a discrete action index into `[ctrl, targ_offset, rot_qubit, rot_axis]`. |
| `utils_topology_restrict.py` | Builds the reduced action dictionary when only a subset of qubit pairs (matching real hardware connectivity) are allowed for CNOTs. |

## `environments/` — the RL environment (Gym-style `CircuitEnv`)

There are 5 environment variants; all share the same overall contract
(`reset() -> state`, `step(action) -> (next_state, reward, done)`) but differ in **how the
TN warm-start is used** and **whether noise is simulated**:

| File | TN warm-start handling | Noise |
|---|---|---|
| `environment_qulacs.py` | TN circuit **decoded into the initial RL state tensor** (structure + optionally real angles). Used for **TensorRL (trainable TN-init)** and, with `zero_param_init=1`, **StructureRL**. | none |
| `environment_qulacs_TN_notin_agent.py` | TN circuit converted **once** to a `Statevector` (`self.TN_state`) and used purely as the reference/initial wavefunction; **not** part of the RL state tensor. Used for **TensorRL (fixed TN-init)**. | none |
| `environment_qulacs_noise.py` | Same as `environment_qulacs.py` | depolarizing/amplitude-damping/shot noise via `noise_models`/`noise_values`/`n_shots` config |
| `environment_qulacs_TN_notin_agent_noise.py` | Same as the fixed-TN env | with noise |
| `environment_qulacs_TN_notin_agent_noise_restricted.py` | Same as fixed-TN + noise | with noise, and a restricted CNOT topology (paired with `agents/*_restricted.py`) |

Common internals of `CircuitEnv` (see `environment_qulacs_TN_notin_agent.py` as the
representative example):

- **`__init__`**: loads the molecule's Hamiltonian npz (`dmrg-to-qc/mol_data/`), loads the
  matching TN-init `.qpy` circuit, extracts its DAG depth-wise (`depth_wise_gates`),
  computes `min_eig`/`max_eig` (true ground energy bounds used for reward/error), builds a
  `curriculum` object (see `environments/utils/curricula.py`) that manages the acceptance
  threshold over training.
- **State tensor layout**: shape `(num_layers, num_qubits+6, num_qubits)`. Rows
  `[0:num_qubits]` = CNOT control/target one-hot encoding per layer; rows
  `[num_qubits:num_qubits+3]` = which qubit has an RX/RY/RZ in that layer; rows
  `[num_qubits+3:num_qubits+6]` = the angle values themselves (dropped from the network
  input unless `conf['agent']['angles']=1`).
- **`reset()`**: zero state tensor, optionally seeded with the TN circuit's structure/angles
  (trainable/StructureRL envs only), reloads the Hamiltonian, resets `moments`/illegal-action
  tracking, computes the initial energy.
- **`step(action)`**: decodes the action into a CNOT or rotation gate, writes it into the
  first free "moment" slot per qubit, re-optimizes all placed rotation angles with
  `scipy.optimize.minimize` (COBYLA by default, `self.optim_alg`/`self.optim_method` from
  `[non_local_opt]` in the config) against the qulacs-simulated energy, computes
  `error = |energy - min_eig|`, a shaped `reward` (`reward_fn`), and `done` (chemical accuracy
  reached, or step budget `num_layers_termination` exhausted).
- **`illegal_action_new()`**: prevents the agent from proposing redundant actions
  (e.g. two colliding gates on the same qubit in the same moment) — shrinks the effective
  action space each step.
- **`make_circuit()`**: rebuilds a qulacs `QuantumCircuit` from the binary state tensor.
- **`utils/curricula.py`**: `VanillaCurriculum` / `MovingThreshold` / `SuccesCountThreshold` —
  different policies for tightening (or not) `accept_err` as training progresses, referenced
  by `conf['env']['curriculum_type']`.

### `environments/VQAs/` — energy evaluation (the "VQE" inner loop)

| File | Role |
|---|---|
| `VQE_qulacs_TN_notin_RL.py` | `Parametric_Circuit.construct_ansatz(state)` builds the qulacs circuit for a given binary-encoded RL state (for the fixed-TN-init variant); `get_exp_val` / `get_energy_qulacs` compute `⟨ψ|H|ψ⟩` (statevector-exact or shot-sampled) — this is what `CircuitEnv.get_energy`/`scipy_optim`'s `cost(x)` call every optimization iteration. |
| `VQE_qulacs.py` | Same, for the trainable-TN-init / StructureRL variant. |
| `VQE_qulacs_noise.py`, `VQE_qulacs_TN_notin_RL_noise*.py` | Noisy-simulation counterparts (apply depolarizing/amplitude-damping channels gate-by-gate, add shot noise). |
| `VQE.py`, `VQE_su4.py`, `VQE_qulacs_su4.py` | Variants targeting the `su4` (`rxx/ryy/rzz`) basis gate set instead of `{cx,rx,ry,rz}`. |

## `configuration_files/` — experiment configs

One `.cfg` (INI-style, parsed by `environments/utils/utils.get_config`) per
molecule/method/bond-dimension combination, under `StructureRL/`, `TensorRL_fixed/`,
`TensorRL_trainable/`. Sections:

- `[general]` — `episodes` (training episode count).
- `[env]` — qubit count, `num_layers` (max circuit depth/step budget), noise config,
  `TN_init`/`TN_bond` (whether/how much to use the Phase-A warm-start),
  `zero_param_init` (StructureRL switch), reward function type, curriculum config,
  `accept_err` (chemical-accuracy threshold).
- `[problem]` — which Hamiltonian (`ham_type`, `geometry`, fermion-to-qubit `mapping`).
- `[agent]` — DQN hyperparameters: network size (`neurons`), batch size, replay memory size,
  learning rate, `agent_type`/`agent_class` (which class in `agents/` to instantiate),
  `n_step`, ε-schedule, target-net update period, `final_gamma` (per-step discount, converted
  to `γ = final_gamma^(1/num_layers)`).
- `[non_local_opt]` — the classical optimizer used inside every RL step to fit rotation
  angles (`optim_alg`, e.g. `COBYLA`; `method`, e.g. `scipy_each_step`; `global_iters`).

## Root-level training scripts

Each script is a thin orchestrator: parses `--seed/--config/--experiment_name`, builds
`CircuitEnv` + the configured agent class, runs the episode loop (`one_episode`/`train`),
and periodically checkpoints `policy_net`, `optim`, and the replay buffer plus a
per-episode stats dump (`Saver` → `summary_<seed>.npy`). They differ only in which
environment module they import and whether a noise/topology-restricted config is expected:

- `Tensor_training_and_structureRL_noiseless.py` / `Tensor_training_and_structureRL_noise.py`
  → `environment_qulacs[.py|_noise.py]` (trainable-TN-init / StructureRL)
- `TensorRL_fixed_noiseless.py` / `TensorRL_fixed_noise.py` / `TensorRL_fixed_noise_restricted.py`
  → `environment_qulacs_TN_notin_agent*.py` (fixed-TN-init)

## Baseline QAS methods (not part of the training loop above)

The paper benchmarks against CRLQAS, SA-QAS, RA-QAS, Vanilla RL, quantumDARTS, TF-QAS — these
are external repos/results (linked in `README.md`), not code living in this repository.
