# Total Training Pipeline — Dataflow

This traces one full run end-to-end: from raw Hamiltonian to a trained RL-QAS agent,
across the two phases (offline TN warm-start generation, then RL training), as actually
implemented in this repo (not just the paper's abstraction).

## Phase A — Offline: build the Hamiltonian and the TN warm-start circuit (run once per problem)

```
 1) python3 dmrg-to-qc/dmrg_to_qc.py
```

Steps inside that script (`dmrg-to-qc/dmrg_to_qc.py:main`):

1. User picks a molecule/model (`molecule_choice`) → resolves to a `mol_data/*.npz` file
   already containing `hamiltonian`, `eigvals`, `weights`, `paulis`
   (produced ahead of time by `making_molecules.py` / `heisenberg_model.py` using
   PySCF/OpenFermion-style electronic-structure + Jordan-Wigner/parity mapping).
2. User picks bond dimension `χ` and number of brickwork layers.
3. `mps2qc.mpo_from_paulis` turns the Pauli-string Hamiltonian into a Matrix Product
   Operator (MPO) (quimb).
4. `mps2qc.gs_dmrg(ham_mpo, bond_dims=[χ], num_sweeps=2)` runs DMRG → ground-state MPS
   `psi_dmrg`.
5. `mps2qc.mps_to_qc(psi_dmrg, ansatz={brickwork, num_layers}, optimizer_opts={StiefelAdam})`
   fits a brickwork tensor-network circuit to `psi_dmrg` by maximizing overlap — this is the
   Riemannian/Cayley-transform Adam optimization on the Stiefel manifold
   (`stiefel_opt.StiefelAdam`) described in the paper's Appendix B.
6. `tnqc_ansatze.qiskit_circ_from_tn_params` converts the fitted TN circuit into a real
   Qiskit `QuantumCircuit`, then **transpiles** it to the hardware-native basis gates
   `{cx, rx, ry, rz}`.
7. Sanity check: statevector energy of the transpiled circuit vs. the DMRG energy.
8. **Output artifacts** written to `dmrg-to-qc/init_state_circ/`:
   `init_<mol_config>_TNbond<χ>.qasm` and the matching `.qpy` (used by the RL environment).

```
mol_data/*.npz (H, eigvals)
        │
        ▼
   MPO (quimb)  ──DMRG──▶  MPS |ψ⟩  (bond dim χ)
        │
        ▼
 Riemannian/Stiefel-Adam fit of brickwork 2-qubit unitaries to maximize |⟨ψ|U|0⟩|
        │
        ▼
 Qiskit circuit, transpiled to {RX,RY,RZ,CNOT}
        │
        ▼
 init_state_circ/init_<mol>_TNbond<χ>.qpy   ◀── consumed by Phase B
```

## Phase B — Online: RL-based Quantum Architecture Search

Entry points (repo root), one per variant:

- `Tensor_training_and_structureRL_noiseless.py` — **TensorRL (trainable TN-init)** and
  **StructureRL** (same script/env; `zero_param_init` config flag picks which one)
- `TensorRL_fixed_noiseless.py` — **TensorRL (fixed TN-init)**, noiseless
- `TensorRL_fixed_noise.py` / `Tensor_training_and_structureRL_noise.py` — noisy-simulation
  counterparts (depolarizing/shot noise added)
- `*_restricted.py` variants — same, but action space restricted to a fixed hardware
  connectivity topology (`agents/utils_topology_restrict.py`)

Run as:
```
python3 TensorRL_fixed_noiseless.py --seed <seed> --config <cfg_name> --experiment_name "TensorRL_trainable/"
```
`<cfg_name>.cfg` is read from `configuration_files/<experiment_name>/` (see `03_modules.md`
for the config schema) via `environments/utils/utils.get_config`.

### One-time setup (before the episode loop)

```
config (.cfg)  ──►  CircuitEnv(conf, device)                  agents.__dict__[agent_type][agent_class](...)
                        │  loads mol_data/*.npz (H, eigvals)         │  builds policy_net / target_net (MLP)
                        │  loads init_state_circ/*.qpy (TN circuit)  │  builds N-step replay memory
                        │  computes min_eig (target), initial energy │  optimizer = Adam
                        ▼                                             ▼
                 environment instance                          agent instance
```

- **Fixed TN-init env** (`environment_qulacs_TN_notin_agent.py`): the TN circuit is turned
  into a `Statevector` once (`self.TN_state`) and used only as the *reference state* that
  energies/rewards are measured against. The RL binary-encoded state starts **empty**.
- **Trainable TN-init / StructureRL env** (`environment_qulacs.py`): the TN circuit's gates
  are decoded depth-by-depth from its DAG and **written directly into the initial RL state
  tensor** (`reset()`), with real angles for TensorRL-trainable or angles forced to `0` for
  StructureRL (`zero_param_init` flag). The agent can then modify those angles too.

### The episode loop (`one_episode`, called by `train()` for `episodes` iterations)

```
┌──────────────────────────── one episode ────────────────────────────────────┐
│                                                                              │
│  state = env.reset()          # binary-encoded circuit tensor (+ energy)    │
│                                                                              │
│  for t in 0 .. num_layers:                                                  │
│                                                                              │
│    illegal = env.illegal_action_new()      # mask redundant/no-op gates     │
│                                                                              │
│    action = agent.act(state, illegal)      # ε-greedy over policy_net       │
│         │  random with prob ε, else argmax_a Q(state,a)                    │
│         ▼                                                                   │
│    next_state, reward, done = env.step(action)                             │
│         │                                                                   │
│         ├─ decode action → (CNOT ctrl/targ) or (rotation qubit/axis)        │
│         ├─ write gate into binary-encoded state tensor                     │
│         ├─ build qulacs circuit from full state (VQAs.*.Parametric_Circuit)│
│         ├─ classical optimizer (COBYLA via scipy) re-optimizes ALL         │
│         │     rotation angles to minimize ⟨H⟩   (this is the VQE inner     │
│         │     loop, run fresh at every RL step)                            │
│         ├─ energy = ⟨ψ(θ*)|H|ψ(θ*)⟩ (qulacs; + noise/shots if enabled)     │
│         ├─ error = |energy − true_min_eig|                                 │
│         ├─ reward: +5 if error < threshold; −5 if out of steps;            │
│         │          else clipped relative energy improvement                │
│         └─ done = (error < threshold) OR (max steps reached)               │
│                                                                              │
│    agent.remember(state, action, reward, next_state, done)                 │
│         → pushed into N-step replay buffer (collapses 5 transitions        │
│           into one n-step return before storing)                          │
│                                                                              │
│    state = next_state                                                      │
│                                                                              │
│    if len(replay_buffer) > batch_size:                                     │
│        loss = agent.replay(batch_size)   # Double-DQN Bellman update       │
│           │  sample batch → policy_net(s,a) vs                             │
│           │  target_net-bootstrapped (r + γⁿ·Q_target(s', argmaxQ_policy)) │
│           │  SmoothL1 loss → Adam step on policy_net                       │
│           │  target_net synced from policy_net every `update_target_net`   │
│           │  steps; ε decays geometrically toward ε_min                    │
│                                                                              │
│    if done: break                                                          │
│                                                                              │
│  curriculum.update_threshold(...)   # optionally tighten accept_err        │
│  saver: log actions/errors/reward/nfev/time to summary_<seed>.npy          │
│  every 5 episodes: checkpoint policy_net, optimizer, replay buffer to disk │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Full ASCII process diagram (both phases)

```
                      ┌─────────────────────────────────────────┐
                      │        mol_data/*.npz  (once)            │
                      │  electronic-structure Hamiltonian:        │
                      │  Pauli strings + weights, exact eigvals   │
                      └───────────────────┬───────────────────────┘
                                          │
                                          ▼
   ╔══════════════════════════ PHASE A — dmrg-to-qc/dmrg_to_qc.py ═══════════════════════════╗
   ║                                                                                          ║
   ║   Pauli Hamiltonian ──mps2qc.mpo_from_paulis──▶  MPO                                     ║
   ║                                                     │                                    ║
   ║                                              mps2qc.gs_dmrg (quimb DMRG, bond dim χ)      ║
   ║                                                     │                                    ║
   ║                                                     ▼                                    ║
   ║                                            MPS ground state |ψ⟩                          ║
   ║                                                     │                                    ║
   ║                              mps2qc.mps_to_qc  (tnqc_ansatze brickwork structure          ║
   ║                              + stiefel_opt.StiefelAdam Riemannian optimization,           ║
   ║                              maximize |⟨ψ|U|0⟩| )                                        ║
   ║                                                     │                                    ║
   ║                                                     ▼                                    ║
   ║                                   Fitted TN circuit (2-qubit unitaries)                   ║
   ║                                                     │                                    ║
   ║                             tnqc_ansatze.qiskit_circ_from_tn_params                        ║
   ║                             (build Qiskit circuit, transpile to {RX,RY,RZ,CX})            ║
   ║                                                     │                                    ║
   ║                                                     ▼                                    ║
   ║                dmrg-to-qc/init_state_circ/init_<mol>_TNbond<χ>.{qasm,qpy}                  ║
   ║                                                                                          ║
   ╚═══════════════════════════════════════════╤════════════════════════════════════════════╝
                                               │
                                               ▼
   ╔══════════════════════ PHASE B — Tensor*/TensorRL_*.py  (RL-QAS training) ═══════════════╗
   ║                                                                                          ║
   ║   configuration_files/<Method>/<cfg>.cfg ──get_config──▶ conf (dict)                     ║
   ║                                                     │                                    ║
   ║              ┌──────────────────────────────────────┼────────────────────────────┐       ║
   ║              ▼                                      ▼                            │       ║
   ║   CircuitEnv(conf)  ◀── loads mol_data H, eigvals         agents.DQN_Nstep(conf)  │       ║
   ║   ◀── loads init_state_circ/*.qpy TN circuit         (policy_net / target_net MLP,│       ║
   ║   (as fixed statevector, or decoded into initial      N-step replay memory, Adam) │       ║
   ║    RL-state tensor, depending on variant)                                          │       ║
   ║              │                                                                    │       ║
   ║              └──────────────────────┬─────────────────────────────────────────────┘       ║
   ║                                     ▼                                                     ║
   ║                     ┌───────────────────────────────────────────┐                         ║
   ║                     │   episode loop (repeat `episodes` times)   │                        ║
   ║                     │                                             │                        ║
   ║                     │  state = env.reset()                       │                        ║
   ║                     │      │                                     │                        ║
   ║                     │      ▼                                     │                        ║
   ║                     │  ┌─────────── per-step loop ────────────┐ │                        ║
   ║                     │  │ illegal = env.illegal_action_new()   │ │                        ║
   ║                     │  │ action  = agent.act(state, illegal)  │ │  ε-greedy DQN           ║
   ║                     │  │ next_state, reward, done =            │ │                        ║
   ║                     │  │      env.step(action)                 │ │                        ║
   ║                     │  │   ├─ append gate to circuit tensor    │ │                        ║
   ║                     │  │   ├─ build qulacs circuit (VQAs.*)    │ │                        ║
   ║                     │  │   ├─ COBYLA re-optimizes angles       │ │  classical VQE          ║
   ║                     │  │   │     to minimize ⟨H⟩ (scipy)       │ │  inner loop             ║
   ║                     │  │   ├─ energy, error vs true ground     │ │                        ║
   ║                     │  │   │     state computed (qulacs)       │ │                        ║
   ║                     │  │   └─ reward = f(error, threshold)     │ │                        ║
   ║                     │  │ agent.remember(...) → replay buffer   │ │                        ║
   ║                     │  │ if buffer big enough:                 │ │                        ║
   ║                     │  │     agent.replay() → Double-DQN       │ │  gradient step on       ║
   ║                     │  │     Bellman update, ε decays          │ │  policy_net             ║
   ║                     │  │ state = next_state; break if done     │ │                        ║
   ║                     │  └───────────────────────────────────────┘ │                        ║
   ║                     │      │                                     │                        ║
   ║                     │      ▼                                     │                        ║
   ║                     │  curriculum.update_threshold()              │                        ║
   ║                     │  saver logs stats → summary_<seed>.npy      │                        ║
   ║                     │  periodic checkpoint: policy_net, optim,    │                        ║
   ║                     │  replay buffer  → results/.../*.pth         │                        ║
   ║                     └───────────────────────────────────────────┘                         ║
   ║                                                                                          ║
   ╚══════════════════════════════════════════════════════════════════════════════════════════╝
                                               │
                                               ▼
                results/<experiment_name>/<config>/summary_<seed>.npy
                (per-episode: actions taken, errors, rewards, nfev, opt angles, timing)
                + thresh_<acc>_<seed>_model.pth / _optim.pth / _replay_buffer.pth  (checkpoints)
```

## Key feedback loop worth calling out

Every single RL **step**, not just every episode, triggers a full classical VQE
re-optimization (COBYLA over all rotation angles placed so far) before the energy/reward is
computed. That inner optimization is the dominant cost per step, which is exactly what
`TensorRL (fixed TN-init)` attacks: by keeping the warm-start circuit *out* of the trainable
parameter set and out of the RL-state tensor, there are fewer trainable angles and a smaller
neural-network input, so both the VQE inner loop and the DQN forward/replay passes get cheaper
— giving the paper's reported 98% per-episode time reduction.
