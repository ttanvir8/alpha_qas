# TensorZero-QAS: Planning, Gadgets, and Learned Priors for Scalable Quantum Architecture Search

**Target venue:** NeurIPS / ICML / ICLR (A* ML conference)
**Working title alternatives:** *AlphaQAS*, *Gadget-Aware TensorRL-QAS*, *TensorZero-QAS: AlphaZero-style Quantum Architecture Search with Tensor-Network Warm-Starts*

**Builds on:** TensorRL-QAS (NeurIPS 2025, this repo) + AlphaTensor-Quantum (Nature Machine Intelligence 2025, DeepMind)

---

## 1. One-paragraph pitch

TensorRL-QAS showed that warm-starting RL-based quantum architecture search (QAS) with a DMRG/MPS
circuit slashes CNOT counts by ~10x and per-episode cost by 98%, but it still relies on a
**model-free ε-greedy DDQN with an MLP** — the weakest possible search/learning stack — and it
builds circuits **one primitive gate at a time** with a **purely energy-based reward**. AlphaTensor-Quantum
demonstrated, on the fault-tolerant side of quantum computing, that the combination of
(i) **MCTS planning (AlphaZero-style)**, (ii) **gadget-aware reward shaping** that lets the agent
compose multi-action motifs, (iii) **synthetic demonstrations** mixing supervised and RL losses,
(iv) **symmetry-exploiting attention architectures**, and (v) **change-of-basis / permutation data
augmentation** can discover circuit constructions that match or beat decades of human design.
**TensorZero-QAS transplants all five ingredients into the NISQ/VQE-era QAS setting of TensorRL-QAS**,
keeping the tensor-network warm-start as the physically motivated initialization but replacing
the search engine, the reward structure, the network, and the training data pipeline. The claim:
this closes the remaining scalability gap of RL-QAS (12→30+ qubits), discovers *reusable circuit
motifs* (the VQE analog of AlphaTensor's rediscovery of Karatsuba), and produces circuits that are
simultaneously shallower and more accurate than TensorRL-QAS under noise.

---

## 2. Gap analysis: what TensorRL-QAS lacks that AlphaTensor-Quantum had

| Dimension | TensorRL-QAS (current repo) | AlphaTensor-Quantum | Opportunity |
|---|---|---|---|
| Search | ε-greedy DDQN, 5-step returns, no planning | Sample-based MCTS (800 sims/move) guided by policy+value net | Planning amortizes expensive env steps: each `env.step()` here costs a full COBYLA VQE re-optimization, so choosing actions *carefully* matters far more than in cheap-step domains |
| Action granularity | Single primitive gates {RX, RY, RZ, CNOT} | Single factors, **but gadget rewards let 3–7 actions collapse into one cheap motif** | Reward-level "gadgets" for VQE: Givens rotations, particle-number-conserving blocks, fSim/XX+YY blocks, brickwork cells — agent learns to *compose* motifs without enlarging the action space |
| Reward | Shaped energy improvement, +5/−5 terminal | Per-move −1 with retroactive gadget bonuses | Add motif-completion bonuses + explicit CNOT/depth cost so compactness is optimized *directly*, not only implicitly via step budget |
| Network | 5-layer MLP, 1000 neurons | Symmetrized axial attention over the tensor state | The RL state here IS a 3D tensor (depth × gate-type × qubit); axial attention with qubit-permutation equivariance scales to more qubits and generalizes across molecules |
| Training data | Only on-policy replay | 5M synthetic demonstrations, mixed supervised+RL loss | Generate random circuits → known energies/structures for free (classically simulable regime); pretrain policy/value nets before touching the expensive VQE environment |
| Augmentation | None | Random change of basis (Clifford), action permutation | Qubit relabeling consistent with Hamiltonian symmetry, gate-commutation-based action reordering, MPS gauge freedom |
| Transfer | One agent per molecule per config | One agent per *application family* | Train one agent across a family (e.g., all hydrogen chains H₄→H₁₀); measure zero-shot / few-shot transfer to unseen geometries |

---

## 3. Proposed method

### 3.1 Core loop (replaces DDQN in `agents/`)

AlphaZero-style actor-learner:

1. **State**: same binary circuit tensor as the current repo (`(num_layers, num_qubits+6, num_qubits)`),
   plus warm-start metadata (bond dimension χ, DMRG energy gap) as conditioning tokens.
2. **Network**: a *symmetrized axial attention* torso (AlphaTensor-Quantum, Methods) adapted to the
   QAS state tensor: attention along the depth axis and the qubit axis, with learnable
   symmetrization `X ← β∘X + (1−β)∘X^T` on the qubit×qubit CNOT sub-block (which is the natural
   symmetric structure here — CNOT adjacency). Two heads:
   - **policy head** over the `3N + 2·C(N,2)` gate actions (masked by `illegal_action_new()`);
   - **distributional value head** over the return, with **risk-seeking value** for acting
     (we want the best circuit ever found, not the average — same argument as AlphaTensor).
3. **Search**: sample-based MCTS (K sampled actions per node, e.g. K=16–32, ~100–400 sims/move —
   tuned down from AlphaTensor's 800 because our env step is expensive). Crucial trick to make
   this affordable: **the tree search uses a cheap surrogate evaluator** (see 3.4), and the full
   COBYLA VQE re-optimization only runs on the action actually committed.
4. **Learner**: replay buffer of played games + synthetic demonstrations; mixed loss
   `L = L_RL(policy, value) + λ·L_supervised(demonstrations)`.

### 3.2 Gadget-aware reward (the flagship idea)

Direct port of AlphaTensor-Quantum's gadgetization mechanism to the VQE gate world.
Every time an action is played, the environment checks (by pattern matching on the last k
actions, exactly like the Toffoli/CS linear-dependency checks) whether the recent action
sequence completes a **known hardware-efficient motif**:

- **Givens rotation block** (4 CNOTs + rotations; particle-number conserving — the workhorse of
  chemistry ansätze): retroactive bonus so the motif's joint cost equals its "compiled" cost.
- **fSim / XX+YY interaction block**: bonus reflecting native availability on
  superconducting hardware (1 native 2-qubit gate instead of 2–3 CNOTs).
- **Brickwork cell** matching the warm-start layer structure: bonus for extending the MPS
  circuit *coherently* rather than scattering gates.
- **Symmetry-preserving blocks** (total-spin / particle-number sector preserving): bonus because
  they provably cannot leave the correct symmetry sector of the ground state.

Reward per step becomes:
`r_t = α·(energy improvement, clipped) − c_CNOT·[action is CNOT] − c_1q·[action is rotation] + gadget_bonus(a_{t−k:t}) + terminal(±)`

This is the mechanism by which AlphaTensor-Quantum rediscovered measurement-based uncomputation
*without being told about it*; the analogous hoped-for headline here is the agent **rediscovering
(or beating) UCCSD-style Givens-rotation ansätze and ADAPT-VQE operator pools from scratch**,
purely from gadget-shaped rewards.

### 3.3 Synthetic demonstrations & pretraining

AlphaTensor-Quantum trained on 5M random tensor/factorization pairs. QAS analog, essentially free
to generate:

1. Sample random circuits from the action space (optionally biased to contain motifs from 3.2,
   mirroring AlphaTensor's gadget-injected demonstrations, probability ~0.9 of ≥1 motif).
2. For small N (≤ 12 qubits, classically simulable with qulacs — already in this repo), compute the
   state each circuit prepares; treat *(prepared state as target, circuit as solution)* as a
   demonstration: the network learns "which action sequences produce which physics" before ever
   seeing a real molecule.
3. Additionally, harvest demonstrations from **DMRG at increasing bond dimension**: the circuit
   fitted at χ=4 relative to χ=2 is a *ground-truth example* of "gates that improve a χ=2
   warm-start" — exactly the task the agent faces. This is a novel demonstration source with no
   AlphaTensor analog, unique to the TN-warm-start setting (uses `dmrg-to-qc/` as a data factory).

### 3.4 Cheap surrogate evaluation inside MCTS

The repo's dominant cost is COBYLA re-optimizing *all* angles at *every* step
(see `02_training_pipeline.md`, "Key feedback loop"). MCTS multiplies env queries by the
simulation count, so naively it's intractable. Fix, in increasing order of ambition:

- **Level 0**: inside the tree, evaluate leaf circuits with *frozen previously-optimized angles* +
  new gate at analytic optimum for a single angle (1D line search is closed-form for
  R-gates: energy is sinusoidal in each angle — "Rotosolve" trick); full COBYLA only on commit.
- **Level 1**: the value head itself *is* the surrogate (pure AlphaZero: no env call at leaves,
  bootstrapped value). Env only evaluates the committed action.
- **Level 2**: an MPS-based simulator (quimb, already a dependency of Phase A) evaluates tree
  nodes at truncated bond dimension — cheap, and *systematically improvable*, giving a
  fidelity/cost dial no prior QAS work has inside a planner. This turns "tensor networks for
  warm-starting" into "tensor networks *inside the search loop*" — a genuinely new contribution
  beyond both parent papers.

### 3.5 Symmetry augmentation

- **Qubit permutations** consistent with the Hamiltonian's symmetry group (analog of AlphaTensor's
  change of basis; for spin models this is the full lattice symmetry group, for molecules
  orbital-swap symmetries): augment (state, policy) pairs at train time, average policies over
  the group at act time.
- **Commutation-based action reordering** (analog of AlphaTensor's action-swap data augmentation):
  gates on disjoint qubits commute → every played episode yields many equivalent action sequences
  for free.

---

## 4. Experimental plan

### 4.1 Benchmarks (reuse repo infrastructure)

- **Molecules** (existing `mol_data/`): 6-BeH₂, 8-H₂O, 10-H₂O, 12-LiH, 8/10-CH₂O — direct
  comparability with the NeurIPS 2025 tables.
- **Scale-up**: hydrogen chains H₄–H₁₆ and TFIM/Heisenberg up to 30 qubits (TensorRL-QAS stopped
  at 20 for TFIM; the MPS surrogate of 3.4-Level-2 is what unlocks going past statevector limits).
- **Noise**: same amplified depolarizing + shot noise protocol as the parent paper (envs already in
  repo: `environment_qulacs_*_noise.py`), plus IBMQ validation of final circuits.

### 4.2 Baselines

TensorRL-QAS (fixed & trainable) — the direct ancestor and strongest baseline; CRLQAS; vanilla
RL-QAS; RA-QAS; SA-QAS; quantumDARTS; TN-VQE; ADAPT-VQE (important addition: the human-designed
"grow the ansatz greedily" method our gadget agent should beat on gate count at equal accuracy).

### 4.3 Ablations (one per transplanted ingredient — this is the reviewers' checklist)

1. MCTS on/off (planner vs. ε-greedy DDQN at *matched wall-clock*, not matched env steps).
2. Gadget rewards on/off; and per-gadget-family ablation.
3. Synthetic demonstrations on/off; demonstration-source ablation (random vs. DMRG-Δχ).
4. Symmetrized axial attention vs. plain MLP (the repo's current net) at 8, 12, 20, 30 qubits.
5. Surrogate level 0 / 1 / 2 inside the tree.
6. Warm-start on/off under the new stack (does planning reduce the *dependence* on the warm-start,
   or compound with it? Either answer is a finding).

### 4.4 Metrics

Energy error vs. chemical accuracy; CNOT count / depth / rotation count; success probability over
seeds; **wall-clock and env-query budget to solution** (the honest planning-cost accounting);
noisy-hardware energy; **motif statistics** — frequency of discovered gadget patterns, and
qualitative analysis of any *novel* recurring motif (the "Karatsuba moment" section).

### 4.5 Headline results targeted

1. ≥2x further CNOT/depth reduction over TensorRL-QAS (fixed) at equal accuracy on 8–12 qubit
   molecules.
2. First RL-QAS results at 30 qubits (via MPS surrogate), beyond any published RL-QAS.
3. One trained agent transferring zero-shot across bond lengths / molecule families.
4. Discovery + human-readable analysis of at least one reusable ansatz motif not present in the
   gadget library (emergent, like measurement-based uncomputation in AlphaTensor-Quantum).

---

## 5. Implementation roadmap mapped to this repo

| Phase | Work | Repo touchpoints |
|---|---|---|
| P0 (2 wk) | Gadget detector + reshaped reward in the existing DDQN stack (cheapest ingredient first; standalone ablation) | `environments/environment_qulacs_TN_notin_agent.py` (`step`, `reward_fn`), new `environments/utils/gadgets.py` |
| P1 (3 wk) | Rotosolve single-angle analytic evaluation to replace per-step full COBYLA inside candidate scoring | `environments/VQAs/VQE_qulacs_TN_notin_RL.py`, `CircuitEnv.scipy_optim` |
| P2 (4 wk) | Symmetrized axial-attention network replacing the MLP; permutation augmentation | `agents/DeepQ.py` (`unpack_network`), new `agents/attention_torso.py` |
| P3 (4 wk) | Sample-based MCTS actor (policy/value heads, risk-seeking value); actor-learner split | new `agents/mcts_agent.py`, root training scripts |
| P4 (3 wk) | Synthetic demonstration generator (random-circuit + DMRG-Δχ) and mixed-loss learner | `dmrg-to-qc/` as data factory, new `demos/` module |
| P5 (4 wk) | MPS surrogate inside the tree (quimb); 20–30 qubit scale-up runs | new `environments/VQAs/VQE_mps_surrogate.py` |
| P6 | Full benchmark + ablation grid, noisy runs, IBMQ validation, writing | `configuration_files/`, HPC |

P0–P2 are each individually publishable-increment ablations; the paper is safe even if P5 (the
riskiest piece) underdelivers.

## 6. Risks and mitigations

- **MCTS cost blow-up**: env steps here cost a VQE optimization, unlike AlphaTensor's free tensor
  subtraction. Mitigated by surrogate levels 0–2 (Sec. 3.4); worst case, fall back to shallow
  search (50 sims) which literature suggests already beats ε-greedy.
- **Gadget library bias**: hand-picked motifs could cap discovery. Mitigation: ablation 3 runs a
  no-gadget agent; motif-statistics analysis checks for emergent patterns *outside* the library.
- **Attention net data hunger**: mitigated by synthetic demonstrations (P4) — that is exactly why
  AlphaTensor needed them too.
- **Novelty positioning**: reviewers may say "AlphaZero applied to known problem." Defense: the
  MPS surrogate *inside* the planner (3.4-L2) and DMRG-Δχ demonstrations (3.3-#3) exist in neither
  parent paper; gadget rewards have never been formulated for the variational/NISQ gate set; and
  the noisy-QAS-at-30-qubits result would be a first.

## 7. Contribution statement (draft, for the paper's intro)

1. **TensorZero-QAS**, the first QAS framework combining tensor-network warm-starts with
   AlphaZero-style planning, unifying the two RL-for-quantum-circuits lineages (variational/NISQ
   and fault-tolerant).
2. **Gadget-shaped rewards for variational circuits**: a general mechanism for injecting
   hardware/physics motifs (Givens, fSim, symmetry-preserving blocks) into QAS reward signals
   without enlarging the action space.
3. **Tensor networks inside the search loop**: an MPS surrogate evaluator with a tunable
   fidelity/cost dial that scales RL-QAS to 30 qubits.
4. **DMRG-difference demonstrations**: a free, physics-grounded source of supervised pretraining
   data for circuit-building agents.
5. State-of-the-art compactness/accuracy/noise-robustness results on standard molecular
   benchmarks, with full ablations isolating each ingredient.
