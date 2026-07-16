# TensorRL-QAS — What This Codebase Does

Source: NeurIPS 2025 paper *"TensorRL-QAS: Reinforcement learning with tensor networks for
improved quantum architecture search"* (Kundu & Mangini). See `README.md` and
`tensor_rl_paper.md` in the repo root for the original sources this summary is derived from.

## Problem being solved

Variational Quantum Algorithms (VQAs) need a parameterized quantum circuit (PQC), `U(θ)`,
that prepares the ground state of a Hamiltonian `H` (e.g. a molecule's electronic Hamiltonian).
Hand-designing that circuit is hard: too shallow → not expressive enough; too deep → destroyed
by hardware noise. **Quantum Architecture Search (QAS)** automates discovering the circuit
structure itself, and prior RL-based QAS approaches scale badly — the action space and
per-episode simulation cost blow up with qubit count.

## The core idea

Don't make the RL agent start from an empty circuit and discover everything, including the
"easy" bulk structure that classical methods already solve well. Instead:

1. **Warm-start** the search with a *classically* obtained approximate ground state
   (via tensor networks / DMRG), converted into a small PQC.
2. Only then let an **RL agent** spend its exploration budget on top of that warm-start,
   adding the remaining gates needed to close the gap to true chemical accuracy.

This is the "Tensor" + "RL" in TensorRL-QAS. It cuts CNOT count/depth by up to 10x, cuts
training time per episode by up to 98%, and scales RL-QAS from ~8-qubit to 20-qubit problems,
compared to RL-QAS that starts from scratch.

## The three-stage procedure (per the paper, Fig. 1 / Sec. 3)

1. **DMRG** (Density Matrix Renormalization Group) finds a Matrix Product State (MPS)
   `|ψ⟩` approximating the ground state of `H`, controlled by a **bond dimension `χ`**
   (expressivity/cost knob).
2. **MPS → PQC mapping** via Riemannian optimization on the Stiefel manifold (Cayley-transform
   Adam): fits a brickwork circuit of 2-qubit unitaries `U = ∏ Uₖ` to maximize overlap
   `|⟨ψ|U|0⟩|` with the DMRG state.
3. **RL-based QAS** takes over: a Double-DQN agent sequentially appends gates from
   `{RX, RY, RZ, CNOT}` on top of the warm-start circuit, using VQE-style classical
   parameter optimization (COBYLA) and an energy-based reward, until it reaches chemical
   accuracy or hits the step budget.

## Three training variants compared in the paper

| Variant | Is the warm-start circuit part of the RL state? | Are its parameters trainable by the agent? |
|---|---|---|
| **TensorRL (trainable TN-init)** | Yes — encoded into the RL state tensor | Yes |
| **TensorRL (fixed TN-init)** | No — only used as a fixed initial statevector | N/A (frozen) |
| **StructureRL** (baseline/ablation) | Yes, same structure as trainable | Yes, but parameters start at **zero** (isolates "does the *structure* help" from "does the *warm-start value* help") |

`TensorRL (fixed TN-init)` is the cheapest: the agent's binary-encoded state and its neural
network input never include the warm-start depth, so simulation and replay are much smaller.

## Where each stage lives in this repo

See `03_modules.md` for a full module-by-module breakdown, and `02_training_pipeline.md` for
the runtime dataflow and an ASCII diagram of one full training run.

- Stage 1+2 (DMRG → PQC): `dmrg-to-qc/` (entry point `dmrg_to_qc.py`)
- Stage 3 (RL-QAS training): repo-root `Tensor*.py` / `TensorRL_*.py` scripts, driving
  `agents/` (DQN) and `environments/` (Gym-style `CircuitEnv` + qulacs-based VQE evaluation)
- Configuration for each molecule/method combination: `configuration_files/`
