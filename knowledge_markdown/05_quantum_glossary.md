# Quantum Computing Glossary — Terms Used in the Reproduction Guide

This explains every quantum-specific term that shows up in
**[04_reproducing_paper_results.md](04_reproducing_paper_results.md)** (and, by extension, in
`01`–`03`). Written for a 3rd-year CS student: you already know linear algebra, algorithms,
and how to read config files — you just haven't done quantum computing before. Each entry
gives a plain-language definition, a CS-flavored analogy where useful, and a pointer to where
it shows up in the reproduction guide.

Read top to bottom — later sections build on earlier ones.

---

## 1. Absolute basics

### Qubit
The quantum equivalent of a classical bit. A classical bit is `0` or `1`. A qubit is a
2-dimensional complex vector `α|0⟩ + β|1⟩` with `|α|² + |β|² = 1` — it can be in a
"superposition" of both basis states at once. `n` qubits together are described by a vector
of `2ⁿ` complex numbers (not `n` separate vectors) — this exponential blow-up is *why*
simulating quantum circuits on a classical computer gets expensive fast, and it's the whole
reason tensor-network approximations (§4) exist.

Everywhere `num_qubits` appears in a `.cfg` file, it's just this: how many qubits the circuit
acts on. A "6-BeH₂" or "8-H₂O" problem means the molecule's electronic structure has been
mapped onto a 6- or 8-qubit register (see §2).

### Quantum gate
The quantum analogue of a logic gate (AND/OR/NOT), but reversible and represented by a unitary
matrix (a matrix `U` with `U†U = I`, i.e. it preserves vector length — no information is
destroyed). Applying a gate = multiplying the statevector by that matrix.

The gate set used throughout this codebase is `{RX, RY, RZ, CNOT}`:
- **RX(θ), RY(θ), RZ(θ)** — single-qubit "rotation" gates, each with one continuous
  parameter θ (an angle). These are the *trainable parameters* of the circuit — analogous to
  weights in a neural network. In the RL state tensor (see `02_training_pipeline.md`), the
  rows labeled by rotation axis and the `thetas` array are exactly these angles.
- **CNOT** (Controlled-NOT) — a *2-qubit* gate: flips a "target" qubit if a "control" qubit is
  `1`. It's the standard way to create entanglement (correlations between qubits that can't be
  explained by treating them independently) — without at least some 2-qubit gates, a circuit
  can never represent most interesting quantum states. Every "CNOT count" reported in the
  paper's tables is a hardware-cost proxy: 2-qubit gates are much noisier and slower on real
  hardware than 1-qubit gates, so fewer CNOTs = a cheaper, more realistic circuit.

### Quantum circuit
A sequence of gates applied to an initial state (almost always `|0...0⟩`, all qubits off).
Drawn left-to-right like a musical score, with one horizontal line per qubit. Two structural
measurements used throughout the paper:
- **Depth** — the number of sequential "layers" of gates (gates that don't share a qubit can
  run in the same layer/in parallel). Lower depth = faster to run, less time for noise to
  accumulate.
- **Gate count** (often split into CNOT count and rotation/"ROT" count) — total number of
  gates, a proxy for both compilation cost and noise exposure.

### Statevector
The full `2ⁿ`-length complex vector describing an `n`-qubit system's exact state. Qiskit's
`Statevector(circuit)` (used throughout `dmrg_to_qc.py` and the environments) computes this by
brute-force matrix multiplication — only feasible because these benchmark problems stay under
~20 qubits (`2²⁰ ≈ 1` million entries, still tractable). "Statevector simulation" = simulating
a quantum circuit exactly on a classical computer this way, as opposed to running it on real
quantum hardware.

### Parameterized Quantum Circuit (PQC)
A quantum circuit whose gates include tunable parameters (the RX/RY/RZ angles above) — same
role as a parameterized function in classical ML. "Finding the PQC that prepares the ground
state" (§2) means: find gate placements *and* angle values such that running the circuit on
`|0...0⟩` produces (approximately) the target quantum state. Every acronym "PQC" in the docs
means exactly this circuit.

### Ansatz
Borrowed from physics/German ("initial guess/form"): a chosen *structure* for a PQC (which
gates, on which qubits, in what pattern) before its parameters are optimized. "Brickwork
ansatz" (§4) and "hardware-efficient ansatz" both refer to specific structural templates.

---

## 2. The chemistry problem being solved

### Hamiltonian
An operator (represented as a matrix, or a weighted sum of simpler matrices) whose
eigenvalues are the possible energies of a physical system, and whose lowest eigenvalue is the
**ground-state energy** — the energy of the system at rest, which is what determines a
molecule's stable structure, reaction energies, etc. In the code, a Hamiltonian for a molecule
is loaded from `mol_data/*.npz` as the `hamiltonian` array (a dense matrix, used for exact
energy checks) plus `paulis`/`weights` (a sparse, human-readable decomposition — see below).
"Find the ground state of `H`" is the entire objective of every training run in this repo.

### Ground state / eigenvalues (`eigvals`)
The ground state is the eigenvector of `H` with the smallest eigenvalue. `eigvals` (stored in
every `mol_data/*.npz`) are the exact eigenvalues of `H`, computed once offline (classically,
by exact diagonalization) so the code always knows the *true* answer to compare against —
that's `min_eig`/`true_min_eig`/`self.min_eig` throughout `CircuitEnv`. The "Error" column in
every paper table is just `|energy_from_circuit − true_ground_energy|`.

### Chemical accuracy
A domain-specific threshold from quantum chemistry: an energy error below **1.6 × 10⁻³
Hartree** (Hartree = the atomic unit of energy) is considered accurate enough to make reliable
predictions about real chemical behavior (reaction rates, bond energies, etc.). This is
*exactly* the `accept_err = 1.6e-3` value you'll see in every H₂O/BeH₂/LiH/CH₂O `.cfg` file —
it's the success criterion the RL agent is trained to reach, not an arbitrary tolerance.
(The Heisenberg/TFIM spin-model configs use different thresholds, `1e-3`/`1e-2`, because
they're not molecules and "chemical accuracy" doesn't apply — see §7.)

### Basis set (e.g. `STO-3G`, `6-31G`)
In quantum chemistry, you can't represent an electron's wavefunction exactly on a computer —
you approximate it as a combination of a finite set of simpler reference functions ("atomic
orbitals"). A **basis set** is the specific choice of that finite reference set. Bigger basis
sets (`6-31G` has more functions per atom than `STO-3G`) → more accurate but more qubits
needed. This is *why* the same water molecule (H₂O) appears as both an "8-qubit" and a
"10-qubit" problem in the paper (Table 12): 8-qubit H₂O uses `STO-3G`, 10-qubit H₂O uses the
larger `6-31G` basis.

### Active electrons / active orbitals
Simulating *every* electron in a molecule is wasteful — most are in low-energy "core"
orbitals that barely participate in chemistry. The **active space** approximation only
explicitly simulates a chosen subset of electrons/orbitals near the frontier (where chemistry
actually happens), freezing the rest. This is a knob in `making_molecules.py`
(`active_electrons`, `active_orbitals`) that directly controls how many qubits the resulting
Hamiltonian needs.

### Fermion-to-qubit mapping (`jordan_wigner`, `parity`)
Electrons are fermions, governed by different mathematical rules (anticommutation) than
qubits. Before you can write a molecular Hamiltonian as gates on qubits, you need a **mapping**
that translates fermionic operators into qubit (Pauli) operators while preserving the physics.
**Jordan-Wigner** and **parity** are two standard, different such translation schemes — you'll
see `mapping = jordan_wigner` in almost every `.cfg`'s `[problem]` section. You don't need to
know the math to reproduce results, just that changing this string would change the qubit
Hamiltonian entirely (mismatching it with a pre-generated `mol_data` file would silently give
wrong physics), so it must match between the Hamiltonian file and the config.

### Qubit tapering (`taper = 1`)
A follow-on trick to fermion-to-qubit mapping: molecular Hamiltonians often have symmetries
(e.g. conservation of particle number/spin) that let you provably remove ("taper off") a few
qubits without losing information, shrinking the problem size. `taper = 1` in the configs just
means this reduction was applied when the Hamiltonian was generated.

### Pauli string / Pauli operators
The **Pauli matrices** `X, Y, Z` (plus identity `I`) are the four basic single-qubit
"building block" operators; any operator on `n` qubits (including a Hamiltonian) can be written
as a weighted sum of **Pauli strings** — tensor products like `X⊗I⊗Z⊗Y` (one Pauli letter per
qubit). This is the sparse, human-readable representation of a Hamiltonian, stored as the
`paulis` (list of strings like `"XIZY"`) and `weights` (their coefficients) arrays in
`mol_data/*.npz`, and is what `mps2qc.mpo_from_paulis` consumes in Phase 1c.

---

## 3. Circuit compilation & storage formats

### Transpile
Converting a circuit written with one gate set/structure into an equivalent circuit using only
a *target* gate set (and respecting other constraints like hardware connectivity), typically
while trying to minimize depth/gate count. Directly analogous to a compiler lowering
high-level code to a specific instruction set. `dmrg_to_qc.py` transpiles the fitted
tensor-network circuit down to `{cx, rx, ry, rz}` (or the `su4` set) because that's the gate
set the RL agent is allowed to build with.

### QASM / QPY
Two file formats Qiskit uses to save a quantum circuit to disk. **QASM** ("Quantum Assembly")
is a human-readable text format (like assembly language for circuits) — you can open a `.qasm`
file and read the gate list. **QPY** is Qiskit's binary/pickle-like serialization — faster to
load, preserves more circuit metadata, but not human-readable. `dmrg_to_qc.py` saves both; the
RL environments (`CircuitEnv.__init__`) load the `.qpy` version of the TN warm-start circuit.

### `su4` basis gates (`rxx`, `ryy`, `rzz`)
An alternative, more expressive 2-qubit gate set (parameterized `XX`/`YY`/`ZZ` rotations,
named after `SU(4)`, the group of 4×4 special-unitary matrices these gates can reach) used in
one ablation (Table 7's expanded action-space experiment, §H.6 of the paper). More expressive
than plain CNOT, at the cost of being less standard on real hardware.

---

## 4. Tensor networks & the DMRG warm-start (Phase 1)

### Tensor network
A way of representing a huge multi-dimensional array (like a `2ⁿ`-entry statevector) as a
network of many small, connected arrays ("tensors") instead of one giant one — conceptually
like factoring a huge matrix into a product of small matrices to save memory, generalized to
higher dimensions. The entire point is to represent (an approximation of) a quantum state
*without* the `2ⁿ` blow-up from §1, as long as the state isn't "too entangled."

### Matrix Product State (MPS)
The simplest, most common tensor network: an `n`-qubit state written as a chain of `n` small
tensors linked in a line (see the equation in `01_overview.md`/the paper §3.1). It's the
tensor-network analogue of writing a long vector as a product of small matrices multiplied in
sequence. This is the data structure DMRG (below) produces, and what gets fitted to a quantum
circuit in Phase 1c.

### Bond dimension (χ)
The one hyperparameter that controls an MPS's size/expressivity: it's the size of the
"connection" between adjacent tensors in the chain. `χ=1` means no correlation between
different parts of the chain at all (a product state); larger χ can represent more
entanglement, at higher classical memory/compute cost (`O(χ²)`–`O(χ³)` scaling depending on the
operation). This is exactly the `TN_bond` field in every `.cfg` and the "Choose bond dimension"
prompt in `dmrg_to_qc.py` — it's the main knob trading warm-start quality for classical cost,
and Table 10 in the paper is literally "what happens as you turn this knob up."

### Matrix Product Operator (MPO)
Same idea as an MPS, but representing an *operator* (matrix) instead of a *state* (vector) —
used to represent the Hamiltonian itself in tensor-network form so DMRG can act on it
efficiently. `mps2qc.mpo_from_paulis` builds this from the Pauli-string Hamiltonian before
DMRG runs.

### DMRG (Density Matrix Renormalization Group)
A classical (non-quantum-computer) numerical algorithm — the workhorse of computational
condensed-matter/quantum-chemistry physics for decades, long before quantum computers existed
— that finds a good MPS approximation to the ground state of a Hamiltonian given as an MPO. It
works by repeatedly sweeping along the MPS chain, locally optimizing one or two tensors at a
time while holding the rest fixed, until the energy stops improving. You never need to
implement this yourself — `mps2qc.gs_dmrg` calls into the `quimb` library, which does it. All
you control is `bond_dims` (the χ above) and `num_sweeps`.

### Overlap / fidelity (`|⟨ψ|U|0⟩|`)
A number between 0 and 1 measuring how close two quantum states are (1 = identical, up to an
irrelevant global phase). "Fitting a circuit to an MPS" (Phase 1c) means numerically
maximizing the overlap between the circuit's output state `U|0⟩` and the target MPS `|ψ⟩` —
directly analogous to minimizing a loss/maximizing a similarity score in ML, just with a
quantum-specific notion of "distance between vectors."

### Brickwork (circuit structure)
A specific, simple ansatz layout: 2-qubit gates arranged in a repeating pattern that looks like
a brick wall when drawn — alternating layers where gates act on qubits (0,1),(2,3),... then
(1,2),(3,4),..., etc. Chosen because it only needs nearest-neighbor qubit connections
(matching real linear-connectivity hardware) and is easy to generate at any depth. This is the
`structure: 'brickwork'` / `num_layers` setting passed to `tnqc.qiskit_circ_from_tn_params` in
`dmrg_to_qc.py`, and "TN layers" in Table 8 means the depth of *this specific* structure.

### Stiefel manifold / Riemannian optimization / Cayley transform
This is the trickiest term here, but you can treat it as a black box: ordinary gradient
descent, if applied naively to a matrix, will generally break the constraint that the matrix
stays **unitary** (`U†U=I`) after the update — and a non-unitary "gate" isn't physically
realizable. The **Stiefel manifold** is just the mathematical name for "the set of all unitary
matrices of a given size" (a curved space, not a flat vector space). **Riemannian
optimization** is gradient descent adapted to *stay on* such a curved space at every step,
and the **Cayley transform** is the specific formula used here to do that "stay on the curved
space" correction. Practically: this is *how* Phase 1c's circuit-fitting optimizer
(`stiefel_opt.StiefelAdam`) guarantees every 2-qubit gate it produces remains a valid quantum
gate throughout training, without you needing to understand the differential geometry behind
it.

---

## 5. Putting it together: VQE and QAS

### VQE (Variational Quantum Eigensolver)
The classic hybrid quantum-classical algorithm this whole pipeline is built around: given a
fixed PQC structure, repeatedly (1) run the circuit and measure the energy `⟨ψ(θ)|H|ψ(θ)⟩`, and
(2) use a classical optimizer to update `θ` to lower that energy — alternating quantum
measurement and classical optimization, like an ML training loop where the "forward pass" is a
quantum circuit evaluation. Every call to `scipy.optimize.minimize` inside `CircuitEnv.step`
(re-optimizing rotation angles with COBYLA after each new gate is added) *is* a VQE inner loop.

### QAS (Quantum Architecture Search)
The circuit-structure equivalent of Neural Architecture Search: instead of fixing the PQC's
gate layout and only optimizing angles (plain VQE), *also* search over which gates to place
and where. This repo's entire RL training loop (Phase 2) is one particular way of doing QAS:
an RL agent chooses gate placements step by step, and a VQE-style inner loop (COBYLA)
optimizes the angles of whatever's been placed so far.

### Warm-start / TN-init
"Don't start the architecture search from an empty circuit — start from a circuit that's
already a decent approximate answer (from DMRG + Phase 1c), and only search for the remaining
improvements." Directly analogous to weight initialization from a pretrained model in ML,
instead of training from random weights. This is the central trick the whole paper (and repo)
is about — see `01_overview.md` for the three ways this warm-start is used (fixed vs.
trainable vs. zeroed/StructureRL).

---

## 6. Noise & real-hardware realism

### Noise (in a quantum circuit)
Real quantum hardware isn't perfect — gates and measurements introduce small, random errors.
"Noiseless simulation" (most configs in this repo) pretends hardware is perfect, useful for
isolating whether the *algorithm* works before worrying about hardware imperfections. "Noisy
simulation" (the `*_noise.cfg` configs) artificially injects modeled errors to test robustness.

### Depolarizing noise
A simple, standard mathematical model of a gate error: with some probability `p`, replace the
qubit's state with a fully random ("maximally mixed") state instead of applying the intended
gate correctly; otherwise the gate works fine. `p` is set separately for 1-qubit gates and
2-qubit gates (2-qubit gates are typically much noisier on real hardware) — that's the
`noise_values = (p1, p2)` pair referenced in Phase 2's noise section of the reproduction guide.

### Shot noise
Even on perfect hardware, you can't read out an exact expectation value `⟨H⟩` from a single
run of a quantum circuit — quantum measurement is inherently probabilistic. You have to run
("shoot") the circuit many times and average the outcomes; with a finite number of shots
(`n_shots` in the configs), your estimate of the energy has statistical noise that shrinks like
`1/√(n_shots)`. This is unrelated to hardware imperfection — it exists even in an idealized
noise-free device, purely from the physics of quantum measurement.

### Qubit connectivity / topology (restricted variants)
Real quantum chips don't let every qubit interact directly with every other qubit — a 2-qubit
gate (CNOT) can typically only be applied between qubits that are physically adjacent on the
chip, forming a fixed connectivity graph ("topology"). The `*_restricted.py`/`*_restricted.cfg`
variants constrain the RL agent's allowed CNOT actions to only that graph's edges, so the
resulting circuit could actually run on a specific real device without extra "routing" (adding
SWAP gates to move information between non-adjacent qubits).

### IBMQ-Brisbane
The specific real IBM Quantum hardware device the paper ran validation circuits on (Appendix
G) to check that circuits discovered in noiseless simulation still work under real hardware
noise/connectivity. This requires IBM Quantum cloud access and is *not* something this local
codebase can do by itself (noted as a gap in Phase 4 of the reproduction guide).

---

## 7. The specific physics models used as benchmarks

### Molecular ground-state problems (BeH₂, H₂O, LiH, CH₂O)
The main benchmark family: find the ground-state energy of small molecules' electronic
Hamiltonians (built via `making_molecules.py`, using PennyLane/OpenFermion-style
quantum-chemistry tooling). "6-BeH₂" etc. names combine the qubit count with the molecule.

### Heisenberg model
A standard, non-chemistry "spin model" from condensed-matter physics: a 1D chain of qubits
("spins") where each neighboring pair interacts equally via all three Pauli axes
(`XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁`), plus a uniform field term. Used in the paper (Table 6) to show
the method generalizes beyond quantum chemistry. Built by
`dmrg-to-qc/heisenberg_model.py:construct_hamiltonian`.

### Transverse Field Ising Model (TFIM)
Another standard spin-model benchmark: neighboring qubits interact only along `Z`
(`ZᵢZᵢ₊₁`), plus a "transverse" field along `X` with strength `h`
(`H = Σ ZᵢZᵢ₊₁ + h Σ Xᵢ`). Used in Tables 6–7 to test scaling up to 20 qubits — different `h`
values (`0.05` vs `0.001`) make the model harder or easier, which is why the guide flags that
the shipped `tfim_j1_h0.001_6q.npz` doesn't match the paper's `h=0.05` benchmark.

---

## Quick reference: acronym decoder

| Acronym | Expansion | Section above |
|---|---|---|
| PQC | Parameterized Quantum Circuit | §1 |
| DMRG | Density Matrix Renormalization Group | §4 |
| MPS | Matrix Product State | §4 |
| MPO | Matrix Product Operator | §4 |
| VQE | Variational Quantum Eigensolver | §5 |
| QAS | Quantum Architecture Search | §5 |
| RL-QAS | Reinforcement-Learning-based QAS | §5 (see `03_modules.md` for the RL/DQN side) |
| TFIM | Transverse Field Ising Model | §7 |
| χ (chi) | Bond dimension | §4 |
| CNOT | Controlled-NOT gate | §1 |
| QASM / QPY | Qiskit circuit file formats | §3 |
