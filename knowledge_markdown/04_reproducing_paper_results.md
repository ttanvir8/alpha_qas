# Reproducing the Paper's Results — Phase-by-Phase Guide

This is a practical run-book for using **this repo** to reproduce the tables/figures in
`tensor_rl_paper.md`. It cross-references what's actually shipped in the repo (configs,
Hamiltonians, TN circuits) against what the paper reports, and flags the gaps you'd hit.

---

## Phase 0 — Environment setup

```bash
conda create -n tensorrl python=3.10
conda activate tensorrl
pip install -r requirements.txt   # qulacs, torch, qiskit==2.0.0, quimb
```

The paper's DMRG step also needs `jax` (imported in `dmrg-to-qc/dmrg_to_qc.py`) and the
Hamiltonian-generation scripts need `pennylane` + `pennylane.qchem` (imported in
`dmrg-to-qc/making_molecules.py`, not listed in `requirements.txt`) — install these if you
need to regenerate Hamiltonians (Phase 1b below):
```bash
pip install jax pennylane
```
Per your global setup, run this inside your `env` alias (uv-based) python environment rather
than a fresh conda env if you prefer — just make sure the 4 packages above (+ jax/pennylane
if needed) are importable there.

---

## Phase 1 — Get the Hamiltonian + TN warm-start circuit for the target system

### 1a. Check what's already shipped (no work needed)

`dmrg-to-qc/mol_data/*.npz` (Hamiltonians) and `dmrg-to-qc/init_state_circ/*.qpy` (TN
warm-start circuits) already exist for:

| System | mol_data (Hamiltonian) | init_state_circ (TN circuit, bond dims available) |
|---|---|---|
| 6-BeH₂ | ✅ | ✅ χ=2 |
| 8-H₂O | ✅ | ✅ χ=2 |
| 10-H₂O | ✅ | ✅ χ=2, χ=3 |
| 8-CH₂O (`CH2_8q`) | ✅ | ✅ χ=2, χ=5 |
| 10-CH₂O (`CH2_10q`) | ✅ | ✅ χ=2, χ=3, χ=5, χ=6 (+ an `su4` variant) |
| 5-Heisenberg | ✅ | ✅ χ=2 |
| 4-LiH | ✅ (`LIH_4q`) | ❌ |

For these, you can skip straight to Phase 2 using the matching `.cfg` in
`configuration_files/{StructureRL,TensorRL_fixed,TensorRL_trainable}/`.

### 1b. What's referenced by configs but NOT shipped — you must generate it

Some `.cfg` files point at Hamiltonians/circuits that are **not present** in this checkout and
must be produced yourself before training will run:

| Config | Needs | Status |
|---|---|---|
| `*/LIH12q_TNbond2.cfg` (12-LiH, the paper's headline 12-qubit result) | `mol_data/LIH_12q_geom_..._jordan_wigner.npz` + `init_state_circ/init_LIH_12q_..._TNbond2.qpy` | ❌ missing — only `LIH_4q` Hamiltonian is shipped |
| `TensorRL_trainable/H2O12q_TNbond{2,3,5}.cfg` (12-qubit H₂O) | `mol_data/H2O_12q_...npz` + matching init circuits | ❌ missing — only 8q/10q H₂O shipped |
| Table 6 (6-TFIM, `h=0.05`) and Table 7 (15/17/20-qubit TFIM, `h=0.001`) | `mol_data/tfim_j1_h0.05_6q.npz` etc. | ❌ only `tfim_j1_h0.001_6q.npz` is shipped, and there is **no `.cfg`** for TFIM at all in `configuration_files/` |

To generate a missing Hamiltonian:
```bash
cd dmrg-to-qc
python3 making_molecules.py     # edit the `mol = "H2O"` line at the top to switch molecule/qubit count/geometry
# or, for spin models:
python3 heisenberg_model.py     # Heisenberg / TFIM construct_hamiltonian(n) — edit n and h as needed
```
This writes a new `mol_data/<name>.npz` (`hamiltonian`, `eigvals`, `weights`, `paulis`).
Geometries and basis sets for every molecule in the paper's tables are listed in
Appendix M.3/Table 12 of `tensor_rl_paper.md` — match those exactly to reproduce the same
numbers.

### 1c. Run DMRG → MPS → PQC (produces the TN warm-start circuit)

```bash
cd dmrg-to-qc
python3 dmrg_to_qc.py
```
Interactive prompts, matching the paper's settings:
- Select the molecule (list printed on screen).
- **Bond dimension χ** — the paper uses **χ=2 by default**, varies it in Table 8 (TN layers,
  not bond dim) and Table 10 (bond-dim scaling study for 12-LiH: χ ∈ {2,4,8,16,32} — you'd
  need to generate each and re-run Phase 2 per χ to reproduce that table).
- **Number of brickwork layers** — the paper's main results use a **1-layer** brickwork
  warm-start (Table 8 shows 2- and 3-layer ablations degrade success rate for 8/10-H₂O).

Output: `init_state_circ/init_<mol_config>_TNbond<χ>.{qasm,qpy}` plus a console sanity check
comparing the statevector energy before/after a qasm round-trip (should match to `1e-6`).

---

## Phase 2 — Run RL-QAS training (the actual experiment)

Pick the script matching the variant you want, and the `.cfg` matching the molecule:

| Paper variant | Script | Config directory |
|---|---|---|
| TensorRL (trainable TN-init) | `TensorRL_training_and_structureRL_noiseless.py` | `configuration_files/TensorRL_trainable/` |
| StructureRL (ablation) | `TensorRL_training_and_structureRL_noiseless.py` (same script — see below) | `configuration_files/StructureRL/` |
| TensorRL (fixed TN-init) | `TensorRL_fixed_noiseless.py` | `configuration_files/TensorRL_fixed/` |
| Noisy variants (Table 2, App. F/G) | `Tensor_training_and_structureRL_noise.py` / `TensorRL_fixed_noise.py` | same dirs, `*_noise.cfg` |
| Restricted-topology variants | `*_restricted.py` | `*_noise_restricted.cfg` |

`TensorRL (trainable)` vs `StructureRL` are **the same environment/script** — the only
difference is the `.cfg`'s `zero_param_init` flag (`0` = trainable/real warm-start angles,
`1` = StructureRL/angles zeroed). The `StructureRL/` config directory ships this pre-set.

Run, e.g. for 8-H₂O fixed TN-init:
```bash
python3 TensorRL_fixed_noiseless.py --seed 0 --config H2O8q_TNbond2 --experiment_name "TensorRL_fixed/"
```
- You'll be prompted `0: CPU / 1: GPU (CUDA)` — pick per Appendix N / H.4 (paper shows CPU is
  viable up to 8-qubit; larger systems used GPU in the paper's runs, though the training loop
  itself works on either device).
- `--seed` — the paper averages metrics over **5 random seeds** (neural-network
  initialization) for the variance reported in Figs. 3–6; run seeds `0..4` and aggregate.
- `--config` must match a `.cfg` filename (without `.cfg`) inside
  `configuration_files/<experiment_name>/`.

### Matching the paper's stopping condition, not just `episodes`

Every shipped `.cfg` sets `episodes = 10000`, but **Table 14 (Appendix N) reports results by
wall-clock hours, capped at 48h per run**, not by a fixed episode count — some
configurations (e.g. StructureRL at 8q/10q, TensorRL-trainable at 10q/12q) hit the 48h cap
before `episodes=10000` completes, others finish early. To reproduce Table 1/14 faithfully:
run until *either* `episodes` is reached *or* ~48 wall-clock hours have elapsed (whichever the
paper reports for that cell), and read off the **best episode's** metrics, not necessarily the
last one.

### Reproducing the noisy experiments (Table 2, Appendix F/G)

Every shipped `*_noise*.cfg` in this repo actually has **`noise_models = 0` and
`noise_values = 0`** (noise disabled) despite the filename — this looks like the noise
parameters were stripped before the configs were committed. To reproduce Table 2's
depolarizing-noise numbers, you must edit the `.cfg` yourself before running
`TensorRL_fixed_noise.py` / `Tensor_training_and_structureRL_noise.py`:
```ini
[env]
noise_values = (0.01,0.05)     # 1-qubit, 2-qubit depolarizing rates (paper: 10x/5x amplified vs IBM Quantum)
n_shots = 10000                 # paper's shot-noise setting
```
(`CircuitEnv.__init__` parses `noise_values` as a string `"(p1,p2)"`, splitting on the first
comma — see `environments/environment_qulacs_TN_notin_agent_noise.py`.) The restricted-topology
noise config additionally needs the real hardware connectivity graph wired into
`agents/utils_topology_restrict.py` / `environments/utils/utils_topology_restrict.py`
to match the IBMQ-Brisbane comparison in Appendix G.

---

## Phase 3 — Collect results and compute the paper's metrics

Each run writes to `results/<experiment_name>/<config>/`:
- `summary_<seed>.npy` — a dict of per-episode training/test stats: `actions`, `errors`,
  `errors_noiseless`, `reward`, `nfev` (classical-optimizer function evaluations),
  `opt_ang`, `time`, `done_threshold`.
- `thresh_<accept_err>_<seed>_model.pth` / `_optim.pth` / `_replay_buffer.pth` — periodic
  checkpoints (every 5 episodes) of the policy network, optimizer, and replay buffer, so a
  run can be resumed (`conf['agent']['init_net'] = 1` reloads these — see the `if
  conf['agent']['init_net']:` block in each training script).

To reproduce a Table-1-style row (Error / Depth / CNOT / ROT) for a given seed:
1. Load `summary_<seed>.npy` (`np.load(..., allow_pickle=True).item()`).
2. Find the episode with the lowest final `errors` value (or the first episode where
   `errors[-1] < accept_err` for a "first success" metric, matching Table 14's
   "time to first success").
3. That episode's **action sequence** (`stats['train'][ep]['actions']`) plus the TN
   warm-start (if fixed-TN-init, prepended; if trainable, already encoded) reconstructs the
   circuit via `CircuitEnv.make_circuit()` — depth/CNOT-count/rotation-count are then read
   directly off that qulacs circuit.
4. `error` in the summary is already `|energy − true_min_eig|`, i.e. exactly the paper's
   "Error" column.

`agent_test`/`one_episode`'s test-mode counterpart is defined in every training script but is
**not invoked automatically** — it's there for you to call manually (e.g. in a small script
that loads a saved `_model.pth` and runs greedy rollout episodes) if you want a clean
train/test split rather than reading metrics off the training trajectory.

---

## Phase 4 — What you can and can't reproduce out-of-the-box

| Paper result | Reproducible with shipped data? |
|---|---|
| Table 1: 6-BeH₂, 8-H₂O, 10-H₂O (TensorRL fixed/trainable, StructureRL) | ✅ yes, configs + data shipped |
| Table 1: 12-LiH | ❌ needs Phase 1b (generate `LIH_12q` Hamiltonian + TN circuit + write a `.cfg`) |
| Table 2 / Appendix F, G: noisy 8-H₂O, IBMQ-Brisbane hardware run | ⚠️ configs exist but noise is zeroed out — must edit `.cfg` (Phase 2); the real-hardware part (Appendix G) additionally needs IBM Quantum access, not part of this repo |
| Table 5 (Appendix H.1): 8-CH₂O, 10-CH₂O | ✅ yes, configs + data shipped |
| Table 6 (Appendix H.5): 5-Heisenberg | ✅ yes; 6-TFIM ❌ (no shipped Hamiltonian at h=0.05, no `.cfg`) |
| Table 7 (Appendix H.6): 15/17/20-qubit TFIM | ❌ needs generating TFIM Hamiltonians at those sizes + new `.cfg`s (none shipped) |
| Table 8: TN-layer ablation (1 vs 2 vs 3 layers) | ✅ mechanism supported (`dmrg_to_qc.py`'s "number of layers" prompt) — just re-run Phase 1c per layer count and Phase 2 per resulting circuit |
| Table 9: pure TN-compiled energy, no RL | ✅ — this is just the "Initial energy" console output printed by `CircuitEnv.__init__` (or the `energy_qiskit`/`energy` sanity-check prints in `dmrg_to_qc.py`), no RL training needed |
| Table 10: bond-dimension scaling for 12-LiH | ❌ blocked on the same missing 12-LiH data as Table 1 |
| Figures 3–6 (reward curves, success-rate scaling, CPU timing) | ✅ derivable from `summary_<seed>.npy` across the 5-seed runs, once Phase 2/3 are done for each method |

---

## Phase 5 — Full command reference (every shipped config, every variant)

Everything below is copy-pasteable, and every command is annotated with the **Phase**
(0–4 above) it depends on and the **paper table/figure** it reproduces (per the Phase 4
reproducibility matrix), so you can trace any command back to *why* it's needed:

- **Phase 0** (env setup) must be done once before anything below.
- **Phase 1a/1b/1c** (Hamiltonian + TN warm-start circuit) is a per-molecule prerequisite —
  each config below states whether its data is already shipped (Phase 1a — run immediately)
  or must be generated first (Phase 1b+1c — see §5.4).
- **Phase 2** is what each command in §5.1–5.3 actually *is* — a concrete instantiation of the
  script/config table there.
- **Phase 3** (extracting Error/Depth/CNOT/ROT from `summary_<seed>.npy`) and **§5.5**
  (aggregating across seeds) are what you run *after* any block below finishes.
- **Phase 4**'s reproducibility table is the source of every "Table N" reference below.

### 5.1 StructureRL — `TensorRL_training_and_structureRL_noiseless.py`, `configuration_files/StructureRL/`

*Depends on: Phase 0 (done once). Script/variant mapping: Phase 2's table, `StructureRL` row.*

```bash
# Each entry: config name, then the paper table it reproduces and its Phase-1 data status.
cfgs=(
  BEH26q_TNbond2            # Table 1, 6-BeH2                  — Phase 1a: shipped
  CH28q_TNbond2             # Table 5 (App. H.1), 8-CH2O       — Phase 1a: shipped
  CH210q_TNbond2_elec4      # Table 5 (App. H.1), 10-CH2O      — Phase 1a: shipped ("mod" geometry)
  H2O8q_TNbond2             # Table 1, 8-H2O                   — Phase 1a: shipped
  H2O8q_TNbond2_noise       # Table 2 (App. F/G), 8-H2O noisy  — Phase 1a data shipped, BUT see
                            #   "Reproducing the noisy experiments" (Phase 2): noise_values must
                            #   be edited in first, this .cfg ships with noise OFF
  H2O10q_TNbond2            # Table 1, 10-H2O                  — Phase 1a: shipped
  LIH12q_TNbond2            # Table 1, 12-LiH                  — Phase 1b+1c REQUIRED, see §5.4
  heisenberg_5q_TNbond2     # Table 6 (App. H.5), 5-Heisenberg — Phase 1a: shipped
)
for cfg in "${cfgs[@]}"; do
  for seed in 0 1 2 3 4; do            # 5-seed average, per Figs. 3-6 / Phase 2's "--seed" note
    echo 0 | python3 TensorRL_training_and_structureRL_noiseless.py \
      --seed $seed --config $cfg --experiment_name "StructureRL/"
  done
done
```
After each config finishes: extract metrics per **Phase 3**, aggregate seeds per **§5.5**.

### 5.2 TensorRL (fixed TN-init) — `TensorRL_fixed_noiseless.py` / `TensorRL_fixed_noise.py` / `TensorRL_fixed_noise_restricted.py`, `configuration_files/TensorRL_fixed/`

*Depends on: Phase 0. Script/variant mapping: Phase 2's table, `TensorRL (fixed TN-init)` row —
this is the variant the paper reports as cheapest/fastest (Phase-2 "Key feedback loop" note in
`02_training_pipeline.md`, and Table 14/App. N timings).*

```bash
# Noiseless — Phase 1a data already shipped for all of these:
cfgs=(
  BEH26q_TNbond2            # Table 1, 6-BeH2
  CH28q_TNbond2             # Table 5, 8-CH2O
  CH210q_TNbond2            # Table 5, 10-CH2O
  CH210q_TNbond2_elec4      # Table 5, 10-CH2O
  H2O8q_TNbond2             # Table 1, 8-H2O
  H2O8q_TNbond2_cpu         # Appendix H.4 / Fig. 6, CPU-only timing (same data as H2O8q_TNbond2)
  H2O10q_TNbond2            # Table 1, 10-H2O
  LIH12q_TNbond2            # Table 1, 12-LiH — Phase 1b+1c REQUIRED, see §5.4
  heisenberg_5q_TNbond2     # Table 6, 5-Heisenberg
)
for cfg in "${cfgs[@]}"; do
  for seed in 0 1 2 3 4; do
    echo 0 | python3 TensorRL_fixed_noiseless.py \
      --seed $seed --config $cfg --experiment_name "TensorRL_fixed/"
  done
done

# Noisy (Table 2 / Appendix F) — requires the Phase 2 "Reproducing the noisy experiments"
# edit (noise_values/n_shots) before running, Phase 1a Hamiltonian data is already shipped:
for seed in 0 1 2 3 4; do
  python3 TensorRL_fixed_noise.py \
    --seed $seed --config H2O8q_TNbond2_noise --experiment_name "TensorRL_fixed/"
done

# Restricted-topology + noise (Appendix G, IBMQ-Brisbane connectivity comparison) — same
# Phase 2 noise-editing requirement, plus the topology graph noted in Phase 2:
for seed in 0 1 2 3 4; do
  python3 TensorRL_fixed_noise_restricted.py \
    --seed $seed --config H2O8q_TNbond2_noise_restricted --experiment_name "TensorRL_fixed/"
done
```
`H2O10q_TNbond3` (also shipped in this directory) reproduces the `TensorRL (fixed TN-init)`
row is not present for χ=3 in Table 1 (only the trainable variant has a χ=3 row there) — run it
if you want the fixed-TN-init/χ=3 data point anyway; same Phase 1a shipped-data status.
After each config finishes: Phase 3 + §5.5 as above.

### 5.3 TensorRL (trainable TN-init) — `TensorRL_training_and_structureRL_noiseless.py` / `Tensor_training_and_structureRL_noise.py`, `configuration_files/TensorRL_trainable/`

*Depends on: Phase 0. Script/variant mapping: Phase 2's table, `TensorRL (trainable TN-init)`
row — the variant Appendix H.2 (reward-curve analysis) reports as having the strongest
trainability/highest final reward.*

```bash
# Noiseless — Phase 1a data already shipped for all of these:
cfgs=(
  BEH26q_TNbond2               # Table 1, 6-BeH2
  CH28q_TNbond2                # Table 5, 8-CH2O
  CH210q_TNbond2               # Table 5, 10-CH2O
  CH210q_TNbond2_elec4         # Table 5, 10-CH2O
  CH210q_TNbond3               # Table 5, 10-CH2O (chi=3 variant)
  H2O8q_TNbond2                # Table 1, 8-H2O
  H2O10q_TNbond2               # Table 1, 10-H2O (chi=2 row)
  H2O10q_TNbond3               # Table 1, 10-H2O (chi=3 row)
  H2O10q_TNbond2_more_steps    # App. M.2 steps-per-episode robustness check, not a numbered table
  H2O10q_TNbond3_more_steps    # App. M.2 steps-per-episode robustness check, not a numbered table
  heisenberg_5q_TNbond2        # Table 6, 5-Heisenberg
)
for cfg in "${cfgs[@]}"; do
  for seed in 0 1 2 3 4; do
    echo 0 | python3 TensorRL_training_and_structureRL_noiseless.py \
      --seed $seed --config $cfg --experiment_name "TensorRL_trainable/"
  done
done

# Noisy 8-H2O (Table 2) — Phase 2 noise-editing required first:
for seed in 0 1 2 3 4; do
  python3 Tensor_training_and_structureRL_noise.py \
    --seed $seed --config H2O8q_TNbond2_noise --experiment_name "TensorRL_trainable/"
done

# 12-qubit LiH (Table 1) and the 12-qubit H2O bond-dimension sweep (chi=2,3,5 — a bonus
# scaling data point in the spirit of Table 10, not itself a numbered table row) —
# ALL need Phase 1b+1c completed first (see §5.4):
for cfg in LIH12q_TNbond2 H2O12q_TNbond2 H2O12q_TNbond3 H2O12q_TNbond5; do
  for seed in 0 1 2 3 4; do
    echo 0 | python3 TensorRL_training_and_structureRL_noiseless.py \
      --seed $seed --config $cfg --experiment_name "TensorRL_trainable/"
  done
done
```
After each config finishes: Phase 3 + §5.5 as above.

### 5.4 Missing-data molecules — generate before running the blocks above

*This section is Phase 1b (Hamiltonian generation) + Phase 1c (DMRG → PQC) run back-to-back,
for exactly the systems Phase 1b's table and Phase 4's reproducibility matrix flagged as ❌.*

These `.cfg`s exist in `configuration_files/` but their Hamiltonian/TN-circuit files do not
exist in this checkout (see Phase 1b/Phase 4). Run once per molecule/bond-dimension before the
matching training command above will work:

```bash
cd dmrg-to-qc

# 12-LiH (needed by LIH12q_TNbond2.cfg in all three method directories):
#   1. Edit making_molecules.py: set mol/geometry to Li/H per Table 12
#      ("Li 0.000 0.000 0.000; H 0.000 0.000 3.400", STO-3G, 12 qubits after tapering)
#      and run it to produce mol_data/LIH_12q_..._jordan_wigner.npz
python3 making_molecules.py
#   2. Fit the TN warm-start circuit at bond dimension 2 (bond dim also drives Table 10 —
#      repeat with 4, 8, 16, 32 to reproduce that scaling study):
python3 dmrg_to_qc.py   # select LIH molecule, chi=2 (or 4/8/16/32 for Table 10), 1 layer

# 12-H2O (needed by H2O12q_TNbond{2,3,5}.cfg, TensorRL_trainable only):
python3 making_molecules.py   # set mol="H2O", geometry per Table 12, active space -> 12 qubits
python3 dmrg_to_qc.py         # select H2O molecule, chi=2, then repeat run for chi=3 and chi=5

# 6-TFIM at h=0.05 (Table 6) and 15/17/20-qubit TFIM at h=0.001 (Table 7):
#   heisenberg_model.py only ships a Heisenberg Hamiltonian builder in its __main__;
#   edit it (or add a small script) to call construct_hamiltonian-style TFIM assembly
#   (H = sum Z_i Z_{i+1} + h * sum X_i) at n=6/15/17/20 and the required h, save to
#   mol_data/tfim_j1_h<h>_<n>q.npz in the same {hamiltonian, eigvals, weights, paulis} format
#   used by the existing tfim_j1_h0.001_6q.npz.
python3 heisenberg_model.py   # after editing for TFIM + desired (n, h)
python3 dmrg_to_qc.py         # then fit the TN circuit for each size, chi=2

cd ..
```
After each generation step, no shipped `.cfg` exists yet for TFIM — you'll also need to copy
an existing `.cfg` (e.g. `heisenberg_5q_TNbond2.cfg`) to a new
`configuration_files/<Method>/tfim_<n>q_TNbond2.cfg`, and update `[env] num_qubits`,
`[problem] ham_type = tfim_j1_h<h>` (matching the `if self.ham_type not in ['heisenberg',
'tfim_j1_h0.05']` special-case check in `CircuitEnv.__init__`, which loads TFIM/Heisenberg
Hamiltonians without a `geometry` suffix — see `environments/environment_qulacs_TN_notin_agent.py:126-129`),
and `[env] accept_err`/`thresholds` to `1e-2` (the paper's TFIM threshold, Appendix M.1).

### 5.5 Aggregating across seeds

*This is Phase 3's metric-extraction step applied across all 5 seeds from §5.1–5.3 at once —
the same `summary_<seed>.npy` structure documented in Phase 3, just batched.*

Once the 5-seed sweep for a given `(method, config)` pair has finished, aggregate with:
```python
import numpy as np, glob

runs = [np.load(f, allow_pickle=True).item()
        for f in glob.glob("results/TensorRL_fixed/H2O8q_TNbond2/summary_*.npy")]

best_errors = [min(min(ep['errors']) for ep in run['train'].values()) for run in runs]
print("mean:", np.mean(best_errors), "std:", np.std(best_errors))
```
This is the mechanism behind every "±σ over 5 seeds" figure/table in the paper (Figs. 3–6,
Table 1's implicit variance) — swap the `glob` path for any `(method, config)` combination
above.
