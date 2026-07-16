# knowledge_markdown/ — TensorRL-QAS Explained

Notes derived from `README.md` and `tensor_rl_paper.md` plus a read-through of the actual
source (`dmrg-to-qc/`, `agents/`, `environments/`, `configuration_files/`, root training
scripts).

1. **[01_overview.md](01_overview.md)** — what problem this solves, the core idea, and the
   three training variants (TensorRL trainable/fixed, StructureRL).
2. **[02_training_pipeline.md](02_training_pipeline.md)** — the full dataflow from Hamiltonian
   to trained agent, including the per-episode/per-step loop, with ASCII diagrams.
3. **[03_modules.md](03_modules.md)** — module-by-module breakdown of every directory/file
   and its role in the pipeline.
4. **[04_reproducing_paper_results.md](04_reproducing_paper_results.md)** — phase-by-phase
   run-book for reproducing the paper's tables/figures with this codebase, including what
   Hamiltonians/configs are already shipped vs. what you'd need to generate yourself.
5. **[05_quantum_glossary.md](05_quantum_glossary.md)** — plain-language glossary of every
   quantum-specific term used in `04` (qubit, Hamiltonian, DMRG, MPS, bond dimension, VQE,
   QAS, noise models, etc.), written for a CS student new to quantum computing.
