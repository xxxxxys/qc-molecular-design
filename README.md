# QC-Assisted Molecular Design

> Reproduction of: **Ajagekar A. & You F.** "Molecular design with automated quantum computing-based deep learning and optimization." *npj Computational Materials* **9**, 143 (2023). [DOI: 10.1038/s41524-023-01099-0](https://doi.org/10.1038/s41524-023-01099-0)

---

## Overview

This repository provides a complete, modular reproduction of the QC-assisted molecular design framework from the paper. The approach combines:

- **(a) GraphConv neural fingerprint** — fixed-weight Duvenaud-style graph convolution producing a 256-dim molecular fingerprint
- **(b) Conditional Restricted Boltzmann Machine (CRBM)** — energy-based generative model trained with CD learning or QC-assisted (simulated annealing) learning
- **(c) QUBO-based molecular generation** — iterative surrogate-guided navigation of chemical space solved with D-Wave Neal simulated annealing

Classical equivalents used (no quantum hardware required):
| Paper | This repo |
|---|---|
| D-Wave Advantage quantum annealer | `dwave-neal` simulated annealing |
| QC-assisted generative training | SA-based negative phase sampling |

---

## Package Structure

```
qc_mol_design/
├── config.py                 # Hyperparameters, property bin definitions
├── data/
│   ├── loader.py             # ZINC-12k loading, property discretisation
│   └── featurizer.py         # Atom features (75-dim), mol→graph, Dataset
├── models/
│   ├── graph_conv.py         # Fixed-weight Duvenaud GraphConv → 256-dim fp
│   ├── crbm.py               # CRBM: energy, free energy, conditionals, QUBO mapping
│   └── predictors.py         # Property prediction MLP heads (QED / LogP / SAS)
├── training/
│   ├── cd_trainer.py         # Contrastive Divergence (CD-1) training
│   └── qc_trainer.py         # SA-based QC-assisted training
├── baselines/
│   └── fingerprint_models.py # ECFP-2048, MACCS-167 baselines (Table 1)
├── optimization/
│   ├── qubo.py               # QUBO formulation + dwave-neal SA solver
│   └── designer.py           # Molecular design: explore (×10) + optimize (×90)
└── evaluation/
    ├── metrics.py            # MAE, Table 1/2 statistics, validity/novelty
    └── visualization.py      # KDE, violin plots, t-SNE, learning curves
run.py                        # Main experiment script
```

---

## Installation

```bash
# Using Conda (recommended)
conda create -n qc-mol python=3.12
conda activate qc-mol

pip install torch numpy pandas scikit-learn tqdm matplotlib scipy
pip install rdkit
pip install dwave-neal dimod
```

---

## Usage

```bash
# Full experiment (Table 1 + Table 2 + Figures)
python run.py

# Debug / quick smoke-test (~30 seconds)
python run.py --debug --mode cd --no-baselines

# Choose training mode: cd | qc | both
python run.py --mode both

# Specify target property and range
python run.py --prop qed  --lo 0.76 --hi 0.79   # Table 2, QED target
python run.py --prop logp --lo 1.799 --hi 2.223  # Table 2, LogP target

# Training only (skip generation)
python run.py --mode both --no-generation
```

Results are saved in `results/`:
- `table1_prediction_mae.csv` — MAE comparison (reproduces Table 1)
- `table2_generation_stats.csv` — Generation statistics (reproduces Table 2)
- `generated_molecules.csv` — All generated SMILES with properties
- `fig2_tsne.png`, `fig3_kde.png`, `fig3_violin_sas.png`, `fig4_gen_dist.png`, `fig5a_learning_curves.png`

---

## Dataset

The code uses a 12,000-molecule ZINC subset (Dwivedi et al. 2022) available at the [paper's GitHub repository](https://github.com/PEESEgroup/qc-camd).

Place `all_gen_mols.csv` in the project root (columns required: `smiles`, `qed`, `logp`, `sas`).

---

## Key Results (Paper)

**Table 1 — Property Prediction MAE**

| Method | QED | SAS | LogP |
|---|---|---|---|
| Rule-based ECFP (2048) | 0.17 | 0.89 | 1.18 |
| Rule-based MACCS (512) | 0.19 | 0.76 | 1.24 |
| GraphConv Neural (256) | 0.75 | 0.88 | 0.81 |
| **CD-learning** | **0.12** | **0.75** | **1.33** |
| **QC-assisted** | **0.10** | **0.66** | **1.27** |

---

## Citation

```bibtex
@article{ajagekar2023molecular,
  title   = {Molecular design with automated quantum computing-based deep learning and optimization},
  author  = {Ajagekar, Akshay and You, Fengqi},
  journal = {npj Computational Materials},
  volume  = {9},
  pages   = {143},
  year    = {2023},
  doi     = {10.1038/s41524-023-01099-0}
}
```
