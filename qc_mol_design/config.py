"""
Global configuration for QC-assisted molecular design reproduction.

Reference: Ajagekar & You, npj Computational Materials 9, 143 (2023).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Property discretisation bins  (approximated from Supplementary Table 1
# and target conditions in Table 2 of the paper)
# ─────────────────────────────────────────────────────────────────────────────
PROP_BINS: Dict[str, List[float]] = {
    "qed":  [0.00, 0.40, 0.527, 0.615, 0.674, 0.721, 0.760, 0.790, 0.820, 0.847, 1.00],
    "logp": [-6.0, -1.0,  0.522, 1.295, 1.799, 2.223, 2.584, 2.946, 3.314, 3.688, 8.0],
    "sas":  [ 1.0,  2.0,  2.5,   3.0,   3.5,   4.0,   5.0,   6.0,   7.0,   8.0, 10.0],
}

# Number of bins per property (len(edges) - 1)
N_BINS_PER_PROP: Dict[str, int] = {k: len(v) - 1 for k, v in PROP_BINS.items()}

# Maximum atom valencies used for structural constraints
ATOM_VALENCY: Dict[str, int] = {
    "C": 4, "N": 3, "O": 2, "F": 1, "P": 3,
    "S": 2, "Cl": 1, "Br": 1, "I": 1, "Si": 4,
}

# Nine heavy atom types referenced in the paper's Methods section
NINE_HEAVY_ATOMS: List[str] = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


@dataclass
class Config:
    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Data ──────────────────────────────────────────────────────────────
    data_path: str = "all_gen_mols.csv"
    n_train:   int = 10_000
    n_valid:   int = 1_000
    n_test:    int = 1_000

    # ── Molecular graph featurisation ─────────────────────────────────────
    atom_feat_dim: int = 75          # standard DeepChem atom feature size
    max_degree:    int = 6           # degree-specific weight matrices: 0 … max_degree

    # ── GraphConv fingerprint (fixed weights, paper §Methods) ─────────────
    gcn_layers:  int = 2             # number of graph conv layers
    gcn_hidden:  int = 128           # hidden dim per conv layer
    fp_dim:      int = 256           # final fingerprint dimension (paper: 256)

    # ── CRBM ──────────────────────────────────────────────────────────────
    n_props:      int = 3            # number of simultaneously modelled properties
    n_hidden:     int = 64           # latent variable dimension h (paper: 64)

    @property
    def n_vis(self) -> int:
        """Total visible layer size = sum of all property bin counts."""
        return sum(N_BINS_PER_PROP[p] for p in ("qed", "logp", "sas"))

    @property
    def prop_bin_counts(self) -> List[int]:
        """Per-property bin counts in the order [qed, logp, sas]."""
        return [N_BINS_PER_PROP[p] for p in ("qed", "logp", "sas")]

    @property
    def prop_offsets(self) -> List[int]:
        """Start index of each property group within y vector."""
        counts = self.prop_bin_counts
        offsets = [0]
        for c in counts[:-1]:
            offsets.append(offsets[-1] + c)
        return offsets

    # ── Training ──────────────────────────────────────────────────────────
    epochs:     int   = 50
    batch_size: int   = 128
    lr:         float = 1e-3
    weight_decay: float = 1e-5
    cd_k:       int   = 1            # contrastive divergence steps
    grad_clip:  float = 1.0

    # ── QC-assisted training (SA-based) ───────────────────────────────────
    qc_sa_reads:  int = 10           # SA reads per CRBM batch update
    qc_sa_sweeps: int = 500          # SA sweeps per read

    # ── Molecular generation ──────────────────────────────────────────────
    n_explore:      int   = 10       # exploration phase steps (paper: 10)
    n_optimize:     int   = 90       # surrogate-guided steps (paper: 90)
    valency_penalty: float = 100.0   # λ in QUBO valency constraint
    sa_reads:       int   = 100      # SA reads for generation QUBO
    sa_sweeps:      int   = 2_000    # SA sweeps for generation QUBO
    n_ref_mols:     int   = 3        # reference molecules to run generation on

    # ── Target property for generation ────────────────────────────────────
    target_prop:  str               = "qed"
    target_range: Tuple[float, float] = (0.76, 0.79)

    # ── Device ────────────────────────────────────────────────────────────
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Debug mode (small data, few epochs) ───────────────────────────────
    debug: bool = False

    def __post_init__(self) -> None:
        if self.debug:
            self.n_train  = 500
            self.n_valid  = 100
            self.n_test   = 100
            self.epochs   = 5
            self.n_explore = 2
            self.n_optimize = 5
            self.sa_reads  = 10
            self.sa_sweeps = 200


# Singleton default configuration
CFG = Config()
