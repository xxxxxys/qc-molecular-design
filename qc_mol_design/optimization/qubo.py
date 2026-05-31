"""
QUBO formulation for the molecular generation problem.

From the paper (Methods – Optimization strategy for molecular design):

  "The main idea behind the proposed solution technique is the use of a
   weighted linear model Σ_{i,j>i} β_{ij} A_{ij} that serves as a surrogate
   model to approximate the objective function F_θ(y*, G_c(X, A))."

  Structural constraints (valency):
  "Σ_{j>i} A_{ij} + Σ_{i>j} A_{ji} ≤ v_i  for each atom i"
  Encoded as a quadratic penalty Q_c = Σ_i Q_i where
  Q_i = (Σ_{j≠i} A_{ij} + Σ_k 2^k s_{ik} − v_i)^2

Variables:
  x_{ij} ∈ {0,1}  for i < j   (bond indicators, n*(n-1)/2 variables)
  s_{ik} ∈ {0,1}              (binary slack encoding residual valence)

The full QUBO objective:
  min_x  Σ_{i<j} β_{ij} x_{ij}  +  λ · Q_c(x)
where Q_c is expanded into a symmetric matrix form.

SA solver:
  Uses D-Wave's Neal (simulated annealing) as a proxy for the D-Wave Advantage
  quantum annealer used in the paper.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from qc_mol_design.config import CFG, ATOM_VALENCY


# ─────────────────────────────────────────────────────────────────────────────
# QUBO variable index management
# ─────────────────────────────────────────────────────────────────────────────

class MoleculeQUBO:
    """
    Builds and maintains the QUBO for a molecule with fixed atom types and
    variable bond topology.

    Variable layout in the QUBO vector:
      [bond_0, bond_1, ..., bond_{n_bond-1}, slack_0_0, slack_0_1, ..., slack_{n-1}_k]
    """

    def __init__(self, atom_symbols: List[str], max_valencies: List[int]) -> None:
        self.atoms         = atom_symbols
        self.max_valencies = max_valencies
        self.n             = len(atom_symbols)

        # ── Bond variable indices (i < j) ────────────────────────────────
        self.bond_pairs: List[Tuple[int, int]] = []
        self.bond_to_idx: Dict[Tuple[int, int], int] = {}
        k = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.bond_pairs.append((i, j))
                self.bond_to_idx[(i, j)] = k
                k += 1
        self.n_bonds = k

        # ── Slack variable indices (encode residual valence per atom) ─────
        self.slack_to_idx: Dict[Tuple[int, int], int] = {}
        for i in range(self.n):
            n_bits = _n_slack_bits(max_valencies[i])
            for b in range(n_bits):
                self.slack_to_idx[(i, b)] = k
                k += 1

        self.n_vars = k

    def bond_idx(self, i: int, j: int) -> int:
        """Return QUBO variable index for bond (i,j), normalising i < j."""
        if i > j:
            i, j = j, i
        return self.bond_to_idx[(i, j)]

    def bond_vector(self, adj: np.ndarray) -> np.ndarray:
        """Flatten upper-triangular adjacency → bond feature vector (n_bonds,)."""
        vec = np.zeros(self.n_bonds, dtype=np.float32)
        for k, (i, j) in enumerate(self.bond_pairs):
            vec[k] = adj[i, j]
        return vec

    def decode_sample(self, sample: Dict[int, int]) -> np.ndarray:
        """Decode a QUBO sample → adjacency matrix (n×n)."""
        adj = np.zeros((self.n, self.n), dtype=np.int32)
        for k, (i, j) in enumerate(self.bond_pairs):
            v = int(sample.get(k, 0))
            adj[i, j] = adj[j, i] = v
        return adj

    # ── QUBO matrix construction ──────────────────────────────────────────

    def build(
        self,
        surrogate_beta: Optional[np.ndarray] = None,   # (n_bonds,) or None
        penalty:        float = 100.0,
    ) -> np.ndarray:
        """
        Build upper-triangular QUBO matrix Q of shape (n_vars, n_vars).

        Objective: x^T Q x  is minimised.
        """
        Q = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)

        # ── Surrogate objective ─────────────────────────────────────────
        if surrogate_beta is not None:
            beta = np.asarray(surrogate_beta).ravel()
            for k in range(min(len(beta), self.n_bonds)):
                Q[k, k] += float(beta[k])

        # ── Valency constraint penalty ──────────────────────────────────
        for i in range(self.n):
            v_max   = self.max_valencies[i]
            n_bits  = _n_slack_bits(v_max)

            # All terms in ( Σ_j bond_{ij} + Σ_b 2^b slack_{ib} - v_max )
            # are collected as (var_index, coefficient) pairs.
            terms: List[Tuple[int, float]] = []
            for j in range(self.n):
                if j == i:
                    continue
                ii, jj = (i, j) if i < j else (j, i)
                terms.append((self.bond_to_idx[(ii, jj)], 1.0))
            for b in range(n_bits):
                terms.append((self.slack_to_idx[(i, b)], float(2 ** b)))

            constant = -float(v_max)

            # Expand (Σ coef·var + constant)^2 into QUBO form
            for p, (ip, cp) in enumerate(terms):
                # Linear part: (cp^2 + 2·constant·cp) var_p  →  diagonal
                Q[ip, ip] += penalty * cp * (cp + 2.0 * constant)
                for q, (iq, cq) in enumerate(terms):
                    if q <= p:
                        continue
                    lo, hi = min(ip, iq), max(ip, iq)
                    Q[lo, hi] += penalty * 2.0 * cp * cq

        return Q


def _n_slack_bits(v_max: int) -> int:
    """Number of binary slack bits to encode [0, v_max]."""
    return int(math.floor(math.log2(max(v_max, 1)))) + 1


# ─────────────────────────────────────────────────────────────────────────────
# SA solver (D-Wave Neal)
# ─────────────────────────────────────────────────────────────────────────────

def solve_qubo(
    Q:          np.ndarray,
    num_reads:  int = 100,
    num_sweeps: int = 2000,
) -> List[Dict[int, int]]:
    """
    Solve a QUBO  min_{x∈{0,1}^n}  x^T Q x.

    Tries D-Wave Neal first; falls back to a built-in Metropolis SA if Neal
    is unavailable.

    Returns a list of solution dicts {var_idx: 0_or_1}, sorted best first.
    """
    try:
        return _solve_neal(Q, num_reads, num_sweeps)
    except ImportError:
        return _solve_builtin_sa(Q, num_reads, num_sweeps)


def _solve_neal(
    Q: np.ndarray, num_reads: int, num_sweeps: int
) -> List[Dict[int, int]]:
    import neal
    n = Q.shape[0]
    Q_dict = {
        (i, j): float(Q[i, j])
        for i in range(n)
        for j in range(i, n)
        if Q[i, j] != 0.0
    }
    sampler  = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q_dict, num_reads=num_reads, num_sweeps=num_sweeps)
    return [dict(s) for s in response.samples()]


def _solve_builtin_sa(
    Q: np.ndarray, num_reads: int, num_sweeps: int
) -> List[Dict[int, int]]:
    """Fallback Metropolis SA implementation (no external dependency)."""
    n = Q.shape[0]

    def energy(x: np.ndarray) -> float:
        return float(x @ Q @ x)

    solutions = []
    for _ in range(num_reads):
        x = np.random.randint(0, 2, n).astype(np.float64)
        T = 1.0
        e = energy(x)
        for _ in range(num_sweeps):
            flip = random.randint(0, n - 1)
            x[flip] ^= 1
            new_e = energy(x)
            delta = new_e - e
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                e = new_e
            else:
                x[flip] ^= 1
            T *= 0.995
        solutions.append(({i: int(x[i]) for i in range(n)}, e))

    solutions.sort(key=lambda t: t[1])
    return [s for s, _ in solutions]
