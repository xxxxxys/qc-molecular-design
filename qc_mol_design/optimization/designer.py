"""
Molecular design pipeline  (Figure 1c of the paper).

From the paper (Methods – Generating molecules):
  "Ten exploration steps are performed for each optimization problem,
   followed by 90 sampling steps to minimize the associated free energy of
   the trained conditional energy-based model."

Algorithm:
  Input:  reference molecule (fixed atom types X), target property y*
  Output: generated molecules satisfying the property target

  Phase 1 – Exploration (n_explore steps):
    • Solve QUBO with no surrogate (uniform β = 0)
    • Collect (bond_vector, CRBM_free_energy) pairs for surrogate fitting

  Phase 2 – Surrogate-guided optimization (n_optimize steps):
    • Fit linear surrogate: F̂(A) ≈ β^T A_vec   (sklearn LinearRegression)
    • Build QUBO with current β as objective coefficients
    • Solve QUBO → new adjacency matrix A*
    • Evaluate A* with trained CRBM and RDKit
    • Add to dataset; refit surrogate

  Validity check:
    Every generated adjacency is converted to canonical SMILES via RDKit.
    Invalid / sanitisation-failing molecules are discarded automatically.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

from qc_mol_design.config import CFG, ATOM_VALENCY, PROP_BINS, N_BINS_PER_PROP
from qc_mol_design.data.featurizer import graph_from_atoms_adj
from qc_mol_design.data.loader import discretize
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.models.graph_conv import NeuralFingerprint
from qc_mol_design.optimization.qubo import MoleculeQUBO, solve_qubo

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Molecule helpers
# ─────────────────────────────────────────────────────────────────────────────

def adj_to_smiles(atom_symbols: List[str], adj: np.ndarray) -> Optional[str]:
    """Convert adjacency matrix + atom list → canonical SMILES, or None."""
    from rdkit import Chem
    try:
        rw = Chem.RWMol()
        for sym in atom_symbols:
            rw.AddAtom(Chem.Atom(sym))
        for i in range(len(atom_symbols)):
            for j in range(i + 1, len(atom_symbols)):
                if adj[i, j]:
                    rw.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_rdkit_props(smiles: str) -> Optional[Dict[str, float]]:
    """Return dict of {qed, logp, sas} for a SMILES string, or None."""
    from rdkit import Chem
    from rdkit.Chem.QED import qed as rdkit_qed
    from rdkit.Chem.Crippen import MolLogP
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            from rdkit.Contrib.SA_Score import sascorer
            sas = float(sascorer.calculateScore(mol))
        except Exception:
            sas = float(mol.GetNumHeavyAtoms()) / 10.0 + 1.0
        return {
            "qed":  float(rdkit_qed(mol)),
            "logp": float(MolLogP(mol)),
            "sas":  sas,
        }
    except Exception:
        return None


def _make_target_y(prop: str, target_range: Tuple[float, float]) -> torch.Tensor:
    """
    Build the concatenated one-hot y* vector for the target property.
    Non-target properties get uniform (average) bins.
    Returns (1, n_vis) FloatTensor.
    """
    mid = (target_range[0] + target_range[1]) / 2.0
    parts = []
    for p in ("qed", "logp", "sas"):
        n = N_BINS_PER_PROP[p]
        if p == prop:
            idx = int(discretize(np.array([mid]), p)[0])
            oh  = np.zeros(n, dtype=np.float32)
            oh[idx] = 1.0
        else:
            oh = np.ones(n, dtype=np.float32) / n  # soft / uniform
        parts.append(oh)
    y = np.concatenate(parts)
    return torch.FloatTensor(y).unsqueeze(0)   # (1, n_vis)


# ─────────────────────────────────────────────────────────────────────────────
# Molecular designer
# ─────────────────────────────────────────────────────────────────────────────

class MolecularDesigner:
    """
    Implements Figure 1c of the paper:
      iterative QUBO-based navigation of chemical space guided by CRBM free energy.
    """

    def __init__(self, gcn: NeuralFingerprint, crbm: CRBM) -> None:
        self.gcn  = gcn.to(CFG.device).eval()
        self.crbm = crbm.to(CFG.device).eval()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _fingerprint(
        self, atom_symbols: List[str], adj: np.ndarray
    ) -> np.ndarray:
        """Compute GCN fingerprint for (atom_symbols, adj) → (fp_dim,)."""
        X, A, deg = graph_from_atoms_adj(atom_symbols, adj)
        X_t   = torch.FloatTensor(X[None]).to(CFG.device)
        A_t   = torch.FloatTensor(A[None]).to(CFG.device)
        deg_t = torch.LongTensor(deg[None]).to(CFG.device)
        with torch.no_grad():
            fp = self.gcn(X_t, A_t, deg_t)
        return fp.cpu().numpy()[0]

    def _free_energy(self, fp: np.ndarray, y_star: torch.Tensor) -> float:
        """Evaluate CRBM free energy for a molecule fingerprint under target y*."""
        f_t = torch.FloatTensor(fp[None]).to(CFG.device)
        y_t = y_star.to(CFG.device)
        with torch.no_grad():
            fe = self.crbm.free_energy(y_t, f_t)
        return float(fe.item())

    # ── Main design loop ─────────────────────────────────────────────────────

    def design(
        self,
        ref_smiles:   str,
        target_prop:  str   = None,
        target_range: Tuple[float, float] = None,
        penalty:      float = None,
        n_explore:    int   = None,
        n_optimize:   int   = None,
        sa_reads:     int   = None,
        sa_sweeps:    int   = None,
    ) -> List[Dict]:
        """
        Run the full design pipeline for a reference molecule.

        Returns a list of generated molecule dicts (smiles, properties, phase).
        """
        from rdkit import Chem

        prop         = target_prop   or CFG.target_prop
        t_range      = target_range  or CFG.target_range
        lam          = penalty       or CFG.valency_penalty
        n_exp        = n_explore     or CFG.n_explore
        n_opt        = n_optimize    or CFG.n_optimize
        sa_r         = sa_reads      or CFG.sa_reads
        sa_s         = sa_sweeps     or CFG.sa_sweeps

        # ── Parse reference molecule ──────────────────────────────────────
        mol = Chem.MolFromSmiles(ref_smiles)
        if mol is None:
            raise ValueError(f"Invalid reference SMILES: {ref_smiles}")

        atom_symbols  = [a.GetSymbol() for a in mol.GetAtoms()]
        max_valencies = [ATOM_VALENCY.get(s, 4) for s in atom_symbols]

        print(f"\n[DESIGN] Reference: {ref_smiles}")
        print(f"[DESIGN] Atoms ({len(atom_symbols)}): {atom_symbols}")
        print(f"[DESIGN] Target: {prop} ∈ {t_range}")

        # ── Pre-build QUBO structure ─────────────────────────────────────
        qubo    = MoleculeQUBO(atom_symbols, max_valencies)
        y_star  = _make_target_y(prop, t_range)
        # fit_intercept=False: paper surrogate is Σβ_{ij}A_{ij}, no constant term
        surrogate = LinearRegression(fit_intercept=False)
        bond_X: List[np.ndarray] = []
        fe_y:   List[float]      = []
        generated: List[Dict]    = []
        # Global bond-vector hash set for cross-step deduplication.
        # Duplicate (bond_vec, energy) pairs in the surrogate dataset are
        # equivalent to upweighting already-visited molecules, biasing β.
        seen_bvec: set = set()
        seen_smi_global: set = set()

        # ── Inner: solve QUBO, evaluate, collect ─────────────────────────
        def _step(beta: Optional[np.ndarray], phase: str) -> int:
            # Adaptive penalty: keep λ ≥ 10 × max|β| so valency constraints
            # always dominate the surrogate objective regardless of β scale.
            if beta is not None and len(beta) > 0:
                eff_lam = max(lam, 10.0 * float(np.abs(beta).max()) + 1e-8)
            else:
                eff_lam = lam

            Q        = qubo.build(beta, eff_lam)
            samples  = solve_qubo(Q, sa_r, sa_s)
            valid    = 0
            for sample in samples[:20]:           # check up to 20 candidates
                adj  = qubo.decode_sample(sample)
                smi  = adj_to_smiles(atom_symbols, adj)
                if smi is None or smi in seen_smi_global:
                    continue
                seen_smi_global.add(smi)

                fp   = self._fingerprint(atom_symbols, adj)
                fe   = self._free_energy(fp, y_star)
                bvec = qubo.bond_vector(adj)

                # Only append unique bond vectors to the surrogate dataset
                bvec_key = bvec.tobytes()
                if bvec_key not in seen_bvec:
                    seen_bvec.add(bvec_key)
                    bond_X.append(bvec)
                    fe_y.append(fe)

                props = compute_rdkit_props(smi)
                if props:
                    in_range = t_range[0] <= props[prop] <= t_range[1]
                    generated.append({
                        "smiles":   smi,
                        "phase":    phase,
                        "in_range": in_range,
                        **props,
                    })
                    valid += 1
                if valid >= 5:
                    break
            return valid

        # ── Phase 1: Exploration ──────────────────────────────────────────
        print("[PHASE 1] Exploration")
        for step in range(n_exp):
            n_v = _step(None, f"explore_{step}")
            print(f"  Step {step+1:2d}/{n_exp} | found {n_v} valid | "
                  f"dataset: {len(bond_X)}")

        # ── Phase 2: Surrogate-guided optimisation ────────────────────────
        print("[PHASE 2] Surrogate-guided optimisation")
        for step in range(n_opt):
            if len(bond_X) >= 2:
                surrogate.fit(bond_X, fe_y)
                beta = surrogate.coef_
            else:
                beta = None

            n_v = _step(beta, f"opt_{step}")
            beta_str = f"{np.mean(beta):.4f}" if beta is not None else "N/A"
            print(f"  Step {step+1:2d}/{n_opt} | β̄={beta_str} | "
                  f"found {n_v} valid | dataset: {len(bond_X)}")

        # ── Summary ───────────────────────────────────────────────────────
        in_range = [m for m in generated if m["in_range"]]
        total    = len(generated)
        print(f"\n[DESIGN] Done. Generated={total}, "
              f"In-range={len(in_range)} ({100*len(in_range)/max(total,1):.1f}%)")
        if in_range:
            for key in ("qed", "logp", "sas"):
                vals = [m[key] for m in in_range]
                print(f"  {key.upper()}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

        return generated
