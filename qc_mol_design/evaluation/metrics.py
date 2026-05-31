"""
Evaluation metrics reproducing Tables 1 and 2 of the paper.

Table 1 – Property Prediction MAE:
  MAE computed over a 1000-molecule test set for QED, SAS, and LogP.

Table 2 – Molecular Generation Efficacy:
  For each target property range:
    • mean ± std of the property value for generated molecules
    • comparison against training set molecules in the same range

Additional metrics:
  • validity:  fraction of generated SMILES that are valid RDKit molecules
  • uniqueness: fraction of distinct SMILES among valid molecules
  • novelty:    fraction of valid unique SMILES not in the training set
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qc_mol_design.config import CFG, PROP_BINS
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.models.graph_conv import NeuralFingerprint
from qc_mol_design.models.predictors import MultiPropertyPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Property prediction MAE  (Table 1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_prediction_mae(
    gcn:    NeuralFingerprint,
    crbm:   CRBM,
    pred:   MultiPropertyPredictor,
    loader: DataLoader,
) -> Dict[str, float]:
    """
    Evaluate property prediction Mean Absolute Error on a data loader.

    Returns dict: {'qed': mae_qed, 'logp': mae_logp, 'sas': mae_sas}.
    """
    gcn.eval(); crbm.eval(); pred.eval()
    dev = CFG.device

    all_preds:  Dict[str, List[float]] = {p: [] for p in ("qed", "logp", "sas")}
    all_truths: Dict[str, List[float]] = {p: [] for p in ("qed", "logp", "sas")}

    with torch.no_grad():
        for X, adj, deg, y_vec, props in loader:
            X, adj, deg = (t.to(dev) for t in (X, adj, deg))
            y_vec = y_vec.to(dev)
            props = props.to(dev)   # (B, 3): [qed, logp, sas]

            f    = gcn(X, adj, deg)
            h    = crbm.encode(y_vec, f)
            phat = pred(h)          # dict: prop → (B,)

            for i, p in enumerate(("qed", "logp", "sas")):
                all_preds[p].extend(phat[p].cpu().tolist())
                all_truths[p].extend(props[:, i].cpu().tolist())

    maes = {}
    for p in ("qed", "logp", "sas"):
        phat_arr = np.array(all_preds[p])
        true_arr = np.array(all_truths[p])
        maes[p]  = float(np.abs(phat_arr - true_arr).mean())
    return maes


# ─────────────────────────────────────────────────────────────────────────────
# Generation metrics  (Table 2)
# ─────────────────────────────────────────────────────────────────────────────

def summarise_generation(
    generated:    List[Dict],
    prop:         str,
    target_range: tuple,
    train_df:     pd.DataFrame,
) -> Dict:
    """
    Compute generation efficacy statistics matching Table 2 of the paper.

    Returns a dict with keys:
      qed_mean, qed_std, logp_mean, logp_std, sas_mean, sas_std,
      n_generated, n_in_range, in_range_fraction
    """
    in_range = [m for m in generated if m.get("in_range", False)]

    # Training-set reference: molecules satisfying same target
    lo, hi     = target_range
    train_ref  = train_df[train_df[prop].between(lo, hi)]

    stats: Dict = {
        "n_generated":    len(generated),
        "n_in_range":     len(in_range),
        "in_range_pct":   100.0 * len(in_range) / max(len(generated), 1),
        "train_n_ref":    len(train_ref),
    }

    for key in ("qed", "logp", "sas"):
        gen_vals = [m[key] for m in in_range] if in_range else [float("nan")]
        trn_vals = train_ref[key].values if len(train_ref) else np.array([float("nan")])
        stats[f"gen_{key}_mean"]   = float(np.nanmean(gen_vals))
        stats[f"gen_{key}_std"]    = float(np.nanstd(gen_vals))
        stats[f"train_{key}_mean"] = float(np.nanmean(trn_vals))
        stats[f"train_{key}_std"]  = float(np.nanstd(trn_vals))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Diversity / novelty metrics
# ─────────────────────────────────────────────────────────────────────────────

def generation_quality_metrics(
    generated:     List[Dict],
    train_smiles:  Set[str],
) -> Dict[str, float]:
    """
    Compute validity, uniqueness, and novelty over generated molecules.
    """
    from rdkit import Chem

    all_smiles   = [m["smiles"] for m in generated]
    valid_smiles = [s for s in all_smiles if Chem.MolFromSmiles(s) is not None]
    unique_valid = list(set(valid_smiles))
    novel        = [s for s in unique_valid if s not in train_smiles]

    n_all    = max(len(all_smiles), 1)
    n_valid  = max(len(valid_smiles), 1)

    return {
        "validity":   len(valid_smiles) / n_all,
        "uniqueness": len(unique_valid) / n_valid,
        "novelty":    len(novel) / max(len(unique_valid), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def print_table1(results: Dict[str, Dict[str, float]]) -> None:
    """
    Pretty-print Table 1 comparison.

    results: { method_name: {'qed': mae, 'logp': mae, 'sas': mae} }
    """
    header = f"{'Method':<25} {'QED':>10} {'SAS':>10} {'LogP':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print("Table 1 – Mean Absolute Error for Property Prediction")
    print(sep)
    print(header)
    print(sep)
    for method, maes in results.items():
        row = (
            f"{method:<25} "
            f"{maes.get('qed',  float('nan')):>10.4f} "
            f"{maes.get('sas',  float('nan')):>10.4f} "
            f"{maes.get('logp', float('nan')):>10.4f}"
        )
        print(row)
    print(sep)
    print("Reference (paper CD-learning): QED≈0.12 | SAS≈0.75 | LogP≈1.33")
    print("Reference (paper QC-assisted): QED≈0.10 | SAS≈0.66 | LogP≈1.27")
    print(sep + "\n")


def print_table2(
    results:      Dict[str, Dict],
    target_prop:  str,
    target_range: tuple,
) -> None:
    """Pretty-print Table 2 row for a single target condition."""
    prop_label = target_prop.upper()
    lo, hi     = target_range
    print(f"[Table 2] {prop_label} ∈ [{lo:.3f}, {hi:.3f}]")
    for method, stats in results.items():
        if f"gen_{target_prop}_mean" in stats:
            m = stats[f"gen_{target_prop}_mean"]
            s = stats[f"gen_{target_prop}_std"]
            print(f"  {method:<20}: {m:.2f} ± {s:.3f}  "
                  f"(in-range: {stats['n_in_range']}/{stats['n_generated']})")
