#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py  –  Main experiment script for the QC-assisted molecular design reproduction.

Reproduces:
  • Table 1 – Property prediction MAE comparison (ECFP, MACCS, GraphConv,
               CD-learning, QC-assisted)
  • Table 2 – Targeted molecular generation for multiple property ranges
  • Figures 2, 3, 4, 5 – Visualisations (saved as PNG files)

Usage:
    # Full run (QUBOProject conda env)
    python run.py

    # Debug / quick smoke-test
    python run.py --debug

    # Control target property and range
    python run.py --prop qed --lo 0.76 --hi 0.79

    # Choose training mode: cd | qc | both
    python run.py --mode both
"""
import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# ── Insert project root so the package is importable without install ─────────
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

from qc_mol_design.config import CFG, PROP_BINS
from qc_mol_design.data import (
    load_zinc12k, split_dataset,
    MoleculeDataset, make_dataloader,
)
from qc_mol_design.models import (
    build_fingerprint_model, CRBM, MultiPropertyPredictor,
)
from qc_mol_design.training import CDTrainer, QCTrainer
from qc_mol_design.baselines import run_fingerprint_baselines
from qc_mol_design.optimization import MolecularDesigner, compute_rdkit_props
from qc_mol_design.evaluation import (
    compute_prediction_mae,
    summarise_generation,
    generation_quality_metrics,
    print_table1,
    print_table2,
    plot_kde,
    plot_violin_sas,
    plot_learning_curves,
    plot_generation_distributions,
)

try:
    from qc_mol_design.evaluation import plot_tsne
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="QC-assisted molecular design")
    p.add_argument("--debug",  action="store_true",
                   help="Small dataset + few epochs for quick testing")
    p.add_argument("--prop",   default="qed",
                   choices=("qed", "logp", "sas"),
                   help="Target molecular property for generation")
    p.add_argument("--lo",     type=float, default=0.76,
                   help="Target property range lower bound")
    p.add_argument("--hi",     type=float, default=0.79,
                   help="Target property range upper bound")
    p.add_argument("--mode",   default="both",
                   choices=("cd", "qc", "both"),
                   help="CRBM training mode: cd | qc | both")
    p.add_argument("--no-baselines", action="store_true",
                   help="Skip fingerprint baseline training (faster)")
    p.add_argument("--no-generation", action="store_true",
                   help="Skip molecular generation (training only)")
    p.add_argument("--save-dir", default="results",
                   help="Directory for saving checkpoints and figures")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Seed and device setup
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Model checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    save_dir: str,
    tag:      str,
    gcn:      torch.nn.Module,
    crbm:     CRBM,
    pred:     MultiPropertyPredictor,
) -> None:
    path = Path(save_dir) / f"checkpoint_{tag}.pt"
    torch.save({
        "gcn":  gcn.state_dict(),
        "crbm": crbm.state_dict(),
        "pred": pred.state_dict(),
    }, path)
    print(f"[CKPT] Saved → {path}")


def load_checkpoint(
    path: str,
    gcn:  torch.nn.Module,
    crbm: CRBM,
    pred: MultiPropertyPredictor,
) -> None:
    ckpt = torch.load(path, map_location=CFG.device)
    gcn.load_state_dict(ckpt["gcn"])
    crbm.load_state_dict(ckpt["crbm"])
    pred.load_state_dict(ckpt["pred"])
    print(f"[CKPT] Loaded → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 target conditions (from the paper)
# ─────────────────────────────────────────────────────────────────────────────

TABLE2_TARGETS = {
    "qed":  [
        (0.527, 0.615),
        (0.674, 0.721),
        (0.721, 0.760),
        (0.760, 0.790),
        (0.820, 0.847),
    ],
    "logp": [
        (0.522, 1.295),
        (1.799, 2.223),
        (2.223, 2.584),
        (2.584, 2.946),
        (3.314, 3.688),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Configuration ────────────────────────────────────────────────────────
    if args.debug:
        CFG.debug = True
        CFG.__post_init__()

    CFG.target_prop  = args.prop
    CFG.target_range = (args.lo, args.hi)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(CFG.seed)
    print("=" * 68)
    print(" QC-Assisted Molecular Design  –  Paper Reproduction")
    print(" Ajagekar & You, npj Computational Materials 9, 143 (2023)")
    print("=" * 68)
    print(f"  Device   : {CFG.device}")
    print(f"  Mode     : {args.mode}")
    print(f"  Target   : {CFG.target_prop} ∈ {CFG.target_range}")
    print(f"  Debug    : {CFG.debug}")
    print(f"  Epochs   : {CFG.epochs}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1  Load and split dataset
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 1  Loading ZINC-12k dataset")
    print("─" * 40)
    data                            = load_zinc12k()
    train_df, valid_df, test_df     = split_dataset(data, seed=CFG.seed)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2  Build graph datasets and data loaders
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 2  Building molecular graph datasets")
    print("─" * 40)
    train_ds = MoleculeDataset(train_df, "Train")
    valid_ds = MoleculeDataset(valid_df, "Valid")
    test_ds  = MoleculeDataset(test_df,  "Test")

    train_loader = make_dataloader(train_ds, shuffle=True)
    valid_loader = make_dataloader(valid_ds, shuffle=False)
    test_loader  = make_dataloader(test_ds,  shuffle=False)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3  Build frozen GraphConv fingerprint model
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 3  Initialising fixed GraphConv fingerprint model")
    print("─" * 40)
    gcn = build_fingerprint_model()
    n_params = sum(p.numel() for p in gcn.parameters())
    print(f"  GCN parameters (frozen): {n_params:,}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4  Fingerprint baselines  (Table 1: ECFP, MACCS)
    # ────────────────────────────────────────────────────────────────────────
    table1_results: Dict[str, Dict[str, float]] = {}

    if not args.no_baselines:
        print("\n" + "─" * 40)
        print("STEP 4  Fingerprint baselines (Table 1 rows 1–2)")
        print("─" * 40)
        for prop in ("qed", "logp", "sas"):
            print(f"\n  Property: {prop.upper()}")
            bl = run_fingerprint_baselines(train_df, test_df, prop)
            table1_results.setdefault("ECFP-2048",  {})[prop] = bl["ecfp_mae"]
            table1_results.setdefault("MACCS-167",  {})[prop] = bl["maccs_mae"]
            print(f"    ECFP-2048 MAE: {bl['ecfp_mae']:.4f}")
            print(f"    MACCS-167 MAE: {bl['maccs_mae']:.4f}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5  Train CRBM(s) and property predictors
    # ────────────────────────────────────────────────────────────────────────
    trained_models: Dict[str, tuple] = {}   # tag → (crbm, pred, history)

    modes_to_run = (
        ["cd", "qc"] if args.mode == "both"
        else [args.mode]
    )

    for mode in modes_to_run:
        tag = "CD-learning" if mode == "cd" else "QC-assisted"
        print("\n" + "─" * 40)
        print(f"STEP 5  Training CRBM ({tag})")
        print("─" * 40)

        crbm = CRBM().to(CFG.device)
        pred = MultiPropertyPredictor().to(CFG.device)
        n_crbm = sum(p.numel() for p in crbm.parameters())
        n_pred = sum(p.numel() for p in pred.parameters())
        print(f"  CRBM parameters: {n_crbm:,}")
        print(f"  Predictor parameters: {n_pred:,}")

        t0 = time.time()
        if mode == "cd":
            trainer = CDTrainer(gcn, crbm, pred)
            history = trainer.train(train_loader, valid_loader, tag=tag)
        else:
            trainer = QCTrainer(gcn, crbm, pred)
            history = trainer.train(train_loader, valid_loader, tag=tag)

        elapsed = time.time() - t0
        print(f"  Training time: {elapsed/60:.1f} min")

        save_checkpoint(save_dir, mode, gcn, crbm, pred)
        trained_models[mode] = (crbm, pred, history)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6  Evaluate property prediction  (Table 1)
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 6  Evaluating property prediction (Table 1)")
    print("─" * 40)

    for mode in modes_to_run:
        tag  = "CD-learning" if mode == "cd" else "QC-assisted"
        crbm, pred, _ = trained_models[mode]
        maes = compute_prediction_mae(gcn, crbm, pred, test_loader)
        table1_results[tag] = maes
        print(f"  [{tag}]  QED={maes['qed']:.4f} | "
              f"SAS={maes['sas']:.4f} | LogP={maes['logp']:.4f}")

    print_table1(table1_results)

    # Save Table 1 CSV
    t1_path = save_dir / "table1_prediction_mae.csv"
    pd.DataFrame(table1_results).T.to_csv(t1_path)
    print(f"[TABLE1] Saved → {t1_path}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 7  Learning curve visualisation  (Figure 5a)
    # ────────────────────────────────────────────────────────────────────────
    cd_hist = trained_models.get("cd",  (None, None, []))[2]
    qc_hist = trained_models.get("qc",  (None, None, []))[2]
    plot_learning_curves(
        cd_hist, qc_hist,
        save_path=str(save_dir / "fig5a_learning_curves.png")
    )

    if args.no_generation:
        print("\n[SKIP] Molecular generation skipped (--no-generation flag).")
        print("[DONE]")
        return

    # ────────────────────────────────────────────────────────────────────────
    # STEP 8  Targeted molecular generation  (Table 2)
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 8  Targeted molecular generation (Table 2)")
    print("─" * 40)

    # Use the best-performing trained CRBM (QC if available, else CD)
    best_mode  = "qc" if "qc" in trained_models else "cd"
    best_crbm, _, _ = trained_models[best_mode]
    designer   = MolecularDesigner(gcn, best_crbm)

    # Select reference molecules from test set
    ref_smiles_list = test_df["smiles"].head(CFG.n_ref_mols).tolist()

    # Property targets for Table 2
    prop      = CFG.target_prop
    t_ranges  = TABLE2_TARGETS.get(prop, [CFG.target_range])

    all_gen_records: List[Dict] = []
    table2_stats:    Dict       = {}
    violin_data:     Dict[str, List[float]] = {}
    dist_data:       Dict[str, Dict[str, List[float]]] = {}

    for t_lo, t_hi in t_ranges:
        target_label = f"{prop}=[{t_lo:.3f},{t_hi:.3f}]"
        print(f"\n  Target: {target_label}")

        target_gen: List[Dict] = []
        for ref_smi in ref_smiles_list:
            gen = designer.design(
                ref_smiles   = ref_smi,
                target_prop  = prop,
                target_range = (t_lo, t_hi),
            )
            target_gen.extend(gen)

        all_gen_records.extend(target_gen)

        # Table 2 stats
        stats = summarise_generation(target_gen, prop, (t_lo, t_hi), train_df)
        table2_stats[target_label] = stats
        print_table2({"QC-assisted": stats}, prop, (t_lo, t_hi))

        # Violin plot data (SAS scores for generated in-range molecules)
        sas_vals = [m["sas"] for m in target_gen if m.get("in_range")]
        if sas_vals:
            violin_data[target_label] = sas_vals

        # Distribution comparison data
        gen_prop_vals  = [m[prop] for m in target_gen if m.get("in_range")]
        trn_ref        = train_df[train_df[prop].between(t_lo, t_hi)][prop].tolist()
        dist_data[target_label] = {
            "generated": gen_prop_vals,
            "training":  trn_ref,
        }

    # ────────────────────────────────────────────────────────────────────────
    # STEP 9  Save generation results and visualisations
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("STEP 9  Saving results and figures")
    print("─" * 40)

    if all_gen_records:
        gen_df = pd.DataFrame(all_gen_records)
        gen_path = save_dir / "generated_molecules.csv"
        gen_df.to_csv(gen_path, index=False)
        print(f"[GEN] {len(gen_df)} molecules saved → {gen_path}")

        train_smiles_set = set(train_df["smiles"].tolist())
        qual = generation_quality_metrics(all_gen_records, train_smiles_set)
        print(f"[GEN] Validity:   {qual['validity']*100:.1f}%")
        print(f"[GEN] Uniqueness: {qual['uniqueness']*100:.1f}%")
        print(f"[GEN] Novelty:    {qual['novelty']*100:.1f}%")

    # Table 2 CSV
    if table2_stats:
        t2_df   = pd.DataFrame(table2_stats).T
        t2_path = save_dir / "table2_generation_stats.csv"
        t2_df.to_csv(t2_path)
        print(f"[TABLE2] Saved → {t2_path}")

    # Figures
    if all_gen_records:
        gen_df_viz = pd.DataFrame(all_gen_records)

        plot_kde(
            train_df, gen_df_viz,
            save_path=str(save_dir / "fig3_kde.png")
        )

        if violin_data:
            plot_violin_sas(
                violin_data,
                save_path=str(save_dir / "fig3_violin_sas.png")
            )

        if dist_data:
            plot_generation_distributions(
                dist_data, prop, t_ranges,
                save_path=str(save_dir / "fig4_gen_dist.png")
            )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 10  Latent space t-SNE  (Figure 2)  — requires sklearn
    # ────────────────────────────────────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE

        print("\n[FIG2] Computing latent representations for t-SNE ...")
        best_crbm.eval()
        lat_list, prop_list = [], []

        with torch.no_grad():
            for X, adj, deg, y_vec, props in valid_loader:
                X, adj, deg = (t.to(CFG.device) for t in (X, adj, deg))
                y_vec = y_vec.to(CFG.device)
                f = gcn(X, adj, deg)
                h = best_crbm.encode(y_vec, f)
                lat_list.append(h.cpu().numpy())
                prop_list.append(props[:, 0].numpy())   # QED

        lat_train  = np.vstack(lat_list)
        prop_train = np.concatenate(prop_list)

        # Build latent for generated molecules
        if all_gen_records:
            from qc_mol_design.data.featurizer import mol_to_graph
            from qc_mol_design.data.loader import make_label_vector

            lat_gen_list, prop_gen_list = [], []
            for rec in all_gen_records[:500]:
                g = mol_to_graph(rec["smiles"])
                if g is None:
                    continue
                X_np, adj_np, deg_np = g
                X_t   = torch.FloatTensor(X_np[None]).to(CFG.device)
                adj_t = torch.FloatTensor(adj_np[None]).to(CFG.device)
                deg_t = torch.LongTensor(deg_np[None]).to(CFG.device)
                y_vec_np = make_label_vector(rec["qed"], rec["logp"], rec["sas"])
                y_t = torch.FloatTensor(y_vec_np[None]).to(CFG.device)
                with torch.no_grad():
                    fp = gcn(X_t, adj_t, deg_t)
                    h  = best_crbm.encode(y_t, fp)
                lat_gen_list.append(h.cpu().numpy()[0])
                prop_gen_list.append(rec["qed"])

            if lat_gen_list:
                plot_tsne(
                    lat_train,
                    np.stack(lat_gen_list),
                    np.array(prop_gen_list),
                    prop_name=prop.upper(),
                    save_path=str(save_dir / "fig2_tsne.png"),
                )
    except ImportError:
        print("[FIG2] scikit-learn not found; skipping t-SNE plot.")
    except Exception as e:
        print(f"[FIG2] t-SNE failed: {e}")

    print("\n" + "=" * 68)
    print(" Experiment complete.")
    print(f" Results saved in: {save_dir.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
