"""
Contrastive Divergence (CD-k) training for the CRBM.

Algorithm (CD-1 as used in the paper's baseline):
  1. Data phase:
     - Given observed data (f, y_0)
     - Compute mean-field hidden activation: h̄_0 = P(h=1 | y_0, f)
     - Sample: h_0 ~ Bernoulli(h̄_0)
  2. Reconstruction phase (k Gibbs steps):
     - Sample y_k ~ P(y | h_{k-1}, f)  [categorical per property group]
     - Compute h̄_k = P(h=1 | y_k, f)
     - Sample: h_k ~ Bernoulli(h̄_k)
  3. Parameter update (positive minus negative statistics):
     ΔW  = (y_0 ⊗ h̄_0 − y_k ⊗ h̄_k) / B
     Δb  = (y_0 − y_k) / B
     Δc  = (h̄_0 − h̄_k) / B
     ΔW1 = f^T (y_0 − y_k) / B
     ΔW2 = f^T (h̄_0 − h̄_k) / B

Property predictor is jointly trained with MSE against continuous property values.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from qc_mol_design.config import CFG
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.models.graph_conv import NeuralFingerprint
from qc_mol_design.models.predictors import MultiPropertyPredictor


# ─────────────────────────────────────────────────────────────────────────────
# CD parameter update (in-place, no autograd)
# ─────────────────────────────────────────────────────────────────────────────

def _cd_step(
    crbm: CRBM,
    y0:   torch.Tensor,   # (B, n_vis)
    f:    torch.Tensor,   # (B, fp_dim)
    lr:   float,
    k:    int = 1,
    clip: float = 1.0,
) -> float:
    """
    One CD-k in-place parameter update for the CRBM.
    Returns the reconstruction cross-entropy (for monitoring only).
    """
    # ── Positive phase (data statistics) ──────────────────────────────
    ph0 = crbm.p_h_given_yf(y0, f)                 # (B, n_hid)
    h0  = crbm.sample_h(ph0)

    # ── Negative phase (fantasy statistics via k Gibbs steps) ─────────
    yk  = y0
    phk = ph0
    hk  = h0
    for _ in range(k):
        logits_y = crbm.p_y_given_hf_logits(hk, f)
        yk  = crbm.sample_y_from_logits(logits_y)
        phk = crbm.p_h_given_yf(yk, f)
        hk  = crbm.sample_h(phk)

    # ── Parameter gradients ────────────────────────────────────────────
    B = float(y0.shape[0])
    with torch.no_grad():
        dy  = (y0 - yk)                             # (B, n_vis)
        dh  = (ph0 - phk)                           # (B, n_hid)

        dW  = y0.t() @ ph0 / B - yk.t() @ phk / B  # (n_vis, n_hid)
        db  = dy.mean(0)                             # (n_vis,)
        dc  = dh.mean(0)                             # (n_hid,)
        dW1 = f.t() @ dy / B                         # (fp_dim, n_vis)
        dW2 = f.t() @ dh / B                         # (fp_dim, n_hid)

        # Gradient clipping by L2 norm
        for g in (dW, db, dc, dW1, dW2):
            norm = g.norm()
            if norm > clip:
                g.mul_(clip / (norm + 1e-8))

        crbm.W .data.add_(lr * dW)
        crbm.b .data.add_(lr * db)
        crbm.c .data.add_(lr * dc)
        crbm.W1.data.add_(lr * dW1)
        crbm.W2.data.add_(lr * dW2)

    # Reconstruction cross-entropy for logging
    with torch.no_grad():
        logits_recon = crbm.p_y_given_hf_logits(ph0, f)   # (B, n_vis)
        # Compute per-group cross-entropy and average
        loss = _multigroup_ce(logits_recon, y0)
    return loss.item()


def _multigroup_ce(logits: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over property groups."""
    offsets = CFG.prop_offsets
    counts  = CFG.prop_bin_counts
    losses  = []
    for start, count in zip(offsets, counts):
        lg  = logits[:, start : start + count]
        tgt = y_onehot[:, start : start + count].argmax(-1)
        losses.append(F.cross_entropy(lg, tgt))
    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Full CD trainer
# ─────────────────────────────────────────────────────────────────────────────

class CDTrainer:
    """
    Trains the CRBM with contrastive divergence and jointly trains the
    multi-property predictor with MSE on the CRBM latent representation.
    """

    def __init__(
        self,
        gcn:   NeuralFingerprint,
        crbm:  CRBM,
        pred:  MultiPropertyPredictor,
    ) -> None:
        self.gcn   = gcn.to(CFG.device).eval()
        self.crbm  = crbm.to(CFG.device)
        self.pred  = pred.to(CFG.device)

        self.pred_optim = torch.optim.Adam(
            pred.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        self.history: List[Dict] = []

    # ── Pre-compute fingerprints ─────────────────────────────────────────
    def precompute_fps(self, loader: DataLoader) -> List[torch.Tensor]:
        """
        Cache fixed GCN fingerprints to avoid recomputation each epoch.
        Returns list of (B, fp_dim) tensors.
        """
        fps_list = []
        self.gcn.eval()
        with torch.no_grad():
            for X, adj, deg, y_vec, props in loader:
                X, adj, deg = (t.to(CFG.device) for t in (X, adj, deg))
                fps_list.append(self.gcn(X, adj, deg).cpu())
        return fps_list

    # ── Training loop ────────────────────────────────────────────────────
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        tag: str = "CD",
    ) -> List[Dict]:
        """
        Run CFG.epochs training iterations.
        Returns training history (list of per-epoch metric dicts).
        """
        print(f"\n[{tag}] Pre-computing fingerprints ...")
        train_fps = self.precompute_fps(train_loader)
        print(f"[{tag}] Training CRBM for {CFG.epochs} epochs ...")

        # Collect all training labels and properties for reuse
        train_labels, train_props = [], []
        for _, _, _, y_vec, props in train_loader:
            train_labels.append(y_vec)
            train_props.append(props)

        for epoch in range(1, CFG.epochs + 1):
            self.crbm.train(); self.pred.train()
            cd_losses, pred_losses = [], []

            for batch_idx, (fp_batch, y_batch, prop_batch) in enumerate(
                zip(train_fps, train_labels, train_props)
            ):
                f = fp_batch.to(CFG.device)
                y = y_batch.to(CFG.device)
                p = prop_batch.to(CFG.device)   # (B, 3): [qed, logp, sas]

                # ── CD update for CRBM ─────────────────────────────────
                cd_loss = _cd_step(self.crbm, y, f, CFG.lr, CFG.cd_k, CFG.grad_clip)
                cd_losses.append(cd_loss)

                # ── Property predictor on CRBM latent ─────────────────
                with torch.no_grad():
                    h = self.crbm.encode(y, f)   # (B, n_hid)

                self.pred_optim.zero_grad()
                preds = self.pred(h)             # dict: prop → (B,)
                prop_loss = sum(
                    F.mse_loss(preds[name], p[:, i])
                    for i, name in enumerate(("qed", "logp", "sas"))
                )
                prop_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pred.parameters(), CFG.grad_clip)
                self.pred_optim.step()
                pred_losses.append(prop_loss.item())

            epoch_info = {
                "epoch":     epoch,
                "cd_loss":   float(sum(cd_losses) / max(len(cd_losses), 1)),
                "pred_loss": float(sum(pred_losses) / max(len(pred_losses), 1)),
            }
            self.history.append(epoch_info)

            if epoch % 10 == 0:
                print(
                    f"  [{tag}] Epoch {epoch:3d}/{CFG.epochs} | "
                    f"CD: {epoch_info['cd_loss']:.4f} | "
                    f"Pred MSE: {epoch_info['pred_loss']:.4f}"
                )

        return self.history
