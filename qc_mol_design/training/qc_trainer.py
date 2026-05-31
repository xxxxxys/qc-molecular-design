"""
QC-assisted CRBM training via Simulated Annealing.

From the paper (Methods – QC-assisted training):
  "We exploit quantum annealers to model p_θ(y, h | f) and draw samples from
   them to train the constructed energy-based model."
  "The target Ising Hamiltonian on the quantum annealers is specified through
   local fields h and pairwise couplings J, computed by mapping the energy
   function of the conditional energy-based model … to the Ising Hamiltonian."
  "Each annealing run used for both generative training and optimization is
   performed for 20 µs on the D-Wave Advantage quantum processor."

Classical approximation used here:
  D-Wave quantum annealing  →  D-Wave's Neal simulated annealing

Training algorithm:
  For each batch (f, y_pos):
    1. Positive phase: h̄_pos = P(h | y_pos, f)   [mean-field, exact]
    2. Negative phase via SA:
       a. Build QUBO Q from current CRBM parameters and f
       b. Sample (y_neg, h_neg) ← SA(Q)    [joint sampling of visible + hidden]
    3. Compute CD-like updates using SA samples as the negative phase:
       ΔW  = (y_pos ⊗ h̄_pos − y_neg ⊗ h̄_neg_mf) / B
       (remaining updates mirror the CD trainer)

Efficiency note:
  Building and solving n_QUBO_reads full QUBO problems per batch is expensive.
  We amortise by using one SA call per batch with num_reads = CFG.qc_sa_reads.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from qc_mol_design.config import CFG
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.models.graph_conv import NeuralFingerprint
from qc_mol_design.models.predictors import MultiPropertyPredictor
from qc_mol_design.training.cd_trainer import _multigroup_ce

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SA-based negative phase sampler
# ─────────────────────────────────────────────────────────────────────────────

def _sample_neg_phase_sa(
    crbm:    CRBM,
    f_batch: torch.Tensor,       # (B, fp_dim)
    n_reads: int  = 5,
    n_sweeps: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample negative phase (y_neg, h_neg) for an entire batch using SA.

    Strategy: average one representative molecule's QUBO over the batch
    (the paper uses per-molecule QUBO but this is too expensive for a full
    batch; we take the batch-mean fingerprint).

    Returns:
      y_neg  (B, n_vis)  — sampled visible (property bins)
      h_neg  (B, n_hid)  — sampled hidden
    """
    import neal

    B        = f_batch.shape[0]
    n_vis    = CFG.n_vis
    n_hid    = CFG.n_hidden
    n_tot    = n_vis + n_hid

    # Use batch-mean fingerprint as a representative condition
    f_mean   = f_batch.mean(0)                      # (fp_dim,)
    Q_tensor = crbm.build_neg_phase_qubo(f_mean)    # (n_tot, n_tot)
    Q_np     = Q_tensor.numpy()

    # Build QUBO dict for Neal
    Q_dict   = {
        (i, j): float(Q_np[i, j])
        for i in range(n_tot)
        for j in range(i, n_tot)
        if Q_np[i, j] != 0.0
    }

    sampler  = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(
        Q_dict, num_reads=max(n_reads, B), num_sweeps=n_sweeps
    )

    # Decode samples → (y, h) pairs
    samples  = list(response.samples())
    y_neg_list, h_neg_list = [], []
    for s_idx in range(B):
        s    = samples[s_idx % len(samples)]
        x    = np.array([s.get(i, 0) for i in range(n_tot)], dtype=np.float32)
        y_raw = x[:n_vis]
        h_raw = x[n_vis:]

        # Project y_raw → valid one-hot per property group
        offsets = CFG.prop_offsets
        counts  = CFG.prop_bin_counts
        y_oh    = np.zeros(n_vis, dtype=np.float32)
        for start, count in zip(offsets, counts):
            group = y_raw[start : start + count]
            if group.sum() == 0:
                group[0] = 1.0
            else:
                # Keep argmax as the one-hot
                best = int(group.argmax())
                group = np.zeros(count, dtype=np.float32)
                group[best] = 1.0
            y_oh[start : start + count] = group

        y_neg_list.append(y_oh)
        h_neg_list.append(h_raw)

    y_neg = torch.FloatTensor(np.stack(y_neg_list))   # (B, n_vis)
    h_neg = torch.FloatTensor(np.stack(h_neg_list))   # (B, n_hid)
    return y_neg, h_neg


def _qc_step(
    crbm:     CRBM,
    y0:       torch.Tensor,   # (B, n_vis)
    f:        torch.Tensor,   # (B, fp_dim)
    lr:       float,
    clip:     float = 1.0,
    n_reads:  int   = 5,
    n_sweeps: int   = 500,
) -> float:
    """
    One QC-assisted in-place update step.
    Returns mean free energy (for monitoring).
    """
    B = float(y0.shape[0])

    # Positive phase
    ph0 = crbm.p_h_given_yf(y0, f)                   # (B, n_hid)

    # Negative phase via SA
    with torch.no_grad():
        y_neg, h_neg_raw = _sample_neg_phase_sa(
            crbm, f.detach().cpu(), n_reads, n_sweeps
        )
        y_neg   = y_neg.to(f.device)
        h_neg_raw = h_neg_raw.to(f.device)

    # Mean-field hidden for negative y (mirrors CD logic)
    phk = crbm.p_h_given_yf(y_neg, f)                # (B, n_hid)

    # Parameter update (same formula as CD)
    with torch.no_grad():
        dW  = y0.t() @ ph0 / B - y_neg.t() @ phk / B
        db  = (y0 - y_neg).mean(0)
        dc  = (ph0 - phk).mean(0)
        dW1 = f.t() @ (y0 - y_neg) / B
        dW2 = f.t() @ (ph0 - phk) / B

        for g in (dW, db, dc, dW1, dW2):
            norm = g.norm()
            if norm > clip:
                g.mul_(clip / (norm + 1e-8))

        crbm.W .data.add_(lr * dW)
        crbm.b .data.add_(lr * db)
        crbm.c .data.add_(lr * dc)
        crbm.W1.data.add_(lr * dW1)
        crbm.W2.data.add_(lr * dW2)

    # Free energy of data points (monitor convergence)
    with torch.no_grad():
        fe = crbm.free_energy(y0, f).mean()
    return float(fe.item())


# ─────────────────────────────────────────────────────────────────────────────
# Full QC trainer
# ─────────────────────────────────────────────────────────────────────────────

class QCTrainer:
    """
    Trains the CRBM using SA-based negative phase (QC-assisted simulation)
    and jointly trains the multi-property predictor.
    """

    def __init__(
        self,
        gcn:  NeuralFingerprint,
        crbm: CRBM,
        pred: MultiPropertyPredictor,
    ) -> None:
        self.gcn  = gcn.to(CFG.device).eval()
        self.crbm = crbm.to(CFG.device)
        self.pred = pred.to(CFG.device)
        self.pred_optim = torch.optim.Adam(
            pred.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        self.history: List[Dict] = []

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        tag: str = "QC",
    ) -> List[Dict]:
        print(f"\n[{tag}] Pre-computing fingerprints ...")
        train_fps, train_labels, train_props = [], [], []
        self.gcn.eval()
        with torch.no_grad():
            for X, adj, deg, y_vec, props in train_loader:
                X, adj, deg = (t.to(CFG.device) for t in (X, adj, deg))
                train_fps.append(self.gcn(X, adj, deg).cpu())
                train_labels.append(y_vec)
                train_props.append(props)

        print(f"[{tag}] Training CRBM (QC-assisted SA) for {CFG.epochs} epochs ...")

        for epoch in range(1, CFG.epochs + 1):
            self.crbm.train(); self.pred.train()
            qc_losses, pred_losses = [], []

            for fp_batch, y_batch, prop_batch in zip(
                train_fps, train_labels, train_props
            ):
                f = fp_batch.to(CFG.device)
                y = y_batch.to(CFG.device)
                p = prop_batch.to(CFG.device)

                fe = _qc_step(
                    self.crbm, y, f, CFG.lr, CFG.grad_clip,
                    n_reads=CFG.qc_sa_reads, n_sweeps=CFG.qc_sa_sweeps
                )
                qc_losses.append(fe)

                with torch.no_grad():
                    h = self.crbm.encode(y, f)

                self.pred_optim.zero_grad()
                preds = self.pred(h)
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
                "fe_mean":   float(sum(qc_losses) / max(len(qc_losses), 1)),
                "pred_loss": float(sum(pred_losses) / max(len(pred_losses), 1)),
            }
            self.history.append(epoch_info)

            if epoch % 10 == 0:
                print(
                    f"  [{tag}] Epoch {epoch:3d}/{CFG.epochs} | "
                    f"Free energy: {epoch_info['fe_mean']:.4f} | "
                    f"Pred MSE: {epoch_info['pred_loss']:.4f}"
                )

        return self.history
