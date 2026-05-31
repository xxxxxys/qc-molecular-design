"""
Conditional Restricted Boltzmann Machine (CRBM).

From the paper (Methods – Energy-based model):
  "We adopt a conditional generative model called conditional restricted
   Boltzmann machine (CRBM) to incorporate molecular property targets as
   binary variables."

Energy function:
  E(y, h | f) = -b^T y  -  (W1 f)^T y  -  c^T h  -  (W2 f)^T h  -  y^T W h

where
  y  ∈ {0,1}^{n_vis}    concatenated one-hot property bins  [QED | LogP | SAS]
  h  ∈ {0,1}^{n_hid}    binary latent variables
  f  ∈ R^{fp_dim}       molecular fingerprint (external conditioning input)
  b  ∈ R^{n_vis}        visible bias
  c  ∈ R^{n_hid}        hidden bias
  W  ∈ R^{n_vis×n_hid}  visible–hidden coupling
  W1 ∈ R^{fp_dim×n_vis} fingerprint → visible effective bias
  W2 ∈ R^{fp_dim×n_hid} fingerprint → hidden effective bias

Effective biases (fingerprint-conditioned):
  b_eff(f) = b + W1^T f   (n_vis,)
  c_eff(f) = c + W2^T f   (n_hid,)

Simplified energy:
  E(y, h | f) = -b_eff(f)^T y  -  c_eff(f)^T h  -  y^T W h

Conditional distributions:
  P(h_j = 1 | y, f) = σ( c_eff(f)_j + [W^T y]_j )
  P(y | h, f) factorises per property group via softmax:
    P(y_g | h, f) = softmax( b_eff_g(f) + W_g h )  for group g ∈ {QED, LogP, SAS}

Free energy (marginal over h):
  F(y, f) = -b_eff(f)^T y  -  Σ_j softplus( c_eff(f)_j + [W^T y]_j )
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from qc_mol_design.config import CFG


class CRBM(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        n_vis = CFG.n_vis
        n_hid = CFG.n_hidden
        fp    = CFG.fp_dim

        self.b  = nn.Parameter(torch.zeros(n_vis))
        self.c  = nn.Parameter(torch.zeros(n_hid))
        self.W  = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.W1 = nn.Parameter(torch.randn(fp, n_vis)   * 0.01)  # f → visible
        self.W2 = nn.Parameter(torch.randn(fp, n_hid)  * 0.01)  # f → hidden

    # ──────────────────────────────────────────────────────────────────────
    # Effective biases
    # ──────────────────────────────────────────────────────────────────────

    def effective_biases(
        self, f: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        b_eff (B, n_vis),  c_eff (B, n_hid)
        f: (B, fp_dim)
        """
        b_eff = self.b + f @ self.W1   # (B, n_vis)
        c_eff = self.c + f @ self.W2   # (B, n_hid)
        return b_eff, c_eff

    # ──────────────────────────────────────────────────────────────────────
    # Energy and free energy
    # ──────────────────────────────────────────────────────────────────────

    def energy(
        self,
        y: torch.Tensor,   # (B, n_vis)
        h: torch.Tensor,   # (B, n_hid)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B,)
        b_eff, c_eff = self.effective_biases(f)
        return (
            -(y * b_eff).sum(-1)
            -(h * c_eff).sum(-1)
            -(y @ self.W * h).sum(-1)
        )

    def free_energy(
        self,
        y: torch.Tensor,   # (B, n_vis)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B,)
        """
        F(y, f) = -b_eff^T y  -  Σ_j softplus(c_eff_j + [W^T y]_j)

        Used as the objective function in the molecular generation QUBO.
        Lower value → molecule better fits target property.
        """
        b_eff, c_eff = self.effective_biases(f)
        vis_term = -(y * b_eff).sum(-1)
        h_input  = c_eff + y @ self.W              # (B, n_hid)
        hid_term = -F.softplus(h_input).sum(-1)
        return vis_term + hid_term

    # ──────────────────────────────────────────────────────────────────────
    # Conditional distributions
    # ──────────────────────────────────────────────────────────────────────

    def p_h_given_yf(
        self,
        y: torch.Tensor,   # (B, n_vis)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B, n_hid) — P(h_j=1 | y, f)
        _, c_eff = self.effective_biases(f)
        return torch.sigmoid(c_eff + y @ self.W)

    def p_y_given_hf_logits(
        self,
        h: torch.Tensor,   # (B, n_hid)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B, n_vis) — unnormalised logits
        b_eff, _ = self.effective_biases(f)
        return b_eff + h @ self.W.t()

    def sample_y_from_logits(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample y from P(y | h, f).

        For each property group the logits are split and a categorical sample
        is drawn.  The result is a concatenated one-hot vector.
        """
        offsets = CFG.prop_offsets
        counts  = CFG.prop_bin_counts
        parts   = []
        for start, count in zip(offsets, counts):
            group_logits = logits[:, start : start + count]   # (B, count)
            idx = torch.distributions.Categorical(logits=group_logits).sample()
            parts.append(F.one_hot(idx, count).float())
        return torch.cat(parts, dim=-1)

    @staticmethod
    def sample_h(probs: torch.Tensor) -> torch.Tensor:
        """Bernoulli sample from P(h | y, f)."""
        return (torch.rand_like(probs) < probs).float()

    # ──────────────────────────────────────────────────────────────────────
    # Encoding  (latent representation for property prediction)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        y: torch.Tensor,   # (B, n_vis)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B, n_hid)  — mean-field h̄ = E[h | y, f]
        """
        Extract latent representation h̄ = σ(c_eff + W^T y).
        Used as input features for the property prediction head.
        """
        return self.p_h_given_yf(y, f)

    # ──────────────────────────────────────────────────────────────────────
    # QUBO mapping (for QC-assisted training)
    # ──────────────────────────────────────────────────────────────────────

    def build_neg_phase_qubo(
        self, f_single: torch.Tensor
    ) -> torch.Tensor:
        """
        Build QUBO matrix Q for the joint (y, h) distribution given ONE
        fingerprint f_single (fp_dim,).

        The QUBO is:  min_{x} x^T Q x   where x = [y; h] ∈ {0,1}^{n_vis + n_hid}

        Mapping E → QUBO:
          Diagonal entries (linear terms):
            Q[i,i]          = -b_eff_i       (for y_i)
            Q[n_vis+j, n_vis+j] = -c_eff_j   (for h_j)
          Off-diagonal (coupling terms):
            Q[i, n_vis+j]   = -W[i,j]        (i < n_vis, j < n_hid; upper-tri)

        Returns: Q (n_vis + n_hid, n_vis + n_hid) float32 CPU tensor.
        """
        with torch.no_grad():
            f1     = f_single.unsqueeze(0)          # (1, fp_dim)
            b_eff  = (self.b + f1 @ self.W1)[0]     # (n_vis,)
            c_eff  = (self.c + f1 @ self.W2)[0]     # (n_hid,)

        n_vis = CFG.n_vis
        n_hid = CFG.n_hidden
        n_tot = n_vis + n_hid
        Q = torch.zeros(n_tot, n_tot)

        # Linear (diagonal) terms
        Q.diagonal()[:n_vis] = -b_eff.cpu()
        Q.diagonal()[n_vis:] = -c_eff.cpu()

        # Bilinear (upper-triangle) coupling terms
        W_cpu = self.W.detach().cpu()               # (n_vis, n_hid)
        for i in range(n_vis):
            for j in range(n_hid):
                Q[i, n_vis + j] = -W_cpu[i, j]

        return Q
