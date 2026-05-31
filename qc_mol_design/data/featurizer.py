"""
Molecular featurisation: atom features, graph conversion, PyTorch Dataset.

GraphConv architecture (Duvenaud et al. 2015):
  - Atoms as nodes with 75-dim feature vectors
  - Bonds as edges; adjacency stored as dense matrix
  - Degree per node required for degree-specific weight matrices
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from qc_mol_design.config import CFG
from qc_mol_design.data.loader import discretize, make_label_vector

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Atom feature vector  (75-dim, standard DeepChem convention)
# ─────────────────────────────────────────────────────────────────────────────

_ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
    "Ca", "Fe", "As", "Al", "I",  "B",  "V",  "K",  "Tl", "Yb",
    "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",  "Li",
    "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt",
    "Hg", "Pb",
]  # 43 named types + 1 "other" = 44

_HYBRIDIZATION = None  # Imported lazily to avoid top-level rdkit dependency


def _get_hybrid_types():
    global _HYBRIDIZATION
    if _HYBRIDIZATION is None:
        from rdkit.Chem import rdchem
        _HYBRIDIZATION = [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
        ]
    return _HYBRIDIZATION


def _one_hot(x, vocabulary: list) -> List[int]:
    return [int(x == v) for v in vocabulary]


def _implicit_valence(atom) -> int:
    """RDKit-version-agnostic implicit valence query."""
    try:
        from rdkit.Chem import ValenceType
        return atom.GetValence(ValenceType.IMPLICIT)
    except Exception:
        try:
            return atom.GetImplicitValence()
        except Exception:
            return 0


def atom_features(atom) -> np.ndarray:
    """
    75-dimensional atom feature vector.

    Composition:
      44  atom type one-hot (43 named + 1 other)
      11  degree one-hot  (0 … 10)
      11  implicit valence one-hot (0 … 10)
       1  formal charge
       5  total H count one-hot (0 … 4)
       1  aromaticity flag
       3  hybridisation one-hot (SP / SP2 / SP3)
     ── 76 total → clipped to 75 (drop last hybridisation entry if needed)
    """
    hybrid = _get_hybrid_types()
    feats: List[int] = (
        _one_hot(atom.GetSymbol(), _ATOM_TYPES) + [int(atom.GetSymbol() not in _ATOM_TYPES)]
        + _one_hot(atom.GetDegree(), list(range(11)))
        + _one_hot(_implicit_valence(atom), list(range(11)))
        + [int(atom.GetFormalCharge())]
        + _one_hot(atom.GetTotalNumHs(), list(range(5)))
        + [int(atom.GetIsAromatic())]
        + _one_hot(atom.GetHybridization(), hybrid)
    )
    arr = np.array(feats, dtype=np.float32)
    return arr[:CFG.atom_feat_dim]  # clip / pad to exact atom_feat_dim


# ─────────────────────────────────────────────────────────────────────────────
# SMILES → graph tensors
# ─────────────────────────────────────────────────────────────────────────────

GraphTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]
# (X: N×atom_dim, adj: N×N, deg: N)


def mol_to_graph(smiles: str) -> Optional[GraphTuple]:
    """
    Convert a SMILES string to graph tensors.

    Returns (X, adj, deg) or None if SMILES is invalid.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    X = np.zeros((n, CFG.atom_feat_dim), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        feat = atom_features(atom)
        X[i, : len(feat)] = feat

    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[u, v] = adj[v, u] = 1.0

    deg = adj.sum(axis=1).astype(np.int32)
    return X, adj, deg


def graph_from_atoms_adj(
    atom_symbols: List[str], adj: np.ndarray
) -> GraphTuple:
    """
    Build graph tensors from a list of atom symbols and an adjacency matrix.
    Used during molecular generation when we have explicit atom types.
    """
    from rdkit import Chem

    n = len(atom_symbols)
    X = np.zeros((n, CFG.atom_feat_dim), dtype=np.float32)

    for i, sym in enumerate(atom_symbols):
        # Create a minimal RDKit atom to extract features
        rw = Chem.RWMol()
        rw.AddAtom(Chem.Atom(sym))
        try:
            rw.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        feat = atom_features(rw.GetAtomWithIdx(0))
        X[i, : len(feat)] = feat

    adj_f = adj.astype(np.float32)
    deg   = adj_f.sum(axis=1).astype(np.int32)
    return X, adj_f, deg


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MoleculeDataset(Dataset):
    """
    Pre-builds graph tensors and multi-property label vectors for all molecules.

    Each item: (X, adj, deg, y_vec, props_continuous)
      X              (N, atom_dim) float32
      adj            (N, N)        float32
      deg            (N,)          int32
      y_vec          (n_vis,)      float32  — concatenated one-hot labels
      props_cont     (3,)          float32  — [qed, logp, sas] continuous values
    """

    def __init__(self, data: pd.DataFrame, desc: str = "") -> None:
        self.records: List[tuple] = []
        for _, row in tqdm(data.iterrows(), total=len(data),
                           desc=f"[DATASET] {desc or 'Building graphs'}"):
            g = mol_to_graph(str(row["smiles"]))
            if g is None:
                continue
            X, adj, deg = g
            y_vec  = make_label_vector(
                float(row["qed"]), float(row["logp"]), float(row["sas"])
            )
            props  = np.array([row["qed"], row["logp"], row["sas"]], dtype=np.float32)
            self.records.append((X, adj, deg, y_vec, props))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Variable-length graph collation
# ─────────────────────────────────────────────────────────────────────────────

def collate_graphs(batch):
    """
    Pad a batch of variable-size graphs to the same N_max.

    Returns:
        X       (B, N_max, atom_dim)  float32
        adj     (B, N_max, N_max)     float32
        deg     (B, N_max)            int64
        y_vec   (B, n_vis)            float32
        props   (B, 3)                float32
    """
    max_n = max(x[0].shape[0] for x in batch)

    Xs, adjs, degs, yvecs, propss = [], [], [], [], []
    for X, adj, deg, y_vec, props in batch:
        n   = X.shape[0]
        pad = max_n - n
        Xs.append(np.pad(X,   ((0, pad), (0, 0))))
        adjs.append(np.pad(adj, ((0, pad), (0, pad))))
        degs.append(np.pad(deg, (0, pad)))
        yvecs.append(y_vec)
        propss.append(props)

    return (
        torch.FloatTensor(np.stack(Xs)),
        torch.FloatTensor(np.stack(adjs)),
        torch.LongTensor(np.stack(degs)),
        torch.FloatTensor(np.stack(yvecs)),
        torch.FloatTensor(np.stack(propss)),
    )


def make_dataloader(dataset: MoleculeDataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        collate_fn=collate_graphs,
        num_workers=0,
    )
