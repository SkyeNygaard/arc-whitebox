"""Minimal proposed K32 abstention feature.

Input is the post-ReLU layer-8 complete-Kerdock activation matrix ordered as
129 complete antipodal bases x 512 rows x 256 neurons. The feature is legal:
it uses only the candidate's already-computed Kerdock trajectory.
"""
from __future__ import annotations
import numpy as np

N_BASES=129
ROWS_PER_BASIS=512
WIDTH=256

def layer8_basis_fold_relative_dispersion(a8: np.ndarray) -> float:
    a8=np.asarray(a8)
    if a8.shape != (N_BASES*ROWS_PER_BASIS, WIDTH):
        raise ValueError(f"expected {(N_BASES*ROWS_PER_BASIS, WIDTH)}, got {a8.shape}")
    block_means=a8.reshape(N_BASES,ROWS_PER_BASIS,WIDTH).mean(axis=1,dtype=np.float64)
    fold_means=np.stack([block_means[idx].mean(axis=0) for idx in np.array_split(np.arange(N_BASES),6)])
    center=fold_means.mean(axis=0)
    return float(np.mean(np.linalg.norm(fold_means-center,axis=1)/(np.linalg.norm(center)+1e-12)))

def should_apply_k32(a8: np.ndarray, frozen_threshold: float) -> bool:
    return layer8_basis_fold_relative_dispersion(a8) <= frozen_threshold
