# Rank and sharing audit

## Exact registry

The frozen selector chooses 128 different rows from 256 without replacement. Consequently each network has exactly 128 distinct diagonal slots `M[i_p,i_p]` and 128 distinct row-direction slots `M[i_p,:] @ v_p`; there is no exact slot deduplication. They share one projection matrix `H @ V.T`, so all row-direction moments can be accumulated in one GEMM/reduction rather than 128 separate adjoints.

## Downstream-weighted rank

The exposed lower-structure corpus gives pooled shared ranks **29/36/44** for 90%/95%/99% energy. A universal rank-16 representation is therefore too small. Per network, however, the downstream-weighted lower matrix has median rank-2 energy **0.99994757**, and oracle local rank 2 has median anchor relative error **0.002620**.

This local rank is mechanism evidence, not a legal compression: its right space is network-specific and contains the unknown center-defect direction. A shared representation still needs roughly 30 modes.

## Precision audit

On the high-reference 8-network pair substitution, exact Gaussian pairs scored **0.889350** and primary Kerdock pairs scored **0.889914**, a ratio difference of only **0.000564**.

On the independent-center 24-network ablation, the 128-probe/full-companion arm changes from **0.726019** with primary pairs to **0.726008** with independent pairs. The median pair-source increment is only **0.174%** of correction norm, with median correction cosine **0.999999479**.

Thus pair precision is already far beyond what is needed for the 0.595 target. The unresolved precision requirement belongs to the center contractions, not to `s_p` or `t_p`.
