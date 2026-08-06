# Sparse cubic centering and layer-31 channel

## Status

The shared library contains the eight-network **0.190×** sparse exact-anchor claim, but not the frozen sparse implementation, probe arrays, high-precision moment corpus, or an untouched-network output file. Therefore no real ARC holdout result is claimed here. The attached script independently verifies the algebra and implements the required channel diagnostics.

## Exact sparse feature

Let \(Z\in\mathbb R^d\) be the Gaussian-input activation at the chosen late layer, with

\[
\mu=E[Z],\quad M=E[ZZ^T],\quad q_i=M_{ii},\quad R_{ij}=E[Z_i^2Z_j].
\]

Let \(h=\rho a(U)\) be the fixed-radius homogeneous cloud, where
\(\rho=E\|X\|\). Homogeneity gives

\[
E_U[h_i]=\mu_i,\quad E_U[h_ih_j]=\frac{\rho^2}{d}M_{ij},\quad
E_U[h_i^2h_j]=\frac{\rho^2}{d+1}R_{ij}.
\]

For center \(m\),

\[
\Phi_{ij}(h;m)=
\frac{h_i^2h_j}{\rho^2}
-\frac{d}{\rho^2(d+1)}\left(m_jh_i^2+2m_ih_ih_j\right)
+\frac{2m_i^2h_j}{d+1}.
\]

For rank-one probe \((u,v)\), \(\phi_{u,v}=u^T\Phi v\). Its anchor is

\[
A_{u,v}(m)=\frac{1}{d+1}\left[
K^{raw}_{u,v}
-(u^Tq)(v^Tm)
-2(u\odot m)^TMv
+2(u^Tm^{\odot2})(v^T\mu)
\right].
\]

Writing \(m=\mu+\delta\) and \(K^C_{u,v}=u^TC_{21}v\),

\[
A_{u,v}(m)=\frac{1}{d+1}\left[
K^C_{u,v}
-(u^Tq)(v^T\delta)
-2(u\odot\delta)^TMv
+4(u^T(\mu\odot\delta))(v^T\mu)
+2(u^T\delta^{\odot2})(v^T\mu)
\right].
\]

The first-order row is

\[
b_{u,v}=
\frac{1}{d+1}\left[
-(u^Tq)v-2\,\mathrm{diag}(u)Mv+4(v^T\mu)(u\odot\mu)
\right].
\]

For \(u=e_a,v=e_b\),

\[
A_{ab}(\mu+\delta)-A_{ab}(\mu)=
\frac{-q_a\delta_b-2M_{ab}\delta_a
+4\mu_a\mu_b\delta_a+2\mu_b\delta_a^2}{d+1}.
\]

## Structural answer

Stacking all 128 first-order rows gives \(B\in\mathbb R^{128\times256}\):

\[
e_{\mathrm{center}}=B\delta+Q(\delta^{\odot2}).
\]

Therefore the mean information is shared and at most 256-dimensional. For coordinate-sparse probes it is bounded by the union of coordinates used by the probes and the span of their \(v\)-directions. It is not 128 independent mean anchors.

The lower-order correction also needs selected second-moment contractions:
\(u_p^Tq\) and \((u_p\odot m)^TMv_p\). Depending on the actual frozen probes, these may collapse to a small shared collection or remain roughly 128 probe-specific scalars. The probe arrays are required to measure that empirical dimension.

## Frozen real holdout

1. Hash and preserve the current sparse script/configuration.
2. Preserve layer, 128 probes, Kerdock rotation, folds, regularization, coefficient fitting, and shrinkage.
3. Exclude every network used for probe creation, selection, shrinkage, M83–M86 tuning, or the activation-region work.
4. Run at least 24 still-protected networks.
5. Report summed-MSE ratio, network-bootstrap interval, wins, median, and worst ratio.
6. Continue only below 0.50×, preferably below 0.30×.

## Same-channel diagnostic

For each untouched network save:

- sparse exact-anchor correction \(c_S\);
- full layer-31 translation correction \(c_{31}\);
- exact single-neuron replay vectors \(g_j\);
- top-8, 12, 16, and 32 simultaneous replay corrections.

Report

\[
\cos(c_S,c_{31}),\qquad
\frac{\|P_{G_K}c_S\|^2}{\|c_S\|^2},
\]

and the MSE gain from adding only \((I-P_{G_K})c_S\) after the top-\(K\) layer-31 oracle. Call them the same channel only if the 32-neuron span explains at least 80% of sparse-correction energy, median cosine exceeds 0.8, and residualized sparse correction adds little.

## Center/cubic ablations

| Pointwise center | Cubic anchor | Lower-order recentering |
|---|---|---|
| oracle | oracle | zero |
| oracle | approximate | zero |
| sample | oracle | oracle moments |
| propagated | oracle | oracle moments |
| independent suffix | oracle | oracle moments |
| sample | oracle | omitted |
| sample | approximate | deployable moments |
| propagated | approximate | deployable moments |
| independent suffix | approximate | independent moments |

The “sample center, recentering omitted” row isolates the reported missing correction. A complete same-cloud raw-moment anchor is degenerate because it reconstructs its own cloud mean.

## Synthetic verification

```json
{
  "seed": 7,
  "dimension": 20,
  "probes": 128,
  "algebra": {
    "raw_vs_connected_anchor_relative_error": 3.2551705886383743e-15,
    "independent_sphere_pointwise_anchor_relative_error": 0.045665722461974596,
    "center_expansion_relative_error": 1.4216837454903088e-15
  },
  "center_map": {
    "matrix_rank": 19,
    "rank90": 6,
    "rank99": 9,
    "shared_error_rank90_over_300_perturbations": 6,
    "top_singular_values": [
      0.6164418673904057,
      0.41372063468819464,
      0.3999331658893365,
      0.3084719545936055,
      0.2799255783575762,
      0.2616172076766607,
      0.16899820991442221,
      0.1556253317244127,
      0.11859909937921409,
      0.06829947454152611
    ]
  },
  "control": {
    "baseline_mse": 0.00010619394118136947,
    "exact_anchor_mse_ratio": 0.23949783936797658,
    "missing_recentering_mse_ratio": 0.8456012031459715,
    "propagated_center_0p65pct_mse_ratio": 0.24194462542149442,
    "approx_cubic_10pct_error_mse_ratio": 0.13351268436568633
  },
  "tolerance_curve": [
    {
      "center_error_scale": 0.0,
      "relative_center_error": 0.0,
      "mse_ratio": 0.24182397123696842
    },
    {
      "center_error_scale": 0.25,
      "relative_center_error": 0.0021628182951542304,
      "mse_ratio": 0.2412561139850303
    },
    {
      "center_error_scale": 0.5,
      "relative_center_error": 0.00432563659030854,
      "mse_ratio": 0.24067893517787786
    },
    {
      "center_error_scale": 0.75,
      "relative_center_error": 0.0064884548854628205,
      "mse_ratio": 0.24009273908297657
    },
    {
      "center_error_scale": 1.0,
      "relative_center_error": 0.008651273180617052,
      "mse_ratio": 0.23949783936797517
    },
    {
      "center_error_scale": 1.25,
      "relative_center_error": 0.010814091475771281,
      "mse_ratio": 0.2388945587435751
    },
    {
      "center_error_scale": 1.5,
      "relative_center_error": 0.012976909770925563,
      "mse_ratio": 0.2382832285940571
    },
    {
      "center_error_scale": 2.0,
      "relative_center_error": 0.017302546361234104,
      "mse_ratio": 0.23703778632586167
    }
  ],
  "shrinkage": [
    {
      "alpha": 0.0,
      "exact_anchor_ratio": 1.0,
      "propagated_center_ratio": 1.0
    },
    {
      "alpha": 0.25,
      "exact_anchor_ratio": 0.7575552305413187,
      "propagated_center_ratio": 0.7584060734879485
    },
    {
      "alpha": 0.5,
      "exact_anchor_ratio": 0.549989947283096,
      "propagated_center_ratio": 0.551532202220853
    },
    {
      "alpha": 0.7,
      "exact_anchor_ratio": 0.40905095074083564,
      "propagated_center_ratio": 0.41103154498353855
    },
    {
      "alpha": 1.0,
      "exact_anchor_ratio": 0.23949783936797658,
      "propagated_center_ratio": 0.24194462542149442
    }
  ],
  "channel_identity_synthetic": {
    "cosine_sparse_vs_full_layer31": 0.9858536567855511,
    "sparse_projection_energy_in_all_layer31_neuron_span": 0.9999999999999986,
    "top_k": {
      "4": {
        "cosine_sparse_vs_layer31": 0.9455421475400205,
        "sparse_projection_energy_in_neuron_span": 0.9260003291044745,
        "layer31_mse_ratio": 0.1708419678345843
      },
      "8": {
        "cosine_sparse_vs_layer31": 0.9899343449698231,
        "sparse_projection_energy_in_neuron_span": 0.9988918974772527,
        "layer31_mse_ratio": 0.011008319696360052
      },
      "12": {
        "cosine_sparse_vs_layer31": 0.990925485084429,
        "sparse_projection_energy_in_neuron_span": 0.9993963696611359,
        "layer31_mse_ratio": 0.008563657454821814
      },
      "16": {
        "cosine_sparse_vs_layer31": 0.9907772282760399,
        "sparse_projection_energy_in_neuron_span": 0.9999959563081032,
        "layer31_mse_ratio": 0.008962323154398554
      }
    },
    "note": "Synthetic diagnostic only. The real ARC conclusion requires the frozen repository probes and untouched networks."
  }
}
```

These channel values are a unit test only, not evidence about the real ARC networks.
