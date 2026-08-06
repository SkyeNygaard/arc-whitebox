"""Test D: the full static linear class over the 129 group means (validates T74 floor).
   Test E: is the error MAGNITUDE observable, even though its SIGN is not?
"""
import numpy as np

d = np.load("groups.npz")
G, Y = d["G"], d["Y"]
M = G.shape[0]
yhat = G.mean(axis=1)
E = yhat - Y
base = np.mean(E ** 2)
print(f"networks {M}   baseline MSE {base:.6e}\n")

# ------------------------------------------------------------------ Test D
print("=" * 78)
print("TEST D — best FIXED linear rule over the 129 group means (static class)")
print("=" * 78)
# stack every (network, neuron) as one observation with 129 regressors
A = G.transpose(0, 2, 1).reshape(M * 256, 129)      # (M*256, 129)
b = Y.reshape(M * 256)
grp = np.repeat(np.arange(M), 256)

w_uniform = np.full(129, 1 / 129)
print(f"  uniform rule (shipped)            MSE {np.mean((A@w_uniform-b)**2):.4e}   ratio 1.0000")

# in-sample optimum: the absolute ceiling of the static class
w_hat = np.linalg.lstsq(A, b, rcond=None)[0]
mse_in = np.mean((A @ w_hat - b) ** 2)
print(f"  in-sample optimal w (ceiling)     MSE {mse_in:.4e}   ratio {mse_in/base:.4f}")

# honest: leave-one-network-out
pred = np.zeros_like(b)
for m in range(M):
    tr, te = grp != m, grp == m
    w = np.linalg.lstsq(A[tr], b[tr], rcond=None)[0]
    pred[te] = A[te] @ w
mse_out = np.mean((pred - b) ** 2)
print(f"  leave-one-network-out             MSE {mse_out:.4e}   ratio {mse_out/base:.4f}")
print(f"  --> max static gain = {base/mse_out:.4f}x  (ledger T74 floor 0.93706 => 1.0672x)")

# ------------------------------------------------------------------ Test E
print("\n" + "=" * 78)
print("TEST E — is the error magnitude observable (even if the sign is not)?")
print("=" * 78)
s2 = G.var(axis=1)                                    # between-basis variance (M,256)
e2 = E ** 2
print(f"  corr( log between-basis var , log e^2 )  pooled : "
      f"{np.corrcoef(np.log(s2.ravel()+1e-30), np.log(e2.ravel()+1e-30))[0,1]:+.4f}")
# per-network mean level
print(f"  corr( mean_i s2 , MSE_m ) across networks       : "
      f"{np.corrcoef(s2.mean(1), e2.mean(1))[0,1]:+.4f}")
print(f"  implied var of the mean  (s2/129) vs actual e^2 : "
      f"ratio {np.mean(s2/129)/np.mean(e2):.2f}x")
print("\n  Magnitude is observable; sign is not. A magnitude-only signal cannot")
print("  reduce MSE without a direction to move in.")

# ------------------------------------------------------------------ Test F
print("\n" + "=" * 78)
print("TEST F — where does the score actually sit, and what would winning need?")
print("=" * 78)
FLOPS_PER_ROW = 170_875_096_064 / 66_048
BUDGET = 272_000_000_000
RESIDUAL = 0.056 * 1e11
C = 66048 * FLOPS_PER_ROW + RESIDUAL
mult = max(0.1, C / BUDGET)
score = base * mult
V = base * 66048
print(f"  V (variance constant)      = {V:.5e}")
print(f"  f (FLOPs per design row)   = {FLOPS_PER_ROW:.4e}")
print(f"  C/B                        = {mult:.4f}")
print(f"  adjusted score (all 100)   = {score:.5e}")
print(f"  score law  V*f/B           = {V*FLOPS_PER_ROW/BUDGET:.5e}   (should match)")
print()
for label, target in [("4.34x (leaderboard #1)", score / 4.34), ("80% reduction", score * 0.20)]:
    print(f"  to reach {label:<24} score {target:.4e} -> need V*f down {score/target:.2f}x")
print()
print("  Any Monte-Carlo-rate method has score = V*f/B, independent of row count.")
print("  So row count, per-network allocation and budget fraction are all NEUTRAL.")
print("  Only two levers exist: lower V (better design) or lower f (cheaper arithmetic).")
