"""Decisive tests on the v26 lead path, using the validated 100-network group means.

Test A  score law:    is  score = V * f / B  (V = MSE x rows) flat in design size?
Test B  source:       oracle capacity of the adaptive direct-output PCA source.
Test C  estimability: can the source coefficients be predicted from legal data?
"""
import numpy as np

FLOPS_PER_ROW = 170_875_096_064 / 66_048  # measured tracked FLOPs per design row
BUDGET = 272_000_000_000
ROWS_PER_BASIS = 512

d = np.load("groups.npz")
G, Y, ids = d["G"], d["Y"], d["ids"]          # (M,129,256) (M,256)
M = G.shape[0]
yhat = G.mean(axis=1)
E = yhat - Y                                   # signed error, (M,256)
print(f"networks {M}   baseline MSE {np.mean(E**2):.6e}\n")

# ---------------------------------------------------------------- Test A
print("=" * 78)
print("TEST A — the score law:  score = MSE x C/B,  C = rows x flops_per_row")
print("=" * 78)
print(f"{'bases':>6} {'rows':>7} {'raw MSE':>12} {'V=MSE*rows':>12} {'mult':>7} {'score':>12} {'vs129':>7}")
rng = np.random.default_rng(0)
base_score = None
for k in [16, 24, 32, 48, 64, 80, 96, 112, 129]:
    if k == 129:
        sub = np.arange(129)
        mse = np.mean((G.mean(axis=1) - Y) ** 2)
    else:
        # average over random subsets to remove selection noise
        vals = []
        for _ in range(20):
            sub = rng.choice(129, size=k, replace=False)
            vals.append(np.mean((G[:, sub, :].mean(axis=1) - Y) ** 2))
        mse = float(np.mean(vals))
    rows = k * ROWS_PER_BASIS
    C = rows * FLOPS_PER_ROW
    mult = max(0.1, C / BUDGET)
    score = mse * mult
    if k == 129:
        base_score = score
    print(f"{k:>6} {rows:>7} {mse:>12.4e} {mse*rows:>12.4e} {mult:>7.4f} {score:>12.4e}"
          f" {score/ (base_score or score):>7.3f}")
print("\nIf V is ~constant, score is FLAT in design size: adding rows buys nothing.")

# ---------------------------------------------------------------- Test B
print("\n" + "=" * 78)
print("TEST B — adaptive direct-output PCA source: oracle capacity")
print("=" * 78)
D = G - yhat[:, None, :]                       # group deviations (M,129,256)

def source_basis(m, k):
    """Top-k right singular vectors of the network's own group deviations."""
    _, _, Vt = np.linalg.svd(D[m], full_matrices=False)
    return Vt[:k].T                            # (256,k)

for k in [8, 16, 24, 32, 36, 48, 64, 96, 128]:
    num = np.zeros(M)
    den = np.zeros(M)
    for m in range(M):
        U = source_basis(m, k)
        proj = U @ (U.T @ E[m])
        num[m] = np.sum((E[m] - proj) ** 2)
        den[m] = np.sum(E[m] ** 2)
    pooled = num.sum() / den.sum()
    per = num / den
    print(f"  rank {k:>3}:  pooled r* = {pooled:.4f}   median {np.median(per):.4f}"
          f"   worst {per.max():.4f}   (random-subspace ref {1-k/256:.4f})")

# ---------------------------------------------------------------- Test C
print("\n" + "=" * 78)
print("TEST C — estimability: are the source coefficients predictable at all?")
print("=" * 78)
K = 36
# Per (network, mode): target coefficient, and every legal scalar we can form.
rows_feat, targets, net_id = [], [], []
for m in range(M):
    Dm = D[m]
    Uh, S, Vt = np.linalg.svd(Dm, full_matrices=False)
    U = Vt[:K].T
    c = U.T @ E[m]                              # signed coefficients (K,)
    load = Uh[:, :K] * S[:K]                    # basis loadings (129,K)
    for j in range(K):
        lj = load[:, j]
        rows_feat.append([
            S[j], S[j] ** 2, np.log(S[j] + 1e-30), j, S[j] / S[0],
            lj.mean(), lj.std(), np.abs(lj).mean(), lj[-1],            # coord basis loading
            np.mean(lj ** 3) / (np.std(lj) ** 3 + 1e-30),              # loading skew
            np.mean(lj ** 4) / (np.std(lj) ** 4 + 1e-30),              # loading kurtosis
            float(np.dot(U[:, j], yhat[m])),                           # mode vs prediction
            float(np.dot(U[:, j], np.ones(256))) ,
            S[:K].sum(), np.linalg.norm(E[m]) * 0 + np.linalg.norm(Dm),
        ])
        targets.append(c[j])
        net_id.append(m)
X = np.asarray(rows_feat)
t = np.asarray(targets)
g = np.asarray(net_id)
X = (X - X.mean(0)) / (X.std(0) + 1e-30)
X = np.hstack([X, np.ones((len(X), 1))])

# leave-one-network-out ridge
def loro_r2(X, t, g, lam=1e-2):
    pred = np.zeros_like(t)
    for m in np.unique(g):
        tr, te = g != m, g == m
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        b = X[tr].T @ t[tr]
        w = np.linalg.solve(A, b)
        pred[te] = X[te] @ w
    ss_res = np.sum((t - pred) ** 2)
    ss_tot = np.sum(t ** 2)          # against zero: the honest "predict nothing" baseline
    return 1 - ss_res / ss_tot, pred

for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    r2, pred = loro_r2(X, t, g, lam)
    print(f"  ridge lam={lam:<6}  leave-one-network-out R^2 vs zero = {r2:+.4f}")

r2, pred = loro_r2(X, t, g, 1e-1)
print(f"\n  sign agreement of best predictor: {np.mean(np.sign(pred)==np.sign(t)):.4f}"
      f"  (chance 0.5)")

# What would the resulting MSE be if we applied the predicted correction?
new_mse = []
for m in range(M):
    U = source_basis(m, K)
    c_pred = pred[g == m]
    corr = U @ c_pred
    new_mse.append(np.mean((E[m] - corr) ** 2))
print(f"  MSE after applying LORO-predicted correction : {np.mean(new_mse):.4e}")
print(f"  baseline MSE                                 : {np.mean(E**2):.4e}")
print(f"  ratio                                        : {np.mean(new_mse)/np.mean(E**2):.4f}")
print(f"  oracle ratio at rank {K} (unreachable bound)  : "
      f"{sum(np.sum((E[m]-source_basis(m,K)@(source_basis(m,K).T@E[m]))**2) for m in range(M))/np.sum(E**2):.4f}")
