"""Generate result figures."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

INK = "#1c1c1c"
ACC = ["#0b6e99", "#c1440e", "#2e7d32", "#8e44ad", "#b8860b"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": "#999", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- figure 1
def fig_structure():
    S = load("structure_256x32.json")
    l = [r["layer"] for r in S]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))

    ax[0].semilogy(l, [r["eff_rank"] for r in S], "o-", ms=3, color=ACC[0])
    ax[0].set(xlabel="layer", ylabel="effective rank of Cov$(a_\\ell)$",
              title="Rank collapse")
    ax[0].axhline(3, ls=":", color=ACC[1])
    ax[0].annotate("2.7 by layer 32", (32, 2.7), (18, 6), color=ACC[1],
                   arrowprops=dict(arrowstyle="->", color=ACC[1], lw=0.8), fontsize=8)

    ax[1].plot(l, [r["h_skew_rms"] for r in S], "o-", ms=3, color=ACC[0], label="RMS skewness")
    ax[1].plot(l, [r["h_exkurt_mean"] for r in S], "s-", ms=3, color=ACC[1],
               label="mean excess kurtosis")
    ax[1].set(xlabel="layer", ylabel="cumulant of $h_\\ell$",
              title="Non-Gaussianity grows with depth")
    ax[1].legend(frameon=False, fontsize=8)

    ax[2].plot(l, [r["frac_var_linear_in_x"] for r in S], "o-", ms=3, color=ACC[2],
               label="frac. Var linear in $x$")
    ax[2].plot(l, [r["top_eig_frac"] for r in S], "^-", ms=3, color=ACC[3],
               label="top eigenvalue share")
    ax[2].set(xlabel="layer", ylabel="fraction", title="Where the variance lives")
    ax[2].legend(frameon=False, fontsize=8)

    fig.suptitle("The pushforward of $N(0,I_{256})$ collapses onto a low-dimensional, "
                 "non-Gaussian object", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "01_structure.png"), bbox_inches="tight")


# ---------------------------------------------------------------- figure 2
def fig_error_anatomy():
    D = load("decompose_s0.json")
    S = load("sensitivity_s0.json")
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    l = np.arange(1, 33)

    ax[0].semilogy(l, D["full"], "o-", ms=3, color=ACC[1], label="GaussProp (all propagated)")
    ax[0].semilogy(l, D["oracle_moments"], "s-", ms=3, color=ACC[0],
                   label="oracle $(\\mu,\\Sigma)$ every layer")
    ax[0].axhline(9.6e-7, ls="--", color=ACC[2], lw=1)
    ax[0].text(2, 1.15e-6, "plain MC @ half budget", color=ACC[2], fontsize=7.5)
    ax[0].set(xlabel="layer", ylabel="MSE", title="50$\\times$ of the error is\nmoment propagation")
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")

    ax[1].plot(l, S["sensitivity"], "o-", ms=3, color=ACC[0])
    ax[1].set(xlabel="layer $\\ell$", ylabel="$\\|\\partial Y_L/\\partial Y_\\ell\\|_F/\\sqrt{n}$",
              title="Errors injected early are damped")
    ax[1].annotate("16$\\times$ damping\nfrom layer 1", (1, 0.063), (5, 0.45),
                   fontsize=8, color=ACC[1],
                   arrowprops=dict(arrowstyle="->", color=ACC[1], lw=0.8))

    ks = sorted(int(k) for k in S["hybrid_oracle"])
    vs = [S["hybrid_oracle"][str(k)] for k in ks]
    ax[2].semilogy(ks, vs, "o-", ms=4, color=ACC[3])
    ax[2].set(xlabel="oracle moments used for layers $1..k$", ylabel="final-layer MSE",
              title="Error accrues at every layer")
    ax[2].axhline(9.6e-7, ls="--", color=ACC[2], lw=1)
    ax[2].text(1, 1.15e-6, "plain MC", color=ACC[2], fontsize=7.5)

    fig.suptitle("Anatomy of the white-box error", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "02_error_anatomy.png"), bbox_inches="tight")


# ---------------------------------------------------------------- figure 3
def fig_predictability():
    P = load("predictability_s0.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))

    inp = [(k, v[1]) for k, v in P.items() if "active dirs" in k or "||x||" in k
           or "256 x-coords" in k]
    ax[0].barh(range(len(inp)), [v for _, v in inp], color=ACC[0], height=0.62)
    ax[0].set_yticks(range(len(inp)))
    ax[0].set_yticklabels([k for k, _ in inp], fontsize=7)
    ax[0].set(xlim=(0, 1), xlabel="$R^2$ predicting $a_L$",
              title="From the input $x$: hopeless\n(no low-degree structure)")

    mids = [(int(k.split("_")[1].split(" ")[0]), v[1]) for k, v in P.items()
            if k.startswith("linear in a_")]
    mids.sort()
    ax[1].plot([m for m, _ in mids], [v for _, v in mids], "o-", ms=5, color=ACC[2])
    ax[1].set(xlabel="layer $\\ell$ used as predictor", ylabel="$R^2$ predicting $a_L$",
              ylim=(0.6, 1.02), title="From layer $\\ell$: almost perfect\n"
              "($R^2=0.991$ at $\\ell=31$)")
    for m, v in mids:
        ax[1].annotate(f"{v:.3f}", (m, v), textcoords="offset points",
                       xytext=(0, -13), ha="center", fontsize=7)

    fig.suptitle("Is the final-layer fluctuation predictable? (decides which control "
                 "variates can work)", fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "03_predictability.png"), bbox_inches="tight")


# ---------------------------------------------------------------- figure 4
def fig_scoreboard():
    R = load("final_bench.json")
    names = sorted({r["name"] for r in R})
    agg = {nm: (float(np.mean([r["score"] for r in R if r["name"] == nm])),
                float(np.mean([r["flops"] for r in R if r["name"] == nm])),
                [r["kind"] for r in R if r["name"] == nm][0]) for nm in names}
    order = sorted(agg, key=lambda k: agg[k][0])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    cols = {"mc": ACC[0], "whitebox": ACC[1], "hybrid": ACC[3]}
    vals = [agg[k][0] for k in order]
    ax[0].barh(range(len(order)), vals, color=[cols[agg[k][2]] for k in order], height=0.66)
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels(order, fontsize=7.5)
    ax[0].set_xscale("log")
    ax[0].set(xlabel="leaderboard score  (mean over 4 MLPs, lower is better)",
              title="Scoreboard")
    for v in (1.24e-8, 2.30e-8, 5.35e-8):
        ax[0].axvline(v, ls=":", color="#555", lw=0.9)
    ax[0].text(1.24e-8, len(order) - 0.2, " AIcrowd #1", fontsize=7.5, color="#555")
    ax[0].invert_yaxis()

    for k in order:
        s, f, kind = agg[k]
        ax[1].scatter(f, s, s=34, color=cols[kind], zorder=3, alpha=0.85)
    # label only the best of each family, plus the worst white-box
    for k, dx, dy in [(order[0], 8, -14), ("GaussProp[exact]", 8, 4),
                      ("ASGM[r=32,K=4096]", 8, -14), ("GaussProp[diag]", 8, 4)]:
        s, f, _ = agg[k]
        ax[1].annotate(k, (f, s), textcoords="offset points", xytext=(dx, dy), fontsize=7)
    B = 2.72e11
    ax[1].axvline(B / 10, ls="--", color="#777", lw=1)
    ax[1].text(B / 10 * 1.15, 3e-4, "score floor:\n$C=B/10$", fontsize=7.5, color="#555")
    ax[1].axhline(1.24e-8, ls=":", color="#555", lw=0.9)
    ax[1].text(2e9, 1.4e-8, "AIcrowd #1", fontsize=7.5, color="#555")
    ax[1].set(xscale="log", yscale="log", xlabel="FLOPs used $C$", ylabel="score",
              title="Score vs compute", ylim=(6e-9, 3e-3))

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n)
               for n, c in [("Monte Carlo family", ACC[0]),
                            ("white-box (moment prop.)", ACC[1]),
                            ("hybrid (ASGM)", ACC[3])]]
    ax[1].legend(handles=handles, frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "04_scoreboard.png"), bbox_inches="tight")


# ---------------------------------------------------------------- figure 5
def fig_variance():
    R = load("final_bench.json")
    seeds = sorted({r["seed"] for r in R})
    modes = ["iid", "anti", "sphere", "anti_sphere"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.35
    base = np.array([np.mean([r["mse"] for r in R if r["name"] == "MC[plain, iid]"
                              and r["seed"] == s]) for s in seeds])
    for j, est in enumerate(("plain", "anchored")):
        red = []
        for m in modes:
            v = np.array([np.mean([r["mse"] for r in R
                                   if r["name"] == f"MC[{est}, {m}]" and r["seed"] == s])
                          for s in seeds])
            red.append(float(np.mean(base / v)))
        ax.bar(np.arange(len(modes)) + (j - 0.5) * w, red, w,
               color=ACC[j], label=est)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(["i.i.d.", "antithetic", "sphere", "antithetic\n+ sphere"])
    ax.axhline(1, color="#777", lw=0.8)
    ax.set(ylabel="MSE reduction vs plain i.i.d. MC",
           title="Variance reduction actually achieved (mean over 4 MLPs)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "05_variance_reduction.png"), bbox_inches="tight")


# ---------------------------------------------------------------- figure 6
def fig_edgeworth():
    E = load("edgeworth.json")
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.3))

    lad = E["oracle_marginal_ladder"]
    ks = list(lad)
    ax[0].bar(range(len(ks)), [lad[k] for k in ks],
              color=[ACC[1], ACC[4], ACC[2], ACC[3]], width=.62)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(ks)))
    ax[0].set_xticklabels(ks, rotation=18, ha="right", fontsize=7.5)
    ax[0].axhline(1.24e-8 / 0.1, ls=":", color="#555", lw=1)
    ax[0].text(-.4, 1.5e-7, "MSE needed to match AIcrowd #1\n(at $\\leq$10% budget)",
               fontsize=7, color="#555")
    ax[0].axhline(E["oracle_noise_floor"], ls="--", color=ACC[0], lw=1)
    ax[0].text(-.4, 2.4e-8, "my oracle-noise floor", fontsize=7, color=ACC[0])
    ax[0].set(ylabel="final-layer MSE", title="Edgeworth marginals\n(oracle moments)")

    P = E["precision"]
    for (nm, col, mk) in [("mu", ACC[1], "o"), ("sigma", ACC[0], "s"),
                          ("kappa3", ACC[2], "^"), ("kappa4", ACC[3], "d")]:
        xs = [p[0] for p in P[nm]]
        ys = [p[1] for p in P[nm]]
        ax[1].loglog(xs, ys, mk + "-", ms=4, color=col,
                     label={"mu": "$\\mu$", "sigma": "$\\sigma$",
                            "kappa3": "$\\kappa_3$", "kappa4": "$\\kappa_4$"}[nm])
    ax[1].axhline(1.24e-7, ls=":", color="#555", lw=1)
    ax[1].text(1.3e-4, 1.6e-7, "beats #1 below this line", fontsize=7, color="#555")
    ax[1].set(xlabel="relative error injected", ylabel="final-layer MSE",
              title="Precision asymmetry:\ncumulants barely need to be right")
    ax[1].legend(frameon=False, fontsize=8, ncol=2)

    ab = E["ablation"]
    ks = list(ab)
    ax[2].barh(range(len(ks)), [ab[k] for k in ks],
               color=[ACC[1], ACC[1], ACC[4], ACC[0], ACC[2]], height=.6)
    ax[2].set_xscale("log")
    ax[2].set_yticks(range(len(ks)))
    ax[2].set_yticklabels(ks, fontsize=7)
    ax[2].axvline(1.24e-7, ls=":", color="#555", lw=1)
    ax[2].text(1.4e-7, -0.45, "#1", fontsize=7.5, color="#555")
    ax[2].invert_yaxis()
    ax[2].set(xlabel="final-layer MSE", title="What's left is\nmoment propagation")

    fig.suptitle("Edgeworth Moment Propagation: the marginal model is solved; "
                 "the moments are not", fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "06_edgeworth.png"), bbox_inches="tight")


if __name__ == "__main__":
    for fn in (fig_structure, fig_error_anatomy, fig_predictability,
               fig_scoreboard, fig_variance, fig_edgeworth):
        try:
            fn()
            print("ok:", fn.__name__)
        except Exception as e:
            print("FAILED:", fn.__name__, type(e).__name__, e)


