"""Figures for the ceiling result."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "ceiling.json")))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

INK = "#16191f"
ACC = ["#1f6f8b", "#c1440e", "#2e7d32", "#6244a0", "#b8860b"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": "#999", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})

fig, ax = plt.subplots(1, 4, figsize=(14.6, 3.4))

# ---- 1. the spectrum -------------------------------------------------------
sh = np.array(R["spectrum_finite_d_256"])
k = np.arange(1, len(sh) + 1)
ax[0].bar(k[:20], sh[:20], color=ACC[0], width=0.72)
ax[0].axvspan(0.4, 5.5, color=ACC[2], alpha=0.13)
ax[0].text(3.0, sh[0] * 0.93, "annihilated by\nthe 5-design", ha="center",
           fontsize=7.5, color=ACC[2])
ax[0].annotate(f"{R['share_above_deg12']:.0%} of the variance\nsits above degree 12",
               (14, sh[13]), (11.5, sh[0] * 0.62), fontsize=7.5, color=ACC[1],
               arrowprops=dict(arrowstyle="->", color=ACC[1], lw=0.8))
ax[0].set(xlabel="spherical-harmonic degree", ylabel="share of Var$(f)$",
          title="Spectrum of a depth-32 ReLU net\n(closed form, no networks run)")

# ---- 2. what a design can remove ------------------------------------------
g = R["gains"]
names = ["i.i.d.", "antipodal\nonly", "Kerdock\n5-design", "7-design\n(deg$\\leq$6)",
         "9-design\n(deg$\\leq$8)"]
vals = [g["iid"], g["antipodal"], g["kerdock"], g["design_deg6"], g["design_deg8"]]
cols = [ACC[4], ACC[4], ACC[2], "#bbb", "#bbb"]
bars = ax[1].bar(range(5), vals, color=cols, width=0.66)
for i, (v, n) in enumerate(zip(vals, names)):
    ax[1].text(i, v + 0.03, f"{v:.2f}x", ha="center", fontsize=7.5)
ax[1].set_xticks(range(5))
ax[1].set_xticklabels(names, fontsize=7.5)
ax[1].set(ylabel="variance reduction vs i.i.d.", ylim=(0, 2.45),
          title="The design axis is finished")
ax[1].text(3.5, 1.30, "infeasible:\n86x and 5,547x\nthe budget", ha="center",
           fontsize=7.5, color=ACC[1])
ax[1].annotate("", xy=(2.35, 1.60), xytext=(2.0, 1.60),
               arrowprops=dict(arrowstyle="->", color=ACC[1], lw=0.9))
ax[1].text(2.0, 1.68, "+16% only", fontsize=7, color=ACC[1])

# ---- 3. the ladder ---------------------------------------------------------
steps = ["two-stream\nSobol", "Kerdock\n5-design", "+ Strassen\nL=3"]
scores = [3.4607e-7, 2.2566e-7, R["scores"]["kerdock_plus_strassen_L3"]]
kinds = [ACC[4], ACC[0], ACC[2]]
ax[2].plot(range(3), scores, "-", color="#aaa", lw=1, zorder=1)
ax[2].scatter(range(3), scores, s=90, c=kinds, zorder=3)
for i, s in enumerate(scores):
    ax[2].annotate(f"{s:.3e}", (i, s), textcoords="offset points",
                   xytext=(0, 12), ha="center", fontsize=7.5)
ax[2].set_xticks(range(3))
ax[2].set_xticklabels(steps, fontsize=7.5)
ax[2].set_yscale("log")
ax[2].set(ylabel="adjusted score", title="Ladder\n(statistics, then arithmetic)")
ax[2].text(1.5, 3.0e-7, "statistics\nexhausted here", fontsize=7.5, color=ACC[1],
           ha="center")

# ---- 4. parameter-free validation ------------------------------------------
labels = ["Kerdock\n(design)", "i.i.d.\n(median)"]
pred = [2.4011e-7, 3.7934e-7]
meas = [2.2826e-7, 4.1820e-7]
x = np.arange(2); w = 0.35
ax[3].bar(x - w/2, pred, w, color=ACC[0], label="predicted (closed form)")
ax[3].bar(x + w/2, meas, w, color=ACC[2], label="measured")
for i,(p_,m_) in enumerate(zip(pred,meas)):
    ax[3].text(i, max(p_,m_)*1.06, f"{abs(p_/m_-1):.0%}", ha="center", fontsize=8, color=ACC[1])
ax[3].set_xticks(x); ax[3].set_xticklabels(labels, fontsize=8)
ax[3].set(ylabel="final-layer MSE", ylim=(0, 5.2e-7),
          title="Absolute prediction,\nnothing fitted")
ax[3].legend(frameon=False, fontsize=7.5, loc="upper left")

fig.suptitle("The remaining problem is arithmetic, not statistics", fontsize=10.5, y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ceiling.png"), bbox_inches="tight")
print("wrote figures/ceiling.png")
