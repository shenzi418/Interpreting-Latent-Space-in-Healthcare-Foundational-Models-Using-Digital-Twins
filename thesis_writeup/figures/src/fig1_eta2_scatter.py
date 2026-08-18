"""Fig 1 — informativeness-fidelity map: per-feature territory eta^2 in MedalCare-XL (x) vs PTB-XL (y).

Reads ONLY the frozen F1 artifact (outputs/analysis/fidelity_audit/f1_fidelity.json, byte-identical to
reports/2026-08-13_audit_artifacts/tmp_f1_fidelity.json). No computation beyond plotting.

Run from repo root:  python thesis_writeup/figures/src/fig1_eta2_scatter.py
Writes: thesis_writeup/figures/fig1_eta2_scatter.pdf (+ .png preview) and figures/fig1_points.csv (trace table).
"""
import json, os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "outputs", "analysis", "fidelity_audit", "f1_fidelity.json")
OUT_DIR = os.path.join(HERE, "..")

d = json.load(open(SRC))
feats = d["features"]
axis = d["axis"]
blind = set(d["lists"]["blind_spots_4c"])
spur = set(d["lists"]["spurious_channels_4c"])

# fixed categorical order + validated palette (dataviz reference instance, light surface)
BLOCKS = [("ST_J60", "ST$_{J60}$ (12)", "#2a78d6", "o"),
          ("Q_amp", "Q$_{amp}$ (12)", "#eb6834", "s"),
          ("R_amp", "R$_{amp}$ (12)", "#1baf7a", "D"),
          ("T_amp", "T$_{amp}$ (12)", "#eda100", "^"),
          ("globals", "intervals/globals (6)", "#e87ba4", "v")]

def block_of(name):
    for key, *_ in BLOCKS:
        if name.startswith(key):
            return key
    return "globals"

rows = []
for f in feats:
    if f["name"] == "T_amplitude_mV":      # exact duplicate of T_amp_II in both domains (53 distinct) — plot once
        continue
    rows.append(dict(name=f["name"], block=block_of(f["name"]),
                     xs=f["eta2_4c_sim"], xlo=f["eta2_4c_sim_ci"][0], xhi=f["eta2_4c_sim_ci"][1],
                     yr=f["eta2_4c_real"], ylo=f["eta2_4c_real_ci"][0], yhi=f["eta2_4c_real_ci"][1],
                     verdict=("blind spot" if f["name"] in blind else "spurious" if f["name"] in spur else "")))

fig, ax = plt.subplots(figsize=(8.2, 6.0))
lim = 0.36
ax.plot([0, lim], [0, lim], color="#9a9a94", lw=1, ls="--", zorder=1)
ax.text(0.155, 0.175, r"$\eta^2_{\mathrm{real}}=\eta^2_{\mathrm{sim}}$", ha="right", va="bottom",
        fontsize=8, color="#6f6f69", rotation=45, rotation_mode="anchor")

for key, label, col, mk in BLOCKS:
    sub = [r for r in rows if r["block"] == key]
    xs = np.array([r["xs"] for r in sub]); ys = np.array([r["yr"] for r in sub])
    xerr = np.array([[r["xs"] - r["xlo"] for r in sub], [r["xhi"] - r["xs"] for r in sub]])
    yerr = np.array([[r["yr"] - r["ylo"] for r in sub], [r["yhi"] - r["yr"] for r in sub]])
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="none", ecolor=col, elinewidth=0.8, alpha=0.55, capsize=0, zorder=2)
    ax.scatter(xs, ys, s=38, marker=mk, facecolor=col, edgecolor="white", linewidth=0.8, zorder=3, label=label)

# the frontal QRS axis (circular eta^2) — starred
ax.errorbar([axis["eta2_4c_sim"]], [axis["eta2_4c_real"]],
            xerr=[[axis["eta2_4c_sim"] - axis["eta2_4c_sim_ci"][0]], [axis["eta2_4c_sim_ci"][1] - axis["eta2_4c_sim"]]],
            yerr=[[axis["eta2_4c_real"] - axis["eta2_4c_real_ci"][0]], [axis["eta2_4c_real_ci"][1] - axis["eta2_4c_real"]]],
            fmt="none", ecolor="#222", elinewidth=0.9, zorder=4)
ax.scatter([axis["eta2_4c_sim"]], [axis["eta2_4c_real"]], s=170, marker="*", facecolor="#222", edgecolor="white",
           linewidth=0.8, zorder=5, label="frontal QRS axis (circular $\\eta^2$)")

# selective direct labels: the largest blind spots / spurious channels + axis
def lab(name, dx, dy, ha="left"):
    r = next((r for r in rows if r["name"] == name), None)
    if r is None:
        return
    ax.annotate(name.replace("_", "\\_") if False else name, (r["xs"], r["yr"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5, ha=ha, color="#333",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
lab("Q_amp_III", 10, -2)
lab("Q_amp_aVF", 10, -2)
lab("Q_amp_II", 14, 6)
lab("R_amp_aVL", 18, -14)
lab("R_amp_III", 18, 0)
lab("ST_J60_aVF", 4, -14)
lab("ST_J60_I", 4, 8)
lab("T_amp_V4", 8, 4)
lab("QRS_duration_ms", 10, -6)
ax.annotate("frontal QRS axis", (axis["eta2_4c_sim"], axis["eta2_4c_real"]), xytext=(4, -34),
            textcoords="offset points", fontsize=8, color="#222", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))

# region annotations
ax.text(0.0009, 0.30, "real-informative, sim-flat:" + chr(10) + "blind spots (28 + axis)", fontsize=8, color="#555", va="top", ha="left")
ax.text(0.30, 0.0004, "sim-informative, real-flat:" + chr(10) + "spurious (14 columns)", fontsize=8, color="#555", ha="right", va="bottom")

# square-root scale on both axes: spreads the dense low-eta^2 cluster; the diagonal remains y = x
fwd = lambda v: np.sqrt(np.clip(v, 0, None)); inv = lambda v: np.square(v)
ax.set_xscale("function", functions=(fwd, inv)); ax.set_yscale("function", functions=(fwd, inv))
ticks = [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3]
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xticklabels([str(t) for t in ticks], fontsize=8); ax.set_yticklabels([str(t) for t in ticks], fontsize=8)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_aspect("equal")
ax.set_xlabel("territory information in MedalCare-XL   ($\\eta^2$ vs 4-class territory, 500-draw run-block bootstrap 95% CI)")
ax.set_ylabel("territory information in PTB-XL   ($\\eta^2$ vs 4-class territory, patient-block bootstrap 95% CI)")
ax.xaxis.label.set_size(8.5); ax.yaxis.label.set_size(8.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, color="#ececea", lw=0.6, zorder=0)
ax.set_axisbelow(True)
leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False, handletextpad=0.4, borderaxespad=0.0, title="feature block", title_fontsize=8)
ax.set_title("Per-feature informativeness fidelity: 53 ECG features + the frontal QRS axis", fontsize=10, loc="left")
fig.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "fig1_eta2_scatter.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig1_eta2_scatter.png"), dpi=150)

# trace table
with open(os.path.join(OUT_DIR, "fig1_points.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "block", "eta2_sim", "sim_lo", "sim_hi", "eta2_real", "real_lo", "real_hi", "verdict_4c"])
    for r in rows:
        w.writerow([r["name"], r["block"], f'{r["xs"]:.5f}', f'{r["xlo"]:.5f}', f'{r["xhi"]:.5f}', f'{r["yr"]:.5f}', f'{r["ylo"]:.5f}', f'{r["yhi"]:.5f}', r["verdict"]])
    w.writerow(["frontal_QRS_axis", "axis", f'{axis["eta2_4c_sim"]:.5f}', f'{axis["eta2_4c_sim_ci"][0]:.5f}', f'{axis["eta2_4c_sim_ci"][1]:.5f}',
                f'{axis["eta2_4c_real"]:.5f}', f'{axis["eta2_4c_real_ci"][0]:.5f}', f'{axis["eta2_4c_real_ci"][1]:.5f}', "blind spot"])
print("points plotted:", len(rows), "+ axis; blind spots in list:", len(blind), "spurious:", len(spur))
print("blind list contains axis/dup?", [n for n in blind if n not in {r['name'] for r in rows}])
print("spurious list contains axis/dup?", [n for n in spur if n not in {r['name'] for r in rows}])
