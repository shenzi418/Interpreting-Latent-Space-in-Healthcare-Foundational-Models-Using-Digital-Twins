"""Fig 2 — per-block territory readouts: in-domain (MedalCare-XL, group-disjoint CV) -> cross-domain (PTB-XL),
under the strict/source scaler (left) and the target scaler (right). Slope chart, one line per feature block,
with the constant-predictor macro-F1 floors as reference lines and the transfer efficiency printed at the right.

Reads ONLY outputs/analysis/fidelity_audit/f2_blocks.json (== reports/2026-08-13_audit_artifacts/tmp_f2_blocks.json).
Run from repo root: python thesis_writeup/figures/src/fig2_block_transfer.py
"""
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "outputs", "analysis", "fidelity_audit", "f2_blocks.json")
OUT = os.path.join(HERE, "..")
d = json.load(open(SRC))
fl = d["floors"]
FLOOR_IN, FLOOR_X = fl["constF1_medalcare"], fl["constF1_ptbxl"]

# fixed order + palette (same as Fig 1); full54 and axis2 in neutral inks
BLOCKS = [("ST_J60", "ST$_{J60}$ ×12", "#2a78d6", "o"),
          ("Q_amp", "Q$_{amp}$ ×12", "#eb6834", "s"),
          ("R_amp", "R$_{amp}$ ×12", "#1baf7a", "D"),
          ("T_amp", "T$_{amp}$ ×12", "#eda100", "^"),
          ("globals", "intervals ×6", "#e87ba4", "v"),
          ("full54", "all 54", "#333333", "P"),
          ("axis2", "axis (R$_I$, R$_{aVF}$) fitted", "#8a8a84", "X")]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), sharey=True)
for ax, mode, title in [(axes[0], "cross_source", "strict / source scaler (raw transport)"),
                        (axes[1], "cross_target", "target scaler (after re-centring on the PTB-XL MI cohort)")]:
    ax.axhline(FLOOR_X, xmin=0.55, xmax=1.0, color="#9a9a94", lw=1, ls="--", zorder=1)
    ax.axhline(FLOOR_IN, xmin=0.0, xmax=0.45, color="#9a9a94", lw=1, ls="--", zorder=1)
    ax.text(0.02, FLOOR_IN + 0.004, f"constant floor {FLOOR_IN:.4f}", fontsize=7, color="#6f6f69", ha="left", va="bottom")
    ax.text(0.56, FLOOR_X - 0.004, f"constant floor {FLOOR_X:.4f}", fontsize=7, color="#6f6f69", ha="left", va="top")
    items = []
    for key, label, col, mk in BLOCKS:
        pb = d["per_block"][key]
        y0, y1 = pb["in"]["macro_f1"], pb[mode]["macro_f1"]
        eff = d["efficiency"][key]["eff_source" if mode == "cross_source" else "eff_target"]
        ax.plot([0, 1], [y0, y1], color=col, lw=1.6, alpha=0.9, zorder=2)
        ax.scatter([0, 1], [y0, y1], color=col, marker=mk, s=42, edgecolor="white", linewidth=0.7, zorder=3, label=label)
        star = "*" if (key == "globals" and mode == "cross_target") else ""
        items.append((y1, f"{label}   eff {eff:.3f}{star}", col))
    # de-collide right-hand labels: sort by y, enforce a minimum gap, then draw with short leaders
    items.sort(key=lambda t: -t[0])
    gap = 0.016
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        if ys[i] > ys[i - 1] - gap:
            ys[i] = ys[i - 1] - gap
    # push back up if we ran below the lowest data
    for i in range(len(ys) - 2, -1, -1):
        if ys[i] < ys[i + 1] + gap:
            ys[i] = ys[i + 1] + gap
    for (y1, txt, col), yy in zip(items, ys):
        ax.annotate(txt, xy=(1, y1), xytext=(1.06, yy), textcoords="data", fontsize=7.2, va="center", ha="left",
                    color="#333", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6, shrinkA=0, shrinkB=2))
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["in-domain\nMedalCare-XL (CV)", "cross-domain\nPTB-XL (n=4324)"], fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#ececea", lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("nearest-anchor macro-F$_1$ (4-class territory)", fontsize=8.5)
axes[0].set_ylim(0.10, 0.48)
fig.suptitle("Per-block territory readouts fitted on the simulator, transported to real ECG", fontsize=10, x=0.01, ha="left")
fig.text(0.01, 0.005, "efficiency = (F1$_{cross}$ − floor$_{cross}$) / (F1$_{in}$ − floor$_{in}$).   *intervals block rides on ~33% imputation cross-domain (target-scaler efficiency not headlined).",
         fontsize=6.8, color="#555")
fig.tight_layout(rect=(0, 0.03, 0.86, 0.95))
fig.savefig(os.path.join(OUT, "fig2_block_transfer.pdf"))
fig.savefig(os.path.join(OUT, "fig2_block_transfer.png"), dpi=150)

with open(os.path.join(OUT, "fig2_points.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["block", "f1_in", "f1_cross_source", "f1_cross_target", "eff_source", "eff_target", "eta2_sim_mean", "eta2_real_mean"])
    for key, *_ in BLOCKS:
        pb = d["per_block"][key]; e = d["efficiency"][key]
        w.writerow([key, f'{pb["in"]["macro_f1"]:.4f}', f'{pb["cross_source"]["macro_f1"]:.4f}', f'{pb["cross_target"]["macro_f1"]:.4f}',
                    f'{e["eff_source"]:.3f}', f'{e["eff_target"]:.3f}', f'{d["block_eta2_features"]["sim_mean"][key]:.4f}', f'{d["block_eta2_features"]["real_mean"][key]:.4f}'])
print("ok")
