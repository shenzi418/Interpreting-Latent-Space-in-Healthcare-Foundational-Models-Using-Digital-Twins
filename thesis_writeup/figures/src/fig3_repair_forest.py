"""Fig 3 — the limits of repair: paired Δ macro-F1 of each repair arm against the unrestricted 1024-d latent readout,
cross-domain (PTB-XL, n=4324), clean medalonly encoder. Target scaler = filled marker with 95% CI; strict/source scaler
= hollow marker. Grey band = norm-matched random-projection null (mean … 95th pct), expressed relative to the
unrestricted readout under the same scaler. Reweighting arms are Δ against the UNweighted readout.

Reads ONLY outputs/analysis/fidelity_audit/f3_repair.json (== reports/2026-08-13_audit_artifacts/tmp_f3_repair.json).
Run from repo root: python thesis_writeup/figures/src/fig3_repair_forest.py
"""
import json, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "outputs", "analysis", "fidelity_audit", "f3_repair.json")
OUT = os.path.join(HERE, "..")
d = json.load(open(SRC))
m = d["exp8_leadfix_medalonly"]
base_t = m["base"]["unrestricted"]["target"]["macro_f1"]
base_s = m["base"]["unrestricted"]["source"]["macro_f1"]

# (row label, paired key stem, null stem, dims)
ARMS = [("Q$_{amp}$ + R$_{amp}$ leads (24-d)", "QR24", "QR24", 24),
        ("Q$_{amp}$ leads (12-d)", "Q12", None, 12),
        ("R$_{amp}$ leads (12-d)", "R12", "R12", 12),
        ("ST$_{J60}$ leads (12-d)", "ST12", "ST12", 12),
        ("inferior leads II, III, aVF (12-d)", "inferior", "inferior", 12)]
RW = [("reweight, axis-pair (ESS 1120/6513)", "axpair"),
      ("reweight, six-global (ESS 208/4968)", "six")]

rows = []
for lab, key, nk, dims in ARMS:
    pt = m["paired"][f"{key}_vs_unrestricted_target"]
    ps = m["paired"][f"{key}_vs_unrestricted_source"]
    null_t = m["nulls"][f"{nk}_randproj"]["target"] if nk else None
    rows.append(dict(label=lab, kind="restriction", dt=pt["delta"], lo=pt["ci_lo"], hi=pt["ci_hi"], p=pt["p_boot"],
                     ds=ps["delta"], slo=ps["ci_lo"], shi=ps["ci_hi"], ps=ps["p_boot"],
                     nb=((null_t["null_mean"] - base_t, null_t["null_p95"] - base_t) if null_t else None),
                     obs_t=m["blocks"][key]["cross_target"]["macro_f1"]))
for lab, key in RW:
    r = m["reweight"][key]
    pt, ps = r["target"]["paired_vs_unweighted"], r["source"]["paired_vs_unweighted"]
    rows.append(dict(label=lab, kind="reweight", dt=pt["delta"], lo=pt["ci_lo"], hi=pt["ci_hi"], p=pt["p_boot"],
                     ds=ps["delta"], slo=ps["ci_lo"], shi=ps["ci_hi"], ps=ps["p_boot"], nb=None,
                     obs_t=r["target"]["score_avgpred"]["macro_f1"]))

fig, ax = plt.subplots(figsize=(7.2, 4.4))
n = len(rows)
ys = list(range(n))[::-1]
for y, r in zip(ys, rows):
    if r["nb"]:
        ax.fill_betweenx([y - 0.32, y + 0.32], r["nb"][0], r["nb"][1], color="#e4e4e0", zorder=1)
    # source (hollow, offset up)
    ax.plot([r["slo"], r["shi"]], [y + 0.16, y + 0.16], color="#8a8a84", lw=1.0, zorder=2)
    ax.scatter([r["ds"]], [y + 0.16], s=34, facecolor="white", edgecolor="#555", linewidth=1.0, zorder=3)
    # target (filled)
    col = "#2a78d6" if r["kind"] == "restriction" else "#eb6834"
    ax.plot([r["lo"], r["hi"]], [y - 0.10, y - 0.10], color=col, lw=1.8, zorder=2)
    ax.scatter([r["dt"]], [y - 0.10], s=46, facecolor=col, edgecolor="white", linewidth=0.8, zorder=3)
    ptxt = "p<0.001" if r["p"] < 0.001 else f"p={r['p']:.2f}"
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.985, y - 0.10, f"Δ={r['dt']:+.4f}  {ptxt}", fontsize=8.5, va="center", ha="right",
            color="#333", transform=tr, zorder=4)
ax.axvline(0, color="#444", lw=1, zorder=1)
ax.axhline(1.5, color="#ccc", lw=0.8, ls=":")
ax.set_yticks(ys); ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5)
ax.set_xlim(-0.165, 0.145)
ax.tick_params(axis="x", labelsize=9.5)
ax.set_xlabel("paired Δ macro-F$_1$, cross-domain PTB-XL (n = 4,324)", fontsize=9.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color="#ececea", lw=0.6); ax.set_axisbelow(True)
# legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h = [Line2D([0], [0], marker="o", color="#2a78d6", lw=1.8, markersize=6, label="rescaled transport, 95% CI"),
     Line2D([0], [0], marker="o", color="#8a8a84", markerfacecolor="white", lw=1.0, markersize=6, label="strict transport"),
     Patch(facecolor="#e4e4e0", label="random-projection null (mean–95th pct, rescaled)")]
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
ax.set_title("Repair arms against the unrestricted 1024-dimensional readout", fontsize=10.5, loc="left")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "fig3_repair_forest.pdf"), bbox_inches="tight", pad_inches=0.03)
fig.savefig(os.path.join(OUT, "fig3_repair_forest.png"), dpi=150, bbox_inches="tight", pad_inches=0.03)
with open(os.path.join(OUT, "fig3_points.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "kind", "delta_target", "ci_lo", "ci_hi", "p_boot", "delta_source", "ci_lo_s", "ci_hi_s", "p_boot_s", "obs_macro_f1_target", "null_band_target"])
    for r in rows:
        w.writerow([r["label"], r["kind"], f'{r["dt"]:.4f}', f'{r["lo"]:.4f}', f'{r["hi"]:.4f}', f'{r["p"]:.4f}', f'{r["ds"]:.4f}', f'{r["slo"]:.4f}', f'{r["shi"]:.4f}', f'{r["ps"]:.4f}', f'{r["obs_t"]:.4f}', r["nb"]])
print("ok")
