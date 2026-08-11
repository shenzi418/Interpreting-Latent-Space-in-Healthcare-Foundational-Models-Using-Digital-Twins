"""Dump every Pipeline A number worth knowing across all 6 configs.

Reads the canonical baseline JSON and the INLP JSON, prints
metadata, per-config in-domain/cross-domain headlines (Z and NK2),
per-class precision/recall/F1 for the cross-domain 4c task, and
the 2c collapse summary.
"""

from __future__ import annotations

import json
from pathlib import Path

BASELINE = Path("outputs/phase_b2/cross_domain_4c_pipelineA.json")
INLP = Path("outputs/phase_b2_inlp/cross_domain_4c_pipelineA.json")


def _g(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt_pct(x):
    return "  -  " if x is None else f"{x:6.3f}"


def _fmt_ci(point, ci, p=None):
    if point is None:
        return "  -  "
    lo = ci[0] if ci else None
    hi = ci[1] if ci else None
    out = f"{point:.3f}"
    if lo is not None and hi is not None:
        out += f" [{lo:.3f},{hi:.3f}]"
    if p is not None:
        out += f" p={p:.3f}"
    return out


def dump_metadata(meta: dict) -> None:
    print("=" * 100)
    print("METADATA")
    print("=" * 100)
    print(f"configs:                   {meta.get('configs')}")
    print(f"sources:                   {meta.get('sources')}")
    print(f"territories_4c:            {meta.get('territories_4c')}")
    print(f"territory_4c_to_2c map:")
    for k, v in meta.get('territory_4c_to_2c', {}).items():
        print(f"  {k:<14} -> {v}")
    print(f"n_train_medalcare_4c:      {meta.get('n_train_medalcare_4c')}")
    print(f"n_test_medalcare_4c:       {meta.get('n_test_medalcare_4c')}")
    print(f"n_ptbxl_primary_4c total:  {meta.get('n_ptbxl_primary_4c')}")
    print(f"n_per_class_truth_ptbxl_4c:{meta.get('n_per_class_truth_ptbxl_4c')}")
    print(f"n_per_class_truth_ptbxl_2c:{meta.get('n_per_class_truth_ptbxl_2c')}")
    print(f"LogReg Cs:                 {meta.get('logreg_Cs')}")
    print(f"class_weight / multi_class:{meta.get('class_weight')} / {meta.get('multi_class')}")
    print(f"solver / max_iter:         {meta.get('solver')} / {meta.get('max_iter')}")
    print(f"internal_cv:               {meta.get('internal_cv')}")
    print(f"n_bootstrap / n_permutation:{meta.get('n_bootstrap')} / {meta.get('n_permutation_macro_f1')}")
    print(f"ptbxl_subclass_csv:        {meta.get('ptbxl_subclass_csv')}")
    print(f"seed:                      {meta.get('seed')}")


def dump_one_config(cfg: str, d: dict) -> None:
    print()
    print("=" * 100)
    print(f"CONFIG: {cfg}")
    print("=" * 100)
    # Find which sources are present in this config
    if not isinstance(d, dict):
        print("  [no result block]")
        return
    sources = [k for k in d.keys() if k in ("Z", "ecg_features")]
    for src in sources:
        block = d[src]
        in_d = block.get("in_domain_4c", {})
        cd_4 = block.get("cross_domain_4c", {})
        cd_2 = block.get("cross_domain_2c", {})
        print(f"\n  Source: {src}")
        print(f"    Best LogReg C from internal CV: {block.get('best_C', '-')}")
        print(f"    -- In-domain MedalCare-test (4c, n={in_d.get('n_total','?')}) --")
        print(f"      macro-F1: {_fmt_ci(in_d.get('macro_f1'), in_d.get('macro_f1_ci95'), in_d.get('permutation_p_macro_f1'))}")
        print(f"      balanced-acc: {_fmt_ci(in_d.get('balanced_accuracy'), in_d.get('balanced_accuracy_ci95'), in_d.get('permutation_p_balanced_accuracy'))}")
        print(f"    -- Cross-domain PTB-XL (4c, n={cd_4.get('n_total','?')}) --")
        print(f"      macro-F1: {_fmt_ci(cd_4.get('macro_f1'), cd_4.get('macro_f1_ci95'), cd_4.get('permutation_p_macro_f1'))}")
        print(f"      balanced-acc: {_fmt_ci(cd_4.get('balanced_accuracy'), cd_4.get('balanced_accuracy_ci95'), cd_4.get('permutation_p_balanced_accuracy'))}")
        per_cls = cd_4.get("per_class", {})
        if per_cls:
            print(f"      per-class:")
            print(f"        {'class':<15} {'prec':>7} {'rec':>7} {'f1':>7} {'support':>8}")
            for cls in ["Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"]:
                pc = per_cls.get(cls, {})
                print(f"        {cls:<15} {_fmt_pct(pc.get('precision'))} {_fmt_pct(pc.get('recall'))} {_fmt_pct(pc.get('f1'))} {pc.get('support', '-'):>8}")
        cm = cd_4.get("confusion_matrix")
        if cm:
            print(f"      confusion_matrix (rows=truth, cols=pred; order = 4c territories above):")
            for cls, row in zip(["Anteroseptal", "Anterolateral", "Inferior", "Inferolateral"], cm):
                print(f"        {cls:<15} {row}")
        print(f"    -- Cross-domain PTB-XL (2c collapse: Anterior vs Inferior, n={cd_2.get('n_total','?')}) --")
        print(f"      macro-F1: {_fmt_ci(cd_2.get('macro_f1'), cd_2.get('macro_f1_ci95'), cd_2.get('permutation_p_macro_f1'))}")
        print(f"      balanced-acc: {_fmt_ci(cd_2.get('balanced_accuracy'), cd_2.get('balanced_accuracy_ci95'), cd_2.get('permutation_p_balanced_accuracy'))}")
        cm2 = cd_2.get("confusion_matrix")
        if cm2:
            print(f"      confusion_matrix (rows=truth, cols=pred; order = [Anterior, Inferior]):")
            for cls, row in zip(["Anterior", "Inferior"], cm2):
                print(f"        {cls:<10} {row}")


def main() -> None:
    for path in (BASELINE, INLP):
        if not path.exists():
            print(f"[MISSING] {path}")
            continue
        print(f"\n\n############## {path} ##############")
        d = json.load(path.open())
        dump_metadata(d["metadata"])
        for cfg in d["metadata"]["configs"]:
            dump_one_config(cfg, d["results"].get(cfg, {}))


if __name__ == "__main__":
    main()
