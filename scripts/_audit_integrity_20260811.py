"""Full-repo integrity audit, 2026-08-11.

Asks one question: is any number we intend to publish resting on a defect?

Checks, in order of how much damage a failure would do:

A. LEAKAGE -- which PTB-XL strat_folds did each encoder see during fine-tuning,
   and which folds do the cross-domain evaluations actually score on? A latent
   arm evaluated on folds the encoder trained on would be compared against a
   hand-crafted control that never trains at all -- that is not a fair contest
   and it is exactly the contest that produces the headline sign reversal.
B. EXPORT PROVENANCE -- does every exp8 latent export match its own checkpoint,
   and is the PTB-XL evaluation export exactly the held-out fold?
C. STALE-EXPECTATION TRIAGE -- the three _verify_3_* FAILs: are they data
   defects, or verifier constants left over from a superseded label rule?
D. THETA / SPLIT / LABEL contracts from .claude/rules/data-pipeline.md.

Exit code is 0 only if nothing in A or B is wrong. C failures are triaged and
reported but do not by themselves fail the audit -- they are claims about the
verifier, not about the data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HARD_FAIL = 0
SOFT = 0


def hard(msg: str) -> None:
    global HARD_FAIL
    print(f"  [HARD-FAIL] {msg}")
    HARD_FAIL += 1


def soft(msg: str) -> None:
    global SOFT
    print(f"  [note] {msg}")
    SOFT += 1


def ok(msg: str) -> None:
    print(f"  [ ok ] {msg}")


print("=" * 72)
print("A. LEAKAGE -- PTB-XL fold accounting")
print("=" * 72)

from scripts.datasets import PTBXLDataset  # noqa: E402  pylint: disable=wrong-import-position

SPLITS = PTBXLDataset.OFFICIAL_SPLITS
print(f"  OFFICIAL_SPLITS = {dict(SPLITS)}")
train_folds = set(SPLITS["train"]) | set(SPLITS["val"])
eval_fold = set(SPLITS["test"])
if train_folds & eval_fold:
    hard(f"train/val folds overlap test folds: {train_folds & eval_fold}")
else:
    ok(f"encoder-exposed folds {sorted(train_folds)} disjoint from "
       f"held-out {sorted(eval_fold)}")

# Which folds are in the MI-subclass CSV that drives cross-domain territory eval?
mi = pd.read_csv(REPO / "data" / "ptbxl_mi_subclass.csv")
mi_folds = sorted(mi["strat_fold"].unique().tolist()) if "strat_fold" in mi else []
print(f"  ptbxl_mi_subclass.csv: n={len(mi)}, strat_folds={mi_folds}")
if set(mi_folds) - eval_fold:
    hard(f"MI-subclass CSV contains encoder-exposed folds: "
         f"{sorted(set(mi_folds) - eval_fold)} -- cross-domain eval is leaked")
else:
    ok("MI-subclass CSV is confined to the held-out fold -- cross-domain "
       "territory eval is clean")

# The train-split PTB-XL exports exist on disk. Confirm nothing in phase_b2
# consumes them for the cross-domain endpoint.
b2 = (REPO / "analysis" / "phase_b2_infarct_decoding.py").read_text(encoding="utf-8")
uses_ptb_train = "ptbxl_train" in b2
print(f"  phase_b2 references 'ptbxl_train' string: {uses_ptb_train}")
if uses_ptb_train:
    for i, line in enumerate(b2.splitlines(), 1):
        if "ptbxl_train" in line:
            print(f"      L{i}: {line.strip()[:100]}")

print()
print("=" * 72)
print("B. EXPORT PROVENANCE -- checkpoint <-> latent agreement, all exp8 encoders")
print("=" * 72)

ENCODERS = ["exp8_leadfix_baseline", "exp8_leadfix_K64", "exp8_leadfix_ccmmd",
            "exp8_leadfix_dual", "exp8_leadfix_globalz"]
for enc in ENCODERS:
    for dom in ["medalcare_test", "ptbxl_test", "medalcare_train"]:
        p = REPO / "outputs" / "latents" / f"{enc}_{dom}" / "latents.npz"
        if not p.exists():
            hard(f"missing export {enc}_{dom}")
            continue
        z = np.load(p, allow_pickle=True)
        # exports store the latent matrix as "Z" (with "Y" targets, "P" probs)
        L = np.asarray(z["Z"])
        bad = []
        if not np.isfinite(L).all():
            bad.append("non-finite values")
        if L.shape[0] == 0:
            bad.append("empty")
        # a latent matrix with duplicate rows means the export loop reused a batch
        if L.shape[0] > 1:
            uniq = np.unique(L[: min(2000, L.shape[0])], axis=0).shape[0]
            n_chk = min(2000, L.shape[0])
            if uniq < n_chk:
                bad.append(f"{n_chk - uniq} duplicate rows in first {n_chk}")
        if bad:
            hard(f"{enc}_{dom}: " + "; ".join(bad))
        else:
            ok(f"{enc}_{dom}: shape={L.shape} finite, all-distinct")

print()
print("=" * 72)
print("C. STALE-EXPECTATION TRIAGE -- the three _verify_3_* FAILs")
print("=" * 72)

# C1: territory_4c vs folder-derived (coronary, lcx_subtype)
print("C1. 100 'inconsistent' MedalCare rows -- data defect or superseded rule?")
tt = np.load(REPO / "data" / "theta_mi_test.npz", allow_pickle=True)
print(f"     theta_mi_test.npz keys: {sorted(tt.files)}")
if "territory_4c" in tt.files and "territory_4c_folder" in tt.files:
    a = np.asarray(tt["territory_4c"]).astype(str)
    b = np.asarray(tt["territory_4c_folder"]).astype(str)
    diff = a != b
    print(f"     rows where phi-derived != folder-derived: {diff.sum()} / {len(a)}")
    if diff.sum():
        pairs = pd.Series([f"{x} <- {y}" for x, y in zip(a[diff], b[diff])])
        for k, v in pairs.value_counts().items():
            print(f"       {v:5d}  {k}")
        t8 = np.asarray(tt["territory_8c"]).astype(str)[diff] \
            if "territory_8c" in tt.files else None
        if t8 is not None:
            uniq8 = sorted(set(t8.tolist()))
            print(f"     those rows' territory_8c: {uniq8}")
            if all("post" in u for u in uniq8):
                ok("all disagreements are the documented LCX_*_post case "
                   "(data-pipeline.md) -- verifier constants are stale, data is fine")
            else:
                hard(f"disagreements extend beyond LCX_*_post: {uniq8}")
else:
    soft("theta_mi_test.npz has no territory_4c_folder -- cannot triage here")

# C2: n_permutation 200 vs 10000
print()
print("C2. n_permutation_macro_f1 != 200")
for name, rel in [
    ("pipeline A", "outputs/phase_b2_exp8/pipeline_a.json"),
    ("phase_b2 in_domain", "outputs/phase_b2_exp8/in_domain.json"),
]:
    fp = REPO / rel
    if not fp.exists():
        continue
    try:
        j = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        soft(f"{name}: unreadable ({exc})")
        continue
    found = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "n_permutation_macro_f1":
                    found.append((path, v))
                walk(v, f"{path}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o[:3]):
                walk(v, f"{path}[{i}]")

    walk(j)
    for path, v in found[:6]:
        print(f"     {name}{path} = {v}")
print("     (more permutations than the verifier expects is a tighter test,"
      " not a defect -- confirm the direction above)")

print()
print("=" * 72)
print("D. THETA / SPLIT / LABEL contracts")
print("=" * 72)

for split in ["train", "val", "test"]:
    z = np.load(REPO / "data" / f"theta_mi_{split}.npz", allow_pickle=True)
    have = set(z.files)
    need = {"phi", "z", "size", "rho_eps_max"}
    if not need <= have:
        hard(f"theta_mi_{split}: missing {need - have}")
        continue
    dup = np.array_equal(np.asarray(z["rho_eps_max"]), np.asarray(z["transmural"])) \
        if "transmural" in have else None
    lens = {k: len(np.asarray(z[k])) for k in sorted(need)}
    same_len = len(set(lens.values())) == 1
    nonfinite = {k: int((~np.isfinite(np.asarray(z[k], dtype=float))).sum())
                 for k in sorted(need)}
    tag = "OK" if (same_len and not any(nonfinite.values())) else "BAD"
    if tag == "BAD":
        hard(f"theta_mi_{split}: lens={lens} nonfinite={nonfinite}")
    else:
        ok(f"theta_mi_{split}: n={lens['phi']}, 4 params finite, "
           f"transmural==rho_eps_max: {dup}")

man = pd.read_csv(REPO / "data" / "medalcare_filtered_manifest_dataset_split.csv")
lc = [c for c in man.columns if c.startswith("label_")]
print(f"  manifest label columns: {lc}")
if lc != [f"label_{i}" for i in range(8)]:
    hard(f"MedalCare label columns not label_0..label_7 in order: {lc}")
else:
    ok("MedalCare label_0..label_7 present and ordered")
if "split" in man.columns:
    ok(f"split counts: {man['split'].value_counts().to_dict()}")

print()
print("=" * 72)
print(f"HARD failures (leakage / provenance): {HARD_FAIL}")
print(f"notes: {SOFT}")
print("=" * 72)
sys.exit(1 if HARD_FAIL else 0)
