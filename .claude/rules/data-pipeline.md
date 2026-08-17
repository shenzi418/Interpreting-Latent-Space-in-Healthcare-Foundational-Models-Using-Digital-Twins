*Apply when editing `scripts/datasets.py`, `scripts/prepare_medalcare.py`, `scripts/build_*.py`, or writing a new Dataset class / manifest builder.*

## Data flow
```
Raw data → prepare_medalcare.py → WFDB format
         → make_splits.py        → manifest CSVs (data/)
         → datasets.py           → PyTorch Dataset objects
         → finetune_*.py         → outputs/<run_id>/{args.json, metrics.json, per_class_metrics.csv,
                                                     checkpoints/linear_best.pt}
         → export_latents.py     → outputs/latents/<prefix>_<domain>/latents.npz
         → analysis/             → outputs/analysis/
```

## Lead order — reindex by `sig_name`, never by position (both domains)

**This rule was inverted on 2026-08-10.** It previously instructed a mandatory
aVF↔aVL swap on the MedalCare side. That instruction was wrong and was the
mechanism of a real bug — see `reports/2026-08-10_lead_order_bug_diagnostic.md`.

- WFDB source order on disk is **already standard**: `[I, II, III, aVR, aVL, aVF, V1–V6]`.
  `scripts/prepare_medalcare.py:53` writes it that way and the manifest's
  `lead_order` column agrees. Verified empirically against the limb-lead
  identities `aVL = (I − III)/2`, `aVF = (II + III)/2`: channel 4 **is** aVL.
- Target order returned by every Dataset: the same standard order. **No permutation.**
- Pattern: reindex by the header's `sig_name`, mirroring
  `PTBXLDataset._reorder_leads`. Load with `wfdb.rdsamp` so `sig_name` survives,
  build `[sig_name.index(l) for l in TARGET_LEADS]`, and **assert** on an
  unexpected lead set so a future format change fails loudly.
- The old positional swap transposed the inferior (aVF) and high-lateral (aVL)
  leads in every MedalCare batch while PTB-XL — which already reindexed by name
  — was untouched. That is a synthetic-only frontal-plane corruption, i.e.
  exactly the kind of artifact a domain classifier picks up for free.

## Normalisation — per-lead z-score on both sides
- `LVEF_12lead_cls_Dataset(per_lead_norm=True)` (default) matches
  `PTBXLDataset._z_score`. The legacy MedalCare behaviour was a single **global**
  scalar mean/std over the whole `(12, T)` array — a different convention from
  the one applied to the real domain, and a second independent source of
  synthetic-vs-real mismatch. `per_lead_norm=False` reproduces it for ablation only.

## Manifest column convention
- 8-class MedalCare binary labels in fixed order: `label_0` (sinus), `label_1` (mi), `label_2` (rbbb), `label_3` (lbbb), `label_4` (lae), `label_5` (iab), `label_6` (fam), `label_7` (avblock).
- `MEDALCARE_REMAP`, `MEDALCARE_KEEP_LABELS`, `MEDALCARE_DROP_LABELS` (`scripts/finetune_multilabel.py:52-56`) index by integer position.
- Renaming or reordering columns silently shifts every remapped class — no schema check exists.

## θ targets (`data/theta_mi_*.npz`)
- θ has **4** members: `phi`, `z`, `size`, `rho_eps_max`. `transmural` is a
  *duplicate* of `rho_eps_max` (`np.array_equal` is True on all three splits) —
  do not report them as two parameters. `phase_b2.load_targets` asserts this.
- `territory_4c` is derived **from φ**, not from the MedalCare folder name
  (`scripts/build_medalcare_isch_targets.py`). `territory_4c_folder` preserves
  the old folder-derived labels; the build **fails** on any disagreement other
  than the known `LCX_*_post` case. `z` and `size` ranges are identical across
  all 8 territory buckets, so the θ-sufficient statistic for territory is (ρ, φ).

## Splits
- Seeded from SHA-256 of the `original_csv_path` column (`scripts/make_splits.py`).
- **Do not regenerate ad-hoc.** `data/medalcare_filtered_manifest_dataset_split.csv` is the source of truth.
- All `outputs/latents/*.npz` and `data/theta_mi_*.npz` artifacts assume the canonical row order — re-shuffling breaks every downstream analysis script.