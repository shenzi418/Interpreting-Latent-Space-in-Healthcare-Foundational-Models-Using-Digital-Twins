*Apply when editing `scripts/datasets.py`, `scripts/prepare_medalcare.py`, `scripts/build_*.py`, or writing a new Dataset class / manifest builder.*

## Data flow
```
Raw data → prepare_medalcare.py → WFDB format
         → make_splits.py        → manifest CSVs (data/)
         → datasets.py           → PyTorch Dataset objects
         → finetune_*.py         → outputs/<run_id>/{best_model.pt, args.json, metrics.json, per_class_metrics.csv}
         → export_latents.py     → outputs/latents/<prefix>_<domain>/latents.npz
         → analysis/             → outputs/analysis/
```

## Lead permutation (MedalCare side, mandatory)
- Source lead order from WFDB: `[I, II, III, aVR, aVF, aVL, V1–V6]`.
- Target order returned by Dataset: `[I, II, III, aVR, aVL, aVF, V1–V6]` (positions 4 ↔ 5 swap).
- Pattern (`scripts/datasets.py:93-95`): set `self.input_leads`, `self.new_leads`, compute `self.lead_indices = [input_leads.index(l) for l in new_leads]`, apply as `data[self.lead_indices, :]`.
- Skipping silently swaps the two augmented vector limb leads in every batch — model trains on corrupted signals with no error message.

## Manifest column convention
- 8-class MedalCare binary labels in fixed order: `label_0` (sinus), `label_1` (mi), `label_2` (rbbb), `label_3` (lbbb), `label_4` (lae), `label_5` (iab), `label_6` (fam), `label_7` (avblock).
- `MEDALCARE_REMAP`, `MEDALCARE_KEEP_LABELS`, `MEDALCARE_DROP_LABELS` (`scripts/finetune_multilabel.py:52-56`) index by integer position.
- Renaming or reordering columns silently shifts every remapped class — no schema check exists.

## Splits
- Seeded from SHA-256 of the `original_csv_path` column (`scripts/make_splits.py`).
- **Do not regenerate ad-hoc.** `data/medalcare_filtered_manifest_dataset_split.csv` is the source of truth.
- All `outputs/latents/*.npz` and `data/theta_mi_*.npz` artifacts assume the canonical row order — re-shuffling breaks every downstream analysis script.