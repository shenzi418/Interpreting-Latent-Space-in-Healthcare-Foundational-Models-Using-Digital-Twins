# Dossier A — Data & Experimental Setup

> Source dossier for the thesis Methods chapter. Compiled 2026-08-18, read-only, from the
> repository at commit `7839113` + working tree (post-leadfix fixes uncommitted; tags
> `pre-leadfix` = 7839113, `freeze-2026-08-13` = d02f0b9).
> **Every number below carries a pointer.** `path:line` = source line; `path→key` = JSON key;
> `[computed]` = derived here by reading a committed artifact with numpy/pandas (no model
> was run, no file written outside this dossier). Disagreements between sources are flagged
> **⚠ CONTRADICTION** inline and collected in §7.

---

## Executive summary (10 lines)

1. **Two domains.** MedalCare-XL synthetic: 16,839 WFDB records, 12 leads, 500 Hz, 10 s (5000 samples), 8 mutually exclusive pathology labels; splits train 12,019 / val 2,434 / test 2,386. PTB-XL v1.0.3 real: 21,799 records, 18,869 patients, 12 leads, 500 Hz high-res, 10 s, 5 diagnostic superclasses, 10 official stratification folds.
2. **MI cohorts.** MedalCare MI = 7,797 rows (train 5,347 / val 1,250 / test 1,200) across 8 territory folders; PTB-XL MI-present = 5,469 rows all-folds, of which **4,324** carry a clean 4-class territory (fold 10 alone = **438**).
3. **θ is 4-dimensional** — `isch[0].{phi, z, size, rho_eps_max}`; `transmural` is a byte-identical duplicate of `rho_eps_max` [computed]. `territory_4c` is derived from **φ wedges** (±2.0 rad, 0.0 rad), not from folder names.
4. **Territory anchors** (MedalCare circular means, degrees): Anteroseptal +57.27, Anterolateral +147.25, Inferolateral −147.26, Inferior −57.30.
5. **Encoder** = ECGFounder Net1D (`filter_list=[64,160,160,400,400,1024,1024]`, 1024-d pooled features, `use_bn=False`), pretrained weights `checkpoint/12_lead_ECGFounder.pth`; only `ConvAdapter1D` blocks (309,568 params) + the 3-class head (3,075 params) train in the shared-head runs.
6. **19 training runs** have `metrics.json` on disk; only 11 have `args.json` (the artifact was never written before 2026-08-10, so exp5/exp6/exp7-era invocations are unrecoverable). exp5/exp6/exp7/joint/ptbxl_baselines are **pre-leadfix (provisional)**; `exp8_leadfix_*` are post-fix.
7. **127 latent export directories** under `outputs/latents/`; keys are `Z`, `P`, `Y` (+ `Z_post_gelu` for bottleneck runs, `+ ecg_id` for the two 2026-08-11 exports). Exports are on the **unfiltered** split row order, not the 3-class-filtered one.
8. **Two hand-crafted feature sets**: `global6` (6 features) and `spatial54` (48 per-lead + the same 6 globals), both delineated once on lead II with NeuroKit2 `method="dwt"` on **raw** (un-z-scored) WFDB voltages.
9. **Three analysis pipelines** are load-bearing: Phase-B2 (multinomial LogReg, C by 5-fold CV, four scaler modes), circular geometry (ridge readout to (cos φ, sin φ), GCV alpha, nearest-anchor macro-F1, constant floors 0.29216 PTB-XL / 0.09319 MedalCare), and the fidelity-audit trio (`fidelity_audit` / `block_transfer` / `channel_repair`).
10. **Compute**: Python 3.10.19, torch 2.9.1+cu128, NVIDIA GeForce RTX 5080 (16,303 MiB), sklearn 1.6.1, numpy 2.2.4, wfdb 4.2.0, neurokit2 0.2.10 (pip-only).

---

# 1. DATASETS

## 1.1 MedalCare-XL (synthetic)

### 1.1.1 What it is

Simulated 12-lead ECGs from anatomically/electrophysiologically parameterised whole-heart
models. Dataset README on disk
(`MedalCare-XL/WP2_largeDataset_Noise/README.txt`):

> ">10k synthetic 12-lead ECGs (lead order: I, II, III, aVR, aVL, aVF, V1-V6, 10 second
> recordings, sampling rate: 500Hz). Data are split by pathologies (avblock, lbbb, rbbb,
> sinus, lae, fam, iab, mi). MI data are further split into subclasses depending on the
> occlusion site (LAD, LCX, RCA) and transmurality (0.3 or 1.0). Each pathology subclass
> contains training, validation and testing data (~ 70/15/15 split). Training, validation
> and testing datasets were defined according to **the model with which QRST complexes were
> simulated**, i.e., ECGs calculated with the same anatomical model but different
> electrophysiological parameters are only present in one of the test, validation and
> training datasets but never in multiple."

Three signal variants exist per run (`_raw`, `_noise`, `_filtered`); this project uses
**`_filtered` only** (0.5 Hz HP / 150 Hz LP, 3rd-order Butterworth per the README) —
`scripts/prepare_medalcare.py:112` (`if file.endswith('_filtered.csv')`).

Project context: `openspec/project.md` ("External Dependencies": raw simulation parameter
files in `MedalCare-XL/WP2_largeDataset_ParameterFiles/`).

### 1.1.2 On-disk layout and preparation

Layout (documented and enforced in `scripts/medalcare_paths.py:12-25`):

```
.../MedalCare-XL/WP2_largeDataset_Noise/<pathology>/[<territory>/]<split>/<run_S##>/run_######_filtered.csv
```

- `<split>` on disk is `train` / **`validation`** / `test` — note "validation", not "val"
  (`scripts/medalcare_paths.py:41-43`).
- `<territory>` exists **only** under `mi/` (`scripts/medalcare_paths.py:24`).

`scripts/prepare_medalcare.py` converts each CSV → WFDB:
- CSV is leads-as-rows, time-as-columns; it is **transposed** and column names assigned from
  `LEAD_ORDER` (`prepare_medalcare.py:144-158`).
- `LEAD_ORDER = ['I','II','III','aVR','aVL','aVF','V1'..'V6']` — `prepare_medalcare.py:53`.
  **This is standard clinical order (aVL before aVF).**
- `SAMPLING_RATE = 500` Hz — `prepare_medalcare.py:56`; `UNITS = 'mV'` — `:57`.
- Manifest header written at `prepare_medalcare.py:281-306`:
  `record_id, wfdb_path, original_csv_path, label_0..label_7, sampling_rate_hz, units, lead_order`.
- `record_id` = `medalcare_%06d` over records sorted by `(original_csv_path, class, wfdb_path)`
  (`prepare_medalcare.py:278, 290`).

Verified WFDB header (`MedalCare-XL/WP2_largeDataset_Noise/mi/LAD_0.3/test/run_S62/run_000001_filtered.hea`)
[computed]: `run_000001_filtered 12 500 5000` → 12 signals, 500 Hz, **5000 samples = 10 s**;
gain `200(0)/mV`, 16-bit.

### 1.1.3 Label columns (fixed order — load-bearing)

`scripts/prepare_medalcare.py:41-50` `PATHOLOGY_KEYWORDS`, re-declared as
`scripts/medalcare_paths.py:33-35` `PATHOLOGIES`:

| column | pathology | meaning (per `prepare_medalcare.py:41-50`) |
|---|---|---|
| `label_0` | `sinus` | normal sinus rhythm |
| `label_1` | `mi` | myocardial infarction |
| `label_2` | `rbbb` | right bundle branch block |
| `label_3` | `lbbb` | left bundle branch block |
| `label_4` | `lae` | left atrial enlargement |
| `label_5` | `iab` | inter-atrial conduction block *(comment in `prepare_medalcare.py:47` says "Incomplete atrioventricular block" — **wrong**; the dataset README says interatrial conduction block, and `medalcare_paths.py` is silent)* |
| `label_6` | `fam` | fibrotic atrial cardiomyopathy *(comment at `prepare_medalcare.py:48` says "Familial/genetic condition" — also inconsistent with the README)* |
| `label_7` | `avblock` | AV block |

Schema is asserted at load time — `scripts/medalcare_paths.py:131-147` `assert_label_schema`,
called from `scripts/finetune_multilabel.py:481` and `scripts/build_medalcare_isch_targets.py:511`.
Every row is one-hot (exactly one positive) — `prepare_medalcare.py:241-255`.

### 1.1.4 Counts

Manifest in force: `data/medalcare_filtered_manifest_dataset_split.csv`, **16,839 rows × 16
columns** [computed].

Per-split, per-label positives [computed from the manifest]:

| label | pathology | train | val | test | **total** |
|---|---|---:|---:|---:|---:|
| label_0 | sinus | 900 | 200 | 200 | 1300 |
| label_1 | **mi** | **5347** | **1250** | **1200** | **7797** |
| label_2 | rbbb | 898 | 200 | 200 | 1298 |
| label_3 | lbbb | 900 | 200 | 200 | 1300 |
| label_4 | lae | 1040 | 130 | 130 | 1300 |
| label_5 | iab | 994 | 124 | 126 | 1244 |
| label_6 | fam | 1040 | 130 | 130 | 1300 |
| label_7 | avblock | 900 | 200 | 200 | 1300 |
| — | **split total** | **12019** | **2434** | **2386** | **16839** |

MedalCare MI territory folder × split [computed]:

| territory folder | train | val | test | total |
|---|---:|---:|---:|---:|
| LAD_0.3 | 899 | 200 | 200 | 1299 |
| LAD_1.0 | 900 | 200 | 200 | 1300 |
| LCX_0.3_ant | 450 | 150 | 100 | 700 |
| LCX_0.3_post | 450 | 100 | 100 | 650 |
| LCX_1.0_ant | 400 | 100 | 100 | 600 |
| LCX_1.0_post | 450 | 100 | 100 | 650 |
| RCA_0.3 | 898 | 200 | 200 | 1298 |
| RCA_1.0 | 900 | 200 | 200 | 1300 |
| **total** | **5347** | **1250** | **1200** | **7797** |

These reproduce `data/theta_mi_build_summary.json` → `[*].territory_8c_counts` exactly.
Coronary totals (`→[0].coronary_counts` train): LAD 1799, LCX 1750, RCA 1798.

### 1.1.5 How the splits are actually made — ⚠ CONTRADICTION

**Two split scripts exist and only one produced the manifest in use.**

- `scripts/add_medalcare_splits.py` derives `split` from the **folder name** in the path
  (`extract_split`, `:41-49`: `train`→train, `validation`→val, `test`→test) and `run_id` from
  `<pathology>/<run_S##>/<run_base>` (`:52-76`). Default output is
  `data/medalcare_filtered_manifest_dataset_split.csv` (`:14`). It validates that no `run_id`
  spans two splits (`:79-88`).
- `scripts/make_splits.py` instead runs `StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=seed)` with `seed` default **42** (`:29, :94`), groups = the *parent directory
  name* of `original_csv_path` (`:69-73`), fold 0 → test, fold 4 → val, rest → train
  (`:34-37`). Default output is `data/medalcare_filtered_manifest.csv` (`:27`).

**The file actually used everywhere is the `add_medalcare_splits.py` output.** Evidence:
(a) it has columns `split, run_id` and *no* `fold`/`group_id` columns, which `make_splits.py`
would have written (`make_splits.py:100-102`); (b) `data/medalcare_filtered_manifest.csv`
does not exist on disk [computed: `ls data/`]; (c) folder-derived split agrees with the
manifest `split` column at **100.0%** (16,839/16,839) [computed].

**⚠ The docstring in `scripts/finetune_multilabel.py:493-495` and the rule file
`.claude/rules/data-pipeline.md` ("Splits — Seeded from SHA-256 of the `original_csv_path`
column (`scripts/make_splits.py`)") are both wrong on two counts:** there is no SHA-256
anywhere in `scripts/` or `analysis/` except `analysis/phase_b2_infarct_decoding.py:233-237`
(RNG-stream keying, unrelated), and `make_splits.py` did not produce the manifest. The correct
statement for the thesis is: *splits are the dataset authors' own train/validation/test
directories, read off the path.*

**⚠ Grouping is not fully disjoint across splits.** The dataset README claims anatomical
models never span splits. On the shipped manifest, 13 distinct `run_S##` anatomy folders
exist and **2 of 13 span splits** [computed]: `run_S64` (test 1000, train 388) and `run_S67`
(train 389, val 1000). `add_medalcare_splits.py`'s `validate_run_splits` does **not** catch
this because its grouping key is the far finer `<pathology>/<run_S##>/<run_base>` (10,342
unique values [computed]), not `run_S##`. Analysis-side group-CV in
`analysis/geom_common.py:105` also namespaces `run_id` per split, so it never sees the
straddle either. This is a leakage caveat worth one sentence in Methods.

### 1.1.6 θ targets

Built by `scripts/build_medalcare_isch_targets.py` → `data/theta_mi_{train,val,test}.npz`
(`:525-527`).

- MI rows identified by path segment, not substring — `is_mi_path`
  (`scripts/medalcare_paths.py:90-92`), used at `build_medalcare_isch_targets.py:264`.
- Parameter file resolved by swapping `WP2_largeDataset_Noise` → `WP2_largeDataset_ParameterFiles`
  and `run_X_filtered.csv` → `run_X_VentricularParameters.txt`
  (`build_medalcare_isch_targets.py:226-241`).
- Parsed keys — `ISCH_KEYS` at `:71`:
  `isch[0].phi`, `isch[0].z`, `isch[0].size`, `isch[0].rho_eps_max`. **θ has exactly 4 members.**
- `transmural` in the NPZ is the **path-encoded** transmurality; `rho_eps_max` is the
  file-parsed one; the build cross-checks them (`:308-311`) and they are **byte-identical on
  all three splits** (`np.array_equal` True) [computed]. Do not report them as two parameters.

NPZ keys and shapes (train) [computed]:

| key | shape | dtype |
|---|---|---|
| `idx_in_split` | (5347,) | int64 — row index inside `df[df.split=='train'].reset_index(drop=True)` (`:16-20`) |
| `phi`, `z`, `size`, `rho_eps_max`, `transmural` | (5347,) | float64 |
| `coronary`, `lcx_subtype`, `run_id`, `territory_4c`, `territory_4c_folder`, `territory_8c` | (5347,) | object |

Ranges (`data/theta_mi_build_summary.json`, train entry):
φ ∈ [−3.13942, +3.13982] rad (circular mean +1.0636);
z ∈ [0.10003, 0.99991] (mean 0.58320);
size ∈ [75.014, 174.979] (mean 124.984);
`rho_eps_max` ∈ {0.3, 1.0} with train counts 2697 / 2650 (`→[0].transmural_counts`).

Per-coronary φ (`→[0].per_coronary_phi`): LAD [+0.000042, +1.99887] mean +0.99951;
RCA [−1.99922, −0.00094] mean −1.00017; LCX [−3.13942, +3.13982] mean +2.83878 (wraps ±π).
**⇒ z and size ranges are identical across all 8 territory buckets, so the θ-sufficient
statistic for territory is (ρ, φ)** (`.claude/rules/data-pipeline.md`).

### 1.1.7 `territory_4c` — derived from φ, not from the folder name

`scripts/build_medalcare_isch_targets.py:143-168`:

```
PHI_4C_OUTER_BOUNDARY = 2.0    # |phi| > 2.0 -> lateral        (:143)
PHI_4C_INNER_BOUNDARY = 0.0    # sign of phi -> anterior/inferior (:144)

[ 0.0, +2.0]  -> Anteroseptal
(+2.0, +pi ]  -> Anterolateral
[-2.0,  0.0)  -> Inferior
[-pi,  -2.0)  -> Inferolateral
```
φ is first wrapped to (−π, π] via `arctan2(sin φ, cos φ)` (`:161`).

The **folder-derived** labelling is kept as `territory_4c_folder`
(`derive_territory_4c`, `:98-117`): LAD→Anteroseptal, RCA→Inferior, LCX+`ant`→Anterolateral,
LCX+`post`→Inferolateral. The build **raises** on any φ-vs-folder disagreement other than the
one documented defect D1 (LCX_*_post rows whose φ is positive → relabelled Anterolateral)
— `:350-387`.

Wedge/φ table recorded in-code at `:120-142`, including the fact that **`LCX_0.3_ant` and
`LCX_0.3_post` occupy the same φ wedge** (both circ-mean +2.57), i.e. they are the same
distribution in all four θ parameters under two different labels — capping the folder-label
4-class task at accuracy 0.9167 / macro-F1 0.8643 with Inferolateral recall pinned at 0.500
(`:135-139`).

Resulting `territory_4c` counts (`data/theta_mi_build_summary.json→[*].territory_4c_counts`):

| territory | train | val | test |
|---|---:|---:|---:|
| Anteroseptal | 1799 | 400 | 400 |
| Anterolateral | 1300 | 350 | 300 |
| Inferior | 1798 | 400 | 400 |
| Inferolateral | 450 | 100 | 100 |

### 1.1.8 Anchor angles

`analysis/geom_common.py:69-88` `medalcare_anchor_angles()` — empirical circular mean of φ
within each `territory_4c` bucket, pooled over train+test. The docstring (`:70-78`) is explicit
that these are **a construction of the simulator's own labelling, not an independent
measurement** (they reproduce the wedge midpoints to 0.043°). Stored values
(`outputs/analysis/circular_geometry/floor_audit.json→anchors_deg`):

| territory | anchor (deg) | ≈ rad |
|---|---:|---:|
| Anteroseptal | +57.27 | +0.9996 |
| Anterolateral | +147.25 | +2.5701 |
| Inferolateral | −147.26 | −2.5702 |
| Inferior | −57.30 | −1.0001 |

### 1.1.9 Loader: `LVEF_12lead_cls_Dataset`

`scripts/datasets.py:86`. Registered as `"medalcare"` in `DATASET_REGISTRY` (`:789-792`).

- Loads with `wfdb.rdsamp` (`:239`), transposes to (leads, time) (`:250`).
- **Reorders by `sig_name`**, not by position — `_reorder_leads` (`:185-207`); raises if
  `sig_name` is missing or a lead is absent (`:189-205`).
- `self.target_leads = ['I','II','III','aVR','aVL','aVF','V1'..'V6']` — `:134-135`.
- `per_lead_norm=True` by default (`:142`); `z_score_normalization` (`:173-183`) does
  per-lead mean/std when True, single **global** scalar over the whole (12, T) array when
  False (the legacy behaviour, retained for the `exp8_leadfix_globalz` ablation).
- NaNs zero-filled with a warning (`:245-248`).
- Optional θ loading from `AtrialParameters.txt` + `VentricularParameters.txt`
  (`_parameter_paths`, `:334-348`; `_load_theta`, `:350-370`) — used only by the physics-head
  runs; **⚠ the `--theta-config` file `config/theta.json` referenced in every `args.json`
  does not exist on disk** (`config/` is empty [computed]).

---

## 1.2 PTB-XL (real)

### 1.2.1 Version and root

Root folder name (hard-coded default in three places):
`ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`
— `scripts/build_ptbxl_mi_subclass.py:56-59`, `scripts/finetune_multilabel.py:40-42`,
`analysis/geom_common.py:32-34`. **Version 1.0.3**.

Files read: `ptbxl_database.csv`, `scp_statements.csv`
(`scripts/datasets.py:603-604`, `scripts/build_ptbxl_mi_subclass.py:277-278`).

Size [computed]: `ptbxl_database.csv` = **21,799 rows × 28 columns**; **18,869 unique
`patient_id`**. Fold sizes (`strat_fold`): 1:2175, 2:2181, 3:2192, 4:2174, 5:2174, 6:2173,
7:2176, 8:2173, 9:2183, 10:2198.

### 1.2.2 Loader: `PTBXLDataset`

`scripts/datasets.py:551`.

| item | value | pointer |
|---|---|---|
| target lead order | `I, II, III, aVR, aVL, aVF, V1..V6` | `:554-567` |
| superclasses | `("NORM","MI","STTC","HYP","CD")` | `:568` |
| official splits | train = folds 1–8, val = fold 9, test = fold 10; also `trainval` 1–9, `all` 1–10 | `:569-575` |
| sampling rate | 500 Hz (default) | `:582` |
| duration | 10.0 s → `target_num_samples` = 5000 | `:583, :594-598` |
| file column | `filename_hr` when `use_high_res=True` (default) | `:584, :636` |
| lead reindex | by `sig_name`, raises if missing | `:744-757` |
| resample | `scipy.signal.resample` if header fs ≠ 500 | `:759-769` |
| length match | truncate if longer, zero-pad if shorter | `:771-780` |
| normalisation | **per-lead** z-score, `(x−μ)/(σ+1e-8)` | `:782-786` |
| label build | multi-hot over `diagnostic_class` of every listed SCP code (no probability threshold) | `:693-709` |

Verified header (`records500/00000/00001_hr.hea`) [computed]: `00001_hr 12 500 5000`,
gain `1000.0(0)/mV`, 16-bit. Both domains therefore land at (12, 5000) at 500 Hz.

### 1.2.3 Shared 3-class label space {NORM, MI, CD}

`scripts/finetune_multilabel.py:50-57`:

```
SHARED_LABELS = ("NORM", "MI", "CD");  N_SHARED = 3
MEDALCARE_REMAP = {0:0, 1:1, 2:2, 3:2, 5:2, 7:2}   # sinus→NORM; mi→MI; rbbb,lbbb,iab,avblock→CD
PTBXL_REMAP     = {0:0, 1:1, 4:2}                  # NORM→NORM; MI→MI; CD(index 4)→CD
MEDALCARE_KEEP_LABELS = (0,1,2,3,5,7)
MEDALCARE_DROP_LABELS = (4, 6)                     # lae, fam
```
`remap_labels()` at `:63-84`. Excluded on the PTB-XL side: `STTC` (index 2) and `HYP`
(index 3) — no clean clinical correspondence (`.claude/rules/experiments.md`).

Row counts after the 3-class filter, **post-leadfix** (printed by
`build_shared_head_loaders`, captured in `outputs/_log_exp8_medalonly.txt:3-8`):

| stream | before | after | dropped |
|---|---:|---:|---:|
| MedalCare train | 12019 | **9939** | 2080 (= lae 1040 + fam 1040) |
| MedalCare val | 2434 | **2174** | 260 |
| MedalCare test | 2386 | **2126** | 260 |
| PTB-XL train (folds 1–8) | 17418 | **14116** | 3302 |
| PTB-XL val (fold 9) | 2183 | **1769** | 414 |
| PTB-XL test (fold 10) | 2198 | **1787** | 411 |

Post-filter positives / `pos_weight` on the medalonly run
(`outputs/_log_exp8_medalonly.txt:10-12`): NORM 900 (w 10.043), MI 5347 (w 0.859),
CD 3692 (w 1.692).

**Pre-leadfix contrast**: `outputs/exp5_3class/per_class_metrics.csv` line 5 shows
`val_ptbxl NORM positives=955 negatives=918` → 1873 rows, vs
`outputs/exp8_leadfix_baseline/per_class_metrics.csv` line 5 `955 / 814` → 1769. The
difference is defect D3 (see §1.4): the old `keep_indices = [idx for src, idx in
PTBXL_REMAP.items()]` yielded `[0,1,2]` = NORM/MI/**STTC**, contradicting its own inline
comment; fixed to `keep_indices = list(PTBXL_REMAP.keys())` = `[0,1,4]`
(`git diff pre-leadfix -- scripts/finetune_multilabel.py`, hunk at the old `:200`;
current code `scripts/finetune_multilabel.py` `filter_ptbxl_3class`).

### 1.2.4 MI subclass → 4-class territory (`scripts/build_ptbxl_mi_subclass.py`)

MI-class SCP codes (`diagnostic_class == 'MI'` in `scp_statements.csv`) — **14 codes**
[computed]: `ALMI, AMI, ASMI, ILMI, IMI, INJAL, INJAS, INJIL, INJIN, INJLA, IPLMI, IPMI,
LMI, PMI`. Their `diagnostic_subclass` values [computed]:
AMI ← {AMI, ASMI, ALMI, INJAS, INJAL, INJLA}; IMI ← {IMI, ILMI, IPLMI, IPMI, INJIN, INJIL};
LMI ← {LMI}; PMI ← {PMI}.

Legacy 3-class mapping — `:64-70`: AMI→Anterior, IMI→Inferior, LMI→Lateral, PMI→Posterior
(excluded).

**Refined 4-class code groups — `:101-106`:**

| group | codes | pointer |
|---|---|---|
| `ANTERIOR_PURE_CODES` | `AMI, ASMI, INJAS` | `:101` |
| `ANTEROLATERAL_CODES` | `ALMI, INJAL` | `:102` |
| `INFERIOR_PURE_CODES` | `IMI, INJIN` | `:103` |
| `INFEROLATERAL_CODES` | `ILMI, INJIL, IPLMI` | `:104` |
| `LATERAL_PURE_CODES` | `LMI, INJLA` | `:105` |
| `EXCLUDED_POSTERIOR_CODES` | `PMI, IPMI` | `:106` |

Decision rules (`:88-99`, implemented `:124-159`), in order:
1. any excluded-posterior code → `""` (dropped);
2. anterior side ∧ inferior side both hit → `""` (multi-territory, dropped);
3. anterior side: `ANTEROLATERAL` or `LATERAL_PURE` present → **Anterolateral**, else **Anteroseptal**;
4. inferior side: `INFEROLATERAL` or `LATERAL_PURE` present → **Inferolateral**, else **Inferior**;
5. lateral-only, no anterior/inferior → `""` (ambiguous, dropped).

2-class collapse (`:116-121, :162-164`): Anteroseptal+Anterolateral → **Anterior**;
Inferior+Inferolateral → **Inferior**.

Thresholds: `--prob-threshold` default **0.0** (any listed code counts as present, `:262-265`);
`--strong-threshold` default **80.0**, feeding the auxiliary `mi_strong` flag only (`:266-270`).

### 1.2.5 Counts — fold 10 vs all folds

**Fold 10 only** (`data/ptbxl_mi_subclass.csv`, summary `data/ptbxl_mi_subclass_summary.json`):

| quantity | value | key |
|---|---:|---|
| rows | 2198 | `→n_rows` |
| MI present | 550 | `→n_mi_present` |
| MI strong (≥80) | 293 | `→n_mi_strong` |
| single-territory (legacy 3c) | 444 | `→n_single_territory_primary` |
| **primary 4-class** | **438** | `→n_primary_4c` |
| primary 2-class | 438 | `→n_primary_2c` |
| 4c: Anteroseptal / Anterolateral / Inferior / Inferolateral | 168 / 42 / 196 / 32 | `→territory_4c_counts` |
| 2c: Anterior / Inferior | 210 / 228 | `→territory_2c_counts` |
| top subclass combos | IMI 229, AMI 203, AMI\|IMI 96, LMI 12, AMI\|LMI 7, PMI 1, IMI\|PMI 1, IMI\|LMI 1 | `→top_subclass_combos` |

**All folds** (`data/ptbxl_mi_subclass_allfolds.csv`, 21,799 rows × 16 cols) [computed]:

| quantity | value |
|---|---:|
| MI present | 5469 |
| **primary 4-class** | **4324** |
| 4c: Inferior / Anteroseptal / Anterolateral / Inferolateral | 1914 / 1624 / 455 / 331 |
| 2c: Inferior / Anterior | 2245 / 2079 |
| unique `patient_id` in the 4324 | 3794 |

Per-fold primary-4c size [computed]: 1:442, 2:417, 3:400, 4:438, 5:455, 6:463, 7:433, 8:423,
9:415, **10:438**. Per-fold × territory [computed], fold 10 row: AL 42 / AS 168 / INF 196 / IL 32.

**Why both exist.** Fold 10 (n=438) is the only legitimate eval set for encoders that saw
PTB-XL folds 1–9 in training. `exp8_leadfix_medalonly` never took a PTB-XL gradient step
(`scripts/finetune_multilabel.py:354-365` `--medalcare-only`), which frees all ten folds →
n=4324, a 9.9× power increase (`analysis/circular_geometry.py:24-25`; CLAUDE.md).

### 1.2.6 Acuity labels

Columns `infarction_stadium1`, `infarction_stadium2` in `ptbxl_database.csv`.
Whole-database distribution [computed]: `stadium1` non-null 5612 of which
`unknown` 3430, Stadium III 980, Stadium II-III 943, Stadium I 166, Stadium II 88,
Stadium I-II 5; `stadium2` non-null 103.

Restricted to the 4324 primary-4c rows [computed]:

| `infarction_stadium1` | n |
|---|---:|
| unknown | 2707 |
| Stadium III | 756 |
| Stadium II-III | 703 |
| Stadium I | 94 |
| Stadium II | 61 |
| Stadium I-II | 3 |
| **graded (≠ unknown)** | **1617** |

(II-III + III) / graded = 1459/1617 = **90.23 %** — reproduces the "1617/4324 graded, 90.2 %
Stadium II-III/III" figure in CLAUDE.md. Consumer: `analysis/acuity_stratified_transport.py`.

---

## 1.3 Lead-order convention and the historical aVL/aVF bug

**Current rule** (`.claude/rules/data-pipeline.md`, "Lead order"): reindex by WFDB `sig_name`,
never by position, on **both** domains; target order `[I, II, III, aVR, aVL, aVF, V1..V6]`;
no permutation.

**The bug** (`reports/2026-08-10_lead_order_bug_diagnostic.md` §1). `LVEF_12lead_cls_Dataset`
— the MedalCare loader used by *every* training run and *every* MedalCare latent export —
used to declare `input_leads = ['I','II','III','aVR','aVF','aVL','V1'..'V6']` and apply
`data = data[self.lead_indices, :]`, permuting positions 4↔5. `wfdb.rdsamp` discards
`sig_name` so nothing cross-checked it. The assumption was false:
`prepare_medalcare.py:53` writes standard order and the manifest's `lead_order` column reads
literally `I,II,III,aVR,aVL,aVF,V1,...,V6`.

**Physics identity check** (the citable evidence). Exact limb-lead identities
`aVR = −(I+II)/2`, `aVL = (I−III)/2`, `aVF = (II+III)/2`. Relative RMS error of each stored
channel against each identity (`reports/2026-08-10_lead_order_bug_diagnostic.md`, table in §1):

| | ch4 vs aVL | ch4 vs aVF | ch5 vs aVL | ch5 vs aVF |
|---|---|---|---|---|
| MedalCare (6 records) | **0.0092–0.2611** | 1.3843–1.7622 | 2.0652–2.9683 | **0.0080–0.1163** |
| PTB-XL (`00001_hr`) | **0.0057** | 1.8434 | 1.1444 | **0.0093** |

→ MedalCare channel 4 **is** aVL, channel 5 **is** aVF. PTB-XL was never affected because
`PTBXLDataset._reorder_leads` (`scripts/datasets.py:744-757`) always reindexed by name.
**Cite this identity, not the downstream p-value** (CLAUDE.md; the Stage-4.2 sweep shows
39/66 arbitrary transpositions also improve transfer).

**Fix** (working tree): `scripts/datasets.py:114-133` (the FIXED note), `:134-135`
(`target_leads`), `:185-207` (`_reorder_leads` mirroring PTB-XL). Four dead classes
(`LVEF_12lead_cls_Dataset_Marta`, `LVEF_12lead_reg_Dataset`, `LVEF_1lead_cls_Dataset`,
`LVEF_1lead_reg_Dataset`) still carry the wrong declaration and are deliberately not fixed —
banner at `scripts/datasets.py:15-31`; none is in `DATASET_REGISTRY` (`:789-792`).

**Second, independent mismatch — normalisation.** MedalCare used a single global scalar
z-score, PTB-XL per-lead (`reports/2026-08-10_lead_order_bug_diagnostic.md` §3, "Second
mismatch"). Now per-lead on both sides by default (`scripts/datasets.py:136-141, :173-183`);
`--global-z` (`scripts/finetune_multilabel.py:379-387`) reproduces the legacy arm as
`exp8_leadfix_globalz`.

**Diagnostic result carried forward** (`reports/2026-08-10_lead_order_bug_diagnostic.md` §3):
with the probe fit on MedalCare-train MI and scored on PTB-XL n=438, `territory_4c`
macro-F1 moves 0.2132 (p=0.756) → **0.3278 (p=0.0020)** when MedalCare leads are corrected.
Adding per-lead normalisation gives the project's best cross-domain macro-AUC (0.5870) but a
*worse* macro-F1 (0.2416) — ranking improves, the argmax boundary degrades.

---

# 2. ENCODER AND TRAINING RUNS

## 2.1 ECGFounder / Net1D architecture

Builder `ft_12lead_ECGFounder` — `finetune_model.py:23-62`. Backbone config
(`finetune_model.py:26-41`, identical in `ft_multihead_ECGFounder` `:109-127` and in
`scripts/export_latents.py:56-69` `NET1D_ARCH`):

```
in_channels   = 12
base_filters  = 64
ratio         = 1
filter_list   = [64, 160, 160, 400, 400, 1024, 1024]
m_blocks_list = [2, 2, 2, 3, 3, 4, 4]
kernel_size   = 16
stride        = 2
groups_width  = 16
use_bn        = False
use_do        = False
```

- Input `(B, 12, L)`; first conv is `MyConv1dPadSame(12→64, k=16, stride=2)` (`net1d.py:367-371`),
  followed by 7 stages.
- **1024-d pooled features**: `deep_features = out.mean(-1)` (`net1d.py:418`), cached as
  `self.last_features` (`:420`) — this is the latent `Z`.
- Head: `self.dense = nn.Linear(in_channels=1024, n_classes)` (`net1d.py:401`); the pretrained
  `dense.*` weights are stripped on load and a fresh head is attached
  (`finetune_model.py:46-50`).
- Checkpoint: `checkpoint/12_lead_ECGFounder.pth` (369,942,585 bytes) — the value of
  `--checkpoint` in every `exp8_*` `args.json`. A 1-lead variant `1_lead_ECGFounder.pth` also
  exists but is unused by any run here.
- `MultiHeadECGFounder` (`net1d.py:429-511`): shared Net1D backbone (`return_features=True`)
  + `head_medal` (Linear 1024→8), `head_ptb` (Linear 1024→5), optional `head_physics`
  (Linear, or Linear→ReLU→Dropout→Linear when `physics_hidden>0`) (`:473-489`).
  `forward(x, task)` selects the head (`:491-511`).

### `ConvAdapter1D`

`net1d.py:98-115`. "Bottleneck residual adapter: 1×1 down → nonlinearity → 1×1 up, residual add."

```
hidden = max(8, channels // reduction)      # reduction default 16      (:105)
down   = Conv1d(channels, hidden, k=1, bias=False)                      (:106)
act    = ReLU                                                           (:107)
drop   = Dropout(p) or Identity                                         (:108)
up     = Conv1d(hidden, channels, k=1, bias=False)                      (:109)
nn.init.zeros_(up.weight)      # initialised to exact identity          (:112)
forward: x + up(drop(act(down(x))))                                     (:115)
```

### Freezing conventions and the known bug

- `freeze_backbone_except_adapters(backbone)` — `finetune_model.py:12-20`: freeze everything,
  then unfreeze every `ConvAdapter1D`.
- **⚠ `linear_prob=True` + `use_adapter=True` is BUGGED** in `ft_12lead_ECGFounder`: the
  combination calls `freeze_backbone_except_adapters(model)` which also freezes `model.dense`
  — `finetune_model.py:52-58`. Silent broken training.
- **Workaround used by every shared-head run**: pass `linear_prob=False`, then
  `freeze_backbone_except_adapters(model)` followed by re-enabling `model.dense`
  — `scripts/finetune_multilabel.py:1364-1375`. Trainable parameter counts printed at
  `:1390-1392`; measured on `exp8_leadfix_medalonly`: **head 3,075 + adapters 309,568 =
  312,643** (`outputs/_log_exp8_medalonly.txt:13`).
- `ft_multihead_ECGFounder` freezes `model.backbone` only (`finetune_model.py:135-140`) —
  unaffected.
- Optimiser param groups: head at `--lr-head` (default 1e-3), adapters at `--lr-encoder`
  (default 1e-5) — `scripts/finetune_multilabel.py:1377-1388`.
- Loss: `BCEWithLogitsLoss(pos_weight=...)` with pos_weight from the combined post-filter class
  balance (`:1395-1396`, `:849-863`).

## 2.2 Fine-tuning modes

| mode | flag | entry point | label space |
|---|---|---|---|
| single-domain | `--dataset {medalcare,ptbxl}` | `main()` fallthrough, `scripts/finetune_multilabel.py:2010+` | native (8 or 5) |
| joint dual-head | `--joint-datasets medalcare+ptbxl --multi-head` | `MultiHeadECGFounder` | native per domain |
| dual-head, shared labels | `--dual-head-shared-labels` | `_run_dual_head_shared_labels` (`:2008-2011`) | 3-class both heads |
| **shared head** | `--shared-head` | `_run_shared_head` (`:1336`) | single 3-class head |
| shared head, MedalCare only | `--shared-head --medalcare-only` | same, PTB-XL train loader set to `None` (`:876`) | 3-class |
| bottleneck head | `scripts/finetune_bottleneck.py --bottleneck-dim K` | `BottleneckHead` (`:77-94`) | 3-class |
| bottleneck + physics multitask | `scripts/finetune_bottleneck_multitask.py` | `--lambda-cls`, `--lambda-bio` | 3-class + 5 bio channels |

`--medalcare-only` semantics (`scripts/finetune_multilabel.py:354-365`): no PTB-XL gradient
step, `pos_weight` computed from MedalCare alone (`:851-857`), PTB-XL val/test still *scored*
each epoch but never enters `val_score`, the scheduler, or checkpoint selection. Guarded:
requires `--shared-head` and is incompatible with `--lambda-mmd>0` (`:1992-2003`).

`BottleneckHead` (`scripts/finetune_bottleneck.py:77-94`):
`Linear(1024, K) → GELU → Linear(K, 3)`; the **pre-GELU** projection is cached as
`last_z_k` and is what `Z` means for bottleneck exports (`:80-94`). Only the head trains
(`:149`).

### MMD losses (`losses/mmd.py`)

- `_rbf_kernel(x, y, sigma=None)` — `:13-41`; median heuristic when `sigma is None` (`:32-38`).
- `mmd_rbf(x, y, sigma=None)` — `:43-76`, unbiased RBF MMD²; returns 0 when either side has
  < 2 rows (`:58-60`).
- `mmd_rbf_class_conditional(feat_x, feat_y, labels_x, labels_y, sigma=None, min_samples=2)`
  — `:77-114`: per-class MMD over the shared label space, mean over classes with ≥
  `min_samples` in *both* domains; returns 0 if no class qualifies.
- Selected by `--lambda-mmd` and `--class-cond-mmd` (`scripts/finetune_multilabel.py:376-378`).
- Metric suite: `metrics/multilabel.py` — accuracy, f1, recall, specificity, precision,
  brier, roc_auc (`compute_multilabel_metrics`, `:152`).

## 2.3 Every training run on disk

19 directories carry `metrics.json`; **only 11 carry `args.json`** — `save_run_args`
(`scripts/finetune_multilabel.py:419-441`) did not exist before the 2026-08-10 audit
(absent from `git show pre-leadfix:scripts/finetune_multilabel.py` [computed]), so the exact
invocation of exp5/exp6/exp7/joint/ptbxl_baselines is **unrecoverable**. Fields marked `—`
below are not recorded anywhere; fields marked `(inferred)` come from
`.claude/rules/experiments.md` or the run-ID glossary, not from an artifact.

### 2.3.1 Config table

| run_id | args.json? | mode | label space | epochs (run/`--epochs`) | batch | lr | λ_mmd | K | medal-only | trained on | leadfix? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ptbxl_baselines/linear/ptbxl_baseline` | no | single | PTB-XL 5 superclass | 30 / — | — | — | — | — | — | PTB-XL folds 1–8 | **pre** |
| `joint_baseline` | no | joint dual-head | native 8 + 5 (+ 51-θ physics head) | 30 / — | — | — | 0 (inferred) | — | — | both | **pre** |
| `joint_adapter_cls` | no | joint dual-head + adapters | native 8 + 5 | 30 / — | — | — | 0 (inferred) | — | — | both | **pre** |
| `joint_adapter_mmd` | no | joint dual-head + adapters + MMD | native 8 + 5 | 18 / — | — | — | >0 (inferred) | — | — | both | **pre** |
| `exp5_3class` | no | dual-head, no alignment (inferred) | 3-class | 17 / — | — | — | 0 | — | — | both | **pre** |
| `exp6_3class` | no | dual-head + ccMMD λ=0.1 (inferred) | 3-class | 17 / — | — | — | 0.1 cc | — | — | both | **pre** |
| `exp7_baseline` | no | shared-head, no alignment | 3-class | 30 / — | — | — | 0 | — | — | both | **pre** |
| `exp7_baseline_norm` | no | shared-head, per-lead-norm ablation *(provenance only in `reports/2026-08-10_repo_audit_and_rerun_plan.md:57`)* | 3-class | 30 / — | — | — | 0 | — | — | both | **pre** (still had swapped leads — `reports/2026-08-10_repo_audit_and_rerun_plan.md:122`) |
| `exp7_ccmmd` | no | shared-head + ccMMD | 3-class | 30 / — | — | — | 0.1 cc | — | — | both | **pre** |
| `exp7_bottleneck_K16` | yes | bottleneck head on `exp7_baseline` ckpt | 3-class | 20 / 20 | 128 | 1e-3 | — | 16 | — | both | **pre** |
| `exp7_bottleneck_K64` | yes | same | 3-class | 15 / 20 | 128 | 1e-3 | — | 64 | — | both | **pre** |
| `exp7_bottleneck_K256` | yes | same | 3-class | 14 / 20 | 128 | 1e-3 | — | 256 | — | both | **pre** |
| `exp7_tier2_K64_A_5050` | yes | bottleneck multitask, `lambda_cls=0.5 lambda_bio=0.5`, `adapter_trainable=True` | 3-class + 5 bio | 15 / 15 | 128 | 1e-3 | — | 64 | — | both | **pre** |
| `exp7_tier2_K64_B_bioonly` | yes | same, `lambda_cls=0.0 lambda_bio=1.0` | bio only | 10 / 15 | 128 | 1e-3 | — | 64 | — | both | **pre** |
| `exp8_leadfix_baseline` | yes | `--shared-head` | 3-class | 30 / 30 | 128 | head 1e-3 / enc 1e-5 | 0.0 | — | no | both | **post** |
| `exp8_leadfix_ccmmd` | yes | `--shared-head --lambda-mmd 0.1 --class-cond-mmd` | 3-class | 19 / 30 | 128 | head 1e-3 / enc 1e-5 | 0.1 cc | — | no | both | **post** |
| `exp8_leadfix_dual` | yes | `--dual-head-shared-labels --lambda-mmd 0` | 3-class ×2 heads | 19 / 20 | 128 | head 1e-3 / enc 1e-5 | 0.0 | — | no | both | **post** |
| `exp8_leadfix_globalz` | yes | `--shared-head --global-z` | 3-class | 18 / 30 | 128 | head 1e-3 / enc 1e-5 | 0.0 | — | no | both | **post** |
| `exp8_leadfix_medalonly` | yes | `--shared-head --medalcare-only` | 3-class | 15 / 30 | 128 | head 1e-3 / enc 1e-5 | 0.0 | — | **yes** | **MedalCare only** | **post** |
| `exp8_leadfix_K64` | yes | bottleneck head on `exp8_leadfix_baseline` ckpt | 3-class | 20 / 20 | 128 | 1e-3 | — | 64 | — | both | **post** |

Shared `args.json` fields across all `exp8_*` full runs: `seed=42`, `weight_decay=1e-5`,
`label_smoothing=0.05`, `grad_clip=1.0`, `early_stop_lr=1e-5`, `head_type=linear`,
`use_adapter=False`+`no_adapter=False` (the *argparse* flags — the shared-head path
hard-codes `use_adapter=True` at `scripts/finetune_multilabel.py:1369`, so the flag is
inert there), `physics_hidden=256`, `physics_loss=mse`, `lambda_phys=1.0`,
`metrics=accuracy,f1,recall,specificity,precision,brier,roc_auc`, `num_workers=0`,
`manifest=data/medalcare_filtered_manifest_dataset_split.csv`,
`checkpoint=checkpoint/12_lead_ECGFounder.pth`.
Bottleneck runs additionally: `patience=5` (K16/K64/K256/exp8_K64) or `4` (tier2),
`checkpoint=outputs/<parent>/checkpoints/linear_best.pt`.

Recorded `_argv` (exp8 only, `args.json→_argv`):
- baseline: `--epochs 30 --batch-size 128 --num-workers 0 --seed 42 --shared-head --run-id exp8_leadfix_baseline` (written 2026-08-11T02:50:14)
- ccmmd: `... --shared-head --run-id exp8_leadfix_ccmmd --lambda-mmd 0.1 --class-cond-mmd` (03:41:32)
- dual: `--epochs 20 ... --dual-head-shared-labels --lambda-mmd 0 --run-id exp8_leadfix_dual` (04:25:53)
- globalz: `... --shared-head --run-id exp8_leadfix_globalz --global-z` (05:04:34)
- medalonly: `... --shared-head --medalcare-only --run-id exp8_leadfix_medalonly` (2026-08-11T21:56:32)

### 2.3.2 Checkpoint selection and headline test metrics

Selection metric is `avg_domain_f1` for every 3-class run (`metrics.json→best.primary_metric.name`),
plain `f1` (MedalCare val) for the joint/native runs, and `val_loss_bio` for the bio-only
tier-2 run. Best-so-far snapshots are `checkpoints/checkpoint_{epoch}_{score}.pth`; the
selected weights are `checkpoints/linear_best.pt`
(`.claude/rules/experiments.md`; **there is no `best_model.pt`**).

| run_id | best epoch | best val score | last snapshot filename | MedalCare test macro-F1 | MedalCare test macro-AUC | PTB-XL test macro-F1 | PTB-XL test macro-AUC |
|---|---:|---:|---|---:|---:|---:|---:|
| `ptbxl_baseline` | 29 | 0.6998 (f1) | `checkpoint_29_0.6998.pth` | — | — | **0.6982** | 0.9115 |
| `joint_baseline` | 18 | 0.6474 (f1) | `checkpoint_18_0.6474.pth` | 0.6624 | 0.9801 | 0.7081 | 0.9167 |
| `joint_adapter_cls` | 27 | 0.6622 | `checkpoint_27_0.6622.pth` | 0.6793 | 0.9814 | 0.7065 | 0.9159 |
| `joint_adapter_mmd` | 14 | 0.6540 | `checkpoint_14_0.6540.pth` | 0.6691 | 0.9816 | 0.7037 | 0.9172 |
| `exp5_3class` | 13 | 0.8423 | `checkpoint_13_0.8423.pth` | 0.8667 | 0.9950 | 0.7860 | 0.9438 |
| `exp6_3class` | 13 | 0.8433 | `checkpoint_13_0.8433.pth` | 0.8659 | 0.9948 | 0.7870 | 0.9438 |
| `exp7_baseline` | 29 | 0.8587 | `checkpoint_29_0.8587.pth` | 0.9166 | 0.9956 | 0.7876 | 0.9351 |
| `exp7_baseline_norm` | 30 | 0.8582 | `checkpoint_30_0.8582.pth` | 0.9381 | 0.9980 | 0.7931 | 0.9340 |
| `exp7_ccmmd` | 29 | 0.8589 | `checkpoint_29_0.8589.pth` | 0.9161 | 0.9954 | 0.7869 | 0.9346 |
| `exp7_bottleneck_K16` | 17 | 0.8663 | `checkpoint_17_0.8663.pth` | 0.9440 | 0.9978 | 0.7904 | 0.9384 |
| `exp7_bottleneck_K64` | 10 | 0.8736 | `checkpoint_10_0.8736.pth` | 0.9371 | 0.9971 | 0.7950 | 0.9405 |
| `exp7_bottleneck_K256` | 9 | 0.8783 | `checkpoint_9_0.8783.pth` | 0.9429 | 0.9981 | 0.7941 | 0.9393 |
| `exp7_tier2_K64_A_5050` | 11 | 0.8866 | *(only `linear_best.pt` kept)* | 0.9887 | 0.9999 | 0.7893 | 0.9303 |
| `exp7_tier2_K64_B_bioonly` | 6 | 0.3552 (`val_loss_bio`, minimised) | *(only `linear_best.pt`)* | 0.1875 | 0.3019 | 0.2774 | 0.3721 |
| `exp8_leadfix_baseline` | 30 | 0.8881 | `checkpoint_30_0.8881.pth` | 0.9275 | 0.9975 | **0.8280** | 0.9404 |
| `exp8_leadfix_ccmmd` | 15 | 0.8797 | `checkpoint_15_0.8797.pth` | 0.9289 | 0.9972 | 0.8234 | 0.9380 |
| `exp8_leadfix_dual` | 15 | 0.8804 | `checkpoint_15_0.8804.pth` | 0.9269 | 0.9979 | 0.8345 | 0.9451 |
| `exp8_leadfix_globalz` | 14 | 0.8791 | `checkpoint_14_0.8791.pth` | 0.9104 | 0.9950 | 0.8234 | 0.9392 |
| `exp8_leadfix_medalonly` | 11 | 0.9230 (MedalCare val f1 alone) | `checkpoint_11_0.9230.pth` | 0.9268 | 0.9980 | **0.4567** | 0.6267 |
| `exp8_leadfix_K64` | 18 | 0.9023 | `checkpoint_18_0.9023.pth` | 0.9521 | 0.9967 | 0.8319 | 0.9435 |

All from `outputs/<run_id>/metrics.json → best.test.{medalcare,ptbxl}.{f1,roc_auc}`;
`best.epoch`, `best.primary_metric.value`.

Notes:
- **The pre/post-leadfix PTB-XL F1 jump (≈0.787 → ≈0.828) is not an alignment result** — the
  pre-fix runs were scoring a different filtered PTB-XL subset (STTC instead of CD, §1.2.3),
  so the two columns are not comparable. Do not present them as a before/after.
- `exp8_leadfix_medalonly`'s PTB-XL numbers (0.4567 / 0.6267) are a *passive readout* — that
  domain never entered the loss or checkpoint selection.
- `exp7_tier2_K64_B_bioonly` classification metrics are below chance by construction
  (`lambda_cls=0.0`); its bio metrics are the point:
  `metrics.json→best.test.bio`: `phi_r2_circular = 0.6856`, `trans_auc = 0.9763`,
  `r2_z = 0.5177`, `r2_size = 0.3757`.
  `exp7_tier2_K64_A_5050` `→best.test.bio`: `phi_r2_circular = 0.6627`, `trans_auc = 0.9775`,
  `r2_z = 0.5292`, `r2_size = 0.3418`. Bio channels and their z-score stats are recorded at
  `metrics.json→bio_channels` = `['phi_sin','phi_cos','z_std','size_std','trans_logit']`
  and `→z_score_stats` = `{z_mean 0.5832, z_std 0.2423, size_mean 124.98, size_std 28.87,
  trans_thresh 0.5}`.
- `joint_baseline` also emits `physics_metrics.json` with a **51-name θ vector**
  (`APD.max, APD.min, APD.rho_d, APD.v_d, APD.z_d, LA/LL/RA/RL/V1–V6 {phi,rho,z},
  cv_t.BulkTissue, stim[0..4].{phi,thr,z,time}`) and per-name `mae_norm`. This is a *different*
  θ from the B2 four-parameter `isch[0].*` set — do not conflate them.

---

# 3. LATENT EXPORTS

`scripts/export_latents.py` (1024-d) / `scripts/export_bottleneck_latents.py` (K-d).
Output path convention `outputs/latents/<prefix>_<domain>[_<split>]/latents.npz`
(`export_latents.py:110-111, :357`). Keys:

| key | meaning |
|---|---|
| `Z` | pooled features — 1024-d `deep_features` (`net1d.py:418`) or the pre-GELU bottleneck projection (`finetune_bottleneck.py:80-94`) |
| `Z_post_gelu` | post-GELU activation, bottleneck exports only |
| `P` | head outputs / probabilities — width = the run's label space (3, 5 or 8) |
| `Y` | ground-truth labels in the run's native space (MedalCare 8, PTB-XL 5) |
| `ecg_id` | PTB-XL `ecg_id` — present **only** in the two exports written after 2026-08-11 |

Row order = the unshuffled DataLoader over
`db[db.strat_fold.isin(folds)].reset_index(drop=True)` (PTB-XL) or
`df[df.split==X].reset_index(drop=True)` (MedalCare) — `export_latents.py:328, :343-344`;
relied on by `analysis/geom_common.py:115-127`, which states it was verified byte-for-byte
against the one export that stores `ecg_id`.
**Exports are on the UNFILTERED split (2386 / 2434 / 2198 / 17418 / 2183 / 12019), not the
3-class-filtered subset of §1.2.3.**

Reference row counts: MedalCare train 12019 / val 2434 / test 2386; PTB-XL train 17418 /
val 2183 / test 2198 / all-folds 21799.

**127 directories** [computed]. All arrays are `float32` except `ecg_id` (`int64`).

### 3.1 Pre-Exp-7 exports (native label spaces)

| directory | Z | P | Y | source encoder |
|---|---|---|---|---|
| `exp1_ptbxl` | (2198, 1024) | (2198, 5) | (2198, 5) | `ptbxl_baselines/linear/ptbxl_baseline` (per `export_latents.py:9-13`) |
| `exp4_ptbxl` | (2198, 1024) | (2198, 5) | (2198, 5) | **uncertain** — one of the `joint_*` runs; no manifest records it |
| `exp5_medalcare` | (2386, 1024) | (2386, 8) | (2386, 8) | `joint_adapter_cls` (per `export_latents.py:21-27`) |
| `exp5_ptbxl` | (2198, 1024) | (2198, 5) | (2198, 5) | `joint_adapter_cls` (`export_latents.py:15-20`) |
| `exp6_medalcare` | (2386, 1024) | (2386, 8) | (2386, 8) | **uncertain** — presumably `joint_adapter_mmd` |
| `exp6_ptbxl` | (2198, 1024) | (2198, 5) | (2198, 5) | **uncertain** — same |

### 3.2 Exp 5/6 3-class exports

| directory | Z | P | Y |
|---|---|---|---|
| `exp5_3class_medalcare` | (2386, 1024) | (2386, 3) | (2386, 8) |
| `exp5_3class_medalcare_train` | (12019, 1024) | (12019, 3) | (12019, 8) |
| `exp5_3class_ptbxl` | (2198, 1024) | (2198, 3) | (2198, 5) |
| `exp6_3class_medalcare` | (2386, 1024) | (2386, 3) | (2386, 8) |
| `exp6_3class_medalcare_train` | (12019, 1024) | (12019, 3) | (12019, 8) |
| `exp6_3class_ptbxl` | (2198, 1024) | (2198, 3) | (2198, 5) |

Source encoders `outputs/exp5_3class`, `outputs/exp6_3class`
(`analysis/phase_b2_infarct_decoding.py:80-81` `CONFIG_LATENT_STEMS`).

### 3.3 Exp 7 shared-head 1024-d exports

Naming quirk: for `exp7_baseline` the stem is bare **`exp7`**
(`analysis/phase_b2_infarct_decoding.py:82`), and the MedalCare *test* split has no `_test`
suffix (`scripts/build_medalcare_isch_targets.py:61-69`).

| directory | Z | P | Y | encoder |
|---|---|---|---|---|
| `exp7_medalcare` (= test) | (2386, 1024) | (2386, 3) | (2386, 8) | `exp7_baseline` |
| `exp7_medalcare_train` | (12019, 1024) | (12019, 3) | (12019, 8) | `exp7_baseline` |
| `exp7_ptbxl` (= fold 10) | (2198, 1024) | (2198, 3) | (2198, 5) | `exp7_baseline` |
| `exp7_ptbxl_train` | (17418, 1024) | (17418, 3) | (17418, 5) | `exp7_baseline` |
| `exp7_ccmmd_medalcare` | (2386, 1024) | (2386, 3) | (2386, 8) | `exp7_ccmmd` |
| `exp7_ccmmd_medalcare_train` | (12019, 1024) | … | … | `exp7_ccmmd` |
| `exp7_ccmmd_ptbxl` | (2198, 1024) | … | … | `exp7_ccmmd` |
| `exp7_ccmmd_ptbxl_train` | (17418, 1024) | … | … | `exp7_ccmmd` |
| `exp7_norm_medalcare` | (2386, 1024) | (2386, 3) | (2386, 8) | `exp7_baseline_norm` |
| `exp7_norm_ptbxl` | (2198, 1024) | (2198, 3) | (2198, 5) | `exp7_baseline_norm` |

INLP-aligned variants (`--output-suffix _inlp` / `_inlpv2`; `analysis/inlp_alignment.py:106,
:620`), identical shapes to their parents:
`exp7_medalcare_inlp`, `exp7_medalcare_inlpv2`, `exp7_medalcare_train_inlp`,
`exp7_medalcare_train_inlpv2`, `exp7_ptbxl_inlp`, `exp7_ptbxl_inlpv2`,
`exp7_ptbxl_train_inlp`, `exp7_ptbxl_train_inlpv2`, `exp7_ccmmd_medalcare_inlp`,
`exp7_ccmmd_medalcare_train_inlp`, `exp7_ccmmd_ptbxl_inlp`, `exp7_ccmmd_ptbxl_train_inlp`.

Lead-order diagnostic exports (from `scripts/_diag_leadswap_ptbxl.py`), all on the
`exp7_baseline` checkpoint:
`exp7_medalcare_unswapped` (2386, 1024), `exp7_medalcare_train_unswapped` (12019, 1024),
`exp7_medalcare_unswapped_perlead` (2386, 1024),
`exp7_medalcare_train_unswapped_perlead` (12019, 1024),
`exp7_ptbxl_leadswap` (2198, 1024).

### 3.4 Bottleneck exports (Exp 7 tier 1 / tier 2)

All carry `Z`, `Z_post_gelu`, `P`, `Y`. Row counts per domain/split as above.

| family | K | directories |
|---|---:|---|
| `exp7_bottleneck_K16_*` | 16 | `medalcare_{train,val,test}`, `ptbxl_{train,val,test}` + `_inlp` variant of each (12 dirs) |
| `exp7_bottleneck_K64_*` | 64 | same 6 + `_inlp` + `_inlpv2` variants (18 dirs) |
| `exp7_bottleneck_K256_*` | 256 | same 6 + `_inlp` (12 dirs) |
| `exp7_tier2_K64_A_5050_*` | 64 | `medalcare_{train,val,test}`, `ptbxl_{train,val,test}` (6 dirs) |
| `exp7_tier2_K64_B_bioonly_*` | 64 | same 6 dirs |
| `exp8_leadfix_K64_*` | 64 | same 6 dirs |

Shapes are `(n, K)` for both `Z` and `Z_post_gelu`, `(n, 3)` for `P`, `(n, 8)`/`(n, 5)` for `Y`,
with n ∈ {12019, 2434, 2386} (MedalCare train/val/test) and {17418, 2183, 2198} (PTB-XL).

### 3.5 Post-leadfix (`exp8_*`) 1024-d exports

| directory | Z | P | Y | extra |
|---|---|---|---|---|
| `exp8_leadfix_baseline_medalcare_{train,val,test}` | (12019/2434/2386, 1024) | (·, 3) | (·, 8) | — |
| `exp8_leadfix_baseline_ptbxl_{train,val,test}` | (17418/2183/2198, 1024) | (·, 3) | (·, 5) | — |
| `exp8_leadfix_ccmmd_*` (same 6) | as above | | | — |
| `exp8_leadfix_dual_*` (same 6) | as above | | | — |
| `exp8_leadfix_globalz_*` (same 6) | as above | | | — |
| `exp8_leadfix_medalonly_medalcare_{train,test}` | (12019/2386, 1024) | (·, 3) | (·, 8) | **no `_val` export** |
| `exp8_leadfix_medalonly_ptbxl_test` | (2198, 1024) | (2198, 3) | (2198, 5) | `ecg_id` (2198,) int64 |
| `exp8_leadfix_medalonly_ptbxl` | **(21799, 1024)** | (21799, 3) | (21799, 5) | `ecg_id` (21799,) int64 — the all-folds export |

`analysis/geom_common.py:40-47` `ENCODERS` lists the six post-leadfix encoders usable by the
geometry pipeline: `exp8_leadfix_{medalonly, baseline, ccmmd, dual, globalz, K64}`.
For the five that lack an all-folds export, `_ptbxl_allfolds_latents` (`:130-155`) stitches
train/val/test and reconstructs `ecg_id` from the database (`:115-127`), raising if row
counts disagree.

---

# 4. HAND-CRAFTED FEATURE SETS

Both extractors read **raw voltages** via `wfdb.rdsamp` directly, deliberately bypassing the
Dataset wrappers, because z-scoring destroys the voltage-scale features
(`scripts/extract_ecg_features_neurokit2.py:16-20`). Both reindex to
`LEADS_12 = ("I","II","III","aVR","aVL","aVF","V1".."V6")` by `sig_name` when available
(`:238-241, :256-258`).

## 4.1 `global6` — `scripts/extract_ecg_features_neurokit2.py`

`FEATURE_NAMES` (`:77-84`), in order:

| idx | name | definition | pointer |
|---:|---|---|---|
| 0 | `QRS_duration_ms` | median (R_offset − R_onset)·1000/fs, clipped to (30, 300) ms | `:155-159` |
| 1 | `QT_interval_ms` | median (T_offset − R_onset)·1000/fs, clipped to (200, 700) ms | `:161-165` |
| 2 | `P_duration_ms` | median (P_offset − P_onset)·1000/fs, clipped to (30, 250) ms | `:167-171` |
| 3 | `ST_J60_avg_mV` | voltage at R_offset + 60 ms, **averaged over V2–V6**, median across beats | `:173-185`, `ST_LEADS` `:73` |
| 4 | `T_amplitude_mV` | median voltage at T-peaks, **lead II only** | `:187-192` |
| 5 | `heart_rate_bpm` | 60 / median RR from R-peaks | `:194-200` |

Delineation (`:126-140`): `nk.ecg_clean` → `nk.ecg_peaks` → `nk.ecg_delineate(method="dwt")`,
all on **lead II** (`LEAD_II_IDX`, `:74`). Fails with `fewer_than_2_R_peaks` if < 2 R-peaks
(`:132-134`).

Outputs `data/ecg_features_{medalcare_train, medalcare_test, ptbxl_test}.npz`, keys
`features (n, 6) float32`, `nk2_ok (n,) bool`, `feature_names` (`:318-321`). Non-MI rows are
NaN by construction (only MI indices are processed).

Coverage (`data/ecg_features_summary.json`):

| dataset/split | rows | MI rows processed | all-6 OK | ok_rate | per-feature finite (QRS/QT/P/ST/T/HR) |
|---|---:|---:|---:|---:|---|
| medalcare train | 12019 | 5347 | 3711 | 0.6940 | 4171 / 4257 / 4281 / 5283 / 5285 / 5319 |
| medalcare test | 2386 | 1200 | 701 | 0.5842 | 797 / 832 / 887 / 1191 / 1192 / 1194 |
| ptbxl test (fold 10) | 2198 | 550 | 399 | 0.7255 | 462 / 464 / 490 / 548 / 548 / 549 |

Failure reasons (same file): medalcare train `fewer_than_2_R_peaks` 23,
`nk_pipeline_failed:ValueError` 5; medalcare test 5 / 1; ptbxl test 0 / 1.

## 4.2 `spatial54` — `scripts/extract_ecg_features_spatial.py`

**Why it exists** — docstring `:1-26`: the 6-feature control has *no* spatial content
(4 global scalars, one anterior-averaged ST, one inferior T) and cannot represent the
anterior-vs-inferior contrast it is asked to predict; it is also 6-vs-1024 wide. The 54-set is
a **strict superset** of the 6-set (`:38-40`), so any drop is estimation cost, never lost
information.

Construction (`:101-107`): `PER_LEAD_KINDS = ("ST_J60","Q_amp","R_amp","T_amp")`,
`FEATURE_NAMES = [f"{kind}_{lead}" for kind in PER_LEAD_KINDS for lead in LEADS_12] + GLOBAL6`,
so `N_FEATURES = 54`, `N_PER_LEAD = 48`. Column layout confirmed from the stored NPZ
[computed] and echoed in `analysis/fidelity_audit.py:100-107`:

| columns | block | names |
|---|---|---|
| 0–11 | **ST_J60 × 12** | `ST_J60_{I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6}` |
| 12–23 | **Q_amp × 12** | `Q_amp_{same 12}` |
| 24–35 | **R_amp × 12** | `R_amp_{same 12}` |
| 36–47 | **T_amp × 12** | `T_amp_{same 12}` |
| 48–53 | **globals × 6** | `QRS_duration_ms, QT_interval_ms, P_duration_ms, ST_J60_avg_mV, T_amplitude_mV, heart_rate_bpm` |

Per-lead definitions (`:165-201`):
- `ST_J60_<lead>` — voltage at `R_offset + round(0.060·fs)`, median across beats, read in every lead (`:166-171`).
- `Q_amp_<lead>` — **most negative** sample in `[R_onset, R_peak)`, median across beats (`:176-189`). Rationale (`:173-175`): pathological Q waves are the *chronic*-MI localisation marker, and PTB-XL MI is predominantly old infarction.
- `R_amp_<lead>` — voltage at the R-peak instant, median across beats (`:192-196`).
- `T_amp_<lead>` — voltage at the T-peak instant, median (`:199-201`).

**All 12 leads are read at the SAME sample index — the fiducial found on lead II**
(`:42-46`): "That is deliberate: it samples the instantaneous QRS/T vector across the frontal
and precordial planes… Per-lead re-delineation would misalign the leads and destroy that."
Delineation itself: `_fiducials` (`:110-121`) — `nk.ecg_clean` → `nk.ecg_peaks` →
`nk.ecg_delineate(method="dwt")` on lead II. The 6 globals are computed by calling
`extract_features_one_ecg` verbatim (`:146-147`) so the shared columns cannot drift.

Outputs: `data/ecg_features_spatial_{medalcare_train, medalcare_test, ptbxl_test}.npz` plus
the all-folds `data/ecg_features_spatial_ptbxl_allfolds.npz` (via `--ptbxl-folds 1,..,10
--ptbxl-subclass-csv ... --ptbxl-out ...`, `:271-289`). The script **asserts** that the
subclass CSV row count matches the fold selection (`:315-320`).

Coverage (`data/ecg_features_spatial_summary.json`, and
`data/ecg_features_spatial_ptbxl_allfolds_summary.json` for the last row):

| dataset / split | total rows | MI processed | all-54 OK | ok_rate | complete blocks: ST_J60 / Q_amp / R_amp / T_amp / global6 |
|---|---:|---:|---:|---:|---|
| medalcare train | 12019 | 5347 | 3711 | 0.6940 | 5283 / 5277 / 5319 / 5285 / 3711 |
| medalcare test | 2386 | 1200 | 701 | 0.5842 | 1191 / 1179 / 1194 / 1192 / 701 |
| ptbxl test (fold 10) | 2198 | 550 | 399 | 0.7255 | 548 / 549 / 549 / 548 / 399 |
| **ptbxl folds 1–10** | **21799** | **5469** | **4087** | **0.7473** | 5415 / 5440 / 5453 / 5416 / 4087 |

Failure reasons (all-folds PTB-XL): `some_features_nan` 1366,
`nk_pipeline_failed:ValueError` 11, `fewer_than_2_R_peaks` 5.
**The binding constraint is always `global6`** (the interval features), never the per-lead
blocks — which is why the analysis pipelines median-impute rather than drop rows.

Stored array shapes [computed]:
`ecg_features_spatial_ptbxl_allfolds.npz` → `features (21799, 54) float32`,
`nk2_ok (21799,) bool` (sum 4087), `feature_names (54,) object`;
`ecg_features_spatial_medalcare_train.npz` → `features (12019, 54)`, `nk2_ok` sum 3711.
Analysis-side row selection: MedalCare via `theta_mi_*.npz→idx_in_split`
(`analysis/fidelity_audit.py:59-60`; same at `analysis/channel_repair.py:91`), PTB-XL via
`territory_4c.notna()` (`analysis/fidelity_audit.py:69-70`), giving the canonical matrices
**Fs (6547, 54)** and **Fr (4324, 54)** — asserted at `analysis/fidelity_audit.py:78-79`.
(6547 = MedalCare train 5347 + test 1200; the val split is not used by the geometry lane.)

**Imputation load on those two matrices** [computed]: all-54-finite rows are
**3254 / 4324 = 75.3 %** for PTB-XL (so **24.7 %**, n=1070, carry at least one imputed column)
and **4412 / 6547 = 67.4 %** for MedalCare. The 75.3 % figure is the one quoted in CLAUDE.md's
imputation-confound check.

## 4.3 Frontal QRS axis

Defined **only** as a derived feature inside the audit scripts, not in the extractors:

```
axis = arctan2(F[:, NAMES.index("R_amp_aVF")], F[:, NAMES.index("R_amp_I")])
```
— `analysis/fidelity_audit.py:81-84` (`i_aVF`, `i_I`, `axis_s`, `axis_r`), with a finiteness
mask at `:85-86`. Logged as "plus derived circular feature: frontal QRS axis =
atan2(R_amp_aVF, R_amp_I)" at `analysis/fidelity_audit.py:107`.
The same pair is the `axis2` block in `analysis/block_transfer.py:95`
(`[NAMES.index("R_amp_I"), NAMES.index("R_amp_aVF")]`) and the `axpair` block in
`analysis/channel_repair.py:197`. It does **not** appear in `analysis/geom_common.py`.

---

# 5. ANALYSIS PIPELINE DEFINITIONS

*(definitions and file pointers only — no results)*

## 5.1 Phase-B2 classifier pipeline — `analysis/phase_b2_infarct_decoding.py` (2924 lines)

**Scope** (docstring `:1-33`): in-domain (B2) and cross-domain (B2-CD) decoding of the four
`isch[0]` targets — φ (circular, sin/cos Ridge), z (Ridge), size (Ridge), `rho_eps_max`
(binary, Logistic) — plus the "Pipeline A" territory classifier.

### Pipeline A — direct territory classifier

`fit_territory_4c_classifier` (`:1184-1236`): **multinomial `LogisticRegression`**,
`penalty="l2"`, `solver="lbfgs"`, `class_weight="balanced"`, `max_iter=4000`,
`multi_class="multinomial"`.
- **C tuning**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)` on the
  *source* training rows only, scoring **macro-F1** over `TERRITORIES_4C`; the argmax C is
  refit on the full training set (`:1200-1235`).
- **C grid**: `LOGREG_CS_TERR_4C = np.logspace(-5, 2, 8)` = 1e-5 … 1e2 (`:146`); the 8-class
  probe reuses it verbatim (`LOGREG_CS_TERR_8C`, `:173`). The grid was widened from the binary
  probe's default because 1024-d Z favours much stronger L2 (comment `:144-145`).
- Label spaces: `TERRITORIES_4C = ["Anteroseptal","Anterolateral","Inferior","Inferolateral"]`
  (`:135`); `TERRITORIES_2C = ["Anterior","Inferior"]` (`:136`);
  `TERRITORY_4C_TO_2C` (`:137-142`); the legacy 3-class
  `TERRITORY_LABELS = ["Anterior","Inferior","Lateral"]` (`:124`) with
  `PHI_BIN_BOUNDARY = 2.0` (`:123`).
- 8-class in-domain audit: `TERRITORIES_8C` (`:151-157`), collapse maps to 4c anatomy
  (`:158-164`) and to 2c transmurality — `TRANSMURALITY_LABELS = ["0.3","1.0"]` (`:165`),
  `TERRITORY_8C_TO_TRANS` (`:166-172`).

### Scaler modes — `standardise_target` (`:451-533`)

`X_target` is always the **label-selected primary subset** (the ~438 / ~4324 rows with a known
territory); `pool` is the full unselected same-split matrix.

| `--scaler-domain` | what it fits on | code | docstring verdict |
|---|---|---|---|
| `source` | the MedalCare-train scaler, reused unchanged | `:510` | "a defect, not a baseline" — the domains disagree on per-coordinate spread by up to ~3× (`:461-467`) |
| **`target`** (CLI default, `:1839`+) | a fresh `StandardScaler` on `X_target` itself | `:512` | per-domain diagonal standardisation, AdaBN/CORAL lineage; transductive, unlabelled-target-only (`:469-473`) |
| `target_pool` (function default) | `StandardScaler` on `pool` | `:514-520` | avoids the subtle leak of fitting statistics on a label-selected subset (`:475-484`) |
| **`target_pool_measured`** = "**strict**" | column-wise `nanmean`/`nanstd` on the **un-imputed** pool, falling back to source stats for columns with < 2 real values | `:522-529`, `fit_scaler_nanaware` `:418-448` | exists because `target_pool` is corrupted for the *feature* arm (~75 % of pooled rows are pure MedalCare medians), which partially reconstructs the `source` defect and only for one arm (`:486-497`) |

The nomenclature in `reports/` and CLAUDE.md: **"strict" = `target_pool_measured`**,
**"legacy"/"`target`" = `target`** — see `scripts/_audit_paired_grid.py:28-31`
(run-dir labels) and `reports/2026-08-11_integrity_audit_and_probe_map.md:358`.
Missing values are median-imputed from the **training** column medians before any scaling —
`median_impute_with_train_medians` (`:384-409`), which also returns the per-feature
imputed-cell percentages for both splits. Source-side `StandardScaler` fit:
`fit_scaler` (`:412-415`).

### Scoring

- `_score_predictions` (`:1239-…`): macro-F1 + balanced accuracy, 1000-resample percentile
  bootstrap CIs, label-shuffle permutation p-values, per-class P/R/F1/support, confusion
  matrix; optional `collapse_map` to score the 2-class collapse **without refitting**.
- Constants: `N_BOOT = 1000` (`:181`), `N_PERM = 1000` for closed-form Ridge (`:182`),
  `N_PERM_BINARY = 10000` for LogReg refits (`:187`, raised from 200 so the p-floor 9.999e-5
  survives Holm), `SEED = 42` (`:188`).
- Ridge permutations use the intercept-aware hat-matrix trick
  `y_test_pred = K @ (y_train − ȳ) + ȳ` (`_ridge_K_matrix`, `_ridge_predict_perm` `:599-602`),
  which is what makes 1000 permutations affordable.
- **RNG discipline**: `derive_rng(*parts, seed=42)` (`:221-241`) keys a `default_rng` on
  SHA-256 of the joined cell identity, so a cell's CIs and p-values do not depend on which
  other cells were requested in the same invocation (the m10 fix). SHA-256 rather than
  `hash()` because Python's string hash is salted per process (`:233-237`).
- **Multiplicity**: `PRIMARY_ENDPOINTS` (`:206-220`) is a 5-member pre-registered family,
  fixed in code before the numbers were looked at (rationale `:190-205`); `holm_bonferroni`
  (`:246+`) corrects over it with an explicit `m_family` so a skipped endpoint cannot shrink
  the correction. The five: in-domain φ / z / size / `rho_eps_max` decodability from the
  1024-d latent, and "cross-domain 4-class territory transfer beats chance (Pipeline A)".

### `paired_macro_f1` — the arm-vs-arm test

`:837-894`. "Is arm A's macro-F1 different from arm B's, **on the same rows**?" The docstring
(`:846-863`) states the motivation directly: every published cross-domain number until
2026-08-11 tested each arm against *its own* label-shuffle null, answering "better than
chance?" not "is one arm better than the other?".

Two nulls:
- **Bootstrap** (`:871-874`): resample row indices once, score both arms on the identical
  resample, `delta = F1(A) − F1(B)`; percentile CI at 2.5/97.5, and
  `p_a_beats_b_bootstrap = (#{δ ≤ 0} + 1)/(n_boot + 1)`.
- **Paired swap permutation** (`:876-882`): flip A/B predictions per row with prob 0.5,
  recompute δ; `p_two_sided_paired_swap = (#{|δ_perm| ≥ |δ_obs|} + 1)/(n_perm + 1)`.
  This conditions on the *predictions*, not the labels — the right null for "these two arms
  are interchangeable".

Sign convention `delta = A − B` (`:863`). It draws its own
`derive_rng(cfg, "paired_Z_vs_features", endpoint)` stream, so adding it shifts no existing
p-value (CLAUDE.md). Circular analogue: `paired_bootstrap_circular` (`:897-923`).

### Other probes in the same file

- `rho_eps_max` binary probe — `fit_logistic_binary` (`:720-790`): LogReg, C by manual 5-fold
  `StratifiedKFold(shuffle=True, random_state=SEED)` scoring **AUC** (`:735-755`),
  `class_weight="balanced"`, `penalty="l2"`, `solver="lbfgs"`, `max_iter=2000`; the
  permutation null **refits the model** `N_PERM_BINARY` = 10000 times (`:774-784`), which is
  why the budget was raised from 200.
- Latent stems resolved by `CONFIG_LATENT_STEMS` (`:78-101`).
- Cross-domain PTB-XL row-alignment guards `_PTBXL_EXPECTED_ROWS` / `_PTBXL_EXPECTED_IDS`
  (`:107-114`) check every export against the subclass CSV, element-wise where `ecg_id` exists.

## 5.2 Circular-geometry pipeline

### `analysis/geom_common.py` — shared substrate

- `Domain` dataclass (`:55-66`): `z` (n, d) latents, `angle` (n,) radians, `territory` (n,)
  strings, `group` (n,) CV key, `name`.
- `load_medalcare(encoder)` (`:91-112`): pooled train+test MI rows, **n = 6547**, continuous
  φ; group = `f"{split}:{run_id}"`.
- `load_ptbxl(encoder, anchors)` (`:158-177`): all-folds MI rows with a 4-class territory,
  **n = 4324**; `angle` = the **anchor angle of the row's territory** (i.e. PTB-XL has no
  continuous target — it is quantised to four anchors); group = `patient_id`.
- `medalcare_anchor_angles()` (`:69-88`) — §1.1.8.
- **`RidgeSVD`** (`:183-240`): multi-output ridge that SVD-factorises the standardised,
  centred design **once**, so re-solving for a permuted target is nearly free
  (`solve(Y)`, `:229-232`) — this is what makes a 500-permutation label-shuffle null
  affordable.
  - **Alpha by generalised cross-validation** on the training rows: `_gcv_alpha` (`:208-221`),
    grid `np.logspace(-2, 5, 24)` (`:193`), `dof = Σ s²/(s²+α) + 1` (the +1 is the intercept),
    `GCV = ‖resid‖²/n / (1 − dof/n)²`.
  - `direction_matrix()` (`:238-240`) returns coefficients in the standardised latent basis —
    the readout subspace.
- `fast_ridge_coef(x, y, alpha)` (`:243-254`): normal-equation ridge for bootstrap loops where
  alpha is already fixed (one Gram + one Cholesky, ~10× cheaper than the SVD).
- Circular scoring: `resultant(delta) = |mean exp(iδ)|` (`:260-262`);
  `med_abs_deg` (`:265-267`); `angles_from_cs(pred) = arctan2(pred[:,1], pred[:,0])` (`:270-272`).
- **`group_folds(groups, n_splits, rng)`** (`:275-283`): group-disjoint CV — "Prevents a
  patient (or simulation run) from appearing in both the training and the held-out half."

### `analysis/circular_geometry.py` — the four probes

Docstring `:1-31`. `N_FOLDS = 5`, `N_PERM = 500` (`:67-68`).
- **P1** in-domain MedalCare: ridge readout to (cos φ, sin φ), continuous target (`cv_readout`, `:93-130`).
- **P2** in-domain PTB-XL: same readout onto territory anchor angles.
- **P3** transport (`:131-170`): MedalCare-fit readout applied to PTB-XL and back, **never
  refit**, under both a source and a target scaler.
- **P4** anchor sensitivity (`anchor_sensitivity`, `:171-223`): all **24** assignments of the
  four anchor angles to the four PTB-XL territory names.
- Two upgrades over the 2026-08-12 first pass, stated at `:19-27`: all-folds PTB-XL (n=4324)
  and group-disjoint CV (patient_id / run_id).
- Standing warning at `:22-27`: every printed R must be read against its constant floor.

### `analysis/floor_audit.py` — the constant-predictor floor

Docstring `:1-49`, the identity at `:9-16`:

> For any predictor emitting the same angle *c* for every row,
> `R = |mean exp(i(c − tᵢ))| = |exp(ic)|·|mean exp(−i tᵢ)| = |mean exp(−i tᵢ)|`
> — the chosen constant cancels, so **every** constant predictor scores exactly the resultant
> of the label marginal.

- `const_floor(theta) = |mean exp(−iθ)|` — `:86-88`.
- Normalised headroom `R_norm = (R − floor)/(1 − floor)` — `:41-43`, `norm()` `:104`.
  0 for a constant predictor, 1 for exact, can go negative.
- **Why the permutation null does not substitute** (`:18-38`): a permutation preserves the
  label marginal and destroys only the pairing, so `E[null resultant vector] ≈ R_pred · R_floor`
  — and since `R_pred ≤ 1` the null sits **systematically below the floor**. A precision note
  at `:27-33` records that the reported statistic is `E|V|`, which exceeds `|E V|` by a Jensen
  term (~+3.9 % for diffuse target-scaler cells), so the check is first-order.
- Stored floors (`outputs/analysis/circular_geometry/floor_audit.json→floors`):
  PTB-XL 4-anchor **0.29215682**, MedalCare continuous φ **0.09319111**, MedalCare quantised
  4-anchor 0.12270139; each verified against a 3600-point grid with spread < 6e-16
  (`verify_invariance`, `:91-102`).
- `verify_null_identity` (`:113-143`) refits two ridges to check the predicted-vs-stored null.
- `cyclic_order` (`:144-158`).

### `analysis/fidelity_audit.py` — per-feature informativeness fidelity

Header `:1-6`: promoted from `reports/2026-08-13_audit_artifacts/scripts/tmp_f1_fidelity.py`,
**independently re-implemented adversarially** in `analysis/fidelity_audit_verify.py`, verdict
`CONFIRMED_WITH_CORRECTION`. `RNG_SEED = 20260813`, `N_BOOT = 500` (`:22-23`).

- Matrices Fs (6547, 54) / Fr (4324, 54), asserted at `:78-79`.
- **ANOVA η²** per feature column, vectorised and NaN-aware: `eta2_all` (`:122-146`) —
  `η² = SSB/SST` computed from per-class finite counts/sums/sum-of-squares so a NaN in one
  column never drops a whole row. `prep_domain` (`:115-120`) builds the finite mask and the
  zero-filled `X`, `X²`.
- **Circular η²** for the frontal axis: `circ_eta2` (`:149-163`) —
  `(Σⱼ Rⱼ − R)/(N − R)` with **unnormalised** resultant lengths.
- Labels: `y4` = index into `TERRITORIES = ["Anteroseptal","Anterolateral","Inferolateral","Inferior"]`
  (`:25`); `y2 = (y4 >= 2)` i.e. 0 = AS+AL, 1 = IL+INF (`:89-90`).
- **Block bootstrap** (`:217-247`): resample **groups** with replacement — simulation `run_id`
  blocks in MedalCare, `patient_id` blocks in PTB-XL — 500 draws; domains resampled
  independently but **differences paired by draw index** (`:263-264`, logged at `:270-273`).
  CIs are 2.5/97.5 percentiles (`:254-257`).
- Marginal-realism statistics (`:280-302`): per-feature SMD `(μ_r − μ_s)/pooled sd`, sd ratio,
  and circular mean/R/angular-sd for the axis (`circ_mean_R`, `:204-207`).
- Rank agreement: Spearman ρ between `η²_sim` and `η²_real`, over the 54 features and over
  55 (features + axis) (`:311-…`).

### `analysis/block_transfer.py` — does the audit *predict* transfer?

Header `:1-6`: verified by `analysis/block_transfer_verify.py`,
verdict `CONFIRMED_WITH_CORRECTION` (fold-seed sensitivity declared; the block orderings are
the robust part).

- **Blocks** (`:86-97`): `ST_J60` ×12, `Q_amp` ×12, `R_amp` ×12, `T_amp` ×12, `globals` ×6,
  `full54`, `axis2 = [R_amp_I, R_amp_aVF]`. The `ST_J60` and `T_amp` selectors exclude the
  global `*_mV` columns via `not n.endswith("_mV")` (`:88, :91`).
- **Pre-stated prediction** written to disk *before* any result is computed — `:111-136`
  (P1–P4). This is a genuine pre-registration inside the script.
- Metrics (`:141-167`): `nearest_anchor_terr(pred_angle)` = argmin over
  `|angle(exp(i(pred − anchor)))|`; `macro_f1` via `sklearn.f1_score(labels=TERRITORIES,
  average="macro")`; `circ_eta2` as above; `circ_R = resultant(pred − true)`;
  `impute(X, med)` = median fill.
- **Floors recomputed in-script and cross-checked against the established values**
  (`:170-188`) → `RESULTS["floors"]`, stored in
  `outputs/analysis/fidelity_audit/f2_blocks.json→floors`:
  `R_floor_ptbxl 0.29215682`, `R_floor_medalcare 0.09319111`,
  `constF1_ptbxl 0.15341456`, `constF1_medalcare 0.12571461`, `n_medalcare 6547`, `n_ptbxl 4324`.
  The macro-F1 floor is the best single-class constant predictor (`:174-179`).
- **In-domain**: shared group-disjoint 5-fold CV (`group_folds(group_m, 5, rng(0))`, `:194-195`)
  reused identically across blocks so block-vs-block comparisons are paired; train-fold medians
  for imputation (`:212-217`).
- **Transport**: refit on all MedalCare with MedalCare medians (`:222-225`), then two scalers
  (`:227-235`) — *source/strict* = MedalCare medians + MedalCare μ/σ; *target* = **diagonal
  CORAL**, i.e. PTB-XL medians, then re-standardise PTB-XL to zero mean/unit sd and map back
  through the source μ/σ: `m.predict(z_t * m.sd_ + m.mu_)`.
- **Transfer efficiency** (`:265-268`, formula logged at `:261-262`):
  `(cross F1 − const floor_PTBXL) / (in-domain F1 − const floor_MedalCare)`;
  plus `eta2_ratio = eta2_cross / eta2_in`.
- `paired_test(terr_true, groups, angA, angB, n_draws=1000, seed=1)` (`:302+`): paired
  Δ = F1(A) − F1(B) with a **group bootstrap CI** and a **group-swap permutation**.
  `f1_boot_ci` (`:374`), `anova_eta2` (`:407`).

### `analysis/channel_repair.py` — can the mechanism be *repaired*?

Header `:1-6`: verified by `analysis/channel_repair_verify.py`, verdict `CONFIRMED`
(all numbers reproduced exactly). `SEED = 20260813`, `N_BOOT = 1000`, `N_NULL = 200`
(`:44-46`).

Three interventions (docstring `:9-15`):
- **A — channel-restricted latent readout.** Everything is fit on MedalCare; at PTB-XL
  inference **only latents are used, features never**. Mechanically: one `RidgeSVD`
  factorisation maps latents → the 54 features on all-54-finite MedalCare rows
  (`fit_W_stage`, `:234-248`), then per-block coefficients are re-solved on the shared SVD
  with a per-block GCV alpha (`:243-247`); `r_of(c, bname, z)` (`:250-253`) produces the
  block-restricted representation for any latent matrix.
- **B — importance reweighting** (falsification arm), `run_reweight` (`:508-577`):
  a `LogisticRegression(C=1.0)` domain classifier on the standardised pooled summary features
  gives `w = p_real/(1 − p_real)`, clipped at the 99th percentile, normalised;
  **ESS = 1/Σw²** is reported alongside `n_rows`; 10 weighted resamples refit the readout, and
  the averaged prediction is compared to the unweighted baseline **on the same row subset**
  via `paired_boot`.
- **C — lead-group restriction**: `inferior` = {ST_J60, Q_amp, R_amp, T_amp} × {II, III, aVF}
  (12 dims) vs `anterior` = same measures × {V1–V4} (16 dims) — `block_cols` `:187-207`.
  Other blocks there: `Q12`, `R12`, `ST12`, `T12`, `axpair`, `QR24`.

Two nulls, both at `N_NULL = 200` draws:
- **Random-projection null** `null_random_projection` (`:425-451`): replace the learned
  block coefficient matrix with a Gaussian matrix of the **same shape and same per-column
  norm**, then run the *entire* downstream pipeline (readout refit + transport + scoring);
  reports `null_mean`, `null_p95`, `p = (1 + #{null ≥ obs})/(n+1)`.
- **Shuffled-source-label refit null** `null_shuffled_phi` (`:453-486`): permute MedalCare φ
  **at group level**, refit the readout with a fresh GCV alpha on the cached factorisation,
  transport, score. Reports `n_unique_groups`.
- `transport_preds` (`:157-164`) implements the same source/target(diagonal-CORAL) pair as
  `block_transfer.py`; `paired_boot` (`:166-185`) is a group bootstrap with a two-sided
  `p_boot = 2·min(P(δ≤0), P(δ≥0))`.
- Scoring helpers: `assign_int` (nearest anchor, `:102-105`), `fast_macro_f1` (`:111`),
  `circ_R` (`:121`), `circ_eta2` (`:125`), `score_all` (`:136`).

## 5.3 The four supporting analysis scripts

**`analysis/tier1_evaluation.py`** (514 lines). Runs the same four-block evaluation suite as
`dim_scan.py` — (1) *alignment* on a test pool subsampled to 2000/domain: single-bandwidth MMD
(median heuristic), multi-bandwidth MMD, C2ST AUROC, kNN-5 mixing; (2) *class structure* on the
3-class shared remap: KMeans-3 accuracy/NMI/ARI computed combined and per domain, logistic
transfer M→P and P→M (macro-AUC, per-class AUC, accuracy), kNN-5 both directions, cosine
intra/inter/cross gaps; (3) *mechanism* on the MedalCare MI subset in-domain: φ circular R²
(sin/cos Ridge), z R², size R², `rho_eps_max` AUC — **plus** (4) the Pipeline-A 4-class anatomy
classifier imported from `phase_b2_infarct_decoding.py`, scored in-domain on MedalCare-test
(n ≈ 1.2 k) and cross-domain on the PTB-XL primary 4c subset (n ≈ 438, 4-class and 2-class
collapsed). It exists to put K ∈ {1024 (the `exp7_baseline` reference), 256, 64, 16} on one
table; outputs `outputs/tier1_eval/{<config>_summary.json, cross_config_table.json,
cross_config_table.md, frontier_tier1.png}` (docstring `:1-43`).

**`analysis/dim_scan.py`** (693 lines). A **post-hoc PCA dimension scan, no retraining**: for
each (config, pca_mode) it fits `StandardScaler + PCA(n_components=1024)` once on the *train*
pool and then slices `Z_proj[:, :K]` for each K, recomputing the same alignment / class-structure
/ mechanism blocks at every K. Defaults: `DEFAULT_KS = (1024, 512, 256, 128, 64, 32, 16, 8)`
(`:107`), `DEFAULT_CONFIGS = ("exp7_baseline", "exp7_ccmmd")` (`:108`),
`DEFAULT_PCA_MODES = ("combined", "medalcare", "ptbxl")` (`:109`). The pre-registered selection
rule is in the docstring (`:29-32`): **K\*** = the smallest K with C2ST ≤ 0.85 **and**
LR M→P ≥ 0.65 **and** φ circular R² ≥ 0.35, with fallback "best LR M→P among K with
C2ST ≤ 0.95". Bootstrap is reduced to 200 and permutations to 50 because 24+ cells are swept,
with the explicit caveat that CIs are a sanity check at the 1024-d row only (`:48-52`).

**`analysis/inlp_alignment.py`** (705 lines). Iterative Nullspace Projection (Ravfogel et al.,
ACL 2020) applied post-hoc to the latents. Algorithm as written at `:14-23`: standardise Z;
then repeatedly fit `LogisticRegression(C=1.0, class_weight='balanced')` to predict domain
identity, form the rank-(D−1) orthogonal projection `Pₜ = I − wₜwₜᵀ/‖wₜ‖²`, project, and
re-measure 5-fold CV domain accuracy; stop when accuracy ≤ `stop_acc`. `P_total = P₁P₂…P_T`,
applied at inference as `Z_aligned = scaler.transform(Z_raw) @ P_total`. Defaults:
`DEFAULT_MAX_ITER = 20`, `DEFAULT_STOP_ACC = 0.55`, `DEFAULT_SEED = 42`,
`DEFAULT_POOL_MODE = "symmetric"`, `DEFAULT_OUTPUT_SUFFIX = "_inlp"` (`:100-106`);
`PRIMARY_CONFIGS = ("exp7_baseline", "exp7_ccmmd")` (`:83`). Two pool modes (`:27-42`):
**symmetric** (default since 2026-08-10, defect A4) fits on MedalCare-train + PTB-XL-train
(folds 1–8) and holds out both test sets; **asymmetric** (v1 legacy) fitted on MedalCare-train
+ PTB-XL-**test**, so any "less separable" claim measured on PTB-XL-test is partly
resubstitution, and the ~85/15 imbalance is what broke the stopping rule. A
`random_projection_control` (`:289`) provides the matched control.

**`analysis/nonlinear_c2st.py`** (247 lines). A falsification instrument for a suspicious
result: `whitened_frontier.py` reported C2ST = **exactly** 0.5000 after removing 2 whitened INLP
directions, and the docstring (`:1-16`) argues that this is what a *degenerate* logistic C2ST
returns by construction — INLP halts precisely when the logistic weight collapses, and a
logistic C2ST on the same data collapses the same way, emitting a constant probability for
which `roc_auc_score` returns 0.5. So the domain signal is re-measured with instruments that do
not share the failure mode (`:18-31`): a **gradient-boosting** C2ST
(`HistGradientBoostingClassifier`, axis-aligned splits, no linear margin), an **MLP** C2ST
(different nonlinear family), a **kNN** C2ST (nonparametric, detects local clustering), and a
**multi-bandwidth RBF MMD with a permutation p-value** (no classifier at all, so classifier
degeneracy cannot produce a false negative). All fit on projected TRAIN and scored on projected
HELD-OUT data, run at k=0 and at the k where linear C2ST hit chance. Helper `is_constant`
(`:79-87`) detects the degenerate case explicitly. Verdict logic is stated in advance
(`:33-37`): if the nonlinear tests also sit at chance the free-alignment result stands;
if any recovers the domain, the honest claim shrinks to "only the linearly decodable part was
removed" and "alignment is free" must not be written. Writes
`outputs/analysis/domain_signal/nonlinear_c2st_<run>.json`.

---

# 6. COMPUTE ENVIRONMENT

| item | value | pointer |
|---|---|---|
| Python | **3.10.19** (conda) | `env-ECGFounder.yml` (`python=3.10.19`); `openspec/project.md` says "Python 3.10" |
| env name / prefix | `ECGFounder`, `C:\Users\Owen\anaconda3\envs\ECGFounder` | `env-ECGFounder.yml` (UTF-16 encoded) |
| torch | **2.9.1+cu128** | `env-ECGFounder.yml`; `reports/EXECUTION_LOG_2026-08-10.md:17` |
| torchvision | 0.24.1+cu128 | `env-ECGFounder.yml` |
| numpy / pandas / scipy / scikit-learn | 2.2.4 / 2.2.3 / 1.15.2 / **1.6.1** | `env-ECGFounder.yml`, `requirements.txt` |
| matplotlib / h5py / tqdm / wfdb | 3.10.1 / 3.11.0 / 4.66.5 / **4.2.0** | `env-ECGFounder.yml`, `requirements.txt` |
| **neurokit2** | **0.2.10 — `requirements.txt` only, NOT in the conda env** | `requirements.txt:3`; absent from `env-ECGFounder.yml` |
| **GPU** | **NVIDIA GeForce RTX 5080, 16,303 MiB** | `reports/EXECUTION_LOG_2026-08-10.md:19` |
| device string at runtime | `cuda:0` | `outputs/_log_exp8_medalonly.txt:1`; `scripts/finetune_multilabel.py:1349` (`torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`) |
| interpreter actually used | `F:\anaconda3\envs\ECGFounder\python.exe` | `reports/EXECUTION_LOG_2026-08-10.md:15` |
| required env var | `KMP_DUPLICATE_LIB_OK=TRUE` (duplicate `libiomp5md.dll` otherwise aborts) | `reports/EXECUTION_LOG_2026-08-10.md:21` |
| OS / shell | Windows 11 Pro 10.0.26200, PowerShell | session environment; `outputs/_log_exp8_medalonly.txt:16-19` shows a PowerShell invocation |
| throughput observed | ≈ 2.2 it/s, 78 steps/epoch at batch 128 on the medalonly run | `outputs/_log_exp8_medalonly.txt:24-45` |

⚠ **Interpreter trap, recorded at `reports/EXECUTION_LOG_2026-08-10.md:24-30`**: bare `python`
on PATH resolves to `F:\anaconda3\python.exe` (base anaconda) with **torch 2.10.0+cpu, no
CUDA**, sklearn 1.7.2, numpy 2.3.5 — *not* the project environment. At least one C2ST
diagnostic was run that way before being re-verified. Any reproduction instruction in the
thesis must give the absolute interpreter path.

No test/lint/format/typecheck tooling is configured (`.claude/rules/commands.md`, "Tests /
lint / format / typecheck — **None configured.**"); validation is via `metrics.json` and
visual checks.

**Reproducibility gaps to disclose**: `env-ECGFounder.yml` is gitignored (`.gitignore`), as are
`reports/`, `outputs/*`, `checkpoint/`, and all three raw dataset roots; `config/theta.json`
(referenced by every `exp8` `args.json`) is **not on disk**; `outputs/theta_stats.json`
likewise referenced but unverified here.

---

# 7. OPEN QUESTIONS / CONTRADICTIONS

1. **Split provenance is documented wrongly in three places.** `.claude/rules/data-pipeline.md`
   and `scripts/finetune_multilabel.py:493-495` both say the MedalCare splits are "seeded from
   SHA-256 of `original_csv_path` (`scripts/make_splits.py`)". They are not: the manifest in
   use was written by `scripts/add_medalcare_splits.py`, which reads the split off the
   directory name, and agreement with the folder is 100.0 % [computed]. `make_splits.py`
   exists but uses `StratifiedGroupKFold(5, random_state=42)` (`:94`) and its output file
   `data/medalcare_filtered_manifest.csv` is not on disk. **Fix the Methods text; do not
   describe a stratified group-K-fold that never ran.**
2. **MedalCare anatomy models are not fully split-disjoint.** The dataset README promises they
   are. On the shipped manifest 2 of 13 `run_S##` folders straddle splits (`run_S64`:
   test 1000 / train 388; `run_S67`: train 389 / val 1000) [computed]. No validator catches
   this because both the manifest `run_id` and the analysis-side group key are finer than
   `run_S##`. Decide whether to (a) report it as a limitation or (b) re-derive groups at the
   `run_S##` level for the in-domain CV claims.
3. **`args.json` does not exist for 8 of the 19 runs** — every pre-2026-08-10 run
   (`exp5_3class`, `exp6_3class`, `exp7_baseline`, `exp7_baseline_norm`, `exp7_ccmmd`,
   `joint_baseline`, `joint_adapter_cls`, `joint_adapter_mmd`, `ptbxl_baselines/...`).
   Epochs, batch size, learning rates, and λ_mmd for those runs are recoverable only from the
   run-ID glossary, which is a *convention*, not an artifact. Any Methods table row for them
   should be marked "not recorded".
4. **`exp4_ptbxl`, `exp6_medalcare`, `exp6_ptbxl` latent directories have no recorded source
   encoder.** `scripts/export_latents.py:9-27` documents only exp1 and exp5. The mapping to
   `joint_baseline` / `joint_adapter_mmd` is a guess and should be verified (or the exports
   excluded) before any figure depends on them.
5. **`exp5_3class`/`exp6_3class` are called "dual-head" by `.claude/rules/experiments.md` but
   their `metrics.json` has the shared-head-style `avg_domain_f1` primary metric and a 3-class
   `P` export.** The distinction (`--dual-head-shared-labels` vs `--shared-head`) cannot be
   confirmed from artifacts because there is no `args.json`. Confirm from memory/notes before
   the 2×2 ablation table is written.
6. **`config/theta.json` is missing** although every `exp8` `args.json` records
   `theta_config=<repo>/config/theta.json`; `config/` is empty. The physics-head θ list is
   only recoverable from `outputs/joint_baseline/physics_metrics.json→theta_names` (51 names).
   Note also that this 51-name θ is a *different object* from the 4-member `isch[0].*` θ used
   in Phase B2 — the thesis must not let the two share the symbol θ without a subscript.
7. **The pre/post-leadfix PTB-XL macro-F1 columns are not comparable.** `exp5/6/7` scored a
   PTB-XL 3-class subset containing STTC and excluding CD (defect D3, `git diff pre-leadfix`
   hunk at old `scripts/finetune_multilabel.py:200`; 1873 vs 1769 val rows [computed]), so the
   0.787 → 0.828 movement mixes a labelling change with the lead fix. Present the exp8 family
   on its own.
8. **`prepare_medalcare.py:47-48` mislabels two pathologies** in its inline comments (`iab` as
   "Incomplete atrioventricular block", `fam` as "Familial/genetic condition"); the dataset
   README says interatrial conduction block and fibrotic atrial cardiomyopathy. The *column
   order* is unaffected, but the glossary in the thesis must follow the README.
9. **Which scaler is primary remains an unresolved supervisor call** (CLAUDE.md; Q1 in
   `reports/2026-08-13_thesis_endgame_decision.txt`). The code's own defaults disagree with
   each other by design: `standardise_target` defaults to `target_pool` "so any new caller
   gets the safe convention", while `--scaler-domain` defaults to `target` "so a bare CLI
   invocation reproduces the `outputs/phase_b2_exp8_tgtscaler/` snapshot"
   (`analysis/phase_b2_infarct_decoding.py:499-505`). Methods must state which was used for
   each reported cell.
10. **`ptbxl_mi_subclass_summary.json` reports `n_single_territory_primary = 444` but
    `n_primary_4c = 438`** for fold 10. These are two different rules (legacy 3-class
    single-territory vs the refined 4-class compartment logic, `:334-336`), not an
    inconsistency — but the two numbers are easy to confuse in prose. Always name the rule.
11. **PTB-XL "angles" are not measured, they are assigned.** `analysis/geom_common.py:158-177`
    sets every PTB-XL row's target angle to its territory's *MedalCare* anchor. Any statement
    of the form "the real domain's angle" is therefore a statement about a 4-point quantisation
    of the synthetic labelling, and the constant floor 0.29216 exists precisely because of that
    concentration. This must be said explicitly wherever an R value for PTB-XL appears.
12. **Feature-extraction coverage is dominated by `global6` failures, not per-lead ones**
    (e.g. PTB-XL all-folds: 4087 rows have all 54 finite, but 5415/5440/5453/5416 have complete
    per-lead blocks). All downstream pipelines median-impute rather than drop. Any claim about
    "n = 4324 rows scored" must be paired with the statement that **24.7 % (1070/4324) carry at
    least one imputed column** (MedalCare: 32.6 %, 2135/6547) [computed], and the imputation
    source (MedalCare train-fold medians) matters for the scaler discussion in §5.1.
