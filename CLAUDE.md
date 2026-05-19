# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Thesis**: *Interpreting the Latent Space in ECG Digital-Twin Foundation Models* — a 1-year MRes project, mid-execution. **Thesis due 2026-08-31**, viva mid-September on campus. Paper submission is a stretch goal if results are strong enough.

**Active state of the research:**

- **Ruled out** — direct latent-space alignment between synthetic MedalCare-XL and real PTB-XL is a confirmed dead-end. Single-bandwidth MMD, multi-bandwidth MMD, class-conditional MMD, and INLP each reduce first-order distance metrics, but C2ST AUROC stays ~1.0 on held-out latents. Do not propose new alignment-via-training approaches without acknowledging this. Full record: `reports/inlp_alignment_summary.md`, `reports/exp7_progress_report.md`.
- **Currently pushing** — biophysical θ-decoding from latents (Phase B / B2). In-domain MedalCare decoding of `isch[0].{phi, z, size, rho_eps_max}` and APD parameters works; cross-domain transfer to PTB-XL is the open problem. The 2026-05-13 Track 3 redux (refined 4-class anatomical territory, ~4× power gain) is the active diagnostic — see `reports/b2cd_redux_log.md` and `analysis/phase_b2_infarct_decoding.py`.

**Working mode**: collaborative research, not code-execution-on-demand. When results are weak, default to discussing honest framings — negative-result writeups, methodology critiques, pivot proposals — *before* running the next experiment. The `reports/` directory is the source of truth for what's been tried. See memory `project_research_state.md` (paper-claim structure, phase status) and `project_methodology_decisions.md` (ruled-out approaches and why).

> **Note for fresh clones**: `reports/` is gitignored — the supervisor-facing writeups cited above (`inlp_alignment_summary.md`, `exp7_progress_report.md`, `b2cd_redux_log.md`) are owner-local only. The headline findings are summarised here; if you don't have `reports/`, treat this file plus the rule files as the authoritative summary.

For tech stack, label spaces, and external dependencies see [`openspec/project.md`](openspec/project.md) — canonical project-context file (now committed; was previously gitignored).

## Project Structure

- `scripts/`, `analysis/` — CLI scripts (data prep, training, latent export, audits) and post-training analysis (alignment, dim-scan, B2 decoding, INLP).
- `data/`, `outputs/`, `checkpoint/`, `reports/` — seeded manifests + θ NPZs (do not regenerate); per-run experiment artifacts (gitignored); pre-trained ECGFounder weights; supervisor-facing writeups.
- `losses/`, `metrics/` — MMD variants; multilabel metric suite (F1, AUC, Brier, recall, precision, specificity).
- `MedalCare-XL/`, `ptb-xl-...` — raw synthetic and real datasets on disk (gitignored); θ targets parsed from `MedalCare-XL/WP2_largeDataset_ParameterFiles/*VentricularParameters.txt`.
- `openspec/` — spec-driven development: `specs/`, `changes/`, `AGENTS.md`.
- `net1d.py`, `finetune_model.py`, `util.py` — root-level: backbone + adapter, model builders, eval/checkpoint helpers.

## Tech Stack

Python 3.10.19, PyTorch 2.9.1+cu128 (NVIDIA CUDA 12.8), wfdb 4.2.0, neurokit2 0.2.10 (pip-only — not in conda env), scikit-learn 1.6.1. No PyTorch Lightning, no Transformers, no `pyproject.toml`. Canonical install: `env-ECGFounder.yml`; pip subset: `requirements.txt`.

## Conventions (load-bearing one-liners; full detail in `.claude/rules/`)

- **Lead permutation** — every MedalCare Dataset re-indexes 12 leads `[…, aVF, aVL, V1…]` → `[…, aVL, aVF, V1…]`. Skipping silently swaps limb leads. See `data-pipeline.md`.
- **`sys.path` injection** — every entry-point script under `scripts/` and `analysis/` prepends `REPO_ROOT` to `sys.path` before importing root modules. See `model-conventions.md`.
- **`outputs/<run_id>/` artifact contract** — every fine-tune emits `best_model.pt`, `args.json`, `metrics.json`, `per_class_metrics.csv`. See `experiments.md`.
- **`linear_prob=True + use_adapter=True` is BUGGED** in `ft_12lead_ECGFounder` — freezes `model.dense`. Use the manual-freeze workaround. See `model-conventions.md`.
- **MedalCare label_0..label_7 order is FIXED** as `(sinus, mi, rbbb, lbbb, lae, iab, fam, avblock)`; `MEDALCARE_REMAP` indexes by integer position. See `data-pipeline.md`.

## Detailed rules

| File | When to read |
|---|---|
| [`.claude/rules/commands.md`](.claude/rules/commands.md) | About to run any CLI (install, data prep, train, export, analyze). |
| [`.claude/rules/experiments.md`](.claude/rules/experiments.md) | Interpreting `outputs/<run_id>/` artifacts; designing or comparing Exp 5/6/7 runs. |
| [`.claude/rules/data-pipeline.md`](.claude/rules/data-pipeline.md) | Editing Dataset classes, manifest builders, or `scripts/build_*.py`. |
| [`.claude/rules/model-conventions.md`](.claude/rules/model-conventions.md) | Editing `net1d.py`, `finetune_model.py`, or writing a new training entry-point. |

OpenSpec workflow: see [`openspec/AGENTS.md`](openspec/AGENTS.md). **Do not write code during the proposal stage.** Skip proposals for bug fixes, typos, config tweaks, and tests for existing behaviour.

## MCP Tool Usage

Three MCP servers are wired into this project: **context7** (current library/framework docs), **papersflow** (academic literature + citation verification), and **brave-search** (general web).

### When writing the thesis / paper
- Before citing any paper: `papersflow.verify_citation` to normalize the reference. Do NOT cite from training-data memory — DOIs and venue attributions drift (e.g., Li et al. was previously miscited as "MICCAI 2024" when the correct venue is IEEE TMI 2024).
- Searching literature on a topic: `papersflow.search_literature`, then `find_related_papers` to expand. Do NOT use brave-search or built-in WebSearch for academic queries.
- Mapping prior work around a known paper (references + citing papers): `papersflow.get_citation_graph` / `expand_citation_graph`.

### When coding
- API, version, or migration questions on `torch`, `scikit-learn`, `neurokit2`, `wfdb`, etc.: `context7.resolve-library-id` → `context7.query-docs`. Use even for libraries you "know" — training data may be stale. Do NOT use WebSearch — it returns out-of-date tutorial blogs.
- Debugging *project* business logic, refactoring, or anything in this repo's source: use `Read` / `Grep` / `Glob`. Context7 is for upstream library docs, not local code.

### When researching outside academia
- Vendor blogs, product pages, news, current events: `brave-search.brave_web_search`.
- Location / business queries: `brave-search.brave_local_search`.
- Do NOT use brave-search for: library docs (→ context7), academic citations (→ papersflow).

### Anti-patterns
- Never fabricate a citation, DOI, or venue — verify via `papersflow.verify_citation` first.
- Never trust your memory of a library API for code that will actually run — query `context7` and quote the result.
