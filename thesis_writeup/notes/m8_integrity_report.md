# M8 integrity pass — report (2026-08-24)

Scope: the full draft (67 pp) as of commit `7587711` + fixes below. Verification script: scratchpad `m8_verify.py` (run twice, before/after fixes).

## 1. Number audit (programmatic, against the frozen artifacts)
**180 checks** parsing `f1_fidelity.json`, `f2_blocks.json`, `f3_repair.json`, `floor_audit.json`, `c2st_leadfix.json`, `leadperm_sweep.json`, six `metrics.json`, the IA §6.4c grid, and `tmp_t4_alpha_grid.txt`, asserting each rendered value in the LaTeX.

- **177 exact matches.**
- **1 genuine error found and fixed**: Table 4.1, K64 PTB-XL AUC — true value 0.94347, table said 0.944 (rounded from a 4-dp print), corrected to **0.943**.
- **2 adjudicated, documented, kept**:
  - α=10⁷ row shows **0.294** = round-half-up of the artifact's printed 0.2935 (all other cells match Python 3-dp rounding of the artifact prints; convention: displayed artifact values rounded half-up).
  - Three-condition table middle row (leadfix, global z: 1.0000 / 0.16604 / 0.0049) is sourced from the 2026-08-10 repo-audit §5 table, not the JSON (which holds only conditions 1 and 3); the tracenote now names both sources.
- 1 script limitation (not a thesis issue): the IA grid regex reads 11/12 rows (the bolded row resists the pattern); its two values (0.652/0.633) verified manually in the table.

## 2. Discipline sweeps (notes/check_draft.py)
- Forbidden phrasings: **0 hits**. Circular R̄-without-floor: **0**. Scaler-unnamed cross-domain numbers: **0**.
- Style: no lexicon hits beyond the two adjudicated statistical "robust"; em-dash 0–0.2/1k; no bold-in-prose; no "we".
- Tracenote coverage: every numeric paragraph carries a trace except the auto-generated A.1 table (its header comment is the provenance).
- `\todo` count = **4, all owner facts**: acknowledgments personalisation (1); Declarations — ethics wording confirmation, dataset licence names, artifact-commit decision (3).

## 3. Bibliography
- 40 entries; **40/40 cited**; 0 uncited; 0 cited-but-missing; **0 malformed DOIs**; every entry carries a dated verification comment (papersflow, or publisher record for the two books). 0 BibTeX warnings.

## 4. Build and layout
- 0 LaTeX errors; 0 undefined or multiply-defined references; figure files all present; floats pinned to their sections (`placeins`); **Bibliography added to the ToC** (was missing; template default).
- Overfull lines: **6 remain, all ≤ 13.5 pt (≤ 4.8 mm)** — locations: sweep-table area and three prose hyphenation cases; judged invisible at print size; listed for final polish only.

## 5. Known open items (owner)
1. Four `\todo`s above (facts only I can't supply).
2. Q1b (P2 scaler primacy) — open with the supervisor; thesis is both-modes-proof throughout.
3. The `reports/2026-08-13_audit_artifacts/` commit decision (recommended yes) — affects the Declarations data statement wording only if declined.

**Verdict: the draft is internally consistent, artifact-traceable and clean to send.** Remaining changes before submission should be owner-fact insertions and any reader-flagged wording only.
