"""Canonical parsing of MedalCare-XL `original_csv_path` values.

Two call sites independently reimplemented pathology detection as the substring
test ``"/mi/" in path.lower()``. That is fragile in a way that matters here: the
manifest stores absolute paths from the machine that generated it, so any
directory *above* the dataset root is also in the string. A user whose data lived
under ``D:/mi/...`` or ``.../submission/`` would have had every row silently
classified as MI (or not), with no error -- and `is_mi_path` gates which rows get
a theta target at all, so the failure would surface as a shape mismatch hundreds
of lines downstream, if at all.

Layout this module assumes (verified against the shipped manifest, n=16,839):

    .../MedalCare-XL/WP2_largeDataset_noise/<pathology>/[<territory>/]<split>/<run>/<file>.csv
                                            ^^^^^^^^^^  ^^^^^^^^^^^  ^^^^^^^
                                            e.g. "mi"   "LAD_0.3"    "train"/"validation"/"test"

Everything here is anchored on the **split directory** and matches whole path
segments -- never substrings. The pathology sits one level above the split, or
two when a territory folder intervenes (MI only). Anchoring rather than
"first/only matching segment" is what makes
``D:/mi/experiments/.../sinus/test/...`` resolve to *sinus*, as it should.

`<territory>` is present only under the `mi/` pathology. The eight pathology
segments are exactly the eight manifest label columns; see PATHOLOGIES.
"""
from __future__ import annotations

from typing import List, Tuple

# Order is load-bearing: it matches manifest columns label_0 .. label_7.
# See `.claude/rules/data-pipeline.md`.
PATHOLOGIES: Tuple[str, ...] = (
    "sinus", "mi", "rbbb", "lbbb", "lae", "iab", "fam", "avblock",
)

CORONARIES: Tuple[str, ...] = ("LAD", "LCX", "RCA")
TRANSMURAL_LABELS: Tuple[float, ...] = (0.3, 1.0)
LCX_SUBTYPES: Tuple[str, ...] = ("ant", "post")

# On-disk split directory names. Note "validation", not "val" -- the manifest's
# `split` column uses "val", the directory tree does not.
SPLIT_DIRS: Tuple[str, ...] = ("train", "validation", "test")


def path_segments(p: object) -> List[str]:
    """Lower-cased path segments, separator-agnostic (manifests carry Windows paths)."""
    return [s for s in str(p).replace("\\", "/").lower().split("/") if s]


def _pathology_index(segs: List[str]) -> int:
    """Index of the pathology segment, anchored on the split directory.

    Anchoring matters. A pathology name can legitimately appear in an ancestor
    directory -- ``D:/mi/experiments/.../sinus/test/...`` is a sinus record, but
    both "mi" and "sinus" are present as segments, so neither "the only match"
    nor "the first match" is correct. The layout pins it exactly: pathology sits
    one level above the split directory, or two when a territory folder
    intervenes (MI only).
    """
    split_positions = [i for i, s in enumerate(segs) if s in SPLIT_DIRS]
    if not split_positions:
        raise ValueError(
            f"No split directory {SPLIT_DIRS} in path segments {segs!r}; cannot "
            f"locate the pathology segment unambiguously."
        )
    # Rightmost split segment: run/file names never collide with these, but an
    # ancestor directory could.
    split_idx = split_positions[-1]
    for back in (1, 2):  # <pathology>/<split> or <pathology>/<territory>/<split>
        i = split_idx - back
        if i >= 0 and segs[i] in PATHOLOGIES:
            return i
    raise ValueError(
        f"No pathology segment from {PATHOLOGIES} one or two levels above the "
        f"split directory {segs[split_idx]!r} in {segs!r}"
    )


def pathology_of(p: object) -> str:
    """The pathology segment of a MedalCare path.

    Raises ValueError when the path does not have the expected layout -- guessing
    would corrupt every target derived from it.
    """
    segs = path_segments(p)
    return segs[_pathology_index(segs)]


def is_mi_path(p: object) -> bool:
    """True iff this row is an MI simulation. Layout-anchored, not substring."""
    return pathology_of(p) == "mi"


def parse_territory_from_path(p: object) -> Tuple[str, str, float]:
    """Extract ``(coronary_artery, lcx_subtype, transmural_label)`` from an MI path.

    Returns
    -------
    coronary_artery : str
        One of ``{"LAD", "LCX", "RCA"}``.
    lcx_subtype : str
        ``"ant"`` or ``"post"`` for LCX, otherwise ``""``.
    transmural_label : float
        ``0.3`` or ``1.0`` -- matches ``isch[0].rho_eps_max``.
    """
    segs = path_segments(p)
    mi_pos = _pathology_index(segs)
    if segs[mi_pos] != "mi":
        raise ValueError(f"Not an MI path (pathology is {segs[mi_pos]!r}): {p!r}")
    if mi_pos + 1 >= len(segs):
        raise ValueError(f"MI path has no territory segment after 'mi': {p!r}")

    folder = segs[mi_pos + 1]  # e.g. 'lad_0.3', 'lcx_1.0_ant', 'rca_0.3'
    pieces = folder.split("_")
    coronary = pieces[0].upper()
    if coronary not in CORONARIES:
        raise ValueError(f"Unknown coronary artery {coronary!r} in {folder!r} ({p!r})")
    try:
        transmural = float(pieces[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse transmural label in {folder!r} ({p!r})") from exc
    if transmural not in TRANSMURAL_LABELS:
        raise ValueError(f"Unexpected transmural label {transmural} in {folder!r}")
    lcx_subtype = pieces[2] if (coronary == "LCX" and len(pieces) >= 3) else ""
    if coronary == "LCX" and lcx_subtype not in LCX_SUBTYPES:
        raise ValueError(f"Unexpected LCX subtype {lcx_subtype!r} in {folder!r}")
    return coronary, lcx_subtype, transmural


def assert_label_schema(columns) -> None:
    """Fail loudly if the manifest's label columns are not label_0 .. label_7.

    `MEDALCARE_REMAP`, `MEDALCARE_KEEP_LABELS` and `MEDALCARE_DROP_LABELS` all
    index by *integer position*, so a renamed, reordered or missing label column
    silently relabels the dataset. No schema check existed anywhere; this is it.
    """
    label_cols = [c for c in columns if str(c).startswith("label_")]
    expected = [f"label_{i}" for i in range(len(PATHOLOGIES))]
    if label_cols != expected:
        raise ValueError(
            f"Manifest label schema mismatch.\n  expected (in order): {expected}\n"
            f"  found:               {label_cols}\n"
            f"Positional remaps (MEDALCARE_REMAP etc.) assume the fixed order "
            f"{PATHOLOGIES}; running with a different schema silently relabels "
            f"every sample."
        )
