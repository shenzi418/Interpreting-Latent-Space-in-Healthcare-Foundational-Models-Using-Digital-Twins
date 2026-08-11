"""Print the 8x8 confusion matrix and per-class F1 for exp7_baseline (Z) so
we can characterise the failure mode in terms of anatomy vs transmurality.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PATH = REPO / "outputs" / "phase_b2" / "in_domain_8c.json"


def main(cfg: str = "exp7_baseline") -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    labels = data["metadata"]["territories_8c"]
    leg = data["results"][cfg]["Z"]
    cm = np.array(leg["in_domain_8c"]["confusion_matrix"])

    print(f"=== {cfg} Z 8c CM (rows=truth, cols=pred) ===")
    header = "             " + " ".join(f"{x[:7]:>8s}" for x in labels)
    print(header)
    for i, row in enumerate(cm.tolist()):
        cells = " ".join(f"{c:>8d}" for c in row)
        print(f"{labels[i]:>12s} {cells}")

    print()
    print("Per-class precision / recall / F1:")
    for k, v in leg["in_domain_8c"]["per_class"].items():
        print(f"  {k:>14s}:  F1={v['f1']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}  n={v['support']}")

    print()
    print("Anatomy-collapse 4c CM (rows=truth, cols=pred):")
    cm4 = np.array(leg["in_domain_4c_anatomy"]["confusion_matrix"])
    labels4 = leg["in_domain_4c_anatomy"]["labels"]
    header4 = "                  " + " ".join(f"{x[:10]:>12s}" for x in labels4)
    print(header4)
    for i, row in enumerate(cm4.tolist()):
        cells = " ".join(f"{c:>12d}" for c in row)
        print(f"{labels4[i]:>16s} {cells}")

    print()
    print("Transmurality-collapse 2c CM (rows=truth, cols=pred):")
    cm2 = np.array(leg["in_domain_2c_transmurality"]["confusion_matrix"])
    labels2 = leg["in_domain_2c_transmurality"]["labels"]
    header2 = "         " + " ".join(f"{x:>6s}" for x in labels2)
    print(header2)
    for i, row in enumerate(cm2.tolist()):
        cells = " ".join(f"{c:>6d}" for c in row)
        print(f"{labels2[i]:>6s} {cells}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "exp7_baseline"
    main(cfg)
