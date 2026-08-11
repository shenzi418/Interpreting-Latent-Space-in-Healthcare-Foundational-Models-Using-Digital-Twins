"""Pull final macro-F1 (medalcare-test + ptbxl-test) from each bottleneck training metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

CONFIGS = [
    ("exp7_baseline", 1024),
    ("exp7_bottleneck_K256", 256),
    ("exp7_bottleneck_K64", 64),
    ("exp7_bottleneck_K16", 16),
]


def _walk_for_macro(d, results: dict[str, float]) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                if "macro_f1" in v and isinstance(v["macro_f1"], (int, float)):
                    results[k] = v["macro_f1"]
                _walk_for_macro(v, results)
            elif isinstance(v, list):
                _walk_for_macro(v, results)


def main() -> None:
    print(f"{'config':>28} {'K':>5} {'medal_f1':>10} {'ptb_f1':>10} {'val_f1':>8} {'top-level keys'}")
    for name, k in CONFIGS:
        path = Path(f"outputs/{name}/metrics.json")
        if not path.exists():
            print(f"{name:>28} {k:>5}  [missing]")
            continue
        d = json.load(path.open())
        if isinstance(d, list):
            last = d[-1]
        elif isinstance(d, dict):
            last = d
        else:
            last = {}
        keys = list(last.keys())[:8] if isinstance(last, dict) else type(last).__name__
        results: dict[str, float] = {}
        _walk_for_macro(last, results)
        m = results.get("medalcare_test", float("nan"))
        p = results.get("ptbxl_test", float("nan"))
        v = results.get("medalcare_val", results.get("val", float("nan")))
        print(f"{name:>28} {k:>5} {m:10.4f} {p:10.4f} {v:8.4f} {keys}")


if __name__ == "__main__":
    main()
