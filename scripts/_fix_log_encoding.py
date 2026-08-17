"""One-shot repair: undo a cp1252/UTF-8 double-encoding inside a UTF-8 file.

`Get-Content | Add-Content` round-trips UTF-8 bytes through the console's cp1252
codepage, so appended blocks land as mojibake ("—" -> "â€”") while the rest of
the file stays valid. This repairs the affected lines in place, guarded on byte
sequences that cannot occur in correct text, and leaves a .bak.

Usage:
    python scripts/_fix_log_encoding.py <path>
"""
from __future__ import annotations

import io
import shutil
import sys


def unmojibake(s: str) -> str:
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def unmojibake(s: str) -> str:
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def main() -> int:
    path = sys.argv[1]
    lines = io.open(path, encoding="utf-8").read().split("\n")

    # The guard IS the round-trip. A line is mojibake iff re-encoding it as
    # cp1252 yields valid UTF-8 that differs from the original -- which requires
    # the exact multi-byte patterns UTF-8 produces, so correct text does not
    # qualify. ("café" fails: é -> 0xE9 alone is not valid UTF-8.) An earlier
    # version pre-filtered on a hand-listed set of mojibake prefixes and missed
    # "Δ" -> "Î”" because Î was not on the list; enumerating lead bytes by hand
    # is exactly the kind of thing to let the codec decide instead.
    fixed, n = [], 0
    for line in lines:
        repaired = unmojibake(line)
        if repaired != line:
            n += 1
        fixed.append(repaired)

    if not n:
        print("nothing to repair")
        return 0

    shutil.copyfile(path, path + ".bak")
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(fixed))
    residual = sum(1 for l in fixed if unmojibake(l) != l)
    print(f"repaired {n} of {len(lines)} lines | residual suspect lines: {residual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
