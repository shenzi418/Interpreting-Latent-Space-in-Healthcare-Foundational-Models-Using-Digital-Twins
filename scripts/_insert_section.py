"""Insert a markdown section into a file immediately before a marker line.

Appending would put Stage 2/3 after Stage 4/5, which were written first because
they finished first. Chronology of *execution* is already in the timestamps; the
document should read in stage order.

Reads and writes UTF-8 explicitly -- `Get-Content | Add-Content` round-trips
through the console codepage and double-encodes every non-ASCII character (see
`_fix_log_encoding.py`).

Usage:
    python scripts/_insert_section.py <target> <section-file> <marker-prefix>
"""
from __future__ import annotations

import io
import shutil
import sys


def main() -> int:
    target, section_path, marker = sys.argv[1], sys.argv[2], sys.argv[3]

    lines = io.open(target, encoding="utf-8").read().split("\n")
    section = io.open(section_path, encoding="utf-8").read().rstrip("\n").split("\n")

    hits = [i for i, l in enumerate(lines) if l.startswith(marker)]
    if len(hits) != 1:
        print(f"marker {marker!r} matched {len(hits)} lines; expected exactly 1")
        return 1
    at = hits[0]

    shutil.copyfile(target, target + ".bak")
    out = lines[:at] + section + ["", "---", ""] + lines[at:]
    io.open(target, "w", encoding="utf-8", newline="\n").write("\n".join(out))

    print(f"inserted {len(section)} lines before line {at + 1} ({lines[at][:60]!r})")
    print(f"total now {len(out)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
