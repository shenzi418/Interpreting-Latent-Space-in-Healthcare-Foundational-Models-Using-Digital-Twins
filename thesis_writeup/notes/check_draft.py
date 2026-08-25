r"""check_draft.py — mechanical integrity checks for the thesis LaTeX sources.

Run from thesis_writeup/:   python notes/check_draft.py [chapters/*.tex ...]

Checks (all report-only; exit code 1 if any hard failure):
  1. forbidden phrasings   — regexes in notes/forbidden_phrases.txt (decision doc §4)
  2. \todo{} count         — must be 0 at submission
  3. \tracenote{} coverage — every numeric token that looks like a result
                             (0.xxxx, p=..., rho=..., eta2=...) should have a \tracenote
                             within the same paragraph; reports paragraphs with numbers but no trace
  4. R-with-floor          — every \circR or "R̄" number must be within 200 chars of the word 'floor'
  5. scaler discipline     — every 'transport'/'cross-domain' macro-F1 sentence should mention a scaler
                             (strict|source|target|both scalers|\qone) — heuristic, review-only
  6. UNVERIFIED citations  — every \cite key must exist in references.bib and its entry must carry a
                             '% verified:' line above it (not '% UNVERIFIED')
"""
import re, sys, glob, os, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
files = sys.argv[1:] or sorted(glob.glob("chapters/*.tex"))

def load(p):
    return io.open(p, encoding="utf-8").read()

# ---------- 1. forbidden phrasings ----------
pats = []
for ln in load("notes/forbidden_phrases.txt").splitlines():
    ln = ln.rstrip("\n")
    if not ln or ln.startswith("#"):
        continue
    try:
        pats.append((ln, re.compile(ln, re.I)))
    except re.error as e:
        print(f"[warn] bad regex in forbidden_phrases.txt: {ln!r}: {e}")

hard_fail = False
print("=" * 78)
print("1. FORBIDDEN PHRASINGS")
for f in files:
    txt = load(f)
    for i, line in enumerate(txt.splitlines(), 1):
        if line.lstrip().startswith("%"):
            continue
        for raw, rx in pats:
            if rx.search(line):
                print(f"  {f}:{i}: /{raw}/ :: {line.strip()[:140]}")
                hard_fail = True

# ---------- 2. todo ----------
print("=" * 78)
print("2. \\todo{} COUNT")
ntodo = 0
for f in files:
    n = len(re.findall(r"\\todo\{", load(f)))
    ntodo += n
    if n:
        print(f"  {f}: {n}")
print(f"  total: {ntodo}")

# ---------- 3. tracenote coverage ----------
print("=" * 78)
print("3. PARAGRAPHS WITH RESULT-LIKE NUMBERS BUT NO \\tracenote{}")
numrx = re.compile(r"(?<![\w.])(0\.\d{2,4}|p\s*[=<>]\s*0\.\d+|\\rho\s*=|\\eta\^?2?\s*=|\\etasq\s*=|\d+\.\d+\s*\\%)")
for f in files:
    paras = re.split(r"\n\s*\n", load(f))
    for k, para in enumerate(paras):
        body = "\n".join(l for l in para.splitlines() if not l.lstrip().startswith("%"))
        if numrx.search(body) and "\\tracenote" not in body:
            snippet = body.strip().replace("\n", " ")[:120]
            print(f"  {f} ¶{k}: {snippet}…")

# ---------- 4. R with floor ----------
print("=" * 78)
print("4. CIRCULAR R WITHOUT 'floor' NEARBY")
rrx = re.compile(r"(\\circR|R̄|\\bar\{R\})[^\n]{0,40}?0\.\d{2,4}")
for f in files:
    txt = load(f)
    for m in rrx.finditer(txt):
        window = txt[max(0, m.start() - 200): m.end() + 200]
        if "floor" not in window.lower() and "\\Rfloor" not in window:
            ln = txt.count("\n", 0, m.start()) + 1
            print(f"  {f}:{ln}: {m.group(0)}")
            hard_fail = True

# ---------- 5. scaler discipline (heuristic) ----------
print("=" * 78)
print("5. CROSS-DOMAIN NUMBERS WITHOUT A NAMED SCALER (review-only)")
srx = re.compile(r"(cross-domain|transport|transfer)[^.]{0,200}?0\.\d{3,4}", re.I)
for f in files:
    txt = load(f)
    for m in srx.finditer(txt):
        s = txt[max(0, m.start() - 150): m.end() + 150].lower()
        if not any(k in s for k in ("strict", "source", "target", "both scaler", "\\qone", "legacy", "in-domain")):
            ln = txt.count("\n", 0, m.start()) + 1
            print(f"  {f}:{ln}: {m.group(0)[:120]}")

# ---------- 6. citations ----------
print("=" * 78)
print("6. CITATION KEYS vs references.bib")
bib = load("references.bib") if os.path.exists("references.bib") else ""
entries = {}
for m in re.finditer(r"(?:^|\n)((?:%[^\n]*\n)*)@\w+\{([^,\s]+),", bib):
    comment, key = m.group(1), m.group(2)
    entries[key] = ("verified" in comment.lower()) and ("unverified" not in comment.lower())
used = set()
for f in files:
    for m in re.finditer(r"\\(?:cite[tp]?\*?|nocite)\{([^}]*)\}", load(f)):
        for k in m.group(1).split(","):
            used.add(k.strip())
for k in sorted(used):
    if k not in entries:
        print(f"  MISSING in bib: {k}")
        hard_fail = True
    elif not entries[k]:
        print(f"  NOT VERIFIED : {k}")
        hard_fail = True
print(f"  {len(used)} keys used, {len(entries)} entries in bib")


# ---------- 7. STYLE report (report-only; see 02_writing_guide.md §8) ----------
print("=" * 78)
print("7. STYLE (report-only)")
lex = []
lexp = "notes/style_lexicon.txt"
if os.path.exists(lexp):
    for ln in load(lexp).splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            try:
                lex.append((ln, re.compile(ln, re.I)))
            except re.error as e:
                print(f"[warn] bad regex in style_lexicon.txt: {ln!r}: {e}")

def strip_tex(txt):
    """Remove comments, tracenotes/todos, tabular environments, math, and commands for prose statistics."""
    txt = re.sub(r"(?m)^\s*%.*$", "", txt)
    txt = re.sub(r"\\tracenote\{[^}]*\}", "", txt)
    txt = re.sub(r"\\todo\{[^}]*\}", "", txt)
    txt = re.sub(r"\\begin\{(table|tabular|figure|tikzpicture)\*?\}.*?\\end\{\1\*?\}", " ", txt, flags=re.S)
    txt = re.sub(r"\$[^$]*\$", " MATH ", txt)
    txt = re.sub(r"\\(cite[pt]?|ref|label|url|texttt|emph|textbf|S)\{[^}]*\}", " X ", txt)
    txt = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", txt)
    txt = re.sub(r"[{}]", "", txt)
    return txt

for f in files:
    raw = load(f)
    prose = strip_tex(raw)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", prose)
    nw = max(1, len(words))
    kw = nw / 1000.0
    em = prose.count("---") + prose.count("—")
    notbut = len(re.findall(r"\bnot\b[^.;]{1,60}\bbut\b|\bis not\b[^.;]{1,40}[,;]\s*it is\b|\brather than\b", prose, re.I))
    part_tail = len(re.findall(r",\s+[a-z]+ing\b[^.]{0,80}\.", prose))
    trip = len(re.findall(r"\b\w+, \w+,? and \w+\b", prose))
    sent_init = len(re.findall(r"(?:^|[.!?]\s+)(Additionally|Moreover|Furthermore|Notably|Importantly|Interestingly|Crucially)\b", prose))
    # parenthetical prose asides (owner rule 2026-08-25): parens with >=2 words that are not
    # cross-refs/citations (those are stripped to "X" or start Figure/Table/Chapter/digit)
    parens_all = re.findall(r"\(([^)]{0,160})\)", prose)
    asides = [c for c in parens_all if len(c.split()) >= 2 and "MATH" not in c
              and not re.match(r"(Figure|Table|Chapter|Equation|Section|X\b|\d)", c.strip())
              and not re.search(r"\b(n|p|CI|SD|IQR|AUC|AUROC|macro|rho|eta)\s*[=<>]|\b95\s*%|\\,\\%", c)]
    bold = len(re.findall(r"\\textbf\{", re.sub(r"\\begin\{(table|tabular)\*?\}.*?\\end\{\1\*?\}", "", raw, flags=re.S)))
    we = len(re.findall(r"\b[Ww]e\b", prose)) if "06_declarations" not in f else 0
    # sentence lengths
    sents = [s for s in re.split(r"(?<=[.!?])\s+", prose) if len(s.split()) >= 3]
    sl = [len(s.split()) for s in sents]
    n_long = sum(1 for n in sl if n > 40)  # over-long sentences: a per-sentence AI/readability tell
    import statistics as st
    sl_mean = st.mean(sl) if sl else 0
    sl_sd = st.pstdev(sl) if len(sl) > 1 else 0
    # paragraph lengths (in words)
    paras = [p for p in re.split(r"\n\s*\n", strip_tex(raw)) if len(p.split()) >= 15]
    pl = [len(p.split()) for p in paras]
    pl_cv = (st.pstdev(pl) / st.mean(pl)) if len(pl) > 1 and st.mean(pl) > 0 else 0
    lex_hits = []
    for pat, rx in lex:
        for m in rx.finditer(prose):
            lex_hits.append(m.group(0))
    print(f"  {f}: words={nw}  em-dash/1k={em/kw:.1f}  not-but/1k={notbut/kw:.1f}  participle-tails={part_tail}  triplets={trip}  "
          f"sent-init-adverbs={sent_init}  bold-in-prose={bold}  'we'={we}  sent len {sl_mean:.1f}±{sl_sd:.1f} (sd/mean {sl_sd/max(sl_mean,1):.2f})  "
          f"long>40w={n_long}/{len(sl)} (max {max(sl) if sl else 0})  asides/1k={len(asides)/kw:.1f}  para-cv={pl_cv:.2f} (n={len(pl)})")
    if lex_hits:
        from collections import Counter
        c = Counter(h.lower() for h in lex_hits)
        print("     lexicon:", ", ".join(f"{k}×{v}" for k, v in c.most_common(12)))
    flags = []
    if em / kw > 5: flags.append("em-dash density > 5/1k")
    if notbut / kw > 1: flags.append("not-but > 1/1k")
    if sent_init: flags.append("sentence-initial adverb connectives")
    if bold: flags.append("bold in prose")
    if we: flags.append("'we' outside Declarations")
    if sl and sl_sd / max(sl_mean, 1) < 0.35: flags.append("sentence lengths too uniform")
    if sl and n_long / len(sl) > 0.08: flags.append(f"over-long sentences: {n_long}/{len(sl)} exceed 40 words")
    if len(asides) / kw > 3: flags.append(f"parenthetical asides > 3/1k ({len(asides)} found)")
    if len(pl) > 3 and pl_cv < 0.35: flags.append("paragraph lengths too uniform")
    if flags:
        print("     FLAGS:", "; ".join(flags))

print("=" * 78)
print("RESULT:", "HARD FAILURES PRESENT" if hard_fail else "no hard failures")
sys.exit(1 if hard_fail else 0)
