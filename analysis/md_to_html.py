"""Convert a Markdown report with local images into a self-contained HTML file.

Images referenced via ![...](relative_path.png) are base64-embedded so the
HTML can be emailed or opened anywhere without external dependencies.

Usage:
    python analysis/md_to_html.py outputs/latent_analysis/LATENT_SPACE_ANALYSIS_REPORT.md
"""

import argparse
import base64
import re
from pathlib import Path


CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    max-width: 960px;
    margin: 40px auto;
    padding: 0 20px;
    line-height: 1.6;
    color: #24292e;
    background: #fff;
}
h1 { border-bottom: 2px solid #e1e4e8; padding-bottom: 8px; }
h2 { border-bottom: 1px solid #e1e4e8; padding-bottom: 6px; margin-top: 32px; }
h3 { margin-top: 24px; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 14px;
}
th, td {
    border: 1px solid #d0d7de;
    padding: 6px 12px;
    text-align: left;
}
th { background: #f6f8fa; font-weight: 600; }
tr:nth-child(even) { background: #f6f8fa; }
img { max-width: 100%; height: auto; margin: 12px 0; display: block; }
blockquote {
    border-left: 4px solid #d0d7de;
    padding: 4px 16px;
    margin: 12px 0;
    color: #57606a;
    background: #f6f8fa;
}
code { background: #f0f2f4; padding: 2px 6px; border-radius: 3px; font-size: 13px; }
strong { font-weight: 600; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 24px 0; }
@media print {
    body { max-width: 100%; margin: 20px; }
    img { max-width: 100%; page-break-inside: avoid; }
    h2, h3 { page-break-after: avoid; }
    table { page-break-inside: avoid; }
}
"""


def embed_images(md_text: str, md_dir: Path) -> str:
    """Replace ![alt](path.png) with base64-embedded data URIs."""
    def replacer(match):
        alt = match.group(1)
        src = match.group(2)
        img_path = md_dir / src
        if not img_path.exists():
            return match.group(0)
        suffix = img_path.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(suffix, "image/png")
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'![{alt}](data:{mime};base64,{b64})'
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replacer, md_text)


def md_to_html(md_text: str) -> str:
    """Minimal markdown-to-HTML converter (no external deps)."""
    lines = md_text.split("\n")
    html_lines = []
    in_table = False
    in_list = False
    in_blockquote = False
    i = 0

    def inline(text):
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        text = re.sub(
            r'\\\(\s*(.+?)\s*\\?\)',
            r'<em>\1</em>',
            text,
        )
        return text

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("# ") and not stripped.startswith("## "):
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            html_lines.append(f"<h1>{inline(stripped[2:])}</h1>")
        elif stripped.startswith("## ") and not stripped.startswith("### "):
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            html_lines.append(f"<h2>{inline(stripped[3:])}</h2>")
        elif stripped.startswith("### ") and not stripped.startswith("#### "):
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            html_lines.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith("#### "):
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            html_lines.append(f"<h4>{inline(stripped[5:])}</h4>")
        elif stripped.startswith("---"):
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            html_lines.append("<hr>")
        elif stripped.startswith("|"):
            if in_list: html_lines.append("</ul>"); in_list = False
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(set(c) <= set("-: ") for c in cells):
                i += 1
                continue
            if not in_table:
                html_lines.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            row = "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
        elif stripped.startswith("- "):
            if in_table: html_lines.append("</table>"); in_table = False
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{inline(stripped[2:])}</li>")
        elif re.match(r'^\d+\.\s', stripped):
            if in_table: html_lines.append("</table>"); in_table = False
            content = re.sub(r'^\d+\.\s', '', stripped)
            if not in_list:
                html_lines.append("<ol>")
                in_list = True
            html_lines.append(f"<li>{inline(content)}</li>")
        elif stripped.startswith("> "):
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            if not in_blockquote:
                html_lines.append("<blockquote>")
                in_blockquote = True
            html_lines.append(f"<p>{inline(stripped[2:])}</p>")
        elif stripped.startswith("!["):
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            html_lines.append(f"<p>{inline(stripped)}</p>")
        elif stripped == "":
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            if in_blockquote: html_lines.append("</blockquote>"); in_blockquote = False
        else:
            if in_table: html_lines.append("</table>"); in_table = False
            if in_list: html_lines.append("</ul>"); in_list = False
            if in_blockquote: html_lines.append("</blockquote>"); in_blockquote = False
            if stripped:
                html_lines.append(f"<p>{inline(stripped)}</p>")
        i += 1

    if in_table: html_lines.append("</table>")
    if in_list: html_lines.append("</ul>")
    if in_blockquote: html_lines.append("</blockquote>")
    return "\n".join(html_lines)


def main():
    parser = argparse.ArgumentParser(description="Convert MD report to self-contained HTML.")
    parser.add_argument("input", type=Path, help="Path to .md file")
    parser.add_argument("--output", type=Path, default=None, help="Output .html path (default: same name)")
    args = parser.parse_args()

    md_dir = args.input.parent
    md_text = args.input.read_text(encoding="utf-8")
    md_text = embed_images(md_text, md_dir)
    body = md_to_html(md_text)

    out = args.output or args.input.with_suffix(".html")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Latent Space Analysis Report</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print("Open in browser, then Ctrl+P to save as PDF if needed.")


if __name__ == "__main__":
    main()
