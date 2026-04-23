#!/usr/bin/env python3
"""Render README.md -> README.pdf with sensible print styles.

Pipeline:  markdown → HTML with GFM-style extensions → xhtml2pdf PDF.

Usage:
    python3 tools/render_readme_pdf.py                 # README.md → README.pdf
    python3 tools/render_readme_pdf.py --in foo.md --out foo.pdf

Dependencies (install with: pip install markdown xhtml2pdf):
    - markdown (HTML conversion with tables + fenced code)
    - xhtml2pdf (pure-Python HTML→PDF; no external binaries)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import markdown
from xhtml2pdf import pisa

CSS = """
@page { size: letter; margin: 0.8in; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif;
       font-size: 10pt; line-height: 1.35; color: #222; }
/* No page-break-before on h1 — xhtml2pdf doesn't honor :first-of-type
   or inline overrides, so the only way to keep page-1 non-blank is
   to not force a break at all. The README uses a single h1 (title)
   and h2 for section boundaries, so this loses nothing. */
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 4pt; }
h2 { font-size: 15pt; border-bottom: 1px solid #aaa; padding-bottom: 3pt;
     margin-top: 18pt; }
h3 { font-size: 12pt; margin-top: 14pt; }
h4 { font-size: 11pt; margin-top: 10pt; }
code { font-family: Menlo, Monaco, monospace; font-size: 8.5pt;
       background: #f4f4f4; padding: 1px 3px; border-radius: 2px; }
pre { background: #f4f4f4; padding: 6pt; border-radius: 3px;
      overflow-x: auto; font-size: 8.5pt; line-height: 1.25;
      white-space: pre-wrap; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; text-align: left;
         vertical-align: top; }
th { background: #eee; }
blockquote { border-left: 3px solid #aaa; color: #555; padding-left: 8pt;
             margin: 6pt 0; }
ul, ol { margin: 4pt 0 8pt 16pt; }
li { margin: 2pt 0; }
hr { border: none; border-top: 1px solid #aaa; margin: 12pt 0; }
strong { color: #000; }
"""


def render(md_path: Path, pdf_path: Path) -> int:
    md_src = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_src,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{md_path.stem}</title><style>{CSS}</style></head>"
        f"<body>{html_body}</body></html>"
    )
    with open(pdf_path, "wb") as out:
        res = pisa.CreatePDF(doc, dest=out, encoding="utf-8")
    return 1 if res.err else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="README.md")
    ap.add_argument("--out", dest="dst", default="README.pdf")
    args = ap.parse_args()
    src = Path(args.src); dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    rc = render(src, dst)
    size = dst.stat().st_size if dst.exists() else 0
    print(f"[pdf] {src} -> {dst}  size={size:,} bytes  rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
