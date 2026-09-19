#!/usr/bin/env python3
"""Build styled HTML source for LibreOffice DOCX conversion."""

from __future__ import annotations

import re
from pathlib import Path

import mistune


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = ROOT / "TECHNICAL_BLOG_DRAFT.md"
HTML = ROOT / "blog-assets" / "technical-blog-docx-source.html"


def main() -> None:
    source = MARKDOWN.read_text()
    source = re.sub(r"<!-- table-source:.*?-->", "", source, flags=re.DOTALL)
    for svg in (ROOT / "blog-assets").glob("*.svg"):
        source = source.replace(
            f"blog-assets/{svg.name}",
            (svg.with_suffix(".png")).resolve().as_uri(),
        )
    body = mistune.html(source)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stress-testing GridSFM</title>
<style>
@page {{ size: A4; margin: 18mm; }}
body {{ font-family: "Liberation Sans", Arial, sans-serif; color: #11152B; font-size: 11pt; line-height: 1.5; }}
h1 {{ color: #25154A; font-size: 25pt; margin: 0 0 18pt; }}
h2 {{ color: #493986; font-size: 18pt; margin: 22pt 0 8pt; }}
h3 {{ color: #493986; font-size: 14pt; margin: 16pt 0 6pt; }}
p {{ margin: 0 0 9pt; }}
li {{ margin-bottom: 4pt; }}
a {{ color: #493986; text-decoration: underline; }}
img {{ display: block; width: 100%; height: auto; margin: 10pt auto 12pt; page-break-inside: avoid; }}
code {{ font-family: "Liberation Mono", monospace; background: #F0ECFA; }}
</style>
</head>
<body>{body}</body>
</html>
"""
    HTML.write_text(document)
    print(HTML)


if __name__ == "__main__":
    main()
