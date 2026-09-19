#!/usr/bin/env python3
"""Package TECHNICAL_BLOG_DRAFT.md and rendered tables as a DOCX."""

from __future__ import annotations

import html
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TECHNICAL_BLOG_DRAFT.md"
OUTPUT = ROOT / "TECHNICAL_BLOG_DRAFT.docx"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def xml_text(value: str) -> str:
    return html.escape(value, quote=False)


class DocumentBuilder:
    def __init__(self) -> None:
        self.body: list[str] = []
        self.relationships: list[str] = []
        self.media: list[tuple[Path, str]] = []
        self.next_relationship = 1
        self.next_drawing = 1

    def relationship(self, target: str, kind: str, external: bool = False) -> str:
        relationship_id = f"rId{self.next_relationship}"
        self.next_relationship += 1
        mode = ' TargetMode="External"' if external else ""
        self.relationships.append(
            f'<Relationship Id="{relationship_id}" Type="{kind}" Target="{html.escape(target, quote=True)}"{mode}/>'
        )
        return relationship_id

    def runs(self, source: str, base_bold: bool = False) -> str:
        pattern = re.compile(r"\[([^]]+)]\(([^)]+)\)|\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*")
        result: list[str] = []
        cursor = 0
        for match in pattern.finditer(source):
            if match.start() > cursor:
                result.append(self.run(source[cursor:match.start()], bold=base_bold))
            if match.group(1) is not None:
                rid = self.relationship(
                    match.group(2),
                    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                    external=True,
                )
                result.append(
                    f'<w:hyperlink r:id="{rid}">{self.run(match.group(1), color="493986", underline=True)}</w:hyperlink>'
                )
            elif match.group(3) is not None:
                result.append(self.run(match.group(3), bold=True))
            elif match.group(4) is not None:
                result.append(self.run(match.group(4), font="Liberation Mono"))
            else:
                result.append(self.run(match.group(5), italic=True))
            cursor = match.end()
        if cursor < len(source):
            result.append(self.run(source[cursor:], bold=base_bold))
        return "".join(result)

    @staticmethod
    def run(
        value: str,
        *,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        color: str | None = None,
        font: str | None = None,
    ) -> str:
        properties = []
        if bold:
            properties.append("<w:b/>")
        if italic:
            properties.append("<w:i/>")
        if underline:
            properties.append('<w:u w:val="single"/>')
        if color:
            properties.append(f'<w:color w:val="{color}"/>')
        if font:
            properties.append(f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}"/>')
        props = f"<w:rPr>{''.join(properties)}</w:rPr>" if properties else ""
        preserve = ' xml:space="preserve"' if value[:1].isspace() or value[-1:].isspace() else ""
        return f"<w:r>{props}<w:t{preserve}>{xml_text(value)}</w:t></w:r>"

    def paragraph(self, source: str, style: str | None = None, prefix: str = "") -> None:
        style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
        self.body.append(f"<w:p><w:pPr>{style_xml}</w:pPr>{self.runs(prefix + source)}</w:p>")

    def image(self, relative_path: str, alt: str) -> None:
        source = ROOT / relative_path
        png = source.with_suffix(".png")
        with Image.open(png) as bitmap:
            width, height = bitmap.size
        max_width = 5_850_000
        cx = max_width
        cy = round(cx * height / width)
        media_name = f"table-{len(self.media) + 1}.png"
        self.media.append((png, media_name))
        rid = self.relationship(
            f"media/{media_name}",
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
        )
        drawing_id = self.next_drawing
        self.next_drawing += 1
        self.body.append(f"""
<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{drawing_id}" name="{html.escape(alt, quote=True)}" descr="{html.escape(alt, quote=True)}"/>
<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{drawing_id}" name="{media_name}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>""")

    def parse(self, markdown: str) -> None:
        markdown = re.sub(r"<!-- table-source:.*?-->", "", markdown, flags=re.DOTALL)
        for line in markdown.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            image_match = re.fullmatch(r"!\[([^]]*)]\(([^)]+)\)", stripped)
            if image_match:
                self.image(image_match.group(2), image_match.group(1))
            elif stripped.startswith("# "):
                self.paragraph(stripped[2:], "Title")
            elif stripped.startswith("## "):
                self.paragraph(stripped[3:], "Heading1")
            elif stripped.startswith("### "):
                self.paragraph(stripped[4:], "Heading2")
            elif re.match(r"^\d+\. ", stripped):
                self.paragraph(stripped, "ListParagraph")
            elif stripped.startswith("- "):
                self.paragraph(stripped[2:], "ListParagraph", prefix="• ")
            else:
                self.paragraph(stripped)

    def document_xml(self) -> str:
        section = """
<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1020" w:right="900" w:bottom="1020" w:left="900" w:header="400" w:footer="400" w:gutter="0"/></w:sectPr>
"""
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS}" xmlns:r="{RNS}" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"><w:body>{''.join(self.body)}{section}</w:body></w:document>"""


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans"/><w:sz w:val="22"/><w:color w:val="11152B"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="150" w:line="330" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="300"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="25154A"/><w:sz w:val="50"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="360" w:after="120"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="493986"/><w:sz w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="240" w:after="100"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="493986"/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="360" w:hanging="180"/></w:pPr></w:style>
</w:styles>"""


def main() -> None:
    builder = DocumentBuilder()
    builder.parse(SOURCE.read_text())
    relationships = "".join(builder.relationships)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>""",
        "word/document.xml": builder.document_xml(),
        "word/styles.xml": styles_xml(),
        "word/_rels/document.xml.rels": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{relationships}</Relationships>""",
        "docProps/core.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Stress-testing GridSFM</dc:title><dc:creator>Seiole Labs</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>""",
    }
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        for source, media_name in builder.media:
            archive.write(source, f"word/media/{media_name}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
