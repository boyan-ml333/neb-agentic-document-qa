from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedSegment:
    """Pre-chunk unit of content with metadata.

    PDFs yield one segment per page; Markdown yields one per top-level section.
    source_metadata carries either {"page": N} or {"heading_path": "A > B"}.
    """
    text: str
    source_metadata: dict = field(default_factory=dict)


class PdfParser:
    """Extract text from PDF files using PyMuPDF, one segment per page."""

    def parse(self, path: Path) -> list[ParsedSegment]:
        import fitz  # pymupdf

        segments: list[ParsedSegment] = []
        with fitz.open(str(path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    segments.append(ParsedSegment(
                        text=text,
                        source_metadata={"page": page_num},
                    ))
        return segments


class MarkdownParser:
    """Parse Markdown files using markdown-it-py AST.

    Splits content on headings, preserving a slash-separated heading_path
    (e.g. "Terms > Interest Rate") so chunks carry navigable context.
    """

    def parse(self, path: Path) -> list[ParsedSegment]:
        from markdown_it import MarkdownIt

        md = MarkdownIt()
        source = path.read_text(encoding="utf-8")
        tokens = md.parse(source)

        segments: list[ParsedSegment] = []
        heading_stack: list[str] = []   # heading_stack[i] = text of level i+1
        current_lines: list[str] = []

        def flush(heading_stack: list[str], lines: list[str]) -> None:
            text = "\n".join(lines).strip()
            if text:
                path_str = " > ".join(heading_stack) if heading_stack else "root"
                segments.append(ParsedSegment(
                    text=text,
                    source_metadata={"heading_path": path_str},
                ))

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            if tok.type == "heading_open":
                # Flush previous section before starting a new one
                flush(heading_stack, current_lines)
                current_lines = []

                level = int(tok.tag[1]) - 1  # h1→0, h2→1, …
                # Grab the inline token that follows
                inline_tok = tokens[i + 1] if i + 1 < len(tokens) else None
                heading_text = inline_tok.content if inline_tok else ""

                # Trim stack to current depth, then push new heading
                heading_stack = heading_stack[:level]
                heading_stack.append(heading_text)
                i += 3  # skip heading_open, inline, heading_close
                continue

            elif tok.type in ("inline", "fence", "html_block", "code_block"):
                current_lines.append(tok.content or "")

            elif tok.type == "paragraph_open":
                pass  # content comes in the inline child token

            i += 1

        flush(heading_stack, current_lines)
        return segments
