from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import IngestedEquation, PaperIngestionResult
from .parser import extract_equations

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def ingest_text(text: str, *, source: str = "inline") -> PaperIngestionResult:
    """Extract equations with coarse source-location metadata from Markdown/text."""
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        m = _HEADING.match(line)
        if m:
            headings.append((idx, m.group(2)))

    equations = extract_equations(text)
    ingested: list[IngestedEquation] = []
    cursor = 0
    for eq in equations:
        pos = text.find(eq.source, cursor)
        if pos < 0:
            pos = text.find(eq.source)
        if pos >= 0:
            line_no = text.count("\n", 0, pos) + 1
            cursor = pos + len(eq.source)
        else:
            line_no = None
        section = None
        if line_no is not None:
            prior = [title for number, title in headings if number <= line_no]
            section = prior[-1] if prior else None
        digest = hashlib.sha256((source + "\n" + eq.source).encode("utf-8")).hexdigest()
        ingested.append(IngestedEquation(source=source, line=line_no, section=section, equation=eq, equation_id=digest))
    return PaperIngestionResult(source=source, equations=ingested, count=len(ingested))


def ingest_file(path: str | Path) -> PaperIngestionResult:
    p = Path(path)
    return ingest_text(p.read_text(encoding="utf-8"), source=str(p))
