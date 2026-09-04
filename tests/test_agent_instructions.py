"""The repository's judgment rules exist twice, so they are checked to match.

`CLAUDE.md` and `AGENTS.md` hold the same document because two different agents
look for different filenames. Nothing made them stay the same, and this file is
about the one thing this repository is least willing to tolerate: a document
that describes a rule the code no longer follows. Two copies is that failure
waiting to happen -- an edit to one is invisible in the other, and the agent
reading the stale copy is told not to try something that now works.

Checked rather than deduplicated. A one-line pointer would be tidier and is not
obviously safe: an agent that reads its instruction file and does not follow the
reference gets no rules at all, which is worse than a copy that might drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "CLAUDE.md"
MIRROR = REPO / "AGENTS.md"


def test_the_canonical_instructions_exist():
    assert CANONICAL.is_file(), "CLAUDE.md is the source of the judgment rules"
    assert CANONICAL.stat().st_size > 1000


@pytest.mark.skipif(not MIRROR.exists(), reason="no AGENTS.md in this checkout")
def test_the_two_copies_of_the_instructions_agree():
    """Byte for byte.

    Not "roughly the same": the rules here are specific, and a paragraph
    present in one copy and absent from the other is exactly the drift that
    makes a reader trust the wrong one. If they are meant to differ, this test
    is the place to say why.
    """
    canonical = CANONICAL.read_bytes()
    mirror = MIRROR.read_bytes()
    if canonical == mirror:
        return

    canonical_lines = canonical.decode("utf-8").splitlines()
    mirror_lines = mirror.decode("utf-8").splitlines()
    only_canonical = [
        line for line in canonical_lines if line.strip() and line not in mirror_lines
    ]
    only_mirror = [
        line for line in mirror_lines if line.strip() and line not in canonical_lines
    ]
    raise AssertionError(
        "CLAUDE.md and AGENTS.md have drifted.\n"
        f"  only in CLAUDE.md ({len(only_canonical)} lines): "
        f"{only_canonical[:3]}\n"
        f"  only in AGENTS.md ({len(only_mirror)} lines): {only_mirror[:3]}\n"
        "Copy the canonical file over the mirror, or delete this test and say "
        "in its place why the two are meant to differ."
    )


@pytest.mark.skipif(not MIRROR.exists(), reason="no AGENTS.md in this checkout")
def test_the_mirror_is_line_ending_clean():
    """Hashes in this repository are computed over exact bytes."""
    assert b"\r\n" not in MIRROR.read_bytes()
