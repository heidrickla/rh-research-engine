#!/usr/bin/env python3
"""Refuse CRLF in tracked text files, because a line ending is part of a hash here.

WHY THIS REPOSITORY CARES. `MathCertificate.certificate_hash`,
`DreEvidenceEnvelope.payload_hash`, and the formula-index digests are computed
over exact bytes. `write_dre_experiment` produces the artifact DRE consumes,
and DRE computes `ModelHash` over the raw bytes of a pack -- so the CRLF and
LF versions of the same file are *different models*. A file that drifts to
CRLF gets a different hash, every record referencing it fails replay, and on
Windows the file looks identical in every editor.

`.gitattributes` pins `* text=auto eol=lf`, which normalises on checkout and
on `git add`. That is not enough on its own, and the gap is specific:

  ** A CLEAN INDEX IS NOT A CLEAN WORKING TREE. **

What `.gitattributes` does not cover is a tool rewriting a file **in place**
with CRLF. The index stays LF, the working tree drifts, and `git status` is
silent because the filter reconciles them on the way past. That drift is
invisible until the working tree leaves git -- which it does whenever a file
is read by the engine rather than by `git show`.

THE MEASUREMENT TRAP -- read this before "verifying" anything here.
On Windows the obvious ways to count carriage returns lie:

  * `grep -c $'\\r' <file>` and anything piped through MSYS or Git Bash can
    ADD CR to the stream, so pure-LF files get reported as CRLF.
  * `git show` / `git cat-file` piped into another command do the same.

`git ls-files --eol` is the authoritative diagnostic, and it stays
authoritative through a pipe: the `i/` and `w/` fields are git's own verdict
computed from the real bytes. This script drives off that rather than
re-implementing text detection -- re-implementing would also mean
re-implementing `.gitattributes` resolution, and getting that subtly wrong is
how a checker starts lying too.

Usage:
  python tools/check-line-endings.py           # every tracked text file
  python tools/check-line-endings.py --staged  # only staged files (pre-commit)
  python tools/check-line-endings.py --fix     # rewrite offenders to LF
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class NotAGitRepository(RuntimeError):
    """This gate reads git's index, and there is no index to read."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode == 0:
        return result.stdout.strip()

    detail = (result.stderr or "").strip().splitlines()
    if any("not a git repository" in line for line in detail):
        # The one case worth its own message: a CalledProcessError traceback
        # names an exit status and nothing a reader can act on.
        raise NotAGitRepository(
            "not a git repository, so `git ls-files --eol` has nothing to "
            "report. This gate compares the index against the working tree; "
            "run it from a checkout, not from an exported or unpacked copy."
        )
    # Any OTHER git failure keeps its original type -- a missing tag in a
    # shallow clone is the common one, and callers already degrade for it
    # deliberately. Reclassifying every git error as "no repository" broke that
    # fallback in CI once.
    raise subprocess.CalledProcessError(
        result.returncode, ["git", *args], result.stdout, result.stderr
    )


def _staged_paths() -> set[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return {line.strip() for line in out.splitlines() if line.strip()}


def _offenders(limit_to: set[str] | None) -> list[str]:
    """Return tracked text files whose working-tree copy contains CRLF.

    Parsed from `git ls-files --eol`, whose output lines look like:
        i/lf    w/crlf  attr/text=auto eol=lf   path/to/file
    """
    bad: list[str] = []
    for line in _git("ls-files", "--eol").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        path = fields[-1].strip()
        flags = fields[0].split()
        if limit_to is not None and path not in limit_to:
            continue
        # w/ is the working-tree verdict. w/none means binary or empty.
        working = next((f for f in flags if f.startswith("w/")), "")
        if working in ("w/crlf", "w/mixed"):
            bad.append(path)
    return sorted(bad)


def _fix(paths: list[str]) -> None:
    for path in paths:
        p = Path(path)
        raw = p.read_bytes()
        p.write_bytes(raw.replace(b"\r\n", b"\n"))
        print(f"  fixed: {path}")


def _fail_closed(exc: NotAGitRepository) -> int:
    print(f"line endings: CANNOT CHECK -- {exc}", file=sys.stderr)
    # Nonzero: "could not verify" is not "verified". Failing closed is the same
    # rule the rest of this repository follows for unreadable state.
    return 2


def main(argv: list[str]) -> int:
    try:
        return _run(argv)
    except NotAGitRepository as exc:
        return _fail_closed(exc)


def _run(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        # Answers "does this tool load and parse arguments" without reading
        # git, so the invocability check does not double as an invariant check.
        print(__doc__.strip())
        return 0
    staged = "--staged" in argv
    fix = "--fix" in argv
    limit_to = _staged_paths() if staged else None

    bad = _offenders(limit_to)
    if not bad:
        scope = "staged files" if staged else "tracked text files"
        print(f"line endings: OK ({scope} are LF)")
        return 0

    if fix:
        print(f"line endings: rewriting {len(bad)} file(s) to LF")
        _fix(bad)
        print("line endings: fixed. Re-stage the files and commit again.")
        return 0

    print(f"line endings: {len(bad)} tracked file(s) have CRLF in the working tree:", file=sys.stderr)
    for path in bad:
        print(f"  {path}", file=sys.stderr)
    print("", file=sys.stderr)
    print("A line ending is part of a hash in this repository: the CRLF and LF", file=sys.stderr)
    print("versions of a DRE artifact or model pack are different models.", file=sys.stderr)
    print("", file=sys.stderr)
    print("Fix with:  python tools/check-line-endings.py --fix", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
