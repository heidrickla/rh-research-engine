"""Every text read and write names its encoding.

WHY THIS EXISTS. The CI matrix carried a Windows leg for two defects invisible
from Linux. One was CRLF drift in bytes that get hashed, which
`check-line-endings.py` already catches on any platform. The other was the
PLATFORM DEFAULT TEXT CODEC: `open(path)` on Windows decodes as cp1252, not
UTF-8, and it mojibaked a UTF-8 `claims.json` -- silently, because cp1252
decodes almost every byte to *something*. A record that reads back as
different characters is the failure this repository exists to refuse.

The Windows leg caught that as an INSTANCE, after a merge, and only if a test
happened to read the file on Windows. This catches it as a CLASS, at authoring
time, on whatever runner is to hand -- which is why the OS axis could be
dropped when CI moved to a Linux-only forge rather than the defect simply
going unchecked.

Not a grep. `open(` appears in strings, comments and docstrings throughout the
corpus, and a textual search would either miss the real calls or drown in the
false ones. This walks the syntax tree.

Covers `open`, `Path.open`, `Path.read_text`, `Path.write_text`, and
`subprocess` in text mode. Binary modes are exempt: `encoding` is meaningless
there, and passing it raises.

WHAT RUFF ALREADY DOES, measured rather than assumed. An earlier note here and
in the handoff said `PLW1514` "sees `open()` but not `Path.read_text()`". That
was false for ruff 0.15.21: it catches `open`, `Path.open`, `Path.read_text`,
`Path.write_text`, `codecs.open` and `tempfile.NamedTemporaryFile`. It is a
PREVIEW rule, so a bare `--select PLW1514` is a silent no-op that warns and
passes -- which is the likeliest way that claim was arrived at. It is now
enabled in `pyproject.toml` via `preview` + `explicit-preview-rules`.

WHAT IT DOES NOT DO, and why this file did not become redundant.
`subprocess.run(..., text=True)` decodes the child's stdout with
`locale.getpreferredencoding(False)` -- the same cp1252 on Windows, the same
silent mojibake, a different door. No ruff rule covers it. Ten live instances
were sitting in this repository when the gate was extended to look, one of them
in `tools/phase1-final-gate.py`, which reads `git` output to decide Phase 1
closure. The gate that was written for this defect class had simply stopped at
file I/O.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Calls that decode or encode text using the platform default when told
#: nothing, and WHERE `encoding` SITS POSITIONALLY in each. `read_bytes` and
#: `write_bytes` are absent deliberately -- they never guess.
#:
#: The positions matter: `read_text("utf-8")` names the encoding perfectly
#: well, and a checker that only looked for the keyword form reported it as a
#: defect. Found by reading what this tool flagged instead of trusting it.
TEXT_CALLS = {
    "open": 3,        # open(file, mode, buffering, encoding, ...)
    "read_text": 0,   # Path.read_text(encoding, errors)
    "write_text": 1,  # Path.write_text(data, encoding, errors)
}

#: Subprocess calls that DECODE the child's output. `call` and `check_call`
#: are absent deliberately -- they return a status, never text, so `text=True`
#: on them decodes nothing and flagging it would be noise.
DECODING_SUBPROCESS_CALLS = {"run", "Popen", "check_output"}

#: Either of these puts a subprocess into text mode. `encoding=` also implies
#: text mode, and names the codec, so it is the fix rather than the defect.
TEXT_MODE_FLAGS = {"text", "universal_newlines"}

SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ruff_cache"}


def _is_binary(call: ast.Call) -> bool:
    """A mode string containing `b` makes `encoding` an error, not an omission."""
    for index, argument in enumerate(call.args):
        # open(path, "rb") -- mode is the second positional argument.
        if index == 1 and isinstance(argument, ast.Constant):
            if isinstance(argument.value, str) and "b" in argument.value:
                return True
    for keyword in call.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str) and "b" in keyword.value.value:
                return True
    return False


def _subprocess_offence(call: ast.Call, name: str) -> str | None:
    """Text mode without a named codec decodes the child with the platform default."""
    if name not in DECODING_SUBPROCESS_CALLS:
        return None
    given = {keyword.arg for keyword in call.keywords if keyword.arg}
    if not (given & TEXT_MODE_FLAGS):
        return None
    if "encoding" in given:
        return None
    flag = "text" if "text" in given else "universal_newlines"
    return f"{name}(... {flag}=True) without encoding="


def offences(source: str, path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as failure:
        return [(failure.lineno or 0, f"could not parse: {failure.msg}")]

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        subprocess_message = _subprocess_offence(node, name)
        if subprocess_message is not None:
            found.append((node.lineno, subprocess_message))
            continue
        if name not in TEXT_CALLS or _is_binary(node):
            continue
        if any(keyword.arg == "encoding" for keyword in node.keywords):
            continue
        if len(node.args) > TEXT_CALLS[name]:
            continue  # given positionally
        found.append((node.lineno, f"{name}() without encoding="))
    return found


def main() -> int:
    targets = sorted(
        path
        for path in REPO.rglob("*.py")
        if not any(part in SKIP for part in path.parts)
    )
    failures = []
    for path in targets:
        for line, message in offences(path.read_text(encoding="utf-8"), path):
            failures.append(f"{path.relative_to(REPO)}:{line}: {message}")

    if failures:
        print(f"text encoding: {len(failures)} unqualified text call(s)")
        for failure in failures:
            print(f"  {failure}")
        print(
            "\nName the encoding. On Windows the default is cp1252, which "
            "decodes UTF-8 bytes into different characters without erroring -- "
            "the record reads back wrong and nothing says so. A subprocess in "
            "text mode decodes the child's output the same way."
        )
        return 1

    print(f"text encoding: OK ({len(targets)} files, every text call names one)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
