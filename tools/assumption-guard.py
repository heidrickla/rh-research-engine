"""Every assumption a result rests on must be resolvable, tested, and bounded.

WHY. `ExperimentResult.assumptions` was free text, so nothing could act on it.
`weil_certified` declared "Re psi(z) <= log|z| for Re z > 0" -- false below
Re z = 1/6 -- in the assumptions list of a `rigorous_numerical` record, and it
sat there because no gate reads that field. Nothing computed was wrong; the
record was.

WHAT IT CHECKS.

  resolvable   every assumption line in a recorded ExperimentResult names a
               registry id. A free-text assumption is one nobody can check.

  tested       every registered assumption names a pytest node id that EXISTS.
               A named test that was renamed away is worse than none: it reads
               as checked.

  bounded      every registered assumption says where it fails. UNEXAMINED is
               permitted and REPORTED, because "nobody looked for the boundary"
               and "there is no boundary" must not read the same. That is the
               distinction the false digamma bound lived in.

AND EACH OF THOSE THREE ENUMERATES, SO EACH NEEDS A FLOOR. The first version of
this tool reported PASS with an empty registry and no recorded assumptions:
"every recorded assumption resolves" is vacuously true of nothing. Breaking the
checked thing does not find that -- an injected violation still fails, because
the hole is not that the check cannot fail today but that it can succeed at
scanning nothing tomorrow, after a rename, a moved file, or a wrong working
directory. A one-file check cannot scan nothing; an enumerating one can.

So the counts have floors that would be absurd at zero, and the floors were set
deliberately too high once and watched to fail before being trusted.

Usage:
    python tools/assumption-guard.py            # fail on any violation
    python tools/assumption-guard.py --list     # print the registry
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rh_research_engine.core.assumptions import (  # noqa: E402
    REGISTRY,
    assumption_id,
)

#: An enumeration that returns nothing must not report PASS. These are floors on
#: what the guard SAW, not on what it found -- a scan of zero registered
#: assumptions, or of zero recorded ones, means the guard did not run rather
#: than that everything is well.
MINIMUM_REGISTERED = 1
MINIMUM_RECORDED = 1


def _collected_node_ids() -> set[str]:
    """Every test pytest can actually collect, so a renamed test is caught."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header"],
        cwd=REPO,
        capture_output=True,
        text=True,
        # `text=True` alone decodes with locale.getpreferredencoding(False),
        # which is cp1252 here: a non-ASCII node id would decode wrong, fail to
        # match the registry, and be reported as a RENAMED TEST -- a false
        # positive that is harder to diagnose than a crash. Nothing in ruff
        # covers this; only tools/check-text-encoding.py does.
        encoding="utf-8",
    )
    ids = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if "::" in line:
            ids.add(line.replace("\\", "/"))
            ids.add(line.replace("\\", "/").split("[")[0])
    return ids


def _recorded_assumption_lines() -> tuple[list[tuple[str, str]], int]:
    """Assumptions on the LATEST record per experiment, and how many older ones
    still carry free text.

    A correction files a NEW record here -- ids are content hashes and history is
    not edited -- so the current state of the research record is the last row for
    each name. Judging every historical row would make this gate unpassable the
    moment any assumption is ever improved, which is a gate that fails for being
    right. Superseded rows are counted and reported instead, because a false
    statement left in the corpus is still a fact worth seeing.
    """
    path = REPO / "research_state" / "experiments.jsonl"
    if not path.exists():
        return [], 0
    latest: dict[str, list[str]] = {}
    history: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name = record.get("name", "?")
        entries = record.get("assumptions", []) or []
        if name in latest:
            history.append((name, latest[name]))
        latest[name] = entries
    superseded = sum(
        1
        for _, entries in history
        for entry in entries
        if assumption_id(entry) is None
    )
    rows = [(name, entry) for name, entries in latest.items() for entry in entries]
    return rows, superseded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="skip the pytest collection check (it costs a few seconds)",
    )
    args = parser.parse_args()

    if args.list:
        for entry in REGISTRY.values():
            print(f"  {entry.cite()}")
            print(f"      checked by {entry.checked_by}")
        return 0

    violations: list[str] = []
    unexamined = [a for a in REGISTRY.values() if not a.examined]

    if not args.skip_collect:
        node_ids = _collected_node_ids()
        if not node_ids:
            violations.append("pytest collected nothing; the tested check cannot run")
        for entry in REGISTRY.values():
            if entry.checked_by not in node_ids:
                violations.append(
                    f"assumption {entry.id!r} names {entry.checked_by}, which pytest "
                    "does not collect. A named test that no longer exists reads as "
                    "checked and is not."
                )

    rows, superseded = _recorded_assumption_lines()

    # THE FLOORS. Checked before the loops, because a loop over nothing reports
    # success and says so in the same words it uses when it has checked
    # everything.
    if len(REGISTRY) < MINIMUM_REGISTERED:
        violations.append(
            f"the registry holds {len(REGISTRY)} assumptions, below the floor of "
            f"{MINIMUM_REGISTERED}. An empty registry passes every check below by "
            "having nothing to check, which is not the same as being sound."
        )
    if len(rows) < MINIMUM_RECORDED:
        violations.append(
            f"{len(rows)} recorded assumptions found across the latest experiment "
            f"records, below the floor of {MINIMUM_RECORDED}. Records carrying "
            "assumptions exist in this repository, so finding none means this guard "
            "read the wrong file or none at all."
        )

    for name, line in rows:
        found = assumption_id(line)
        if found is None:
            violations.append(
                f"record {name!r} carries a free-text assumption that names no "
                f"registry id: {line[:80]!r}. Nothing can check it."
            )
        elif found not in REGISTRY:
            violations.append(
                f"record {name!r} cites assumption {found!r}, which is not registered."
            )

    print(
        f"  {len(REGISTRY)} registered assumptions ({len(unexamined)} with no boundary "
        f"examined), {len(rows)} recorded on the latest records"
    )
    if superseded:
        print(
            f"    {superseded} free-text assumptions remain in SUPERSEDED records. "
            "History is not edited here, so they stay; the current record is what "
            "this gate judges."
        )
    for entry in unexamined:
        print(f"    UNEXAMINED: {entry.id} -- {entry.statement}")
    if unexamined:
        print(
            "    (permitted, and reported: an assumption nobody has probed for a "
            "boundary must not read like one known to have none)"
        )

    if violations:
        print("\nassumption guard FAILED:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print("  every recorded assumption resolves, and every registered one names a live test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
