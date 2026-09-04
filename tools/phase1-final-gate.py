#!/usr/bin/env python3
"""Run every Phase 1 closure gate in one command and write the closure report.

WHY THIS EXISTS. Phase 1 was reviewed by repeated adversarial passes, each one
finding a bypass the previous pass had not thought of. That process has no
natural end: it stops when the reviewer runs out of ideas, not when the system
is sound. This tool replaces it with a bounded, re-runnable claim --

    With the production DRE and verifier trust registries empty, no public
    constructor, public function, CLI command, migration path, or closure path
    can produce a qualifying discharge, accepted rigorous authority,
    RH-equivalent frontier advancement, or a worker-declared proof.

-- and a report recording whether that claim currently holds. After this gate
passes, a later defect is a normal defect against a named invariant, with a
regression test, rather than a reason to reopen the architecture.

The report is a human artifact, not a drift-checked snapshot: it records the
head it ran against and is regenerated on every run, so CI does not compare it
against the committed copy.

Usage:
  python tools/phase1-final-gate.py            # run everything, write the report
  python tools/phase1-final-gate.py --quick    # skip the full suite (steps are subsets)
  python tools/phase1-final-gate.py --fast     # only the gates that finish in seconds
  python tools/phase1-final-gate.py --no-report  # run, write nothing
  python tools/phase1-final-gate.py --print    # render the report to stdout too
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

REVIEWS = REPO / "docs" / "reviews"
JSON_OUT = REVIEWS / "phase1-final-gate.json"
MARKDOWN_OUT = REVIEWS / "PHASE1_FINAL_CLOSURE.md"

#: The tag the frozen baseline was cut from. Never regenerated.
BASELINE_TAG = "rh-lab-v1-baseline-df7016d"

PYTEST_SUMMARY = re.compile(r"(\d+) passed")


@dataclass
class Step:
    """One gate. `why` is what a failure here would mean, not what it runs."""

    key: str
    title: str
    why: str
    argv: list[str]
    #: True when this step re-runs tests that step 3 already covered. Recorded
    #: so the report does not read as if the counts add up to a larger suite.
    subset_of_suite: bool = False
    #: Steps taking tens of seconds. Excluded from `--fast`, which exists so a
    #: git hook can run the gate without the author reaching for `--no-verify`
    #: out of impatience. A guard that is routinely bypassed guards nothing.
    slow: bool = False
    #: Filled in by `run`.
    passed: bool | None = None
    detail: str = ""
    tests: int | None = None
    output: str = field(default="", repr=False)


def _tool(name: str, *args: str) -> list[str]:
    return [sys.executable, str(REPO / "tools" / name), *args]


def _pytest(*args: str, parallel: bool = False) -> list[str]:
    """`parallel` only where the run is long enough to pay for the workers.

    xdist costs a few seconds of interpreter startup per worker, so the
    targeted `-k` steps below are faster serially; the full suite is not.
    """
    workers = ["-n", "auto"] if parallel else []
    return [sys.executable, "-m", "pytest", "-q", *workers, *args]


STEPS: list[Step] = [
    Step(
        "line-endings",
        "Line endings",
        "A line ending is part of a hash here: certificate hashes, DRE payload "
        "hashes, and formula-index digests are computed over exact bytes, so CRLF "
        "and LF copies of one artifact are two different models.",
        _tool("check-line-endings.py"),
    ),
    Step(
        "lint",
        "Ruff",
        "Style is not the point; an unused import or an undefined name is usually "
        "the visible end of a half-finished edit.",
        [sys.executable, "-m", "ruff", "check", "."],
    ),
    Step(
        "tests",
        "Full test suite",
        "Every behavioural and adversarial invariant in the repository.",
        # The gate re-runs the whole suite after the matrix jobs already have,
        # and it is the last thing in the workflow, so its wall clock is added
        # to every other job's rather than overlapped with it. Twelve minutes
        # of that was this one step running serially.
        _pytest(parallel=True),
        slow=True,
    ),
    Step(
        "schema-snapshot",
        "Schema surface unchanged",
        "The laboratory program migrates schemas and stored state, which is exactly "
        "where a field gets dropped or a vocabulary widened by accident.",
        _tool("snapshot-schemas.py", "--check"),
    ),
    Step(
        "status-inventory",
        "Status vocabularies unchanged",
        "A status added without a mapping entry is a status classified by accident "
        "of spelling.",
        _tool("inventory-status.py", "--check"),
    ),
    Step(
        "frozen-baseline",
        "Knowledge invariants unchanged since the frozen baseline",
        "The df7016d manifest was written before the program started and is never "
        "regenerated, so it is the one record an edit to durable memory cannot also "
        "update.",
        _tool("snapshot-baseline.py", "--since-baseline"),
    ),
    Step(
        "current-manifest",
        "Authoritative state unchanged",
        "Names the exact record and invariant that moved, rather than reporting that "
        "something did.",
        _tool("snapshot-baseline.py", "--check"),
    ),
    Step(
        "migration-manifest",
        "Knowledge relocation is consistent",
        "The relocation is hash-bound to a manifest listing all 42 IDs, every "
        "semantic hash, every dependency edge, and the four no-go records.",
        _tool("migrate-knowledge-path.py", "--check"),
    ),
    Step(
        "durable-memory",
        "Durable memory integrity",
        "Durable memory is the authoritative research record. If its seal, status "
        "vocabulary, or dependency graph is broken, nothing downstream can be trusted.",
        [sys.executable, "-m", "rh_research_engine.cli", "knowledge", "validate"],
    ),
    Step(
        "retired-path",
        "Deprecated-path AST scan",
        "A hardcoded pre-relocation path matches nothing after the move, so every "
        "recorded dead-end route reads as novel prior work.",
        _pytest("tests/test_source_containment.py", "-k", "retired"),
        subset_of_suite=True,
    ),
    Step(
        "legacy-enums",
        "Legacy-enum containment scan",
        "A deprecated vocabulary that keeps spreading into new modules is a second "
        "copy of the rules, free to drift from the canonical one.",
        _pytest("tests/test_source_containment.py", "-k", "legacy or deprecated"),
        subset_of_suite=True,
    ),
    Step(
        "spelling",
        "Status-substring semantic scan",
        "Classification by spelling promoted 14 of 21 knowledge statuses to rigorous, "
        "including one whose name says an external check has not happened.",
        _pytest("tests/test_source_containment.py", "-k", "spelling or detector"),
        subset_of_suite=True,
    ),
    Step(
        "attack-matrix",
        "Public API authority attack matrix",
        "Every public entry point, attacked the way a motivated user would attack it: "
        "construct the record that says what you want, label it convincingly, and hand "
        "it to the thing that reads labels.",
        _pytest("tests/test_public_api_attack_matrix.py"),
        subset_of_suite=True,
    ),
    Step(
        "empty-trust",
        "Empty-trust invariant",
        "The central Phase 1 acceptance claim. Includes the one registry combination "
        "that *does* discharge under a scoped test trust set, because it has to fail "
        "without it or the test set is not what is doing the work.",
        _pytest(
            "tests/test_public_api_attack_matrix.py",
            "tests/test_contracts.py",
            "-k",
            "no_public_composition or real_trust_registry or trust or activation",
        ),
        subset_of_suite=True,
    ),
    Step(
        "determinism",
        "Determinism smoke tests",
        "Same inputs, same model pack, same engine fingerprint -- same bytes. Checked "
        "across separate processes and PYTHONHASHSEED values, because set iteration "
        "order is seeded per interpreter and in-process tests all share one.",
        _pytest("tests/test_determinism.py", "tests/test_replay_identity.py"),
        subset_of_suite=True,
    ),
    Step(
        "worker-boundary",
        "Worker self-promotion",
        "The headline finding of the original review: a numerical run could be "
        "exported as evidence_class=proved, firing a pack rule that treats it as a "
        "theorem.",
        _pytest("-k", "worker"),
        subset_of_suite=True,
    ),
    Step(
        "missing-memory",
        "Missing-memory end-to-end",
        "An absent research record is not an empty one. Every authoritative command "
        "must exit nonzero, write no artifact, and mutate no state.",
        _pytest("tests/test_fail_closed_state.py"),
        subset_of_suite=True,
        slow=True,
    ),
]


def display_command(argv: list[str]) -> str:
    """The command as a reader would type it, not as subprocess received it.

    `sys.executable` is an absolute, machine-specific interpreter path, and a
    `-k` expression arrives as one argument whose spaces are invisible once
    joined -- so `-k legacy or deprecated` reads as three arguments that would
    not reproduce the run.
    """
    parts: list[str] = []
    for index, part in enumerate(argv):
        if index == 0 and part == sys.executable:
            parts.append("python")
            continue
        if part.endswith(".py") and Path(part).is_absolute():
            part = str(Path(part).relative_to(REPO)).replace("\\", "/")
        parts.append(f'"{part}"' if " " in part else part)
    return " ".join(parts)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run(step: Step) -> Step:
    result = subprocess.run(
        step.argv,
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    step.passed = result.returncode == 0
    step.output = (result.stdout + result.stderr).strip()
    lines = [line for line in step.output.splitlines() if line.strip()]
    step.detail = lines[-1].strip() if lines else "(no output)"
    match = PYTEST_SUMMARY.search(step.output)
    if match:
        step.tests = int(match.group(1))
    return step


def trust_registries() -> dict:
    from rh_research_engine.contracts import activation_status
    from rh_research_engine.mathcert import verifier_activation_status

    return {"dre": activation_status(), "verifier": verifier_activation_status()}


def contract_versions() -> dict:
    from rh_research_engine import __version__
    from rh_research_engine.contracts.artifacts import SCHEMA_VERSION
    from rh_research_engine.properties.models import PropertyGraph

    snapshot = json.loads((REPO / "docs" / "contracts" / "schema-snapshot.json").read_text("utf-8"))
    return {
        "package": __version__,
        "artifact_schema_version": SCHEMA_VERSION,
        "property_graph_schema_version": PropertyGraph().schema_version,
        "schema_snapshot_models": len(snapshot.get("models", {})),
        "schema_snapshot_enums": len(snapshot.get("enums", {})),
    }


def ci_context() -> dict:
    """What CI run produced this, when there is one.

    Recorded rather than inferred. A report claiming a CI run it cannot name is
    worth less than one that says plainly it was produced locally.
    """
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        return {"under_ci": False, "note": "generated locally; no CI run to cite"}
    return {
        "under_ci": True,
        "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        "run_id": run_id,
        "run_number": os.environ.get("GITHUB_RUN_NUMBER", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "job": os.environ.get("GITHUB_JOB", ""),
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
    }


#: Integrations that are implemented at the contract boundary and deliberately
#: not connected. Listed so "it does not work" is distinguishable from "it is
#: switched off on purpose", which is the difference between a bug and a policy.
INERT_INTEGRATIONS = [
    {
        "name": "DRE obligation discharge",
        "state": "implemented, trust registry empty",
        "effect": "No obligation can be discharged, so no RH-equivalent premise "
        "can gain frontier credit through a discharge.",
    },
    {
        "name": "Arb/FLINT rigorous verification",
        "state": "adapter present, no backend detected",
        "effect": "Every envelope reports unknown. A caller-supplied "
        "status=accepted maps to UNKNOWN rather than to RIGOROUS_NUMERICAL.",
    },
]

ACTIVATION_REQUIREMENTS = [
    "Authenticated receipt verification: a signature checked against a key held "
    "outside this repository, retrieval from a sealed DRE decision store addressed "
    "by dre_decision_hash, or deterministic replay reproducing dre_proof_hash.",
    "A pinned model-pack hash and a pinned engine fingerprint.",
    "Proof artifact retrieval or replay, not merely a reference to one.",
    "A documented trust-root rotation procedure.",
    "A documented revocation procedure.",
    "Negative tests for stale signatures and stale receipts.",
    "Adding a fingerprint to the set literal is necessary and NOT sufficient.",
]

OPEN_LIMITATIONS = [
    "receipt_authentication records which mechanism established a receipt, and "
    "NONE never verifies. Declaring one of the other three is a claim about "
    "provenance, not the check itself -- the checker is the activation work above. "
    "Until it exists the empty trust registry, not this field, is what makes "
    "discharge authority inert.",
    "No mathematical engine is connected. Phase 1 establishes what the records are; "
    "the workers that produce them are later workstreams, and an empty slot is "
    "preferable to a placeholder that emits artifacts which look real.",
    "Independence groups are derived from the verifier family and version. Two "
    "genuinely distinct implementations sharing a family name would be counted as "
    "one witness, which is the safe direction but not a precise one.",
    "The closure report records the head it ran against; committing it necessarily "
    "advances that head by one commit.",
    "Branch protection is not configured and will not be. GitHub gates both the "
    "protection and rulesets APIs behind a paid plan for private repositories, "
    "and this repository stays private because it holds unpublished research; "
    "publishing it to obtain a merge button would trade priority on the work for "
    "a CI feature. `tools/githooks/pre-push` compensates by refusing a push to "
    "main when the fast gate fails, but it is strictly weaker: one machine, "
    "skippable with --no-verify, blind to the other platforms. Every gate still "
    "runs on every push and pull request, so nothing is unverified -- only "
    "unblocked.",
    "No COUNTERFACTUAL closure mode was added. EXPLORATORY relaxes rigour and "
    "deliberately does not relax circularity, so there is currently no way to trace "
    "the consequences of an undischarged RH-equivalent premise at all. That is the "
    "safe direction; a quarantined non-promotable mode is Phase 2 work if the "
    "capability is ever wanted.",
]


def build_report(steps: list[Step], *, quick: bool) -> dict:
    failed = [step.key for step in steps if step.passed is False]
    suite = next((s for s in steps if s.key == "tests"), None)
    return {
        "report": "phase1-final-gate",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_sha": _git("rev-parse", "HEAD"),
        "head_short": _git("rev-parse", "--short", "HEAD"),
        "working_tree_clean": _git("status", "--porcelain") == "",
        "baseline_tag": BASELINE_TAG,
        "baseline_sha": _git("rev-parse", BASELINE_TAG),
        "quick_mode": quick,
        "test_count": suite.tests if suite else None,
        "contract_versions": contract_versions(),
        "trust_registries": trust_registries(),
        "ci": ci_context(),
        "gates": [
            {
                "key": step.key,
                "title": step.title,
                "why": step.why,
                "command": display_command(step.argv),
                "passed": step.passed,
                "detail": step.detail,
                "tests": step.tests,
                "subset_of_suite": step.subset_of_suite,
                "skipped": step.passed is None,
            }
            for step in steps
        ],
        "inert_integrations": INERT_INTEGRATIONS,
        "activation_requirements": ACTIVATION_REQUIREMENTS,
        "open_limitations": OPEN_LIMITATIONS,
        "failed_gates": failed,
        "verdict": "BLOCKED" if failed else "READY TO MERGE",
    }


def render_markdown(report: dict) -> str:
    def status(gate: dict) -> str:
        if gate["skipped"]:
            return "skipped"
        return "pass" if gate["passed"] else "**FAIL**"

    lines = [
        "# Phase 1 Final Closure",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
        "Generated by `python tools/phase1-final-gate.py`. Regenerated on every run,",
        "so this file records the head it ran against rather than being compared",
        "against a committed copy.",
        "",
        "## Identity",
        "",
        "| | |",
        "|---|---|",
        f"| Branch | `{report['branch']}` |",
        f"| Head | `{report['head_sha']}` |",
        f"| Working tree | {'clean' if report['working_tree_clean'] else 'dirty'} |",
        f"| Frozen baseline | `{report['baseline_tag']}` (`{report['baseline_sha'][:12]}`) |",
        f"| Tests | {report['test_count']} passed |",
        f"| Package | {report['contract_versions']['package']} |",
        f"| Artifact schema | v{report['contract_versions']['artifact_schema_version']} |",
        f"| Property graph schema | v{report['contract_versions']['property_graph_schema_version']} |",
        f"| Schema surface | {report['contract_versions']['schema_snapshot_models']} models, "
        f"{report['contract_versions']['schema_snapshot_enums']} enums |",
    ]
    ci = report["ci"]
    if ci["under_ci"]:
        lines.append(
            f"| CI run | {ci['workflow']} #{ci['run_number']} "
            f"(run {ci['run_id']}, attempt {ci['run_attempt']}) |"
        )
    else:
        lines.append(f"| CI run | {ci['note']} |")

    lines += [
        "",
        "## The acceptance invariant",
        "",
        "> With the production DRE and verifier trust registries empty, no public",
        "> constructor, public function, CLI command, migration path, or closure path",
        "> can produce a qualifying discharge, accepted rigorous authority,",
        "> RH-equivalent frontier advancement, or a worker-declared proof.",
        "",
        "## Trust registries",
        "",
        "| Registry | Active | Entries |",
        "|---|---|---|",
        f"| DRE discharge authority | "
        f"{report['trust_registries']['dre']['discharge_authority_active']} | "
        f"{report['trust_registries']['dre']['trusted_engine_count']} |",
        f"| Rigorous verifier adapters | "
        f"{report['trust_registries']['verifier']['rigorous_verification_active']} | "
        f"{report['trust_registries']['verifier']['registered_adapter_count']} |",
        f"| Receipt signing keys | "
        f"{bool(report['trust_registries']['dre']['registered_signing_key_count'])} | "
        f"{report['trust_registries']['dre']['registered_signing_key_count']} |",
        "",
        "All three are module-private. No public function accepts a trust set, and",
        "each reports a count rather than its contents, so reading the status grants",
        "nothing. A receipt has to pass every applicable gate, which is why the",
        "signature backend could be built without changing the posture: it is",
        "available, and there is nothing registered for it to verify against.",
        "",
        f"Signature backend available: "
        f"`{report['trust_registries']['dre']['signature_backend_available']}` — a "
        "fact about the environment, reported honestly, granting nothing on its own.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    for gate in report["gates"]:
        detail = gate["detail"].replace("|", "\\|")[:160]
        lines.append(f"| {gate['title']} | {status(gate)} | {detail} |")

    lines += ["", "### What each gate is for", ""]
    for gate in report["gates"]:
        note = " *(subset of the full suite)*" if gate["subset_of_suite"] else ""
        lines += [f"**{gate['title']}**{note} — `{gate['command']}`", "", gate["why"], ""]

    lines += ["## Intentionally inert integrations", ""]
    for item in report["inert_integrations"]:
        lines += [f"**{item['name']}** — {item['state']}", "", item["effect"], ""]

    lines += [
        "## Activation requirements",
        "",
        "Before the first trusted DRE fingerprint is added:",
        "",
    ]
    lines += [f"{index}. {text}" for index, text in enumerate(report["activation_requirements"], 1)]

    lines += ["", "## Open limitations", ""]
    lines += [f"- {text}" for text in report["open_limitations"]]

    if report["failed_gates"]:
        lines += [
            "",
            "## Failing gates",
            "",
            *[f"- `{key}`" for key in report["failed_gates"]],
        ]

    lines += [
        "",
        "## After merge",
        "",
        "The committed gate is the durable review boundary. A later finding is",
        "classified against the invariant it violates, gets a regression test, and is",
        "fixed as a normal defect -- not as a reason to reopen the architecture review.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the full suite; the remaining test steps are subsets of it.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Only the gates that finish in seconds. For git hooks.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Write no files. A hook must not leave the tree dirty after a push.",
    )
    parser.add_argument("--print", dest="show", action="store_true")
    args = parser.parse_args(argv)

    steps = [s for s in STEPS if not (args.quick and s.key == "tests")]
    if args.fast:
        steps = [s for s in steps if not s.slow]

    width = max(len(step.title) for step in steps)
    for step in steps:
        print(f"{step.title:<{width}}  ... ", end="", flush=True)
        run(step)
        print("ok" if step.passed else "FAIL")
        if not step.passed:
            print(step.output[-4000:], file=sys.stderr)

    report = build_report(STEPS, quick=args.quick)
    markdown = render_markdown(report)

    if not args.no_report:
        REVIEWS.mkdir(parents=True, exist_ok=True)
        JSON_OUT.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="",
        )
        MARKDOWN_OUT.write_text(markdown, encoding="utf-8", newline="")

    if args.show:
        print()
        print(markdown)

    print()
    if not args.no_report:
        print(f"wrote {JSON_OUT.relative_to(REPO)}")
        print(f"wrote {MARKDOWN_OUT.relative_to(REPO)}")
    print(f"verdict: {report['verdict']}")
    return 1 if report["failed_gates"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
