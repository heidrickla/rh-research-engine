"""Every knob an experiment has is either reachable or refused on purpose.

THE COMMANDS ARE COPIES. Seventeen of the eighteen experiment commands in
`cli.py` are the same three lines -- call `run`, append to the store, emit -- with
a hand-written signature restating the module's own. Nothing forces those two
signatures to agree, so a parameter added to `run` is unreachable until somebody
remembers to type it again in `cli.py`, and forgetting looks exactly like
deciding.

IT HAD ALREADY HAPPENED. `weil_positivity.tolerance` and
`weil_certified.eval_limit` were both unreachable and undocumented, and both
turn out to be parameters that SHOULD stay internal -- `tolerance` is the gate
separating REFUTED from UNRESOLVED, and a command-line flag that loosens a
refutation threshold is the "loosen a tolerance" non-fix this repository is
written against; `eval_limit` is tied to the working precision by measurement,
so setting it independently invites a run whose budget and precision disagree.
Right outcome, reached by nobody deciding it.

SO THE RULE IS THAT ABSENCE MUST BE DECLARED. Every defaulted parameter of an
experiment's `run` is either passed in the command's call, or named in the
command's docstring with a reason it is not. That is the same shape as
`Assumption.fails_outside` being required and the pattern ledger's `retired`,
`range` and `columns` having no defaults: a decision has to be written down, not
achieved by leaving something out.

The docstring is the right place for the reason rather than a registry, because
it is what `--help` prints and where somebody looks when asking why a flag they
expected is missing.

MUTATIONS WATCHED FAILING: an unanswerable git skipping every module, the
tracked filter skipping every module, the predicate accepting every parameter,
and the signature loop examining none. Weakening `judged >= 15` alone is NOT
caught, and that is correct rather than a hole -- on its own it produces no wrong
answer, it only removes the protection that catches the second mutation. A floor
is defence in depth, and defence in depth cannot be detected by breaking it while
nothing else is broken.

TWO OF THIS FILE'S OWN DEFECTS ARE WHY IT LOOKS LIKE THIS. It first floored the
tracked LISTING rather than the count actually judged, and `if True: continue`
sailed through with the listing full and the loop empty -- a count of inputs is
not a count of inspections. And its positive control first re-implemented the
predicate inline, so it asserted nothing about the real check; both callers now
share `_undeclared`.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib

import pytest

#: Data-injection seams. These exist so tests can hand in ordinates instead of
#: recomputing them, and they take arrays a command line cannot express.
INJECTION_SEAMS = frozenset({"ordinates", "zeros"})

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ROOT / "src" / "rh_research_engine" / "cli.py"


def _tracked_experiments() -> frozenset[str]:
    """Experiment modules the REPOSITORY has, not the ones this disk has.

    `cli.py` can carry a command for a module another session has written and
    not yet committed, and judging that module makes this gate's result depend
    on whose working tree it runs in -- which is the "a gate that reads the
    environment is a gate about the environment" failure, and it happened: this
    file passed while a peer's uncommitted command was stashed and failed the
    moment it came back.

    Returns an empty set when git cannot answer, which the floor test below
    turns into a failure rather than a silent full skip.

    THE SCOPE IS THE INDEX, NOT HEAD, AND THAT IS ONE STAGING EARLIER THAN IT
    SOUNDS. `git ls-files` reads the index, so a file that has been `git add`ed
    and never committed IS judged -- verified by staging a throwaway experiment
    and watching it appear. A commit message in this repository claims the check
    on new work "arrives one commit late"; that is wrong, and it is corrected
    here rather than there because this is where the next reader looks.

    The practical consequence is an ordering, not a limitation: stage, then run
    the suite, then commit. `fit_bias_lab` failed in CI only because the suite
    was run before `git add`, when the file was genuinely invisible.
    """
    import subprocess

    try:
        listing = subprocess.run(
            ["git", "ls-files", "src/rh_research_engine/experiments"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git here
        return frozenset()
    if listing.returncode != 0:  # pragma: no cover - not a checkout
        return frozenset()
    return frozenset(
        pathlib.PurePosixPath(line).stem
        for line in listing.stdout.splitlines()
        if line.endswith(".py")
    )


def _experiment_commands():
    """Each `experiment_app` command paired with the `run` call inside it."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any("experiment_app.command" in ast.unparse(d) for d in node.decorator_list):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "run"
                and isinstance(call.func.value, ast.Name)
            ):
                yield node, call.func.value.id, call
                break


def _undeclared(command, module, call) -> list[str]:
    """The check itself, so the positive control below exercises THIS code.

    An earlier version of that control re-implemented this loop inline and
    therefore asserted nothing about the real check -- the same defect this
    project already fixed once in a Lemma 10 refusal test. Mutating the
    predicate must break both callers, and it does.
    """
    wired = {keyword.arg for keyword in call.keywords if keyword.arg}
    doc = ast.get_docstring(command) or ""
    missed = []
    for name, parameter in inspect.signature(module.run).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue
        if name in INJECTION_SEAMS or name in wired or name in doc:
            continue
        missed.append(name)
    return missed


def test_the_commands_are_found_at_all():
    """An enumerating check needs a floor, or a rename makes it vacuous.

    This file's whole content is a loop over commands. If the AST walk stops
    matching -- the decorator renamed, the app renamed, the file moved -- every
    assertion below passes over an empty sequence and reports agreement.
    """
    found = list(_experiment_commands())
    assert len(found) >= 15, f"only {len(found)} experiment commands found"


def test_every_run_parameter_is_wired_or_declared():
    """Unreachable and refused-on-purpose must not look the same.

    A parameter absent from the command because somebody forgot, and one absent
    because exposing it would let a caller loosen a gate, are different facts.
    Only the second is allowed here, and only if it says so.
    """
    tracked = _tracked_experiments()
    undeclared = []
    judged = 0
    for command, module_name, call in _experiment_commands():
        if module_name not in tracked:
            # Another session's uncommitted experiment. Not the repository's yet,
            # so not this gate's business -- see `_tracked_experiments`.
            continue
        try:
            module = importlib.import_module(f"rh_research_engine.experiments.{module_name}")
        except ModuleNotFoundError:  # pragma: no cover - a module not in this checkout
            continue
        judged += 1
        undeclared += [
            f"{module_name}.{name} (command {command.name})"
            for name in _undeclared(command, module, call)
        ]
    # THE FLOOR GOES ON WHAT WAS JUDGED, not on what was listed. An earlier
    # version floored `tracked` instead, and a mutation replacing the skip with
    # `if True: continue` sailed through: the listing stayed full while the loop
    # examined nothing. A count of inputs is not a count of inspections.
    assert judged >= 15, f"only {judged} commands actually judged"
    assert not undeclared, (
        "these run() parameters are neither passed by the CLI nor named in the "
        "command's docstring, so nobody can tell whether they are refused on "
        "purpose: " + ", ".join(undeclared)
    )


def test_a_forgotten_parameter_would_be_caught():
    """Watch the gate fail, through the same predicate the real check uses.

    A command shaped exactly like the seventeen real ones, missing `tolerance`
    from both its call and its docstring. If `_undeclared` ever stops flagging,
    this fails and so does the sweep -- which is the point of sharing it.
    """
    source = '''
@experiment_app.command("made-up")
def exp_made_up(size: int = 8) -> None:
    """No mention of the other knob."""
    result = weil_positivity.run(size=size)
    _store().append_experiment(result)
    _emit(result)
'''
    command = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef))
    call = next(
        n for n in ast.walk(command)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
    )
    from rh_research_engine.experiments import weil_positivity

    missed = _undeclared(command, weil_positivity, call)
    assert "tolerance" in missed, missed
    assert "prime_limit" in missed, missed
    assert "size" not in missed, "size IS wired and must not be flagged"


@pytest.mark.parametrize(
    ("command_name", "parameter"),
    [
        ("weil-positivity", "tolerance"),
        ("weil-certified", "eval_limit"),
        ("exponent-scan", "signal"),
    ],
)
def test_the_deliberate_refusals_say_why(command_name, parameter):
    """A named parameter with no reason beside it is not a decision either.

    Naming it satisfies the check above, so this asks for the second half: the
    docstring has to explain the refusal, or the rule degrades into a list of
    words that silence a test.
    """
    command = next(
        c for c, _, _ in _experiment_commands() if c.decorator_list and command_name in
        ast.unparse(c.decorator_list[0])
    )
    doc = ast.get_docstring(command) or ""
    assert parameter in doc
    assert "DELIBERATELY NOT AN OPTION" in doc, command_name
    # A reason is prose, not a label. The shortest of the three runs to ~40 words.
    reason = doc.split("DELIBERATELY NOT AN OPTION", 1)[1]
    assert len(reason.split()) >= 25, f"{command_name}: reason is too thin to be one"


def test_the_tracked_listing_is_not_empty():
    """A floor on the skip, or an unanswerable git makes the gate above vacuous.

    `_tracked_experiments` returns an empty set when git cannot be reached, and
    an empty set skips every command -- so the check would pass by examining
    nothing, which is how `formula-guard`'s name-policy check once passed over
    zero modules.
    """
    tracked = _tracked_experiments()
    assert len(tracked) >= 15, f"only {len(tracked)} tracked experiment modules"
    assert "weil_positivity" in tracked
