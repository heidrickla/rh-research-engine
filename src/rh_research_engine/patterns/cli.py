"""`rhre patterns` -- audit a premise, scan measurements for exact structure."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .corpus_sweep import CorpusColumns, build_observations
from .detect import (
    audit_premise,
    escalate,
    scan_columns,
    scan_for_regularities,
    scan_universe,
)
from .ledger import OpenLedger, RevisitVerdict
from .models import Observation, PremiseVerdict, RegularityKind
from .noise import NoiseGround, NoiseRegistry, NoiseRule, Retirement

app = typer.Typer(no_args_is_help=True)
console = Console()

#: Where the triage accumulates.
NOISE_PATH = Path("research_state/pattern_noise.json")
#: Where the questions nobody has answered accumulate.
OPEN_PATH = Path("research_state/pattern_open.json")

#: How a revisit reads. Refutations are the loud ones -- an entry that died is
#: the result the ledger exists to be able to deliver. RETIRED is not one of
#: them and is not coloured like one: the relation holds, and somebody said
#: what it is.
_VERDICT_COLOUR = {
    RevisitVerdict.EXTENDED: "green",
    RevisitVerdict.CONTRADICTED: "red",
    RevisitVerdict.BROKEN: "red",
    RevisitVerdict.INCOMPARABLE: "dim",
    RevisitVerdict.RETIRED: "yellow",
}


@app.command("audit")
def audit(
    quantity: str = typer.Argument(..., help="Name of the quantity being fitted."),
    values: str = typer.Option(..., "--values", help="Comma-separated measurements."),
    target: str | None = typer.Option(
        None, "--target", help="The value the task expects, if it expects one."
    ),
) -> None:
    """Ask whether a quantity has anything in it to fit, before fitting it."""
    result = audit_premise(
        quantity, [item for item in values.split(",") if item.strip()], target=target
    )
    colour = {
        PremiseVerdict.LIVE: "green",
        PremiseVerdict.EMPTY: "yellow",
        PremiseVerdict.DEGENERATE: "yellow",
        PremiseVerdict.INCONCLUSIVE: "dim",
    }[result.verdict]
    console.print(f"[{colour}]{result.verdict.value.upper()}[/]  {result.quantity}")
    console.print(f"  {result.evidence}")
    console.print(f"  sampled {result.sampled}; confidence {result.confidence.value}")
    if result.verdict in (PremiseVerdict.EMPTY, PremiseVerdict.DEGENERATE):
        console.print(
            "\n[yellow]Nothing to fit here.[/] That is a result about the task, "
            "not a failure to produce one -- report it rather than fitting "
            "anyway."
        )


@app.command("scan")
def scan(
    observations: Path = typer.Argument(
        ..., help='JSON list of {"name", "values", "requested"} objects.'
    ),
    all_findings: bool = typer.Option(
        False, "--all", help="Include relations that do not hold in every case."
    ),
    ledger: bool = typer.Option(
        True,
        "--ledger/--no-ledger",
        help="Judge open questions against this scan, and file new ones.",
    ),
) -> None:
    """Scan every column and every pair for an exact relation.

    Not only the quantity the task named: the point of the scan is the column
    nobody asked for.
    """
    try:
        payload = json.loads(observations.read_text(encoding="utf-8"))
        columns = [Observation.model_validate(entry) for entry in payload]
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        console.print(f"[red]cannot read observations:[/] {exc}")
        raise typer.Exit(1) from exc

    findings = scan_for_regularities(columns)
    registry = NoiseRegistry.load(NOISE_PATH)
    suppression = registry.apply(findings)
    shown = suppression.kept if all_findings else escalate(suppression.kept)

    if ledger:
        _keep_ledger(
            suppression.kept,
            scan_universe(columns),
            scan_columns(columns),
            suppression.retired,
        )

    # Always say how much was filtered and why. A filter that cannot report
    # what it hid is one that will eventually hide the discovery.
    if suppression.removed_total:
        console.print(
            f"[dim]{suppression.removed_total} finding(s) retired by "
            f"{len(registry.rules)} noise rule(s):[/]"
        )
        for reason, count in sorted(suppression.removed.items()):
            console.print(f"  [dim]{count} x {reason}[/]")
        console.print()

    if not shown:
        console.print(
            f"nothing above the surprise floor across {len(columns)} column(s)."
        )
        return

    table = Table(title=f"{len(shown)} regularity/regularities")
    table.add_column("surprise", justify="right")
    table.add_column("kind")
    table.add_column("statement")
    table.add_column("tight at")
    table.add_column("from")
    for finding in shown:
        witnesses = ", ".join(finding.witnesses[:10])
        if len(finding.witnesses) > 10:
            witnesses += ", ..."
        also = "".join(
            f"\n(also carried as {', '.join(others)})"
            for _, others in sorted(finding.aliases.items())
        )
        table.add_row(
            str(finding.surprise),
            finding.kind.value,
            finding.statement + also,
            witnesses or "-",
            "unrequested" if finding.from_unrequested else "asked for",
        )
    console.print(table)
    console.print(
        "\n[yellow]Conjectures, not results.[/] A relation holding in every "
        "sampled case is a reason to look for a proof, and is not one -- these "
        "records cannot be filed at a rigorous confidence."
    )


def _keep_ledger(
    findings: list,
    universe: list[str],
    measured: list[str],
    retired: Sequence[Retirement],
) -> None:
    """Judge the open questions against this scan, then file the new ones.

    Judged BEFORE filing, so an entry created by this scan is not immediately
    asked what this scan says about it -- the answer would be INCOMPARABLE,
    since a range does not strictly contain itself, and a ledger whose every
    fresh entry arrives carrying a non-verdict is one nobody reads.

    Filed from what the noise registry KEPT. A finding already retired with a
    reason is not an open question, and re-filing it here would undo the triage
    through a second door.

    Judged with what it RETIRED as well, and this is not symmetric with the
    line above. A retired finding is absent from `findings`, and absent is the
    BROKEN verdict -- so without this the act of explaining a relation would
    write "was a regularity of the narrower range" into the record about a
    relation that is untouched and true. Explained and refuted are opposite
    outcomes, and they shared a verdict.
    """
    book = OpenLedger.load(OPEN_PATH)
    judged = book.revisit(findings, universe, measured, retired)
    added = book.record(findings, universe)
    if not judged and not added:
        return
    book.save(OPEN_PATH)

    for entry, outcome in judged:
        colour = _VERDICT_COLOUR[outcome.verdict]
        console.print(f"[{colour}]{outcome.verdict.value.upper()}[/]  {entry.statement}")
        console.print(f"  [dim]{outcome.note}[/]")
    if added:
        console.print(
            f"[cyan]{len(added)} new open question(s)[/] filed in {OPEN_PATH}"
        )
    if judged or added:
        console.print()


#: Places carried for irrational columns, and the doubled precision the whole
#: scan is repeated at.
#:
#: A column of logs has to be a decimal, and two decimals agreeing to thirty
#: places are reported by the scan as EXACTLY equal -- a different claim from
#: the one the data supports. `zeta(n)` is the concrete case: it rounds to
#: 1.000...0 for n past about a hundred, so `zeta >= mu` looked tight wherever
#: mu(n) = 1 and the ledger filed it as an open question about the squarefree
#: numbers. Only findings present at both precisions are believed.
SWEEP_DIGITS = 30


@app.command("sweep")
def sweep(
    top: int = typer.Option(1000, "--top", help="Sweep n = 2..top-1."),
    zeros: bool = typer.Option(
        True, "--zeros/--no-zeros", help="Include columns built from zeta's zeros."
    ),
    ledger: bool = typer.Option(
        True, "--ledger/--no-ledger", help="Judge and file open questions."
    ),
) -> None:
    """Point the scan at the corpus's own quantities and see what it finds.

    Not a table anyone assembled for it: every callable the indexed corpus uses,
    derived arithmetic nobody asked for, both sides of every inequality the
    corpus asserts, and the zeros.

    WHAT `--top` COSTS, measured because nobody had written it down and it is
    not guessable from the default:

        --top   1,000      21 s        (the default)
        --top   2,000      39 s
        --top   4,000      89 s
        --top  20,000      51 min, peak 2.0 GB

    Roughly linear to about 4,000 and superlinear beyond it. THE THREE SMALL
    POINTS DO NOT PREDICT THE LARGE ONE: their log-log slope is 1.03, which
    extrapolates to 7.5 minutes at 20,000 and is wrong by a factor of seven.
    Three doublings cannot show that an exponent is not constant, so a fourth
    was estimated from a trend that had not been established over the range it
    was read at -- which is the failure this repository keeps finding in its own
    numbers, committed here while measuring a cost before spending it.

    Budget from the table, not from the slope.
    """
    columns = CorpusColumns(top)
    console.print(f"[dim]shortcuts verified at {', '.join(columns.verify_shortcuts())}[/]")
    if zeros:
        columns.load_zeros()
        console.print(f"[dim]{len(columns.ordinates)} zero ordinates located[/]")

    observations = build_observations(columns, SWEEP_DIGITS)
    findings = scan_for_regularities(observations)

    # The same scan at double the precision; only what survives is believed.
    doubled = {
        (f.kind.value, tuple(sorted(f.columns)), f.support)
        for f in scan_for_regularities(build_observations(columns, SWEEP_DIGITS * 2))
    }
    stable, lost = [], []
    for finding in findings:
        key = (finding.kind.value, tuple(sorted(finding.columns)), finding.support)
        (stable if key in doubled else lost).append(finding)
    if lost:
        console.print(
            f"[yellow]{len(lost)} finding(s) did not survive doubling the "
            "precision[/] — an artifact of the rounding, not the mathematics:"
        )
        for finding in lost:
            console.print(f"  [dim]{finding.statement}[/]")

    registry = NoiseRegistry.load(NOISE_PATH)
    suppression = registry.apply(stable)
    if suppression.removed_total:
        console.print(
            f"[dim]{suppression.removed_total} retired by "
            f"{len(registry.rules)} noise rule(s)[/]"
        )

    if ledger:
        _keep_ledger(
            suppression.kept,
            scan_universe(observations),
            scan_columns(observations),
            suppression.retired,
        )

    shown = escalate(suppression.kept)
    console.print(
        f"\n[bold]{len(shown)}[/] finding(s) above the surprise floor, from "
        f"{len(observations)} columns over {len(columns.rows)} rows\n"
    )
    for finding in shown:
        origin = "unrequested" if finding.from_unrequested else "corpus"
        console.print(f"[{finding.surprise:6d}] ({origin}) {finding.statement}")
        if finding.aliases:
            for name, others in sorted(finding.aliases.items()):
                console.print(
                    f"         [dim]{name} also carried as "
                    f"{', '.join(others)} -- the same values on every row, so "
                    "this is one relation and not several[/]"
                )
        if finding.character:
            console.print(f"         [cyan]{finding.character}[/]")
        if finding.witnesses:
            listed = ", ".join(finding.witnesses[:12])
            if len(finding.witnesses) > 12:
                listed += f", ... ({len(finding.witnesses)} total)"
            console.print(f"         [dim]tight at: {listed}[/]")
    console.print(
        "\n[yellow]Conjectures, not results.[/] A relation holding in every "
        "sampled case is a reason to look for a proof, and is not one."
    )


@app.command("open")
def show_open(
    closed: bool = typer.Option(
        False, "--closed", help="Include entries a wider range has settled."
    ),
) -> None:
    """The questions nothing has answered yet.

    A relation tight on a set no predicate names. Ranked by the widest range it
    has survived, because that is the only thing that makes one of these
    stronger than another.
    """
    book = OpenLedger.load(OPEN_PATH)
    entries = [e for e in book.entries if closed or e.open]
    if not entries:
        console.print(
            "nothing open."
            if book.entries
            else f"no ledger yet at {OPEN_PATH} -- run `rhre patterns scan`."
        )
        return

    entries.sort(key=lambda entry: (-entry.widest, -entry.surprise))
    # Counted, not assumed. With `--closed` the table carries settled entries
    # too, and calling all of them open would report three explained artifacts
    # as questions nobody has answered -- a headline contradicting the column
    # beside it.
    still_open = sum(1 for entry in entries if entry.open)
    settled = len(entries) - still_open
    table = Table(
        title=f"{still_open} open finding(s)"
        + (f", {settled} settled" if settled else "")
    )
    table.add_column("widest", justify="right")
    table.add_column("statement")
    table.add_column("tight on")
    table.add_column("revisits", justify="right")
    table.add_column("latest")
    for entry in entries:
        witnesses = ", ".join(entry.witnesses[:8])
        if len(entry.witnesses) > 8:
            witnesses += ", ..."
        verdict = entry.verdict
        table.add_row(
            str(entry.widest),
            entry.statement,
            witnesses or "-",
            str(len(entry.history)),
            f"[{_VERDICT_COLOUR[verdict]}]{verdict.value}[/]" if verdict else "-",
        )
    console.print(table)
    console.print(
        "\n[yellow]Open, not promising.[/] A relation nothing names is a place "
        "to look; widening the range can refute one of these and cannot "
        "establish it."
    )


@app.command("dismiss")
def dismiss(
    columns: str = typer.Option(..., "--columns", help="Comma-separated column names."),
    reason: str = typer.Option(..., "--reason", help="WHY this is noise."),
    ground: NoiseGround = typer.Option(
        NoiseGround.TRIAGED, "--ground", help="How it was established."
    ),
    kind: RegularityKind | None = typer.Option(
        None, "--kind", help="Restrict to one kind; omit to retire any over those columns."
    ),
) -> None:
    """Retire a finding that has been verified as noise.

    A reason is required. "Noise" without one is indistinguishable from a
    result somebody found inconvenient, and the registry is meant to be
    re-readable by whoever inherits it.
    """
    names = [item.strip() for item in columns.split(",") if item.strip()]
    if not names:
        console.print("[red]--columns named nothing.[/]")
        raise typer.Exit(1)

    registry = NoiseRegistry.load(NOISE_PATH)
    rule = NoiseRule(columns=names, reason=reason, ground=ground, kind=kind)
    if not registry.add(rule):
        console.print("[yellow]already retired[/] — no change.")
        return
    registry.save(NOISE_PATH)
    console.print(
        f"retired [bold]{' + '.join(names)}[/] ({kind.value if kind else 'any kind'})"
    )
    console.print(f"  ground: {ground.value}")
    console.print(f"  reason: {reason}")
    console.print(f"\n{len(registry.rules)} rule(s) in {NOISE_PATH}")


@app.command("noise")
def show_noise() -> None:
    """List what has been retired, and why."""
    registry = NoiseRegistry.load(NOISE_PATH)
    if not registry.rules:
        console.print("nothing retired yet.")
        return
    table = Table(title=f"{len(registry.rules)} noise rule(s)")
    table.add_column("columns")
    table.add_column("kind")
    table.add_column("ground")
    table.add_column("reason")
    for rule in registry.rules:
        table.add_row(
            " + ".join(rule.columns),
            rule.kind.value if rule.kind else "any",
            rule.ground.value,
            rule.reason,
        )
    console.print(table)
