from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .contracts.epistemic import Confidence
from .contracts.lifecycle import HypothesisLifecycle
from .core.bootstrap import seed_claims
from .core.bounds import correlation_remainder_to_theta
from .core.knowledge import (
    CANONICAL_KNOWLEDGE_PATH,
    NO_GO_STATUSES,
    KnowledgeBase,
    KnowledgeIntegrityError,
)
from .core.nogo import explain as explain_nogo
from .core.promote import evaluate_export
from .core.scoring import score_claim
from .core.store import ResearchStore
from .dre import ClaimEffect, DreEvidenceEnvelope, WorkerClassError, write_dre_experiment
from .experiments import (
    adversarial_synthetic,
    arc_diagnostics,
    baez_duarte,
    bombieri_finite_model,
    correlation_lab,
    correlation_scan,
    counterterm_discovery,
    exponent_scan,
    fit_bias_lab,
    gamma_filter,
    height_recovery_lab,
    li_criterion,
    local_variance,
    safe_binomial,
    synthetic_zero,
    weil_certified,
    weil_positivity,
    weil_sensitivity,
)
from .patterns.cli import app as patterns_app
from .properties.cli import app as properties_app
from .supervisor import FalsificationTest, Hypothesis, ProofGap
from .supervisor.store import HypothesisQueueStore

app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
dre_app = typer.Typer(no_args_is_help=True)
knowledge_app = typer.Typer(no_args_is_help=True)
symbolic_app = typer.Typer(no_args_is_help=True)
supervisor_app = typer.Typer(no_args_is_help=True)
verifier_app = typer.Typer(no_args_is_help=True)
app.add_typer(experiment_app, name="experiment")
app.add_typer(dre_app, name="dre")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(symbolic_app, name="symbolic")
app.add_typer(supervisor_app, name="supervisor")
app.add_typer(verifier_app, name="verifier")
# The properties sub-app shipped unregistered, so `rhre properties ...` did not
# exist and the subsystem was reachable only by importing it.
app.add_typer(properties_app, name="properties")
app.add_typer(patterns_app, name="patterns")
console = Console()


def _emit(result) -> None:
    """Print a record WITHOUT rich markup interpretation.

    `console.print` reads `[...]` as style markup and silently deletes it, so a
    record citing assumption `[digamma-tail]` printed as ` |Re psi...` with the
    citation gone. The stored JSON was correct throughout -- only the display
    lost it, which is the trap this repository keeps meeting from the other
    side: a reader who trusted the terminal would have "fixed" a record that was
    never wrong. Anything bracketed in any field was affected, not just these.
    """
    console.print_json(result.model_dump_json())


def _store() -> ResearchStore:
    return ResearchStore(Path("research_state"))


def _knowledge_base() -> KnowledgeBase:
    """Resolve durable memory, turning a missing record into a clean exit.

    Construction is inside the guard, not outside it. `KnowledgeBase()` resolves
    the canonical path eagerly and raises when nothing is there, so wrapping
    only the `load()` call left the commonest failure -- no research record at
    all -- surfacing as an unhandled traceback.
    """
    try:
        return KnowledgeBase()
    except KnowledgeIntegrityError as exc:
        console.print(f"[red]durable memory unavailable:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _knowledge_items(kb: KnowledgeBase):
    """Load durable memory, turning an integrity failure into a clean exit.

    Durable memory is never silently repaired, so a corrupt or quarantined file
    stops the command rather than returning a partial view that looks whole.
    """
    try:
        return kb.load()
    except KnowledgeIntegrityError as exc:
        console.print(f"[red]durable memory failed integrity check:[/red] {exc}")
        console.print("Run [bold]rhre knowledge validate[/bold] for the full report.")
        raise typer.Exit(code=1) from exc


def _queue():
    """Load the research plan, turning a missing or unreadable one into an exit.

    Evaluation commands read this. "No actionable hypothesis" and "the plan
    could not be read" look identical on the way out, so the second one has to
    stop rather than report the first.
    """
    from .supervisor.models import QueueSchemaError

    try:
        return HypothesisQueueStore().load()
    except QueueSchemaError as exc:
        console.print(f"[red]hypothesis queue unreadable:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def init(force: bool = False) -> None:
    store = _store()
    if store.claims_path.exists() and not force:
        console.print("research_state/claims.json already exists; use --force to overwrite")
        raise typer.Exit(code=1)
    store.save_claims(seed_claims())
    console.print("Initialized deterministic RH research state.")


@app.command("score-correlation-bound")
def score_correlation_bound(
    theta: float = typer.Option(..., "--theta"),
    rigorous: bool = typer.Option(
        False, "--rigorous", help="The input exponent is a proved estimate, not a fitted slope."
    ),
) -> None:
    try:
        result = correlation_remainder_to_theta(theta, rigorous=rigorous)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(
        f"remainder exponent={result.remainder_exponent:.6g} -> Theta <= {result.theta_upper:.6g}"
        + (" (RH endpoint)" if result.rh_endpoint else "")
    )
    if not rigorous:
        console.print(
            "[yellow]Input treated as unproved; --rigorous is required before this can reach the RH endpoint.[/yellow]"
        )
    console.print("[yellow]This scores a bound; it does not prove the input estimate.[/yellow]")


@app.command()
def status() -> None:
    store = _store()
    claims = store.load_claims()
    if not claims:
        console.print("No claims yet. Run: rhre init")
        return
    table = Table("ID", "Status", "Score", "Statement")
    for claim in claims:
        score = score_claim(claim)
        table.add_row(claim.id, claim.status.value, f"{score.total:.2f}", claim.statement)
    console.print(table)


@supervisor_app.command("add")
def supervisor_add(
    hypothesis_id: str = typer.Option(..., "--id"),
    statement: str = typer.Option(..., "--statement"),
    assumption: list[str] = typer.Option([], "--assumption"),
    proof_gap: list[str] = typer.Option([], "--proof-gap"),
    falsification: list[str] = typer.Option(
        [],
        "--falsification",
        help="ID:COST:DESCRIPTION, with DESCRIPTION containing any additional colons.",
    ),
    rh_equivalent: bool = typer.Option(False, "--rh-equivalent"),
    discharged_obligation: list[str] = typer.Option([], "--discharged-obligation"),
    lifecycle: HypothesisLifecycle = typer.Option(HypothesisLifecycle.PROPOSED, "--lifecycle"),
    epistemic_status: Confidence = typer.Option(Confidence.CONJECTURAL, "--epistemic-status"),
) -> None:
    """Queue a hypothesis.

    `--lifecycle` says where work stands; `--epistemic-status` says how well
    established the statement is. They replace the single `--state` flag, which
    carried both at once and could not express "actively worked on, and still
    only conjectural".
    """
    tests: list[FalsificationTest] = []
    for item in falsification:
        test_id, cost_text, description = item.split(":", 2)
        tests.append(FalsificationTest(id=test_id, cost=int(cost_text), description=description))
    gaps = [
        ProofGap(id=f"gap-{idx:03d}", description=description)
        for idx, description in enumerate(proof_gap, start=1)
    ]
    queue_store = HypothesisQueueStore()
    # The one command that may create the plan, so absence is legitimate here
    # and nowhere else.
    queue = queue_store.load(allow_missing=True)
    queue.upsert(
        Hypothesis(
            id=hypothesis_id,
            statement=statement,
            assumptions=list(assumption),
            lifecycle=lifecycle,
            epistemic_status=epistemic_status,
            proof_gaps=gaps,
            falsification_tests=tests,
            rh_equivalent=rh_equivalent,
            discharged_obligations=list(discharged_obligation),
        )
    )
    queue_store.save(queue)
    console.print(f"Queued {hypothesis_id}")


@supervisor_app.command("list")
def supervisor_list() -> None:
    queue = _queue()
    table = Table(
        "ID", "Lifecycle", "Confidence", "Frontier?", "Advances?", "Actionable?", "Cheapest test"
    )
    for hypothesis in queue.sorted_hypotheses():
        test = hypothesis.cheapest_falsification_test
        table.add_row(
            hypothesis.id,
            hypothesis.lifecycle.value,
            hypothesis.epistemic_status.value,
            "yes" if hypothesis.frontier_relevant else "no",
            "yes" if hypothesis.advances_frontier else "no",
            "yes" if hypothesis.actionable else "no",
            f"{test.id} ({test.cost})" if test else "",
        )
    console.print(table)


@supervisor_app.command("next")
def supervisor_next(json_out: bool = typer.Option(False, "--json")) -> None:
    step = _queue().next_step()
    if step is None:
        console.print("No actionable frontier-relevant hypothesis is ready.")
        raise typer.Exit(code=1)
    if json_out:
        console.print_json(data=step.model_dump(mode="json"))
        return
    console.print(f"{step.hypothesis_id} -> {step.falsification_test_id}")
    console.print(step.description)
    if step.command:
        console.print(step.command)


@app.command("audit-no-go")
def audit_no_go() -> None:
    """Check every claim against the no-go rules and the durable route memory."""
    from .symbolic import match_route

    # Fail closed before anything is reported: the audit is only meaningful
    # against a readable research record.
    _knowledge_items(_knowledge_base())
    claims = _store().load_claims()
    hit = False
    for claim in claims:
        for rule, why in explain_nogo(claim):
            hit = True
            console.print(f"[red]{claim.id}[/red] -> {rule.id} [{why}]: {rule.message}")
        # Statement-level lookup against durable memory, so a refuted route
        # reworded under a new name is still surfaced.
        #
        # No `except` here, deliberately. This used to swallow every exception,
        # print a yellow "route matching unavailable", and carry on to report
        # "No no-go rule violations detected" with exit code 0 -- so the one
        # state in which the durable no-go records cannot be consulted was also
        # the state that reported a clean audit. Missing memory is not an empty
        # one, and an audit that cannot read its evidence has not passed.
        for match in match_route(claim.statement, limit=3):
            if match.is_no_go:
                hit = True
                console.print(
                    f"[red]{claim.id}[/red] -> durable no-go {match.knowledge_id} "
                    f"(score {match.score:.2f}): {match.title}"
                )
    if not hit:
        console.print("No no-go rule violations detected.")


@knowledge_app.command("list")
def knowledge_list(
    domain: str | None = typer.Option(None, "--domain"),
    status_filter: str | None = typer.Option(None, "--status"),
) -> None:
    items = _knowledge_items(_knowledge_base())
    if domain is not None:
        items = [item for item in items if item.domain == domain]
    if status_filter is not None:
        items = [item for item in items if item.status == status_filter]
    table = Table("ID", "Status", "Domain", "Title")
    for item in items:
        table.add_row(item.id, item.status, item.domain, item.title)
    console.print(table)


@knowledge_app.command("show")
def knowledge_show(item_id: str) -> None:
    items = _knowledge_items(_knowledge_base())
    wanted = item_id.casefold()
    item = next((i for i in items if i.id.casefold() == wanted), None)
    if item is None:
        console.print(f"[red]Unknown knowledge item: {item_id}[/red]")
        raise typer.Exit(code=1)
    console.print(item.model_dump_json(indent=2))


@knowledge_app.command("search")
def knowledge_search(query: str) -> None:
    kb = _knowledge_base()
    _knowledge_items(kb)
    items = kb.search(query)
    table = Table("ID", "Status", "Domain", "Title")
    for item in items:
        table.add_row(item.id, item.status, item.domain, item.title)
    console.print(table)


@knowledge_app.command("validate")
def knowledge_validate() -> None:
    """Full integrity audit: seal, JSON shape, status vocabulary, dependencies."""
    kb = _knowledge_base()
    problems = kb.audit()
    if problems:
        for problem in problems:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(code=1)
    items, _ = kb.load_with_quarantine()
    # By membership in the closed no-go set, not by the value's spelling: a
    # second refuted status added later has to be counted, not overlooked
    # because it is spelled differently.
    no_go = [i for i in items if i.status in NO_GO_STATUSES]
    console.print(
        f"Validated {len(items)} knowledge items, {len(no_go)} recorded no-go routes, "
        "seal verified, all dependency references resolve."
    )


@knowledge_app.command("seal")
def knowledge_seal() -> None:
    """Record the integrity checksum of durable memory after a deliberate edit."""
    kb = _knowledge_base()
    digest = kb.seal()
    console.print(f"Sealed {kb.path} -> {kb.seal_path.name}")
    console.print(digest)


@symbolic_app.command("check-certificate")
def symbolic_check_certificate(certificate_file: Path, expression: str) -> None:
    """Check that a certificate actually covers the expression being asked about."""
    from .mathcert.models import MathCertificate
    from .symbolic import check_certificate_against_expression

    cert = MathCertificate.model_validate_json(certificate_file.read_text(encoding="utf-8"))
    result = check_certificate_against_expression(cert, expression)
    console.print_json(data=result.model_dump(mode="json"))
    if not result.usable:
        raise typer.Exit(code=1)


@symbolic_app.command("verify-envelope")
def symbolic_verify_envelope(
    envelope_file: Path,
    allow: list[str] = typer.Option([], "--allow", help="Permitted verifier family."),
) -> None:
    """Validate an external verifier envelope's provenance."""
    from .mathcert.verifiers import VerifierEnvelope, validate_external_envelope

    envelope = VerifierEnvelope.model_validate_json(envelope_file.read_text(encoding="utf-8"))
    errors = validate_external_envelope(envelope, allowed_families=set(allow))
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    console.print(f"Envelope accepted. independence_group={envelope.independence_group}")


@verifier_app.command("capability")
def verifier_capability() -> None:
    from .mathcert import detect_arb_flint

    capability = detect_arb_flint()
    console.print_json(data=capability.__dict__)


@verifier_app.command("arb-flint-interval")
def verifier_arb_flint_interval(
    expression: str = typer.Option(..., "--expression"),
    lower: str = typer.Option(..., "--lower"),
    upper: str = typer.Option(..., "--upper"),
    precision_bits: int = typer.Option(256, "--precision-bits"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    from .mathcert import interval_certificate

    envelope = interval_certificate(
        expression=expression,
        lower=lower,
        upper=upper,
        precision_bits=precision_bits,
    )
    data = envelope.model_dump(mode="json")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        console.print(f"Wrote {out}")
    console.print_json(data=data)


@symbolic_app.command("match-route")
def symbolic_match_route(text: str) -> None:
    """Look a proposed route up against durable memory before claiming novelty."""
    from .symbolic import match_route

    # Without the record there is nothing to compare against, and "no matches"
    # would read as "this route is novel" -- the single most expensive wrong
    # answer this command can give.
    try:
        matches = match_route(text)
    except KnowledgeIntegrityError as exc:
        console.print(f"[red]durable memory unavailable:[/red] {exc}")
        console.print("Novelty cannot be assessed without the research record.")
        raise typer.Exit(code=1) from exc
    if not matches:
        console.print("No durable-memory matches. This does not mean the route is novel.")
        return
    table = Table("ID", "Status", "Score", "Action", "Title")
    for match in matches:
        table.add_row(
            match.knowledge_id, match.status, f"{match.score:.2f}", match.action, match.title
        )
    console.print(table)
    if any(m.is_no_go for m in matches):
        console.print("[red]This overlaps a recorded no-go route.[/red]")
        raise typer.Exit(code=1)


@symbolic_app.command("ingest")
def symbolic_ingest(
    path: Path,
    index: bool = typer.Option(True, "--index/--no-index", help="Index as it ingests."),
    identifier: str | None = typer.Option(None, "--identifier", help="DOI, arXiv ID, ISBN."),
    title: str | None = typer.Option(None, "--title"),
    author: list[str] = typer.Option([], "--author"),
    theorem_label: str | None = typer.Option(
        None, "--theorem", help="The label the source uses, e.g. 'Theorem 3.2'."
    ),
    tag: list[str] = typer.Option([], "--tag"),
) -> None:
    """Extract equations from a document and index them with their provenance.

    Indexing is the default because an ingestion nobody indexed is a parse that
    left no trace. `--no-index` is for inspecting what would be extracted.
    """
    from .symbolic import Citation, FormulaIndex, SourceKind, ingest_file

    result = ingest_file(path)
    parsed = [e for e in result.equations if e.equation.parse_error is None]
    console.print(f"{result.count} equation(s) found in {path}, {len(parsed)} parsed cleanly.")
    if not index:
        return

    citation = Citation(
        source_kind=SourceKind.PAPER,
        source_id=str(path),
        identifier=identifier,
        title=title,
        authors=list(author),
        theorem_label=theorem_label,
    )
    skipped: list[str] = []
    added = FormulaIndex().add_ingestion(result, tags=list(tag), skipped=skipped, citation=citation)
    console.print(f"Indexed {len(added)} record(s).")
    for note in skipped:
        console.print(f"[yellow]skipped {note}[/yellow]")
    if identifier is None or theorem_label is None:
        # Naming what is missing, rather than storing a reference that looks
        # complete and cannot be followed back to a specific result.
        console.print(
            "[yellow]Citation is incomplete: "
            + ("no --identifier; " if identifier is None else "")
            + ("no --theorem label" if theorem_label is None else "")
            + ". Structural matches will not pin the exact source theorem.[/yellow]"
        )


@symbolic_app.command("index-knowledge")
def symbolic_index_knowledge() -> None:
    """Index the formulas durable memory declares, with their citations."""
    from .symbolic import FormulaIndex

    items = _knowledge_items(_knowledge_base())
    skipped: list[str] = []
    added = FormulaIndex().add_knowledge(items, skipped=skipped)
    declared = sum(len(item.formulas) for item in items)
    console.print(f"Indexed {len(added)} formula(s) from {len(items)} durable-memory record(s).")
    for note in skipped:
        console.print(f"[yellow]skipped {note}[/yellow]")
    if declared == 0:
        # An empty result with a stated cause. "Indexed 0" alone reads as a
        # broken reader, and this is not one.
        console.print(
            "[yellow]Durable memory declares no formulas, so there was nothing to "
            "index. The reader only reads the `formulas` field; it does not scrape "
            "prose statements, because a sentence fragment that matches "
            "structurally looks like a result and is not one.[/yellow]"
        )


@symbolic_app.command("level-spacing")
def symbolic_level_spacing(
    height: float = typer.Option(30000.0, "--height", help="Use the zeros with 0 < Im <= height."),
    bins: int = typer.Option(25, "--bins"),
) -> None:
    """How the gaps between consecutive zeros are distributed.

    A different statistic from `pair-correlation`, which averages over all
    pairs at a separation. This is about CONSECUTIVE zeros, and it is where
    level repulsion shows: independent points land arbitrarily close together
    and eigenvalues of a random Hermitian matrix do not.
    """
    from .symbolic.level_spacing import REPULSION_WINDOW, check_level_spacing

    try:
        result = check_level_spacing(height, bins=bins)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    console.print(
        f"[bold]{result.zeros}[/] zeros below T = {result.height:g}; "
        f"mean spacing after unfolding {result.mean_spacing:.6f}"
    )
    console.print(f"  compared against: [cyan]{result.model}[/]")
    console.print(
        f"  mean deviation {result.mean_deviation:.5f}, "
        f"worst {result.worst_deviation:.5f} at s = {result.worst_at:.2f}"
    )
    console.print("\n  [bold]what the residual is made of[/]")
    console.print(
        f"    vs the surmise        {result.mean_deviation:.5f}\n"
        f"    vs the EXACT law      {result.deviation_from_exact:.5f}\n"
        f"    the curve itself      {result.curve_error:.5f}   "
        "(mean |exact - surmise|)\n"
        f"    the histogram         {result.noise_floor:.5f}   "
        f"(what {result.zeros - 1} spacings would show if the law held)"
    )
    left = result.deviation_from_exact - result.curve_error - result.noise_floor
    console.print(
        f"    left over            ~{left:.5f}   — see [bold]--bands[/] for whether that is a shape"
    )
    console.print(
        f"\n[green]level repulsion[/]  P(s < {REPULSION_WINDOW}) = "
        f"{result.repulsion:.6f}, against {result.poisson_repulsion:.6f} for "
        f"independent points — [bold]{result.repulsion_factor:.0f}x rarer[/]"
    )

    table = Table(title="spacing density")
    table.add_column("s", justify="right")
    table.add_column("measured", justify="right")
    table.add_column("surmise", justify="right")
    table.add_column("difference", justify="right")
    for centre, measured, predicted in zip(
        result.centres, result.measured, result.predicted, strict=True
    ):
        table.add_row(
            f"{centre:.2f}",
            f"{measured:.5f}",
            f"{predicted:.5f}",
            f"{measured - predicted:+.5f}",
        )
    console.print(table)
    console.print(
        f"[yellow]Filed as {result.confidence.value}.[/] The GUE law for the "
        "zeros is a conjecture and an asymptotic one, so a finite sample "
        "agreeing with it is evidence and never a proof. The subtraction "
        "above is arithmetic on three numbers, not a decomposition anything "
        "guarantees — `--bands` is the test that does not assume how they "
        "combine."
    )


@symbolic_app.command("spacing-bands")
def symbolic_spacing_bands(
    ceiling: float = typer.Option(90000.0, "--ceiling", help="Top of the last band."),
) -> None:
    """Is the leftover residual a shape, or is it the histogram?

    Three bands of zeros that share no data. Noise is independent between
    them, so a residual with the same shape in all three is not the noise --
    and this needs no assumption about how a deviation and a noise floor
    combine, which subtracting one from the other does.
    """
    from .symbolic.level_spacing import residual_shape

    thirds = [
        (0.0, ceiling * 2 / 9),
        (ceiling * 2 / 9, ceiling * 5 / 9),
        (ceiling * 5 / 9, ceiling),
    ]
    try:
        result = residual_shape(thirds)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    table = Table(title="disjoint bands of zeros")
    table.add_column("band")
    table.add_column("spacings", justify="right")
    table.add_column("deviation", justify="right")
    table.add_column("noise floor", justify="right")
    for band, count, deviation, floor in zip(
        result.bands,
        result.spacings,
        result.deviations,
        result.noise_floors,
        strict=True,
    ):
        table.add_row(
            f"({band[0]:.0f}, {band[1]:.0f}]",
            str(count),
            f"{deviation:.5f}",
            f"{floor:.5f}",
        )
    console.print(table)

    console.print("  pairwise correlation of the signed residual:")
    console.print("    " + ", ".join(f"{value:+.4f}" for value in result.correlations))
    console.print(
        f"  null, from samples drawn FROM the exact law at the same sizes: "
        f"mean |r| {result.null_mean:.4f}, worst {result.null_worst:.4f}"
    )

    if result.is_a_shape:
        console.print(
            "\n[green]A SHAPE.[/] Every band pair correlates beyond anything "
            "the null produced, on zeros that share no data — so what is left "
            "after the curve and the histogram is a property of the zeros."
        )
    else:
        console.print(
            "\n[yellow]NOT ESTABLISHED.[/] At least one band pair sits within "
            "what independent noise produced, so the residual is not "
            "distinguishable from the histogram at these sizes."
        )
    console.print(
        f"\n[yellow]Filed as {result.confidence.value}.[/] A correlation on a "
        "finite sample, detecting a correction to a limit no computation "
        "reaches."
    )


@symbolic_app.command("odlyzko-check")
def symbolic_odlyzko_check(
    data: Path = typer.Option(
        ..., "--data", help="Directory holding zeros1, zeros3, zeros4, zeros5."
    ),
) -> None:
    """Hold the spacing residual against zeros this engine did not compute.

    Two questions `level-spacing` cannot answer about its own residual: whether
    it is an artefact of this zero-finder, and whether it is a finite-height
    correction or a permanent departure. Odlyzko published the first 100000
    zeros and 10000 more at each of 10^12, 10^21 and 10^22 -- sixteen orders
    of magnitude past anything computed here.

    The tables are not vendored; they are somebody else's data. Fetch them
    from the URL printed below.
    """
    from .symbolic.odlyzko import SOURCE, compare

    try:
        result = compare(data)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    agrees = result.ordinate_agreement <= result.stated_accuracy * 1.05
    console.print("[bold]our ordinates against his[/]")
    console.print(
        f"  worst difference {result.ordinate_agreement:.3e} against his "
        f"stated accuracy {result.stated_accuracy:.0e}  "
        f"[{'green' if agrees else 'red'}]"
        f"{'agree to his stated precision' if agrees else 'DISAGREE'}[/]"
    )
    console.print(
        f"  residual from his numbers vs ours: r = "
        f"{result.residual_correlation:.5f}  "
        "— the shape is in the zeros, not in the zero-finder"
    )

    table = Table(title="the residual at heights this engine cannot reach")
    table.add_column("zero index")
    table.add_column("residual", justify="right")
    table.add_column("noise floor", justify="right")
    table.add_column("of the low-height shape", justify="right")
    for index, residual, floor, fraction, uncertainty in zip(
        result.indices,
        result.residuals,
        result.noise_floors,
        result.surviving_fraction,
        result.surviving_uncertainty,
        strict=True,
    ):
        table.add_row(
            index,
            f"{residual:.5f}",
            f"{floor:.5f}",
            f"{fraction:+.3f} ± {uncertainty:.3f}",
        )
    console.print(table)

    if result.shape_is_gone:
        console.print(
            "[green]The shape does not survive.[/] Every index is consistent "
            "with none of it remaining — so the residual measured at "
            "T < 10^5 is a finite-height correction, as read."
        )
    else:
        console.print(
            "[yellow]Some of the shape survives.[/] At least one index sits "
            "further from zero than the sample allows for noise, which the "
            "finite-height reading does not predict."
        )
    console.print(
        f"\n[yellow]Filed as {result.confidence.value}.[/] Two finite samples "
        "agreeing, measured against a conjecture about a limit neither "
        "reaches. Ten thousand zeros bound the surviving amplitude at about a "
        "third and no better; 'it is zero' would claim a precision the sample "
        "does not carry."
    )
    console.print(f"[dim]tables: {SOURCE}[/]")


@symbolic_app.command("spacing-decay")
def symbolic_spacing_decay(
    zeros: Path = typer.Option(
        None,
        "--zeros",
        help="A file of ordinates, one per line. Omit to compute them here.",
    ),
    ceiling: float = typer.Option(1100000.0, "--ceiling", help="Top of the highest band."),
) -> None:
    """How fast does the spacing residual die with height?

    `level-spacing` shows the residual is real and `odlyzko-check` shows it is
    gone by index 10^12. Between those is a rate. Six disjoint bands, an
    amplitude in each with the sampling noise subtracted, and a power of log T
    fitted through them.

    Reaching T ~ 10^6 needs about two million zeros. `--zeros` reads a file of
    ordinates (Odlyzko's `zeros6` is one); without it they are computed here,
    which takes considerably longer.
    """
    import numpy as np

    from .symbolic.spacing_decay import measure_decay

    if zeros is not None:
        ordinates = np.loadtxt(zeros)
        console.print(f"[dim]{len(ordinates)} ordinates from {zeros}[/]")
    else:
        from .symbolic.riemann_siegel import zero_ordinates

        console.print(f"[dim]computing the zeros below {ceiling:g}...[/]")
        ordinates = np.asarray([float(value) for value in zero_ordinates(ceiling)], dtype=float)
        console.print(f"[dim]{len(ordinates)} ordinates[/]")

    edges = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, ceiling]
    bands = list(zip(edges[:-1], edges[1:], strict=True))
    try:
        result = measure_decay(ordinates, bands)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    table = Table(title="residual amplitude by band, noise subtracted")
    table.add_column("band")
    table.add_column("spacings", justify="right")
    table.add_column("median T", justify="right")
    table.add_column("log T", justify="right")
    table.add_column("amplitude", justify="right")
    for band, count, height, value, error in zip(
        result.bands,
        result.spacings,
        result.heights,
        result.amplitudes,
        result.uncertainties,
        strict=True,
    ):
        table.add_row(
            f"({band[0]:.0f}, {band[1]:.0f}]",
            str(count),
            f"{height:.0f}",
            f"{np.log(height):.2f}",
            f"{value:.5f} ± {error:.5f}",
        )
    console.print(table)

    console.print(
        f"  [bold]amplitude = c (log T)^-{result.exponent:.2f}[/]   "
        f"alpha = {result.exponent:.3f} ± {result.exponent_error:.3f}   "
        f"chi² {result.chi_squared:.2f} / {result.degrees_of_freedom}"
    )
    console.print(
        f"  with the exponent pinned at 1:   chi² "
        f"{result.inverse_log_chi_squared:.2f} / "
        f"{result.inverse_log_degrees_of_freedom}"
    )

    if result.steeper_than_inverse_log:
        sigma = (result.exponent - 1.0) / result.exponent_error
        console.print(f"\n[green]Steeper than 1/log T[/] over this range, at {sigma:.1f} sigma.")
    else:
        console.print(
            "\n[yellow]Not distinguishable from 1/log T[/] at this range and these sample sizes."
        )
    console.print(f"\n[yellow]Filed as {result.confidence.value}.[/] {result.caveat}.")
    console.print(f"[dim]log T spans a factor of {result.lever:.2f} here.[/]")


@symbolic_app.command("moments")
def symbolic_moments(
    k: int = typer.Option(2, "--k", help="Measure the 2k-th moment."),
    top: float = typer.Option(160000.0, "--top", help="Highest T measured."),
) -> None:
    """Moments of |zeta(1/2+it)| against the Keating-Snaith conjecture.

    The constant factorises as an arithmetic Euler product times a
    random-matrix term, so unlike every other statistic here the two can be
    told apart. `k = 1` and `k = 2` are theorems, which gives the measurement
    a case where the answer is known underneath the case where it is not.
    """
    import numpy as np

    from .symbolic.moments import (
        arithmetic_factor,
        fit_moment,
        keating_snaith_constant,
        measured_moment,
        moment_constant,
        random_matrix_factor,
        second_moment_asymptotic,
    )

    console.print(f"[bold]the constant[/]  c_{k} = a_{k} * prod j!/(j+k)!")
    console.print(
        f"  arithmetic factor a_{k} = {arithmetic_factor(k):.9f}"
        + ("   (= 1/zeta(2) = 6/pi^2)" if k == 2 else "")
    )
    console.print(
        f"  random matrix    g_{k} = {keating_snaith_constant(k)}, "
        f"g_{k}/(k^2)! = {random_matrix_factor(k)}"
    )
    predicted = moment_constant(k)
    console.print(f"  c_{k} = {predicted:.9f}", end="")
    if k == 1:
        console.print("   [green](theorem: 1)[/]")
    elif k == 2:
        console.print(f"   [green](theorem: 1/(2 pi^2) = {1 / (2 * np.pi**2):.9f})[/]")
    else:
        console.print("   [yellow](conjectural)[/]")

    # The control: the integrator against a theorem with its lower-order term.
    console.print("\n[bold]the integrator, against the k = 1 theorem[/]")
    for height in (top / 8, top):
        measured = measured_moment(1, height)
        exact = second_moment_asymptotic(height)
        naive = float(np.log(height))
        console.print(
            f"  T = {height:9.0f}   measured {measured:.6f}   "
            f"log(T/2pi)+2g-1 {exact:.6f}   [green]diff {measured - exact:+.1e}[/]   "
            f"against log T alone: [red]{measured - naive:+.3f}[/]"
        )
    console.print(
        "  [dim]the leading term alone is out by 1.68 in 9 at the one k where "
        "the answer is known -- eighteen per cent[/]"
    )

    # A degree-k^2 polynomial needs more than k^2 points, so the number of
    # heights scales with k rather than being fixed at six.
    heights = np.geomspace(top / 50, top, max(6, k * k + 3))
    try:
        result = fit_moment(k, heights)
    except ValueError as exc:
        console.print(f"\n[red]{exc}[/]")
        raise typer.Exit(1) from exc

    table = Table(title=f"the {2 * k}-th moment")
    table.add_column("T", justify="right")
    table.add_column("measured", justify="right")
    table.add_column(f"/ c_{k} (log T)^{k * k}", justify="right")
    for height, measured, ratio in zip(
        result.heights, result.measured, result.leading_ratio, strict=True
    ):
        table.add_row(f"{height:.0f}", f"{measured:.6g}", f"{ratio:.3f}")
    console.print(table)

    console.print(f"  a polynomial fit reads off c_{k} = [bold]{result.extracted:.6f}[/]")
    console.print(
        f"  the same fit at k = 2, where the answer is a theorem, is out by "
        f"[{'green' if result.extraction_is_usable else 'red'}]"
        f"{100 * result.calibration_error:.0f}%[/]"
    )
    if not result.extraction_is_usable:
        console.print(
            "\n[red]So that number is not usable.[/] Over this range "
            "log(T/2pi) spans a factor of about 1.5, and a degree-"
            f"{k * k} fit across it cannot separate the leading term from the "
            "rest. Reporting it would be reporting the method's error."
        )

    # The sharp test: the whole polynomial, not one term of it.
    if k == 2:
        from .symbolic.moments import check_fourth_moment

        sharp = check_fourth_moment(heights[-4:])
        sharper = Table(title="the FULL degree-4 polynomial (both share a leader)")
        sharper.add_column("T", justify="right")
        sharper.add_column("measured", justify="right")
        sharper.add_column("proven", justify="right")
        sharper.add_column("err", justify="right")
        sharper.add_column("naive RMT", justify="right")
        sharper.add_column("err", justify="right")
        for height, value, proven, perr, rmt, rerr in zip(
            sharp.heights,
            sharp.measured,
            sharp.proven,
            sharp.proven_error,
            sharp.naive_rmt,
            sharp.rmt_error,
            strict=True,
        ):
            sharper.add_row(
                f"{height:.0f}",
                f"{value:.3f}",
                f"{proven:.3f}",
                f"{perr:.1e}",
                f"{rmt:.3f}",
                f"{rerr:.1e}",
            )
        console.print(sharper)
        if sharp.follows_the_proven_polynomial:
            console.print(
                "  [green]The data follows the theorem, not naive RMT.[/] Both "
                "polynomials have the same leading coefficient and differ only "
                "below it -- so the gap is the arithmetic, and RMT does not "
                "supply it. The RMT departure does not shrink with height."
            )
        console.print(f"  [dim]{sharp.source}[/]")
    else:
        console.print(
            f"  [yellow]No full polynomial available at k = {k}.[/] It is "
            "conjectured by Conrey-Farmer-Keating-Rubinstein-Snaith; Hiary and "
            "Odlyzko use the coefficients without printing them. Until they are "
            "transcribed, the only comparison here is against a leading term -- "
            "which the k = 1 and k = 2 calibrations show says nothing."
        )

    console.print(
        f"\n[yellow]Filed as {result.confidence.value}.[/] A finite integral "
        "against a conjecture about the limit, and for k >= 3 the conjecture "
        "is open."
    )


@symbolic_app.command("explicit-formula")
def symbolic_explicit_formula(
    zeros: int = typer.Option(20000, "--zeros", help="Zeros to sum over."),
) -> None:
    """Rebuild the prime staircase from the zeros and compare it with psi(x).

    The corpus records von Mangoldt's explicit formula. This is not a test of
    the formula -- it is a theorem -- but of whether the corpus RECORDS it
    correctly, which is a different question: a missing constant or a flipped
    sign would parse, index, fingerprint and export exactly as well.
    """
    from .symbolic.explicit_formula import check_explicit_formula, recorded_shape

    for name, present in recorded_shape().items():
        mark = "[green]ok[/]" if present else "[red]MISSING[/]"
        console.print(f"  {mark}  {name.replace('_', ' ')}")

    result = check_explicit_formula(zeros=zeros)
    table = Table(title=f"psi(x) against the corpus, over {result.zeros_used} zeros")
    table.add_column("x", justify="right")
    table.add_column("psi(x)", justify="right")
    table.add_column("from the zeros", justify="right")
    table.add_column("residual", justify="right")
    for point, direct, rebuilt, residual in zip(
        result.points,
        result.direct,
        result.reconstructed,
        result.residuals,
        strict=True,
    ):
        table.add_row(f"{point:.1f}", f"{direct:.5f}", f"{rebuilt:.5f}", f"{residual:+.5f}")
    console.print(table)
    console.print(
        f"[yellow]Filed as {result.confidence.value}.[/] The sum is "
        "conditionally convergent and truncated by index, so a residual is "
        "evidence about the record only once it is well below the truncation "
        "error -- about 1.0 at 100 zeros, 0.12 at 1000, 0.024 at 5000."
    )


@symbolic_app.command("verify-line")
def symbolic_verify_line(
    height: float = typer.Option(
        1000.0, "--height", help="Check every zero with 0 < Im <= height."
    ),
) -> None:
    """Are all the zeros below a height on the critical line, and simple?

    Two counts by independent routes: the argument principle counts zeros in
    the STRIP by tracking arg zeta along a path that never touches the critical
    line, and the sign changes of Z count zeros ON it. Agreement means yes --
    numerically, over a finite range, in floating point.
    """
    from .symbolic.argument_principle import verify_zeros_on_the_line

    result = verify_zeros_on_the_line(height)
    colour = "green" if result.agrees else "red"
    console.print(f"[{colour}]{'AGREE' if result.agrees else 'DISAGREE'}[/]  T = {result.height:g}")
    console.print(f"  strip (argument principle): {result.strip}")
    console.print(f"  on the line (sign changes of Z): {result.on_line}")
    console.print(f"\n  {result.evidence}")
    for note in result.notes:
        console.print(f"    [dim]{note}[/]")
    console.print(
        f"\n[yellow]Filed as {result.confidence.value}.[/] A finite "
        "floating-point computation over a finite range is not a statement "
        "about every zero, and the Riemann hypothesis is one. For the same "
        "question under certified interval arithmetic, at a lower height and a "
        "rigorous confidence, see [bold]rhre symbolic certify-line[/]."
    )


def _refuse_without_backend(exc: Exception) -> None:
    """No backend, no certificate. Refuse rather than degrade.

    Falling back to the floating-point path would answer the question the user
    asked and answer it at the wrong confidence, with nothing in the output
    saying so. The whole content of a certified verification is that the
    arithmetic was certified.
    """
    console.print(f"[red]no certified-enclosure backend:[/] {exc}")
    console.print(
        # `[verify]` is a rich markup tag as far as the console is
        # concerned, and it silently ate the extra: the refusal told the
        # reader to run `pip install -e .`, which does not install the
        # backend it is refusing for want of.
        r"  install one with [bold]pip install -e '.\[verify]'[/], or use "
        "[bold]rhre symbolic verify-line[/], which answers the same question "
        "in floating point and says so."
    )
    raise typer.Exit(1)


@symbolic_app.command("certify-count")
def symbolic_certify_count(
    height: float = typer.Option(
        1000000.0, "--height", help="Count the zeros with 0 < Im <= height."
    ),
) -> None:
    """How many zeros the strip holds below a height -- rigorously, and free.

    The cheap half of `certify-line`, split out because it costs nothing at any
    height worth asking about and the other half does not. This is what
    confirms the floating-point argument principle at every figure this
    repository has recorded.
    """
    from .mathcert.arb_flint import envelope_confidence
    from .symbolic.certified_line import certified_count_envelope

    try:
        envelope = certified_count_envelope(height)
    except RuntimeError as exc:
        _refuse_without_backend(exc)

    counted = envelope.certificate.value.lower.numerator
    console.print(f"[green]N({height:g}) = {counted}[/]")
    console.print(f"  verifier: {envelope.verifier_family} {envelope.verifier_version}")
    console.print(f"  status: {envelope.status.value}")
    console.print(f"  confidence: {envelope_confidence(envelope).value}")
    for check in envelope.checks:
        console.print(f"    [dim]{check}[/]")
    console.print(
        "\n[yellow]Zeros in the STRIP, counted with multiplicity.[/] Where they "
        "are is a different question, and a far more expensive one -- see "
        "[bold]rhre symbolic certify-line[/]."
    )


@symbolic_app.command("certify-line")
def symbolic_certify_line(
    height: float = typer.Option(
        1000.0, "--height", help="Certify every zero with 0 < Im <= height."
    ),
) -> None:
    """Are all the zeros below a height on the critical line, and simple?

    The same question as `verify-line`, under certified interval arithmetic
    rather than float64. Reaches a much lower height, and reaches a different
    kind of answer: this record is filed `rigorous_numerical`, which nothing
    else in this engine can be.

    Costs milliseconds per zero and grows with height. T = 10^4 is about a
    minute; the count alone, at any height, is `certify-count`.
    """
    from .symbolic.certified_line import BackendUnavailable, certify_zeros_on_the_line

    try:
        result = certify_zeros_on_the_line(height)
    except BackendUnavailable as exc:
        _refuse_without_backend(exc)

    colour = "green" if result.verified else "red"
    console.print(
        f"[{colour}]{'CERTIFIED' if result.verified else 'NOT CERTIFIED'}[/]  T = {result.height:g}"
    )
    console.print(f"  zeros in the strip: {result.counted}")
    console.print(f"  certified on the line: {result.certified}")
    for name, passed in result.checks.items():
        console.print(f"    [{'green' if passed else 'red'}]{'ok ' if passed else 'NO '}[/] {name}")
    console.print(f"\n  {result.evidence}")
    console.print(f"  [dim]verifier: {result.backend} {result.backend_version}[/]")
    console.print(
        f"\n[yellow]Filed as {result.confidence.value}.[/] Rigorous about a "
        "finite computation, which is what that means -- and deliberately not "
        "in the RIGOROUS set. It says nothing about zeros above the height, "
        "and the Riemann hypothesis is about all of them."
    )


@symbolic_app.command("verify-zeros")
def symbolic_verify_zeros(
    path: Path = typer.Argument(..., help="A .npy of zero ordinates to check."),
    tolerance: float = typer.Option(
        None,
        "--tolerance",
        help="Ulps of displacement to allow. Default is the measured 64.",
    ),
    show: int = typer.Option(10, "--show", help="How many failures to list."),
) -> None:
    """Are these stored ordinates still the zeros they were written as?

    Ordinates are computed once over hours and read back many times, from a
    disk on a machine WITHOUT ECC MEMORY. A flipped bit is silent: the array
    loads, the statistics run, and the answer is wrong by whatever that bit was
    worth. This evaluates Z at every one and compares the residual against what
    that ordinate's own float64 precision allows.

    It does NOT check that the set is COMPLETE -- zeros can be missing between
    two that are both genuine. `ZeroCount` answers that question and this one
    says nothing about it.
    """
    import numpy as np

    from .symbolic.riemann_siegel import ORDINATE_ULP_TOLERANCE, verify_ordinates

    ordinates = np.load(path)
    limit = ORDINATE_ULP_TOLERANCE if tolerance is None else tolerance
    check = verify_ordinates(ordinates, tolerance=limit)

    colour = "green" if check.ok else "red"
    console.print(
        f"[{colour}]{'VERIFIED' if check.ok else 'CORRUPT'}[/]  "
        f"{check.count:,} ordinates from {path}"
    )
    console.print(f"  worst residual: {check.worst:.1f} ulps of allowance")
    console.print(f"  tolerance: {limit:g}")

    if not check.ok:
        console.print(f"\n  [red]{check.failed.size:,} failed[/]")
        for index in check.failed[:show]:
            console.print(
                f"    [{index}] t = {float(ordinates[index]):.9f}  "
                f"{check.ratio[index]:,.1f} ulps"
            )
        if check.failed.size > show:
            console.print(f"    ... and {check.failed.size - show:,} more")
        raise typer.Exit(1)

    console.print(
        "\n[dim]Displacements below the tolerance are deliberately not "
        "flagged: an ordinate a few ulps from the zero IS the zero in "
        "float64, and failing there would report on the representation "
        "rather than on the data.[/]"
    )


@symbolic_app.command("curve-zeta")
def symbolic_curve_zeta(
    a: int = typer.Option(1, "--a", help="Coefficient a in y^2 = x^3 + ax + b."),
    b: int = typer.Option(1, "--b", help="Coefficient b."),
    prime: int = typer.Option(5, "--prime", help="The field is F_p, p > 3."),
) -> None:
    """The zeta function of one curve over a finite field, exactly.

    Where the Riemann hypothesis is a THEOREM. Every number here is an
    integer or an algebraic number in radicals -- nothing is sampled and
    nothing is enclosed, so the critical line is settled by arithmetic rather
    than by evidence.
    """
    import sympy as sp

    from .symbolic.finite_field_zeta import SingularCurve, curve_zeta

    try:
        result = curve_zeta(a, b, prime)
    except (SingularCurve, ValueError) as exc:
        console.print(f"[red]not an elliptic curve:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[bold]y^2 = x^3 + {a}x + {b}[/] over F_{prime}")
    console.print(f"  #E(F_{prime}) = {result.points}")
    console.print(f"  a_p = p + 1 - #E = {result.trace}")
    console.print(f"  P(T) = {sp.sympify(result.numerator)}")
    console.print(f"  Z(T) = {sp.sympify(result.zeta)}")
    console.print(
        "\n  reciprocal roots: " + ", ".join(str(root) for root in result.reciprocal_roots)
    )
    for root in result.reciprocal_roots:
        modulus = sp.simplify(sp.Abs(root) ** 2)
        console.print(f"    |alpha|^2 = {modulus}   (p = {prime})")

    colour = "green" if result.on_the_critical_line else "red"
    verdict = "ON THE CRITICAL LINE" if result.on_the_critical_line else "OFF IT"
    console.print(f"\n[{colour}]{verdict}[/]  a_p^2 = {result.trace**2} <= 4p = {4 * prime}")
    console.print(
        f"  functional equation Z(1/(pT)) = Z(T): "
        f"{'holds' if result.functional_equation_holds else 'FAILS'}"
    )
    console.print(f"  filed as [bold]{result.confidence.value}[/] — exact algebra, not a sample")
    console.print(f"\n[yellow]{result.caveat}[/]")


@symbolic_app.command("weil-control")
def symbolic_weil_control(
    limit: int = typer.Option(30, "--limit", help="Every prime 3 < p <= limit."),
) -> None:
    """Check every curve over every small field. The positive control.

    `experiments/synthetic-adversary` puts zeros OFF the line to see whether a
    criterion produces a false positive. This is its complement, and it was
    missing: a checker that cannot confirm the critical line where it provably
    holds is broken, and nothing in this engine could have told.
    """
    import sympy as sp

    from .symbolic.finite_field_zeta import (
        SingularCurve,
        curve_zeta,
        frobenius_trace,
        satisfies_the_analogue,
    )

    tested = singular = 0
    failures: list[str] = []
    sampled: list[tuple[int, int, int, int]] = []
    for prime in sp.primerange(5, limit + 1):
        for a in range(prime):
            for b in range(prime):
                try:
                    trace = frobenius_trace(a, b, int(prime))
                except SingularCurve:
                    singular += 1
                    continue
                tested += 1
                if not satisfies_the_analogue(trace, int(prime)):
                    failures.append(f"y^2 = x^3 + {a}x + {b} over F_{prime}")
                if a == b == 1:
                    sampled.append((a, b, int(prime), trace))

    # The fast path held against the slow one. Without this the sweep would be
    # thousands of curves through an integer inequality that nothing ever
    # compared against the zeta function it is supposed to be about.
    disagreements = []
    for a, b, prime, trace in sampled:
        full = curve_zeta(a, b, prime)
        if full.trace != trace or not full.functional_equation_holds:
            disagreements.append(f"F_{prime}")
        if full.on_the_critical_line != satisfies_the_analogue(trace, prime):
            disagreements.append(f"F_{prime} (shortcut disagrees)")
    if disagreements:
        console.print(
            f"[red]the integer test disagrees with the zeta function at "
            f"{', '.join(disagreements)}[/]"
        )
        raise typer.Exit(1)

    if failures:
        console.print(f"[red]{len(failures)} curve(s) OFF the critical line[/]")
        for line in failures[:10]:
            console.print(f"  {line}")
        raise typer.Exit(1)

    console.print(
        f"[green]{tested} curves, every one on the critical line[/] "
        f"(and {singular} singular equations refused rather than counted)"
    )
    console.print(
        "  settled by [bold]a_p^2 <= 4p[/] in integers — no sampling, no enclosure, no tolerance"
    )
    console.print(
        f"  the integer test agrees with the full zeta function at "
        f"{len(sampled)} sampled curve(s), functional equation included"
    )
    console.print(
        "\n[yellow]This is a control on the machinery, not evidence about "
        "zeta.[/] Weil's theorem is proved by intersection theory on C x C, "
        "which has no counterpart over Q."
    )


@symbolic_app.command("pair-correlation")
def symbolic_pair_correlation(
    height: float = typer.Option(50000.0, "--height", help="Use the zeros with 0 < Im <= height."),
    bins: int = typer.Option(30, "--bins"),
    lower_order: bool = typer.Option(
        False,
        "--lower-order",
        help=(
            "Also compare against Conrey-Snaith with every lower-order term. "
            "Slow: a Euler product over a million primes per quadrature point."
        ),
    ),
) -> None:
    """Measure the zeros against the pair correlation the corpus asserts.

    The corpus's first indexed formula is `1 - (sin(pi u)/(pi u))^2`, and
    nothing had ever held it against anything. The expression is read from the
    index rather than retyped, so this checks the corpus and not a copy of it.

    That curve is UNIVERSAL, though -- random matrices and quantum billiards
    obey it too -- so following it says the zeros are a spectrum and says
    nothing about primes. `--lower-order` adds the Conrey-Snaith form, which
    carries the arithmetic and has no free parameters.
    """
    from .symbolic.pair_correlation import check_pair_correlation

    result = check_pair_correlation(height, bins=bins, lower_order=lower_order)
    console.print(
        f"{result.zeros} zeros below T = {result.height:g}, "
        f"pairs within {result.window:g} mean spacings"
    )
    console.print(
        f"  mean deviation {result.mean_deviation:.4f}, "
        f"worst {result.worst_deviation:.4f} at u = {result.worst_at:.2f}"
    )
    table = Table(title="pair correlation")
    table.add_column("u", justify="right")
    table.add_column("measured", justify="right")
    table.add_column("corpus", justify="right")
    table.add_column("difference", justify="right")
    if lower_order:
        table.add_column("Conrey-Snaith", justify="right")
        table.add_column("difference", justify="right")
    for index, (centre, measured, predicted) in enumerate(
        zip(result.centres, result.measured, result.predicted, strict=True)
    ):
        row = [
            f"{centre:.2f}",
            f"{measured:.4f}",
            f"{predicted:.4f}",
            f"{measured - predicted:+.4f}",
        ]
        if lower_order:
            curve = result.lower_order[index]
            row += [f"{curve:.4f}", f"{measured - curve:+.4f}"]
        table.add_row(*row)
    console.print(table)
    if lower_order:
        floor = result.noise_floor
        console.print(
            f"  from Montgomery {result.mean_deviation:.4f} "
            f"({result.mean_deviation / floor:.2f}x noise), "
            f"from Conrey-Snaith {result.lower_order_deviation:.4f} "
            f"({result.lower_order_deviation / floor:.2f}x noise)"
        )
        console.print(
            f"  the two curves are {result.curve_separation:.4f} apart and the "
            f"measured sampling noise is {floor:.4f}"
        )
        if not result.curves_are_distinguishable:
            # Without this the two deviations above read as a comparison when
            # the sample cannot make one.
            console.print(
                "[yellow]  The curves are closer together than the noise, so "
                "neither deviation is evidence about which one the zeros "
                "follow.[/]"
            )
    console.print(
        f"[yellow]Filed as {result.confidence.value}.[/] Montgomery's pair "
        "correlation is a conjecture, and an asymptotic one: a finite sample "
        "agreeing with it is evidence for it and never a proof of it."
    )


@symbolic_app.command("proof-queue")
def symbolic_proof_queue(
    out: Path | None = typer.Option(None, "--out", help="Directory for .lean files."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Sort indexed equations by what Lean can actually be handed.

    Export-ready means Lean source was emitted. It does not mean verified:
    nothing here runs Lean, and `ring` closing the goal is decided by a compiler
    this package never invokes.
    """
    from .symbolic import FormulaIndex, build_proof_queue

    queue = build_proof_queue(FormulaIndex().load())
    if json_out:
        console.print_json(data=queue.model_dump(mode="json"))
        return

    console.print(queue.summary())
    table = Table("Record", "Verdict", "Theorem", "Why")
    for entry in queue.entries:
        table.add_row(
            entry.record_id[:12],
            entry.verdict.value,
            entry.theorem_name if entry.export_ready else "",
            (entry.reason or "")[:60],
        )
    console.print(table)
    if out is not None:
        written = queue.write(out)
        console.print(f"Wrote {len(written)} Lean file(s) to {out}")
    console.print(
        "[yellow]Emitted, not verified. No Lean compiler ran; each export remains "
        "an open obligation until one does.[/yellow]"
    )


@symbolic_app.command("proof-gaps")
def symbolic_proof_gaps(steps_file: Path) -> None:
    """Report which steps of a proof sketch are not yet rigorous."""
    from .symbolic import extract_proof_gaps
    from .symbolic.models import ProofStep

    raw = json.loads(steps_file.read_text(encoding="utf-8"))
    gaps = extract_proof_gaps([ProofStep.model_validate(item) for item in raw])
    console.print_json(data=[gap.model_dump(mode="json") for gap in gaps])
    if gaps:
        raise typer.Exit(code=1)


@symbolic_app.command("minimize")
def symbolic_minimize(statement: str) -> None:
    """Suggest a weaker sufficient target for a conjecture."""
    from .symbolic import minimize_conjecture

    console.print_json(data=minimize_conjecture(statement).model_dump(mode="json"))


@symbolic_app.command("extract")
def symbolic_extract(
    text: str | None = typer.Argument(None), file: Path | None = typer.Option(None, "--file")
) -> None:
    """Extract math expressions/equations from Markdown or text."""
    from .symbolic import extract_equations

    if file is not None:
        payload = file.read_text(encoding="utf-8")
    elif text is not None:
        payload = text
    else:
        console.print("[red]Provide TEXT or --file PATH.[/red]")
        raise typer.Exit(code=2)
    console.print_json(data=[item.model_dump(mode="json") for item in extract_equations(payload)])


@symbolic_app.command("simplify")
def symbolic_simplify(expression: str) -> None:
    """Simplify an expression with a named rewrite trace."""
    from .symbolic import simplify_with_trace

    console.print_json(data=simplify_with_trace(expression).model_dump(mode="json"))


@symbolic_app.command("equivalent")
def symbolic_equivalent(left: str, right: str) -> None:
    """Check symbolic equivalence conservatively."""
    from .symbolic import equivalent

    console.print_json(data=equivalent(left, right).model_dump(mode="json"))


@symbolic_app.command("fingerprint")
def symbolic_fingerprint(expression: str) -> None:
    from .symbolic import fingerprint

    console.print_json(data=fingerprint(expression).model_dump(mode="json"))


@symbolic_app.command("assumptions")
def symbolic_assumptions(expression: str) -> None:
    from .symbolic import extract_assumptions

    console.print_json(data=[a.model_dump(mode="json") for a in extract_assumptions(expression)])


@symbolic_app.command("residue")
def symbolic_residue(expression: str, variable: str, pole: str) -> None:
    from .symbolic import residue

    console.print_json(data=residue(expression, variable, pole).model_dump(mode="json"))


@symbolic_app.command("transform")
def symbolic_transform(name: str) -> None:
    from .symbolic import TRANSFORM_REGISTRY

    factory = TRANSFORM_REGISTRY.get(name)
    if factory is None:
        console.print(f"[red]Unknown transform: {name}[/red]")
        console.print("Available: " + ", ".join(sorted(TRANSFORM_REGISTRY)))
        raise typer.Exit(code=2)
    console.print_json(data=factory().model_dump(mode="json"))


@dre_app.command("export-latest")
def dre_export_latest(
    claim: str = typer.Option(..., "--claim"),
    experiment_name: str | None = typer.Option(None, "--experiment-name"),
    effect: ClaimEffect = typer.Option(ClaimEffect.SUPPORTS, "--effect"),
    primary_metric: str | None = typer.Option(None, "--primary-metric"),
    bound_exponent: float | None = typer.Option(None, "--bound-exponent"),
    rh_equivalent: bool = typer.Option(False, "--rh-equivalent"),
    counterexample_found: bool = typer.Option(False, "--counterexample-found"),
    out: Path = typer.Option(Path("dre/experiments/latest.yaml"), "--out"),
) -> None:
    """Export a stored experiment as a DRE evidence envelope.

    There is deliberately no `--class`, `--method-family`, `--worker-version`,
    `--theta-upper`, or `--independently-verified` flag. Those were free strings
    and floats with no binding to the experiment, so a numerical run could be
    stamped `proved` with a fabricated Theta bound, and one numpy run could be
    relabelled into several "independent" witnesses. Provenance now travels with
    the experiment record.
    """
    store = _store()
    if experiment_name is None:
        result = store.latest_experiment()
    else:
        matches = [r for r in store.load_experiments() if r.name == experiment_name]
        result = matches[-1] if matches else None
    if result is None:
        console.print("[red]No matching experiment results found.[/red]")
        raise typer.Exit(code=1)
    try:
        envelope = DreEvidenceEnvelope.from_experiment(
            result,
            claim_id=claim,
            claim_effect=effect,
            primary_metric_name=primary_metric,
            bound_exponent=bound_exponent,
            rh_equivalent=rh_equivalent,
            counterexample_found=counterexample_found,
            artifact_ref=str(store.experiments_path),
        )
    except (KeyError, TypeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except WorkerClassError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc

    matching_claim = next((c for c in store.load_claims() if c.id == claim), None)
    decision = evaluate_export(
        envelope,
        claim=matching_claim,
        knowledge_path=CANONICAL_KNOWLEDGE_PATH,
    )
    for line in decision.render():
        colour = "red" if line.startswith("[block]") else "yellow"
        console.print(f"[{colour}]{line}[/{colour}]")
    if not decision.allowed:
        console.print("[red]Export blocked. Nothing was written.[/red]")
        raise typer.Exit(code=3)

    write_dre_experiment(envelope, out, observed_at=store.experiment_index(result))
    console.print(f"Wrote {out}")
    console.print(f"payload_hash={envelope.payload_hash}")
    console.print(f"provenance_hash={envelope.provenance_hash}")
    console.print(f"evidence_class={envelope.evidence_class.value} (worker-declared)")
    console.print(f"independence_group={envelope.independence_group}")
    console.print(
        "[yellow]DRE must decide epistemic status; this export does not promote the result.[/yellow]"
    )


@dre_app.command("show-pack")
def dre_show_pack() -> None:
    console.print("DRE pack: dre/model-packs/riemann-research")
    console.print("Validate it with the exact DRE checkout used for the run.")


@experiment_app.command("safe-binomial")
def exp_safe_binomial(k_max: int = 40, dps: int = 80) -> None:
    result = safe_binomial.run(k_max=k_max, dps=dps)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("gamma-filter")
def exp_gamma_filter(x: float = 1000.0, q: float = 2.0) -> None:
    result = gamma_filter.run(x=x, q=q)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("exponent-scan")
def exp_exponent_scan(
    x_min: float = 100.0,
    x_max: float = 3000.0,
    points: int = 10,
    q: float = 2.0,
    phases: int = 8,
) -> None:
    """Fit S_q's exponent, over several grid phases so the spread is visible.

    `signal` IS DELIBERATELY NOT AN OPTION. It is the injection seam the positive
    controls run through -- a pure power law, so a fit that cannot recover a known
    exponent is caught -- and it takes a callable, which a command line cannot
    supply. Exposing it would offer a flag nobody can use while implying the
    scan measures whatever it is handed.
    """
    result = exponent_scan.run(x_min=x_min, x_max=x_max, points=points, q=q, phases=phases)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("baez-duarte")
def exp_baez_duarte(
    sieve_limit: int = 8_000_000,
    k_low: float = 1e4,
    k_high: float = 2e6,
    points: int = 200,
) -> None:
    """Is |c_k| k^(3/4) bounded? RH says yes; a growing envelope would refute it."""
    result = baez_duarte.run(
        sieve_limit=sieve_limit, k_low=k_low, k_high=k_high, points=points
    )
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("li-criterion")
def exp_li_criterion(order: int = 120, bits: int = 1500) -> None:
    """Li's lambda_n from zeta and Gamma alone. RH is lambda_n >= 0 for all n."""
    result = li_criterion.run(order=order, bits=bits)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("weil-positivity")
def exp_weil_positivity(
    size: int = 12, sigma: float = 0.03, prime_limit: int = 200_000
) -> None:
    """Is the Weil form PSD? A negative eigenvalue would refute RH.

    `tolerance` IS DELIBERATELY NOT AN OPTION. It is the gate separating REFUTED
    from UNRESOLVED, and a command-line flag that loosens a refutation threshold
    is the "loosen a tolerance" non-fix this repository is written against.
    Change it in the module, with a reason, or not at all.
    """
    result = weil_positivity.run(size=size, sigma=sigma, prime_limit=prime_limit)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("weil-certified")
def exp_weil_certified(
    size: int = 8, sigma: str = "0.03", prime_limit: int = 200_000, precision: int = 128
) -> None:
    """The Weil form in ball arithmetic: a verdict that is a proof, not a reading.

    `eval_limit` IS DELIBERATELY NOT AN OPTION. `eval_budget` ties it to the
    working precision by measurement -- 100k/200k/400k/1.6M/3.2M at 128-512 bits
    -- so a flag setting it independently invites a run whose budget and
    precision disagree, which returns a finite answer instead of a refusal.
    """
    result = weil_certified.run(
        size=size, sigma=sigma, prime_limit=prime_limit, precision=precision
    )
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("bombieri-finite-model")
def exp_bombieri_finite_model(
    t: float = 2.3,
    count: int = 20,
) -> None:
    """Bombieri's 2000 experiment, run as a control on his own two theorems.

    `t` is the support, so the prime cutoff is `exp(2t)`; `count` is how many
    ordinate pairs the truncation keeps. Both defaults sit where his figures do.

    The run refuses rather than reports if Lemma 10 fails -- an on-line-only
    zero set producing a negative eigenvalue means the matrix is wrong, and
    every threshold built on it would be meaningless.
    """
    result = bombieri_finite_model.run(t=t, count=count)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("weil-sensitivity")
def exp_weil_sensitivity(
    size: int = 20, sigma: float = 0.03, pairs: int = weil_sensitivity.PAIR_LIMIT
) -> None:
    """How far off the line a zero must be before the Weil form would notice."""
    result = weil_sensitivity.run(size=size, sigma=sigma, pairs=pairs)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("height-recovery")
def exp_height_recovery(
    height: float = 300_000.0,
    band_width: float = 0.5,
    anchors: int = 3,
    prime_limit: int = 10_000,
    ladder: str = typer.Option(
        None, "--ladder", help="An .npz of rungs from build-ladder.py, for the full lever."
    ),
    zeros: str = typer.Option(
        None, "--zeros", help="An .npy of ordinates to use instead of computing them."
    ),
) -> None:
    """Regress the l fitted from each band's pair correlation on its true l."""
    result = height_recovery_lab.run(
        height=height,
        band_width=band_width,
        anchors=anchors,
        prime_limit=prime_limit,
        ladder=ladder,
        zeros=zeros,
    )
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("fit-bias-lab")
def exp_fit_bias_lab(
    ladder: str = typer.Option(
        None, "--ladder", help="An .npz of prebuilt rungs from build-ladder.py."
    ),
    zeros: str = typer.Option(
        None, "--zeros", help="An .npy of ordinates, to measure the low bands."
    ),
    max_slices: int = typer.Option(
        None, "--max-slices", help="Cap slices per cell. A full run is ~15 minutes."
    ),
) -> None:
    """How wrong fit_ell is at a known height, as a function of band size.

    NO DEFAULT PATH, deliberately. A default resolving to a file that exists in
    two home directories would make the same command silently measure on one box
    and silently refuse on another, and those must not print the same. Absent
    both artifacts it REFUSES: `refused: 1.0` and no `bias_at_*` key at all, so
    nothing downstream can read a refusal as a flat b(N).

    SIX PARAMETERS ARE DELIBERATELY NOT OPTIONS, and naming them is the point:
    `gain_low`, `gain_high`, `max_railed`, `min_slices`, `prime_limit` and
    `counts`. The first five encode what counts as a measurement -- the gain
    window a cell must sit in, how many fits may be censored at the grid edge,
    and how many slices a cell needs before its mean means anything. Exposing
    them invites a run that reports b over cells where the estimator does not
    respond, which is not a bias but a statement that nothing was measured.
    `counts` is the ladder of band sizes the whole b(N) curve is defined over;
    a run with a different ladder is a different experiment, not this one
    reconfigured.

    Change them in the module, with a reason, or not at all.
    """
    result = fit_bias_lab.run(ladder=ladder, zeros=zeros, max_slices=max_slices)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("correlation-lab")
def exp_correlation_lab(X: int = 20_000, q: float = 4.0, h_max: int | None = None) -> None:
    result = correlation_lab.run(X=X, q=q, h_max=h_max)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("correlation-scan")
def exp_correlation_scan(
    X_min: int = 2_000,
    X_max: int = 50_000,
    points: int = 7,
    q: float = 4.0,
    h_factor: float = 12.0,
    phases: int = 6,
) -> None:
    """Shell-correlation slopes, and the Theta they map to when they are readable."""
    result = correlation_scan.run(
        X_min=X_min, X_max=X_max, points=points, q=q, h_factor=h_factor, phases=phases
    )
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("counterterm-discovery")
def exp_counterterm_discovery(
    X_min: int = 2_000,
    X_max: int = 40_000,
    points: int = 8,
    q: float = 4.0,
    phases: int = 6,
) -> None:
    """Fit a lower-order counterterm basis to the screening remainder."""
    result = counterterm_discovery.run(
        X_min=X_min, X_max=X_max, points=points, q=q, phases=phases
    )
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("local-variance")
def exp_local_variance(
    X: int = 5_000, q: float = 4.0, width: float = 1.0, samples: int = 33
) -> None:
    result = local_variance.run(X=X, q=q, width=width, samples=samples)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("arc-diagnostics")
def exp_arc_diagnostics(
    X: int = 8_000, q: float = 4.0, fft_size: int = 65_536, major_width: float = 0.02
) -> None:
    result = arc_diagnostics.run(X=X, q=q, fft_size=fft_size, major_width=major_width)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("synthetic-zero")
def exp_synthetic_zero(
    eta: float = 0.02,
    gamma: float = 14.134725141734693,
    q: float = 20.0,
    T_min: float = 0.0,
    T_max: float = 120.0,
    points: int = 1000,
) -> None:
    result = synthetic_zero.run(eta=eta, gamma=gamma, q=q, T_min=T_min, T_max=T_max, points=points)
    _store().append_experiment(result)
    _emit(result)


@experiment_app.command("synthetic-adversary")
def exp_synthetic_adversary(
    eta: float = typer.Option(0.02, "--eta"),
    gamma: float = typer.Option(14.134725141734693, "--gamma"),
    critical_gamma: list[float] = typer.Option([], "--critical-gamma"),
    off_line: list[str] = typer.Option([], "--off-line", help="BETA:GAMMA[:MULTIPLICITY]"),
    q: float = typer.Option(20.0, "--q"),
    tolerance: float = typer.Option(0.0, "--tolerance"),
    criterion: list[str] = typer.Option([], "--criterion"),
) -> None:
    zeros = adversarial_synthetic.parse_off_line_zeros(off_line)
    if not zeros:
        zeros = adversarial_synthetic.default_off_line_system(eta, gamma)
    result = adversarial_synthetic.run(
        critical_gamma=critical_gamma or [gamma],
        off_line=zeros,
        q=q,
        tolerance=tolerance,
        criteria=criterion or None,
    )
    _store().append_experiment(result)
    _emit(result)


if __name__ == "__main__":
    app()
