from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..core.knowledge import KnowledgeBase, KnowledgeIntegrityError
from ..supervisor.properties import extract_from_hypothesis
from ..supervisor.store import HypothesisQueueStore
from ..symbolic.formula_index import FormulaIndex
from .closure import implication_closure
from .discriminator import analyze_discriminators
from .extract import extract_from_formula, extract_from_knowledge
from .inventory import object_inventory
from .mining import mine_invariants, mine_symmetries
from .models import ClosureMode, PropertyGraph, PropertyKind
from .singularity import propagate_singularities
from .store import PropertyGraphStore

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("build")
def build(
    mode: ClosureMode = typer.Option(ClosureMode.RIGOROUS, "--mode"),
    out: Path = typer.Option(Path("research_state/property_graph.json"), "--out"),
) -> None:
    formulas = FormulaIndex().load()
    try:
        knowledge = KnowledgeBase().load()
    except KnowledgeIntegrityError as exc:
        # No graph is written. A property graph built without durable memory is
        # missing every knowledge-derived property and every no-go route, and
        # nothing downstream could tell that apart from a graph where those
        # genuinely do not apply.
        console.print(f"[red]durable memory failed integrity check:[/red] {exc}")
        console.print("Run [bold]rhre knowledge validate[/bold] for the full report.")
        raise typer.Exit(code=1) from exc
    objects = object_inventory(formulas, knowledge)
    properties = []
    for record in formulas:
        properties.extend(extract_from_formula(record))
    for item in knowledge:
        properties.extend(extract_from_knowledge(item))
    # The queue is an optional input here: a graph can be built from the
    # formula index and durable memory alone.
    for hypothesis in HypothesisQueueStore().load(allow_missing=True).hypotheses:
        properties.extend(extract_from_hypothesis(hypothesis))
    properties.extend(mine_symmetries(objects))
    properties.extend(mine_invariants(objects))
    properties.extend(propagate_singularities(objects))
    graph = implication_closure(PropertyGraph(objects=objects, properties=properties), mode=mode)
    store = PropertyGraphStore(out)
    store.save(graph)
    console.print(
        f"Wrote {out} with {len(graph.objects)} objects, "
        f"{len(graph.properties)} properties, {len(graph.edges)} edges."
    )
    console.print(f"graph_hash={graph.graph_hash()}")


@app.command("list")
def list_properties(
    graph_path: Path = typer.Option(Path("research_state/property_graph.json"), "--graph"),
    kind: PropertyKind | None = typer.Option(None, "--kind"),
    object_id: str | None = typer.Option(None, "--object-id"),
    rigorous_only: bool = typer.Option(False, "--rigorous-only"),
) -> None:
    store = PropertyGraphStore(graph_path)
    table = Table("ID", "Object", "Kind", "Status", "Value")
    for prop in store.query(object_id=object_id, kind=kind, rigorous_only=rigorous_only):
        table.add_row(prop.id, prop.object_id, prop.kind.value, prop.status.value, prop.value)
    console.print(table)


@app.command("show")
def show(
    graph_path: Path = typer.Option(Path("research_state/property_graph.json"), "--graph"),
    property_id: str = typer.Argument(...),
) -> None:
    graph = PropertyGraphStore(graph_path).load()
    prop = next((item for item in graph.properties if item.id == property_id), None)
    if prop is None:
        console.print(f"[red]Unknown property: {property_id}[/red]")
        raise typer.Exit(code=1)
    console.print_json(data=prop.model_dump(mode="json"))


@app.command("discriminators")
def discriminators(
    graph_path: Path = typer.Option(Path("research_state/property_graph.json"), "--graph"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    graph = PropertyGraphStore(graph_path).load()
    results = analyze_discriminators(graph)
    if json_out:
        console.print(json.dumps([item.model_dump(mode="json") for item in results], indent=2))
        return
    table = Table("Object", "Property", "Status", "Proof?", "Reason")
    for item in results:
        table.add_row(
            item.object_id,
            item.property_id,
            item.status.value,
            "yes" if item.promoted_to_proof else "no",
            item.reason,
        )
    console.print(table)


if __name__ == "__main__":
    app()
