# Property Extraction

The property extraction subsystem turns the existing symbolic, knowledge, math
certificate, and DRE-safe records into a deterministic property graph.

Each property records:

- the mathematical object it describes;
- a typed property kind;
- an epistemic status;
- source provenance and assumptions;
- optional implication edges to derived properties.

Closures support two modes:

- `rigorous`: only known, proved, certified, or rigorous derived properties
  without assumptions may imply new properties;
- `exploratory`: symbolic, heuristic, and synthetic properties may generate
  candidates, but the resulting properties remain heuristic or synthetic.

Synthetic discriminator analysis is deliberately fail-closed. A critical-line
versus off-line synthetic contrast can produce a candidate discriminator, but it
cannot promote an RH-equivalent claim to proved.

Supervisor hypotheses enter the graph as symbolic research state. The graph
retains their `frontier_relevant`, `advances_frontier`, and `actionable`
metadata, but open assumptions and proof gaps keep them out of rigorous closure.
An RH-equivalent hypothesis only becomes frontier-relevant when named
`discharged_obligations` are present.

CLI entry points:

```bash
python -m rh_research_engine.properties.cli build --mode rigorous
python -m rh_research_engine.properties.cli list --kind growth_bound --rigorous-only
python -m rh_research_engine.properties.cli discriminators
```

The graph is stored at `research_state/property_graph.json`.

## What the graph is built from

Objects come from the **formula index** and from durable memory's `formulas`
field — never from `title` or `statement`.

That distinction was learned the expensive way. The symbol regex used to run
over the prose too, so every English word became a `MathObject` of kind
`CLAIM` whose "expression" was the sentence it came from: `rather`, `width`,
and `Schr` (a truncated "Schrödinger") were all registered as mathematical
objects. A real build produced **558 objects and 481 properties from 42
records**, 471 of the properties being the identical `value="unknown"`,
`status=blocked` record emitted whenever domain analysis failed on a sentence.
Ten real properties sat underneath.

Reading only the field declared to hold formulas gives **5 objects and 9
properties** on the same input — `Lambda_`, `Theta`, `theta`, `x`, `y`, every
one a symbol that actually appears in a formula.

The shipped durable memory declares no formulas, so it currently contributes
nothing. That is the honest state and it keeps the gap visible; populating those
fields is what makes knowledge-derived properties appear.

A failed domain analysis now yields no record at all. `unknown` asserts nothing
about an object, and a graph where 98% of the rows say nothing is a graph nobody
can read. Failures are reported through `propagate_singularities(...,
unanalysed=[])` for a caller that wants the count.
