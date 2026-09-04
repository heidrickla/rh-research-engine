#!/usr/bin/env python3
"""Inventory every status, role, and flag vocabulary in the repository.

WHY. The original adversarial review opened on five disjoint status
vocabularies with no mapping between them, and the split recurred in fresh code
within a day. Before a contract layer can unify them, they have to be
enumerated -- all of them, mechanically, not from memory.

WHAT IT REPORTS.

  vocabularies   every StrEnum, its defining module, and its exact members
  axis_fields    every Pydantic field that carries a status/role/lifecycle/
                 frontier axis, and the model it belongs to
  substring_classifiers
                 places that decide an epistemic question by string shape --
                 `status.value.startswith("exact")`, `"derived" in status` and
                 friends. This is the defect class, not a style nit: it
                 promoted 14 of 21 statuses to rigorous by accident of
                 spelling, and it means any status added later is classified
                 by how it happens to be named.

Usage:
  python tools/inventory-status.py            # write docs/contracts/status-inventory.json
  python tools/inventory-status.py --print    # human-readable summary
  python tools/inventory-status.py --check    # fail if the inventory drifted
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import sys
from enum import StrEnum
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PACKAGE = "rh_research_engine"
SRC = REPO / "src" / PACKAGE
DEFAULT_OUT = REPO / "docs" / "contracts" / "status-inventory.json"

#: Field names that carry one of the epistemic axes. Matched exactly, because a
#: substring rule here would be the very thing this tool exists to find.
AXIS_FIELDS = {
    "status": "epistemic",
    "epistemic_status": "epistemic",
    "evidence_class": "epistemic",
    "verification_status": "epistemic",
    "state": "lifecycle",
    "lifecycle": "lifecycle",
    "role": "role",
    "mathematical_role": "role",
    "kind": "classification",
    "rh_equivalent": "frontier",
    "frontier_relevant": "frontier",
    "advances_frontier": "frontier",
    "actionable": "frontier",
    "property_extractable": "frontier",
    "discharged_obligations": "frontier",
    "claim_effect": "classification",
}

#: The axis categories where classification by spelling is the defect class.
#:
#: "classification" is excluded on purpose. `kind` and `claim_effect` say what
#: *shape* a record has -- equation or expression, supports or refutes -- and
#: reading one decides which branch of a parser runs, never how established a
#: statement is. The defect this tool exists to find is deciding an epistemic
#: question by how a value happens to be spelled, which is how 14 of 21
#: knowledge statuses were promoted to rigorous.
SEMANTIC_AXIS_CATEGORIES = frozenset({"epistemic", "lifecycle", "role", "frontier"})

def _iter_modules() -> list[str]:
    names = [PACKAGE]
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        names.append(".".join([PACKAGE, *parts]) if parts else PACKAGE)
    return sorted(set(names))


def collect_vocabularies() -> dict:
    out: dict[str, dict] = {}
    for module_name in _iter_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isclass(obj):
                continue
            if obj.__module__ != module_name:
                continue
            if issubclass(obj, StrEnum) and obj is not StrEnum:
                out[f"{module_name}.{name}"] = {
                    "module": module_name,
                    "name": name,
                    "member_count": len(list(obj)),
                    "members": [m.value for m in obj],
                }
    return dict(sorted(out.items()))


def collect_axis_fields() -> dict:
    from pydantic import BaseModel

    out: dict[str, dict] = {}
    for module_name in _iter_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isclass(obj):
                continue
            if obj.__module__ != module_name or not issubclass(obj, BaseModel):
                continue
            if obj is BaseModel:
                continue
            hits = {
                field: AXIS_FIELDS[field]
                for field in obj.model_fields
                if field in AXIS_FIELDS
            }
            if hits:
                out[f"{module_name}.{name}"] = {
                    "module": module_name,
                    "model": name,
                    "axes": dict(sorted(hits.items())),
                }
    return dict(sorted(out.items()))


def _ends_in_value_attr(node: ast.AST) -> bool:
    """True for `<anything>.value`, the shape a status comparison takes."""
    return isinstance(node, ast.Attribute) and node.attr == "value"


def _is_axis_expression(node: ast.AST) -> bool:
    """True for an expression naming one of the epistemic axes.

    Matches `status`, `item.status`, `item.status.value` and the other axis
    field names -- with or without `.value`. The original detector required
    `.value`, which missed the two real defects found in this repository:
    `"equivalent" in item.status` in the route matcher, and
    `match.status == "false_route"` in the no-go audit. A `StrEnum` compares
    equal to its own string, so leaving `.value` off changes nothing about the
    behaviour and everything about whether the check fires.
    """
    if _ends_in_value_attr(node):
        node = node.value
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    else:
        return False
    return AXIS_FIELDS.get(name) in SEMANTIC_AXIS_CATEGORIES


def collect_substring_classifiers() -> list[dict]:
    """Find real classification-by-spelling, using the AST rather than text.

    Regex over source lines flagged this module's own docstring, which
    *describes* the defect. A detector that cannot tell code from prose will
    either cry wolf or be silenced, and a silenced detector finds nothing.
    """
    findings: list[dict] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            findings.append(
                {
                    "file": str(path.relative_to(REPO)).replace("\\", "/"),
                    "line": exc.lineno or 0,
                    "reason": f"could not parse: {exc.msg}",
                    "source": "",
                }
            )
            continue
        lines = source.splitlines()
        rel = str(path.relative_to(REPO)).replace("\\", "/")

        def record(node: ast.AST, why: str, *, rel: str = rel, lines: list[str] = lines) -> None:
            # Loop variables bound as defaults: the closure is called within
            # this iteration, but binding them explicitly keeps that true if
            # the call ever moves.
            lineno = getattr(node, "lineno", 0)
            findings.append(
                {
                    "file": rel,
                    "line": lineno,
                    "reason": why,
                    "source": lines[lineno - 1].strip()[:160] if 0 < lineno <= len(lines) else "",
                }
            )

        for node in ast.walk(tree):
            # `status.startswith("exact")` / `status.value.endswith(...)`
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"startswith", "endswith"}
                and _is_axis_expression(node.func.value)
            ):
                record(node, f"{node.func.attr} on a status axis")
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                operator, comparator = node.ops[0], node.comparators[0]
                # `"derived" in status`
                if (
                    isinstance(operator, ast.In)
                    and isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and _is_axis_expression(comparator)
                ):
                    record(node, "substring test against a status axis")
                # `status == "false_route"`. Against a *literal* only: comparing
                # against a variable is a filter, not a classification, and the
                # difference is whether a status added later is handled or
                # silently falls through.
                elif (
                    isinstance(operator, ast.Eq | ast.NotEq)
                    and isinstance(comparator, ast.Constant)
                    and isinstance(comparator.value, str)
                    and _is_axis_expression(node.left)
                ):
                    record(node, "status axis compared to a string literal")
    return sorted(findings, key=lambda hit: (hit["file"], hit["line"]))


def build_inventory() -> dict:
    from rh_research_engine import __version__

    vocabularies = collect_vocabularies()
    axis_fields = collect_axis_fields()
    substring = collect_substring_classifiers()
    return {
        "inventory_version": "1",
        "package_version": __version__,
        "summary": {
            "vocabulary_count": len(vocabularies),
            "total_members": sum(v["member_count"] for v in vocabularies.values()),
            "models_carrying_axes": len(axis_fields),
            "substring_classifiers": len(substring),
        },
        "vocabularies": vocabularies,
        "axis_fields": axis_fields,
        "substring_classifiers": substring,
    }


def render(inventory: dict) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def describe_drift(before: dict, after: dict) -> list[str]:
    """Name the vocabulary, the member, the model, and the file:line that moved.

    A migration of the status contract is exactly when "something changed" is
    least useful. The reader needs to know that `KnowledgeStatus` lost
    `false_route`, or that a substring classifier reappeared in
    `extract.py:257`, not that the inventory differs.
    """
    lines: list[str] = []

    b_vocab, a_vocab = before.get("vocabularies", {}), after.get("vocabularies", {})
    for name in sorted(set(b_vocab) - set(a_vocab)):
        lines.append(f"  REMOVED vocabulary {name}")
        lines.append(f"      had members: {b_vocab[name]['members']}")
    for name in sorted(set(a_vocab) - set(b_vocab)):
        lines.append(f"  ADDED   vocabulary {name}")
        lines.append(f"      members: {a_vocab[name]['members']}")
        if ".contracts." not in name:
            lines.append(
                "      NOTE: this vocabulary lives outside contracts/, which is what "
                "the contract layer exists to prevent. Move it or map it (ADR-001)."
            )
    for name in sorted(set(b_vocab) & set(a_vocab)):
        old, new = b_vocab[name]["members"], a_vocab[name]["members"]
        if old == new:
            continue
        gone, added = [m for m in old if m not in new], [m for m in new if m not in old]
        lines.append(f"  CHANGED vocabulary {name}  ({len(old)} -> {len(new)} members)")
        if gone:
            lines.append(f"      removed members: {gone}")
            lines.append("      NOTE: removing a value orphans every record still carrying it.")
        if added:
            lines.append(f"      added members:   {added}")
            lines.append("      NOTE: a new value needs an explicit mapping entry, or it fails closed.")

    b_axes, a_axes = before.get("axis_fields", {}), after.get("axis_fields", {})
    for name in sorted(set(b_axes) - set(a_axes)):
        lines.append(f"  REMOVED axis-carrying model {name}  (axes: {b_axes[name]['axes']})")
    for name in sorted(set(a_axes) - set(b_axes)):
        lines.append(f"  ADDED   axis-carrying model {name}  (axes: {a_axes[name]['axes']})")
    for name in sorted(set(b_axes) & set(a_axes)):
        old_axes, new_axes = b_axes[name]["axes"], a_axes[name]["axes"]
        if old_axes == new_axes:
            continue
        lines.append(f"  CHANGED axes on {name}")
        for field in sorted(set(old_axes) - set(new_axes)):
            lines.append(f"      dropped axis field: {field} ({old_axes[field]})")
        for field in sorted(set(new_axes) - set(old_axes)):
            lines.append(f"      added axis field:   {field} ({new_axes[field]})")

    def _key(hit: dict) -> tuple:
        return (hit["file"], hit["line"], hit["reason"])

    b_sub = {_key(h): h for h in before.get("substring_classifiers", [])}
    a_sub = {_key(h): h for h in after.get("substring_classifiers", [])}
    for key in sorted(set(a_sub) - set(b_sub)):
        hit = a_sub[key]
        lines.append(f"  NEW substring classifier  {hit['file']}:{hit['line']}")
        lines.append(f"      {hit['reason']}")
        lines.append(f"      {hit['source']}")
        lines.append(
            "      This classifies an epistemic question by spelling. Replace it "
            "with an explicit mapping (ADR-001)."
        )
    for key in sorted(set(b_sub) - set(a_sub)):
        hit = b_sub[key]
        lines.append(f"  RESOLVED substring classifier  {hit['file']}:{hit['line']}  ({hit['reason']})")

    if not lines:
        lines.append("  (counts differ but no structural change was identified; diff the JSON)")
    return lines


def summarize(inventory: dict) -> str:
    lines = ["Vocabularies:"]
    for key, value in inventory["vocabularies"].items():
        short = key.replace(f"{PACKAGE}.", "")
        lines.append(f"  {short:<52} {value['member_count']:>3} members")
    lines.append("")
    lines.append(f"Models carrying an axis field: {inventory['summary']['models_carrying_axes']}")
    lines.append("")
    substring = inventory["substring_classifiers"]
    if substring:
        lines.append(f"Substring classifiers ({len(substring)}) -- classify by spelling:")
        for hit in substring:
            lines.append(f"  {hit['file']}:{hit['line']}  {hit['reason']}")
            lines.append(f"      {hit['source']}")
    else:
        lines.append("Substring classifiers: none (all classification is explicit)")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print", dest="show", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    inventory = build_inventory()
    text = render(inventory)

    if args.show:
        print(summarize(inventory))
        return 0

    if args.check:
        if not args.out.exists():
            print(f"no inventory at {args.out}; run without --check first", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") == text:
            print(
                f"status inventory unchanged: {inventory['summary']['vocabulary_count']} "
                f"vocabularies, {inventory['summary']['substring_classifiers']} substring classifiers"
            )
            return 0
        stored = json.loads(args.out.read_text(encoding="utf-8"))
        print("status inventory DRIFTED from the recorded baseline.", file=sys.stderr)
        for line in describe_drift(stored, inventory):
            print(line, file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "If the change is intended, re-run without --check and commit the "
            "inventory alongside it.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="")
    print(f"wrote {args.out}")
    print(
        f"  {inventory['summary']['vocabulary_count']} vocabularies, "
        f"{inventory['summary']['total_members']} members, "
        f"{inventory['summary']['models_carrying_axes']} models carrying axes, "
        f"{inventory['summary']['substring_classifiers']} substring classifiers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
