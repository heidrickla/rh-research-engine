#!/usr/bin/env python3
"""Capture the current contract surface as a deterministic snapshot.

WHY. Phase 1 of the laboratory program unifies the artifact schemas and
migrates stored state. That is exactly the kind of change that can silently
drop a field, widen an enum, or relax a validator -- and the durable memory it
touches is the authoritative research record. A snapshot taken *before* the
migration turns "did we lose anything?" from a judgement call into a diff.

WHAT IS CAPTURED. Every Pydantic model reachable from the package, as JSON
Schema, plus every StrEnum with its exact members. Both are sorted, so the
snapshot is byte-stable across runs and machines and can be committed and
diffed like any other artifact.

WHAT IS NOT CAPTURED. Behaviour. A validator that still exists but stops
raising will not show up here, which is what the test suite is for. Read a
clean schema diff as "the shape held", never as "the guarantees held".

Usage:
  python tools/snapshot-schemas.py                 # write docs/contracts/
  python tools/snapshot-schemas.py --out DIR       # write elsewhere
  python tools/snapshot-schemas.py --check         # fail if the surface drifted
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from enum import StrEnum
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PACKAGE = "rh_research_engine"
DEFAULT_OUT = REPO / "docs" / "contracts"


def _iter_modules() -> list[str]:
    """Every module in the package, found by walking the source tree.

    Deliberately not `pkgutil.walk_packages`: `core/` and `math/` have no
    `__init__.py`, so they are implicit namespace packages and `walk_packages`
    skips them *silently*. That quietly dropped `ClaimStatus`, `EvidenceClass`,
    and `KnowledgeStatus` -- the three vocabularies this snapshot exists to
    protect -- and a snapshot missing them would have diffed clean while they
    changed underneath. A filesystem walk does not care about `__init__.py`.
    """
    root = REPO / "src" / PACKAGE
    names: list[str] = [PACKAGE]
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join([PACKAGE, *parts]) if parts else PACKAGE)
    return sorted(set(names))


def _collect() -> tuple[dict, dict, dict]:
    from pydantic import BaseModel

    models: dict[str, dict] = {}
    enums: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for module_name in _iter_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            # Recorded, not printed and forgotten. A module that fails to import
            # contributes nothing, so its models would otherwise show up as
            # "REMOVED" -- a true statement with a badly misleading reason.
            errors[module_name] = f"{type(exc).__name__}: {exc}"
            continue
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isclass(obj):
                continue
            # Only record a class in the module that defines it, so a
            # re-export does not look like a second, divergent contract.
            if obj.__module__ != module_name:
                continue
            qualified = f"{obj.__module__}.{name}"
            if issubclass(obj, BaseModel) and obj is not BaseModel:
                try:
                    models[qualified] = obj.model_json_schema()
                except Exception as exc:
                    models[qualified] = {"error": f"{type(exc).__name__}: {exc}"}
            elif issubclass(obj, StrEnum) and obj is not StrEnum:
                enums[qualified] = [member.value for member in obj]
    return models, enums, errors


def build_snapshot() -> dict:
    from rh_research_engine import __version__

    models, enums, errors = _collect()
    return {
        "snapshot_version": "1",
        "package_version": __version__,
        "model_count": len(models),
        "enum_count": len(enums),
        "import_errors": dict(sorted(errors.items())),
        "enums": {name: enums[name] for name in sorted(enums)},
        "models": {name: models[name] for name in sorted(models)},
    }


def render(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _model_fields(schema: dict) -> dict[str, str]:
    """Field name -> compact type description, for readable diffs."""
    out: dict[str, str] = {}
    for name, spec in (schema.get("properties") or {}).items():
        if "$ref" in spec:
            kind = spec["$ref"].rsplit("/", 1)[-1]
        elif "anyOf" in spec:
            parts = []
            for option in spec["anyOf"]:
                parts.append(option.get("type") or option.get("$ref", "?").rsplit("/", 1)[-1])
            kind = " | ".join(parts)
        else:
            kind = spec.get("type", "?")
        out[name] = kind
    return out


def describe_drift(before: dict, after: dict) -> list[str]:
    """Say exactly what moved, down to the enum member and model field.

    "The schema changed" is not actionable when the thing being validated is a
    migration of the authoritative research record. The caller needs to know
    that `KnowledgeStatus` lost `false_route`, not that something, somewhere,
    is different.
    """
    lines: list[str] = []

    b_enums, a_enums = before.get("enums", {}), after.get("enums", {})
    for name in sorted(set(b_enums) - set(a_enums)):
        lines.append(f"  REMOVED enum {name}")
        lines.append(f"      had {len(b_enums[name])} members: {b_enums[name]}")
    for name in sorted(set(a_enums) - set(b_enums)):
        lines.append(f"  ADDED   enum {name}")
        lines.append(f"      {len(a_enums[name])} members: {a_enums[name]}")
    for name in sorted(set(b_enums) & set(a_enums)):
        old, new = b_enums[name], a_enums[name]
        if old == new:
            continue
        lines.append(f"  CHANGED enum {name}  ({len(old)} -> {len(new)} members)")
        gone, added = [m for m in old if m not in new], [m for m in new if m not in old]
        if gone:
            lines.append(f"      removed members: {gone}")
        if added:
            lines.append(f"      added members:   {added}")
        if not gone and not added:
            lines.append(f"      reordered: {old} -> {new}")

    b_models, a_models = before.get("models", {}), after.get("models", {})
    for name in sorted(set(b_models) - set(a_models)):
        lines.append(f"  REMOVED model {name}")
    for name in sorted(set(a_models) - set(b_models)):
        lines.append(f"  ADDED   model {name}")
    for name in sorted(set(b_models) & set(a_models)):
        old_schema, new_schema = b_models[name], a_models[name]
        if old_schema == new_schema:
            continue
        lines.append(f"  CHANGED model {name}")
        old_fields, new_fields = _model_fields(old_schema), _model_fields(new_schema)
        for field in sorted(set(old_fields) - set(new_fields)):
            lines.append(f"      removed field: {field}: {old_fields[field]}")
        for field in sorted(set(new_fields) - set(old_fields)):
            lines.append(f"      added field:   {field}: {new_fields[field]}")
        for field in sorted(set(old_fields) & set(new_fields)):
            if old_fields[field] != new_fields[field]:
                lines.append(
                    f"      retyped field: {field}: {old_fields[field]} -> {new_fields[field]}"
                )
        old_required = set(old_schema.get("required") or [])
        new_required = set(new_schema.get("required") or [])
        for field in sorted(old_required - new_required):
            lines.append(f"      now optional:  {field}")
        for field in sorted(new_required - old_required):
            lines.append(f"      now required:  {field}")
        if old_fields == new_fields and old_required == new_required:
            lines.append("      constraints or defaults changed (see the JSON diff)")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against the stored snapshot and fail on any difference.",
    )
    args = parser.parse_args(argv)

    snapshot = build_snapshot()
    text = render(snapshot)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # An unimportable module contributes nothing, so the snapshot is incomplete
    # and any "unchanged" verdict is unreliable. Fail loudly with the actual
    # exception rather than reporting a clean surface over a partial scan.
    if snapshot["import_errors"]:
        print("cannot trust this snapshot: modules failed to import", file=sys.stderr)
        for module_name, reason in snapshot["import_errors"].items():
            print(f"  {module_name}: {reason}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Fix the import error first. Until then a missing model is "
            "indistinguishable from a deleted one.",
            file=sys.stderr,
        )
        return 1

    target = args.out / "schema-snapshot.json"
    seal = args.out / "schema-snapshot.json.sha256"

    if args.check:
        if not target.exists():
            print(f"no snapshot at {target}; run without --check first", file=sys.stderr)
            return 1
        stored = target.read_text(encoding="utf-8")
        if stored == text:
            print(
                f"schema surface unchanged: {snapshot['model_count']} models, "
                f"{snapshot['enum_count']} enums"
            )
            return 0
        print("schema surface DRIFTED from the stored snapshot.", file=sys.stderr)
        stored_snapshot = json.loads(stored)
        for line in describe_drift(stored_snapshot, snapshot):
            print(line, file=sys.stderr)
        print("", file=sys.stderr)
        print("If the change is intended, re-run without --check and commit.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="")
    seal.write_text(digest + "\n", encoding="utf-8", newline="")
    print(f"wrote {target}")
    print(f"  {snapshot['model_count']} models, {snapshot['enum_count']} enums")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
