from __future__ import annotations

import hashlib
import re

from ..core.knowledge import KnowledgeItem
from ..symbolic.formula_index import FormulaRecord
from .models import MathObject, ObjectKind, Provenance

_KNOWN_OBJECTS: dict[str, ObjectKind] = {
    "zeta": ObjectKind.FUNCTION,
    "Gamma": ObjectKind.FUNCTION,
    "Lambda": ObjectKind.FUNCTION,
    "R_q": ObjectKind.FUNCTION,
    "S_q": ObjectKind.FUNCTION,
    "L_q": ObjectKind.FUNCTION,
    "Theta": ObjectKind.PARAMETER,
    "rho": ObjectKind.PARAMETER,
}

_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:_[A-Za-z0-9]+)?")


def stable_object_id(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"obj:{digest}"


def object_inventory(
    formulas: list[FormulaRecord] | None = None,
    knowledge: list[KnowledgeItem] | None = None,
) -> list[MathObject]:
    seen: dict[str, MathObject] = {}

    def add(name: str, *, expression: str | None, kind: ObjectKind, provenance: Provenance) -> None:
        if name in {"O", "log", "sin", "cos", "exp"}:
            return
        oid = stable_object_id(name)
        current = seen.get(oid)
        if current is None:
            seen[oid] = MathObject(
                id=oid, name=name, kind=kind, expression=expression, provenance=[provenance]
            )
        else:
            current.provenance.append(provenance)

    for record in formulas or []:
        prov = Provenance(
            source_type="formula_index",
            source_id=record.id,
            source_ref=record.source,
            method="symbolic-token-inventory",
        )
        for name in sorted(_SYMBOL.findall(record.expression)):
            add(
                name,
                expression=record.expression,
                kind=_KNOWN_OBJECTS.get(name, ObjectKind.EXPRESSION),
                provenance=prov,
            )

    for item in knowledge or []:
        prov = Provenance(
            source_type="knowledge",
            source_id=item.id,
            source_ref=item.domain,
            method="knowledge-statement-inventory",
        )
        # `formulas` only. Running the symbol regex over `title` and
        # `statement` -- ordinary English -- minted a MathObject for every word
        # in them: 'rather', 'width', 'Schr' (a truncated "Schrodinger") all
        # became mathematical objects of kind CLAIM, and their "expression" was
        # the sentence they came from. That produced 558 objects from 42
        # records, 98% of them fabricated, and every downstream analysis then
        # ran on the fabrications.
        #
        # Same rule as `FormulaIndex.add_knowledge`: read the field that is
        # declared to contain formulas. A structural match against a fragment of
        # a sentence looks exactly like a result and is not one.
        #
        # Durable memory currently declares no formulas, so this yields nothing
        # from knowledge today. That is the honest state -- it makes the empty
        # `formulas` field visible instead of papering over it.
        for text in item.formulas:
            for name in sorted(_SYMBOL.findall(text)):
                add(
                    name,
                    expression=text,
                    kind=_KNOWN_OBJECTS.get(name, ObjectKind.CLAIM),
                    provenance=prov,
                )

    return sorted(seen.values(), key=lambda item: item.id)
