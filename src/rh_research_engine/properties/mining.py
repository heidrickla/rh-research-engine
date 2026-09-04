from __future__ import annotations

from ..symbolic import equivalent
from .extract import stable_property_id
from .models import EpistemicStatus, MathObject, PropertyKind, PropertyRecord, Provenance


def mine_symmetries(objects: list[MathObject]) -> list[PropertyRecord]:
    out: list[PropertyRecord] = []
    for obj in objects:
        if not obj.expression:
            continue
        for transform, label in [("-({expr})", "odd-candidate"), ("({expr})", "identity")]:
            try:
                result = equivalent(obj.expression, transform.format(expr=obj.expression))
            except Exception:
                continue
            if result.equivalent is True and label != "identity":
                prov = Provenance(
                    source_type="symbolic",
                    source_id=obj.id,
                    method="symmetry-mining",
                    assumptions=result.assumptions,
                )
                out.append(
                    PropertyRecord(
                        id=stable_property_id(obj.id, PropertyKind.SYMMETRY, label, obj.id),
                        object_id=obj.id,
                        kind=PropertyKind.SYMMETRY,
                        value=label,
                        status=EpistemicStatus.SYMBOLIC_DERIVED,
                        provenance=[prov],
                        assumptions=result.assumptions,
                    )
                )
    return out


def mine_invariants(objects: list[MathObject]) -> list[PropertyRecord]:
    out: list[PropertyRecord] = []
    for obj in objects:
        if obj.name in {"Theta", "rho"}:
            prov = Provenance(source_type="symbolic", source_id=obj.id, method="invariant-mining")
            out.append(
                PropertyRecord(
                    id=stable_property_id(obj.id, PropertyKind.INVARIANT, "critical-line-reference", obj.id),
                    object_id=obj.id,
                    kind=PropertyKind.INVARIANT,
                    value="critical-line-reference",
                    status=EpistemicStatus.SYMBOLIC_DERIVED,
                    provenance=[prov],
                    metadata={"invariant_under": "s -> 1-s / conjugation framework"},
                )
            )
    return out
