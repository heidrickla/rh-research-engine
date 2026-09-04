from __future__ import annotations

from ..symbolic import domain_conditions
from .extract import stable_property_id
from .models import EpistemicStatus, MathObject, PropertyKind, PropertyRecord, Provenance


def propagate_singularities(
    objects: list[MathObject], *, unanalysed: list[str] | None = None
) -> list[PropertyRecord]:
    """Singularity conditions for each object whose expression can be analysed.

    An object whose domain analysis fails yields NO record. It used to yield one
    with `value="unknown"` and `status=BLOCKED`, which asserts nothing: 471 of
    481 properties in a real graph were that record, identical in every field
    but the object they pointed at. A property record says something about an
    object; "unknown" is the absence of one, and burying the ten real properties
    under it made the graph unreadable.

    Failures are reported through ``unanalysed`` rather than dropped, so a
    caller that wants the count can still have it.
    """
    out: list[PropertyRecord] = []
    failures = unanalysed if unanalysed is not None else []
    for obj in objects:
        if not obj.expression:
            continue
        try:
            conditions = domain_conditions(obj.expression)
        except Exception as exc:
            failures.append(f"{obj.name}: {type(exc).__name__}")
            continue
        for condition in conditions:
            prov = Provenance(
                source_type="symbolic",
                source_id=obj.id,
                method="singularity-propagation",
            )
            out.append(
                PropertyRecord(
                    id=stable_property_id(obj.id, PropertyKind.SINGULARITY, condition, obj.id),
                    object_id=obj.id,
                    kind=PropertyKind.SINGULARITY,
                    value=condition.replace(" != 0", " = 0"),
                    status=EpistemicStatus.SYMBOLIC_DERIVED,
                    provenance=[prov],
                    conditions=[condition],
                )
            )
    return out
