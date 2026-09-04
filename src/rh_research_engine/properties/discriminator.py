from __future__ import annotations

from .models import DiscriminatorResult, EpistemicStatus, PropertyGraph, PropertyKind, Provenance


def analyze_discriminators(graph: PropertyGraph) -> list[DiscriminatorResult]:
    results: list[DiscriminatorResult] = []
    for prop in graph.properties:
        if prop.kind not in {PropertyKind.SYMMETRY, PropertyKind.INVARIANT, PropertyKind.GROWTH_BOUND}:
            continue
        status = (
            EpistemicStatus.RIGOROUS_DERIVED
            if prop.is_rigorous
            else EpistemicStatus.SYNTHETIC
        )
        prov = Provenance(
            source_type="synthetic",
            source_id=prop.id,
            method="critical-line-vs-off-line-discriminator",
            assumptions=["synthetic off-line zero model; cannot promote RH-equivalent claims"],
        )
        results.append(
            DiscriminatorResult(
                object_id=prop.object_id,
                property_id=prop.id,
                critical_line_value="rho = 1/2 + i gamma",
                off_line_value="rho = 1/2 + eta + i gamma",
                status=status,
                promoted_to_proof=False,
                reason=(
                    "property is a discriminator candidate only; numerical or synthetic "
                    "off-line behavior cannot prove or promote an RH-equivalent claim"
                ),
                provenance=[prov],
            )
        )
    return sorted(results, key=lambda item: (item.object_id, item.property_id))
