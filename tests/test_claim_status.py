"""`known` means somebody else proved it, and now has to say where.

`ClaimStatus` had no class for reasoning that is neither measured nor read from
a source, so an inference went to `known` -- the status that maps to
`Confidence.KNOWN`, defined as "established external mathematics, needs a
literature citation", and the one `ArtifactRecord` refuses to let any worker
assert about its own output. The guard existed; it was not on the path a claim
takes.

Flagged by the x2plus1-research session, which hit the same gap adapting this
repo's pattern and added the class it was missing.
"""

from __future__ import annotations

import pytest

from rh_research_engine.contracts.epistemic import Confidence
from rh_research_engine.contracts.mappings import (
    CLAIM_STATUS_TO_CONFIDENCE,
    CLAIM_STATUS_TO_ROLE,
    confidence_from_claim_status,
    role_from_claim_status,
)
from rh_research_engine.core.bootstrap import seed_claims
from rh_research_engine.core.models import Claim, ClaimStatus


def test_known_without_a_citation_is_refused():
    """The status promised literature the model had nowhere to hold."""
    with pytest.raises(ValueError, match="no citation"):
        Claim(id="X1", statement="Somebody proved this.", status=ClaimStatus.KNOWN)
    with pytest.raises(ValueError, match="no citation"):
        Claim(
            id="X2",
            statement="Somebody proved this.",
            status=ClaimStatus.KNOWN,
            citation="   ",
        )
    cited = Claim(
        id="X3",
        statement="Somebody proved this.",
        status=ClaimStatus.KNOWN,
        citation="Titchmarsh, The Theory of the Riemann Zeta-Function, 2nd ed., Thm 9.2",
    )
    assert cited.status is ClaimStatus.KNOWN


def test_evidence_is_not_a_citation():
    """They answer different questions: what went in, versus where it is proved.

    C006 had an evidence line and no citation, and the evidence was true --
    which is why nothing looked wrong.
    """
    with pytest.raises(ValueError, match="no citation"):
        Claim(
            id="X4",
            statement="Therefore the leading term cancels.",
            status=ClaimStatus.KNOWN,
            evidence=["The weighted average has a universal -1/2 log H secondary term."],
        )


def test_an_inference_needs_no_citation_and_claims_no_force():
    """It may rest on cited inputs and still be a conclusion drawn here."""
    inferred = Claim(
        id="X5",
        statement="Therefore the leading term cancels.",
        status=ClaimStatus.INFERRED,
        evidence=["The weighted average has a universal -1/2 log H secondary term."],
    )
    assert inferred.status is ClaimStatus.INFERRED
    assert confidence_from_claim_status(ClaimStatus.INFERRED) is Confidence.HEURISTIC
    assert confidence_from_claim_status(ClaimStatus.KNOWN) is Confidence.KNOWN


def test_every_status_is_mapped():
    """The tables raise on an unmapped value, so a new status must be placed."""
    for status in ClaimStatus:
        assert status in CLAIM_STATUS_TO_CONFIDENCE, status
        assert status in CLAIM_STATUS_TO_ROLE, status
        assert confidence_from_claim_status(status) is not None
        assert role_from_claim_status(status) is not None


def test_the_bootstrap_registry_carries_no_uncited_known_claim():
    """C006 was the one, and it is the reason this file exists."""
    claims = seed_claims()
    known = [c for c in claims if c.status is ClaimStatus.KNOWN]
    assert all(c.citation.strip() for c in known), [c.id for c in known]
    c006 = next(c for c in claims if c.id == "C006")
    assert c006.status is ClaimStatus.INFERRED
