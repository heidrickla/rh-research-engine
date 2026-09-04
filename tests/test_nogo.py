from rh_research_engine.core.models import Claim
from rh_research_engine.core.nogo import violations


def test_boundary_unitarity_nogo():
    claim = Claim(id="x", statement="bad", tags={"boundary_unitarity_only"})
    assert any(v.id == "boundary-unitarity" for v in violations(claim))
