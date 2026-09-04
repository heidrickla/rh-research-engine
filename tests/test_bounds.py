import pytest

from rh_research_engine.core.bounds import correlation_remainder_to_theta


def test_zero_remainder_is_rh_endpoint_only_when_rigorous():
    x = correlation_remainder_to_theta(0.0, rigorous=True)
    assert x.theta_upper == 0.5
    assert x.rh_endpoint


def test_fitted_zero_remainder_is_not_an_rh_endpoint_claim():
    """A fitted slope of 0 is a diagnostic, not a proved estimate."""
    x = correlation_remainder_to_theta(0.0)
    assert x.theta_upper == 0.5
    assert not x.rh_endpoint
    assert not x.rigorous


def test_positive_remainder_translates_to_strip():
    x = correlation_remainder_to_theta(0.2, rigorous=True)
    assert abs(x.theta_upper - 0.6) < 1e-12
    assert not x.rh_endpoint


@pytest.mark.parametrize("bad", [-0.3, -1e-9, float("nan")])
def test_out_of_domain_exponent_is_rejected_not_clamped(bad):
    """Clamping rewrote noise-driven negative fits into exactly the RH endpoint."""
    with pytest.raises(ValueError):
        correlation_remainder_to_theta(bad)
