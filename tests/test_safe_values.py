import math

from rh_research_engine.math.safe_values import implied_theta_from_decay


def test_decay_mapping():
    assert math.isclose(implied_theta_from_decay(0.75), 0.5)
    assert math.isclose(implied_theta_from_decay(2 / 3), 2 / 3)
