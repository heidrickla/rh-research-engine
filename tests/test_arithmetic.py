import math

from rh_research_engine.math.arithmetic import von_mangoldt_sieve


def test_von_mangoldt():
    lam = von_mangoldt_sieve(10)
    assert math.isclose(lam[2], math.log(2))
    assert math.isclose(lam[4], math.log(2))
    assert math.isclose(lam[8], math.log(2))
    assert math.isclose(lam[3], math.log(3))
    assert lam[6] == 0.0
