import numpy as np

from rh_research_engine.math.correlation import (
    autocorrelation_fft,
    gamma_shell,
    singular_series_sieve,
)


def test_autocorrelation_fft_matches_direct():
    x = np.array([1.0, -2.0, 3.0, 0.5])
    got = autocorrelation_fft(x)
    want = np.array([np.dot(x[: len(x)-h], x[h:]) for h in range(len(x))])
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_singular_series_basic_values():
    s = singular_series_sieve(12)
    assert s[1] == 0.0
    assert s[2] > 1.0
    assert s[6] > s[2]  # factor p=3 increases the singular series


def test_gamma_shell_positive():
    vals = gamma_shell(np.array([0.2, 1.0, 2.0]), 3.0)
    assert np.all(vals > 0)
