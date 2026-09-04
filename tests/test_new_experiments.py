from rh_research_engine.experiments import (
    arc_diagnostics,
    counterterm_discovery,
    local_variance,
    synthetic_zero,
)


def test_synthetic_zero_recovers_eta():
    result = synthetic_zero.run(eta=0.03, gamma=14.0, q=20.0, T_max=100.0, points=1200)
    assert abs(result.metrics["recovered_eta"] - 0.03) < 0.005


def test_local_variance_positive():
    result = local_variance.run(X=500, q=3.0, width=0.2, samples=9)
    assert result.metrics["window_energy"] >= 0.0


def test_arc_power_partition():
    result = arc_diagnostics.run(X=500, q=3.0, fft_size=4096, major_width=0.05)
    total = result.metrics["total_fourier_power"]
    low = result.metrics["low_frequency_power"]
    high = result.metrics["high_frequency_power"]
    assert abs(total - (low + high)) < 1e-9 * max(1.0, total)


def test_counterterm_fit_runs():
    # points=4 here originally, which a 3-column basis leaves one degree of
    # freedom -- now refused rather than fitted. See test_counterterm_discovery.
    result = counterterm_discovery.run(X_min=300, X_max=900, points=6, q=3.0)
    assert result.metrics["samples"] >= 3
    assert result.metrics["fit_rmse"] >= 0.0
    assert result.metrics["degrees_of_freedom"] >= 3
