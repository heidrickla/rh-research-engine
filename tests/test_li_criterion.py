"""Li's lambda_n, checked against the one value with a closed form.

`lambda_1 = 1 + gamma/2 - log(4 pi)/2` exactly, which is the only anchor that
does not come from someone's table. That matters here: validating against
recalled values for lambda_2 and lambda_4 reported MISMATCH against a
computation whose Arb radius was 1e-120. The computation was right and the
remembered constants were wrong.

The rest is checked by construction rather than by authority -- the
coefficients must not depend on the truncation order or the working precision,
and the series must reproduce zeta's pole structure at s = 1.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rh_research_engine.experiments.li_criterion import (
    archimedean,
    li_coefficients,
    run,
)

pytest.importorskip("flint", reason="Li coefficients need Arb for certified balls")


def test_lambda_one_matches_its_closed_form():
    """The only value here with an independent exact expression."""
    import flint

    previous = flint.ctx.prec
    flint.ctx.prec = 300
    try:
        exact = (
            flint.arb(1)
            + flint.arb.const_euler() / 2
            - (flint.arb(4) * flint.arb.pi()).log() / 2
        )
        expected = float(exact.mid())
    finally:
        flint.ctx.prec = previous

    _, value, _ = li_coefficients(8, 400)[0]
    assert abs(value - expected) < 1e-15, (value, expected)


def test_the_coefficients_do_not_depend_on_order_or_precision():
    """A series truncated too early, or computed too coarsely, would drift.

    This is the check that caught `flint.ctx.cap`: python-flint truncates every
    series at that global, which defaults to 10, so an order-300 request
    silently returned ten coefficients while the timings rose with precision
    and made it look like work was happening.
    """
    small = li_coefficients(12, 400)
    large = li_coefficients(40, 1200)
    assert len(small) == 11, f"order 12 must give 11 coefficients, got {len(small)}"
    assert len(large) == 39
    for (n, a, _), (m, b, _) in zip(small, large[: len(small)], strict=True):
        assert n == m
        assert abs(a - b) < 1e-14, (n, a, b)


def test_every_computed_lambda_is_positive_and_certified():
    result = run(order=40, bits=800)
    assert result.metrics["negative_count"] == 0
    assert result.metrics["violated"] == 0.0
    assert result.metrics["min_lambda"] > 0
    assert result.metrics["max_radius"] < 1e-30, "the balls must be far tighter than the values"


def test_a_negative_coefficient_would_be_reported(monkeypatch):
    """Positivity is the whole criterion, so the report must be able to say no."""
    from rh_research_engine.experiments import li_criterion

    real = li_criterion.li_coefficients

    def with_one_negative(order, bits):
        rows = real(order, bits)
        n, value, radius = rows[5]
        return rows[:5] + [(n, -abs(value), radius)] + rows[6:]

    monkeypatch.setattr(li_criterion, "li_coefficients", with_one_negative)
    result = li_criterion.run(order=20, bits=600)
    assert result.metrics["negative_count"] == 1
    assert result.metrics["violated"] == 1.0
    assert result.metrics["min_lambda"] < 0


def test_the_archimedean_term_is_most_of_lambda_n():
    """Which is why this criterion is a weak falsification test at reachable n.

    If the smooth part were a small fraction, positivity would be a sharp
    question. It is not: by n = 39 it is already the overwhelming majority, and
    it contains no arithmetic at all.
    """
    rows = li_coefficients(40, 800)
    n = np.array([k for k, _, _ in rows], dtype=float)
    lam = np.array([v for _, v, _ in rows])
    fraction = archimedean(n)[-1] / lam[-1]
    assert 0.9 < fraction < 1.1, fraction
    # And the residual is small next to the value it would have to overturn.
    assert abs(lam[-1] - archimedean(n)[-1]) < 0.2 * lam[-1]


def test_the_archimedean_term_is_the_stated_formula():
    """Written down once, so a later edit cannot quietly change what is subtracted."""
    n = np.array([10.0, 100.0])
    expected = (n / 2) * (np.log(n) - math.log(2 * math.pi) + 0.5772156649015329 - 1) + 1
    assert np.allclose(archimedean(n), expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# The verdict used to come from a ball MIDPOINT.
#
# `li_coefficients` returns `(n, lambda_n, radius)` from one Arb ball, and `run`
# counted negative midpoints while the radius sat in the metrics unread.
# Measured: `--order 120 --bits 64` gives 48 negative midpoints whose balls have
# radius up to 6e+24, so the experiment recorded `violated: 1.0` -- a claimed
# refutation of the Riemann hypothesis -- from enclosures covering everything.
# It failed the other way too: at 128 bits it recorded a clean `violated: 0.0`
# with 14 balls straddling zero, having established nothing.
# ---------------------------------------------------------------------------


def test_a_refutation_must_be_proved_not_merely_negative():
    """A negative midpoint inside a ball that covers zero is UNRESOLVED.

    This is the case that recorded `violated: 1.0` before the repair. Forced from
    a hand-built tuple rather than by hunting for parameters that happen to be
    singular on the machine the suite runs on.
    """
    from rh_research_engine.experiments.li_criterion import classify

    verdict, refuted, straddling = classify([(1, 0.023, 1e-30), (2, -22.9, 2.17e9)])
    assert verdict == "UNRESOLVED", (verdict, refuted, straddling)
    assert refuted == 0
    assert straddling == 1


def test_a_ball_wholly_below_zero_is_a_refutation():
    """The other branch. `lambda + radius < 0` is the whole enclosure below zero,
    which is the only thing that could honestly refute RH here."""
    from rh_research_engine.experiments.li_criterion import classify

    verdict, refuted, _ = classify([(1, 0.023, 1e-30), (2, -5.0, 0.5)])
    assert verdict == "REFUTED", verdict
    assert refuted == 1


def test_every_ball_strictly_positive_is_positive():
    from rh_research_engine.experiments.li_criterion import classify

    assert classify([(1, 0.023, 1e-30), (2, 1.5, 1e-20)])[0] == "POSITIVE"


def test_low_precision_is_unresolved_and_never_a_refutation():
    """The real computation at a precision the CLI accepts.

    `--order 120 --bits 64` is reachable from the command line. Measured: 48
    negative midpoints, radii to 6e+24, 69 balls straddling zero. It must come
    back UNRESOLVED and it must NOT come back REFUTED.
    """
    from rh_research_engine.experiments.li_criterion import classify

    rows = li_coefficients(120, 64)
    verdict, refuted, straddling = classify(rows)
    assert verdict == "UNRESOLVED", (verdict, refuted, straddling)
    assert refuted == 0, "a refutation of RH was claimed from rounding"
    assert straddling > 0


def test_a_clean_null_at_insufficient_precision_is_also_refused():
    """The quieter half, and the one more likely to go unnoticed.

    At 128 bits no midpoint is negative, so the old code recorded
    `violated: 0.0` -- which reads as a falsification test that did not fire.
    Fourteen balls straddle zero there, so it had established nothing either way.
    A null from rounding is as wrong as a refutation from rounding.
    """
    from rh_research_engine.experiments.li_criterion import classify

    rows = li_coefficients(120, 128)
    assert all(value >= 0 for _, value, _ in rows), "no midpoint should be negative here"
    verdict, _refuted, straddling = classify(rows)
    assert verdict == "UNRESOLVED", verdict
    assert straddling > 0


def test_sufficient_precision_scales_with_the_order():
    """Measured, not asserted: roughly `bits >= 1.4 * order`.

    order 40 is safe from 64 bits, 80 from 128, and 120 from 160 -- all three
    values the module's observations quote, so none of them is claimed without
    being pinned here.

    AND THE THRESHOLD IS PINNED FROM BELOW, which is the half that makes it a
    threshold. That order 120 is POSITIVE at 160 bits shows only that 160 is
    enough; it says nothing about 160 being NEEDED, and a test that cannot fail
    when the requirement drops is not measuring a requirement. One step below,
    at 128 bits, 14 balls straddle zero and the verdict must be UNRESOLVED.
    """
    from rh_research_engine.experiments.li_criterion import classify

    assert classify(li_coefficients(40, 64))[0] == "POSITIVE"
    assert classify(li_coefficients(80, 128))[0] == "POSITIVE"
    assert classify(li_coefficients(120, 160))[0] == "POSITIVE"
    # Below the rule, the verdict must refuse rather than quietly succeed.
    assert classify(li_coefficients(120, 128))[0] == "UNRESOLVED"
    assert classify(li_coefficients(80, 64))[0] == "UNRESOLVED"
