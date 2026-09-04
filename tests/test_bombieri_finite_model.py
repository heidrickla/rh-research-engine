"""Bombieri's two exact theorems, used as gates on our assembly of his matrix.

These are the only checks in this project with a published, proved answer behind
them. Everything else compares a number to another number we computed. So when
one of these fails the matrix is wrong, and it is worth keeping them sharp.
"""

from __future__ import annotations

import numpy as np
import pytest

from rh_research_engine.experiments.bombieri_finite_model import (
    FAKE_DISPLACEMENT,
    FAKE_HEIGHT,
    TRUNCATIONS,
    assert_lemma_10,
    kernel,
    negative_count,
    on_line,
    retention,
    run,
    spectrum,
    with_quadruple,
)


@pytest.fixture(scope="module")
def zeros():
    from rh_research_engine.symbolic.riemann_siegel import first_zero_ordinates

    return first_zero_ordinates(64)


def test_the_matrix_is_symmetric_and_real_for_real_ordinates(zeros):
    """(7.1) makes `(1/4+x^2)K*` symmetric, so `H` is. Real ordinates keep it real.

    Assembled from `K*` directly this would be neither, and every entry would
    need the kernel at complex argument `t(i/2 +/- x)`.
    """
    matrix = kernel(on_line(zeros, 12), 2.3)
    assert np.abs(matrix - matrix.T).max() < 1e-12 * np.abs(matrix).max()
    assert np.abs(matrix.imag).max() == 0.0


def test_lemma_10_no_negative_eigenvalue_for_real_distinct_ordinates(zeros):
    """The gate that catches a sign error inventing detections from an on-line set.

    Unconditional in `t`, so it is checked across the whole range the figures
    cover rather than at one convenient value.
    """
    for count in (5, 10, 20):
        for t in (0.5, 1.0, 2.3, 4.0, 5.5):
            assert assert_lemma_10(zeros, count, t) >= 0.0


def test_lemma_10_refuses_rather_than_returning_a_wrong_number(zeros):
    """Fail-closed, watched firing, through the module's own seam.

    Lemma 10 is a theorem, so no legitimate ordinate set reaches this branch --
    which is exactly why it needs forcing rather than leaving to chance. A first
    attempt fed a complex ordinate and asserted the raise; it did not raise, and
    the test only appeared to check something because it re-implemented the
    function's logic inline instead of calling it. Two defects, one of them the
    kind that makes a green suite meaningless.

    So the wrong matrix is supplied through `_assemble`, and the assertion is on
    the real function.
    """
    assert assert_lemma_10(zeros, 10, 2.3) >= 0.0

    def negated(gammas, t):
        return -kernel(gammas, t)

    with pytest.raises(ValueError, match="Lemma 10 violated"):
        assert_lemma_10(zeros, 10, 2.3, _assemble=negated)

    def not_real(gammas, t):
        return kernel(gammas, t) + 1j * np.eye(len(gammas))

    with pytest.raises(ValueError, match="not real"):
        assert_lemma_10(zeros, 10, 2.3, _assemble=not_real)


def test_lemma_10_second_half_a_repeated_ordinate_gives_a_zero_eigenvalue(zeros):
    """Eigenvalue 0 occurs IFF some gamma is repeated, with multiplicity `m - 1`.

    The half of Lemma 10 that is easy to leave untested, and the one that pins
    the normalisation: a matrix scaled by any constant still has no negative
    eigenvalue, but only the right one puts an exact zero here.
    """
    distinct = on_line(zeros, 8)
    repeated = np.concatenate([distinct, [distinct[2]]])
    values, _ = spectrum(repeated, 2.3)
    floor = len(values) * float(np.finfo(float).eps) * float(np.abs(values).max())
    assert abs(values[0]) <= floor, values[:3]
    assert abs(spectrum(distinct, 2.3)[0][0]) > floor


def test_theorem_8_one_quadruple_gives_exactly_two_negative_eigenvalues(zeros):
    """Exact and unconditional in `t` -- wherever float64 can resolve it."""
    for count in (5, 10, 20):
        for t in (1.0, 2.3, 4.0, 5.5):
            negatives, _, _ = negative_count(with_quadruple(zeros, count), t)
            assert negatives == 2, (count, t, negatives)


def test_precision_running_out_is_unresolved_and_not_a_refutation(zeros):
    """The §13 trap, reproduced inside the check written to reproduce it.

    At count 20, t 0.5 the two negative eigenvalues are near 1e-15 and 1e-18,
    below the float64 floor. Theorem 8 is unconditional, so a gate that called
    this a counterexample would be the not-tested-versus-refuted rule failing one
    level down. The missing negatives must be accounted for INSIDE the floor.
    """
    negatives, unresolved, floor = negative_count(with_quadruple(zeros, 20), 0.5)
    assert negatives < 2, "the premise of this test has changed"
    assert negatives + unresolved >= 2, (negatives, unresolved, floor)

    result = run(t=0.5, count=20, ordinates=zeros)
    assert "UNRESOLVED" in result.observations[0]
    assert "DISAGREES" not in result.observations[0]


def test_the_figures_plot_the_eigenvalue_not_its_reciprocal(zeros):
    """Settled by trend, which needs no axis scale -- and the scale is unreadable.

    §13 calls the plotted quantity `lambda` while (7.2) sets `Lambda = 1/lambda`.
    Fig. 1 grows in magnitude with `t`. So whichever of the two grows is the one
    plotted, and they cannot both.
    """
    low, _ = spectrum(with_quadruple(zeros, 10), 1.0)
    high, _ = spectrum(with_quadruple(zeros, 10), 5.5)
    assert abs(high[0]) > abs(low[0])
    assert abs(1.0 / high[0]) < abs(1.0 / low[0])


def test_a_smaller_displacement_needs_a_wider_support(zeros):
    """Monotonicity nothing in the assembly enforces, so it is evidence.

    A zero further off the line is easier to see, so at fixed support it should
    retain more of its eigenvalue under refinement. If this ever inverts, the
    thing being measured is not the physical one.
    """
    kept = [retention(zeros, 0.7, displacement=d)[0] for d in (0.005, 0.02, 0.1)]
    assert kept == sorted(kept), kept


def test_retention_is_reported_but_never_interpolated(zeros):
    """`t_c` is where the LIMIT changes character, not a corner in a finite curve.

    Suzuki Thm 1.3 gives the eigenvalue continuous in `t`, so no truncation has a
    kink to bisect for. The run must report the retained fraction and say so,
    rather than emitting a threshold that is really the crossing of a convention.
    """
    result = run(t=2.3, count=20, ordinates=zeros)
    assert "retained_at_largest_truncation" in result.metrics
    assert not any(k.startswith("t_c") or k == "threshold" for k in result.metrics)
    assert any("not a t_c estimator" in o.lower() or "NOT a" in o for o in result.observations)


def test_the_run_records_the_cutoff_and_finds_the_quadruple(zeros):
    result = run(t=2.3, count=20, ordinates=zeros)
    assert result.metrics["negative_eigenvalues"] == 2.0
    assert result.metrics["on_line_margin"] >= 0.0
    assert result.metrics["imaginary_drift"] < 1e-9
    assert result.metrics["prime_cutoff"] == pytest.approx(float(np.exp(2 * 2.3)))
    assert result.metrics["count"] == 20.0


def test_the_published_point_is_the_one_run(zeros):
    """Bombieri's own rho_0 = 0.52 + 3.14i, not a point chosen here.

    A control run at parameters we picked is not a control.
    """
    assert FAKE_HEIGHT == 3.14
    assert FAKE_DISPLACEMENT == 0.02
    gammas = with_quadruple(zeros, 5)
    added = gammas[-4:]
    assert set(np.round(added.real, 10)) == {3.14, -3.14}
    assert set(np.round(added.imag, 10)) == {0.02, -0.02}
    assert len(gammas) == 2 * 5 + 4


def test_the_truncation_sweep_is_a_sweep(zeros):
    """One truncation cannot address a question about the limit."""
    assert len(TRUNCATIONS) >= 3
    _, values = retention(zeros, 2.3)
    assert len(values) == len(TRUNCATIONS)


# --------------------------------------------------------------------------
# The certified route. Everything above is float64; these are the ball-arithmetic
# counterparts, and they exist because a count is what Theorem 8 asserts and a
# gate on the project's central claim should not rest on one solver succeeding.

flint = pytest.importorskip("flint", reason="ball arithmetic needs python-flint")

CERTIFIED_DIGITS = 60


@pytest.fixture
def certified():
    """`flint.ctx` precision is GLOBAL, so it is raised and put back.

    A fixture in `test_weil_certified` that raised it and left it there turned a
    25-minute suite into 47, because everything downstream got slower. Setting
    global state without restoring it makes later tests' cost depend on what ran
    before them.
    """
    saved = flint.ctx.dps
    flint.ctx.dps = CERTIFIED_DIGITS
    yield flint
    flint.ctx.dps = saved


def test_the_certified_ordinates_are_on_the_line_and_tight(certified):
    """Arb certifies `Re = 1/2` while isolating, so both arrive in one call.

    That matters because the ordinates are the uncertified input here, not the
    arithmetic: float64 carries ~1e-13 at gamma ~ 143 and this module's
    eigenvalues reach 1e-17.
    """
    from rh_research_engine.experiments.bombieri_finite_model import certified_ordinates

    gammas = certified_ordinates(8, certified)
    assert len(gammas) == 8
    assert float(gammas[0].real.mid()) == pytest.approx(14.134725141734693, abs=1e-14)
    for g in gammas:
        assert float(g.real.rad()) < 1e-25


def test_theorem_8_counted_without_an_eigensolver(certified):
    """The count is exact because the roots are real -- and it counts, not detects.

    Descartes' rule normally leaves an even deficiency; for an all-real spectrum
    it vanishes, and Lemmas 8-9 prove the spectrum is real. Three quadruples
    rather than one, because one tests whether a negative exists and three tests
    the COUNTING, which is what the theorem actually asserts. Distinct heights,
    since the theorem counts distinct conjugate pairs.
    """
    from rh_research_engine.experiments.bombieri_finite_model import (
        certified_kernel,
        negative_by_descartes,
    )

    for quadruples in (0, 1, 2, 3):
        matrix = certified_kernel(4, 2.3, certified, quadruples=quadruples)
        count, stuck = negative_by_descartes(matrix, certified)
        assert stuck is None, (quadruples, stuck)
        assert count == 2 * quadruples, (quadruples, count)


def test_the_two_certified_routes_agree_where_both_work(certified):
    """Different flint entry point, different algorithm, different failure mode.

    So agreement is evidence about the assembly rather than about one solver.
    """
    from rh_research_engine.experiments.bombieri_finite_model import (
        certified_kernel,
        certified_smallest,
        negative_by_descartes,
    )

    matrix = certified_kernel(6, 2.3, certified, quadruples=1)
    count, stuck = negative_by_descartes(matrix, certified)
    assert stuck is None and count == 2

    values = matrix.eig(algorithm="rump", nonstop=False)
    assert sum(1 for v in values if v.real < 0) == count
    assert float(certified_smallest(matrix, certified).mid()) < 0


def test_the_descartes_route_names_the_coefficient_it_cannot_sign(certified):
    """A refusal that says what is missing, not a count of zero.

    At 60 digits the determinant at size 44 is a ball straddling zero, so the
    count is genuinely unknown. `test_more_digits_resolve_what_sixty_could_not`
    is the other half: this is a COST, not a ceiling, and the companion test
    shows the same case PROVED at 150.

    An earlier version of this docstring claimed the limit was permanent and
    unaffected by precision. That came from a ladder which built its `arb`
    literals once at 60 digits and reused them at 400 and 1000, so the input ball
    never shrank and neither did the answer.
    """
    from rh_research_engine.experiments.bombieri_finite_model import (
        certified_kernel,
        negative_by_descartes,
    )

    count, stuck = negative_by_descartes(certified_kernel(20, 2.3, certified), certified)
    assert count is None, "the premise of this test has changed"
    assert stuck == 0, stuck

    # And it is not the cluster: at t = 2.3 nothing is near zero and `eig` works.
    matrix = certified_kernel(20, 2.3, certified)
    assert sum(1 for v in matrix.eig(algorithm="rump", nonstop=False) if v.real < 0) == 2


def test_the_reality_gate_refuses_rather_than_reporting_a_wide_ball(certified):
    """Non-real eigenvalue means the assembly is wrong, not imprecise.

    With a quadruple present `H` is complex symmetric but NOT Hermitian, so
    nothing structural forces a real spectrum -- Lemmas 8-9 do. An imaginary part
    EXCLUDING zero is Arb proving a contradiction with a theorem; one that is
    merely wide is a precision statement. Only the first may raise.
    """
    from rh_research_engine.experiments.bombieri_finite_model import (
        certified_kernel,
        certified_smallest,
    )

    matrix = certified_kernel(4, 2.3, certified)
    assert float(certified_smallest(matrix, certified).mid()) < 0

    broken = certified_kernel(4, 2.3, certified)
    for j in range(broken.nrows()):
        broken[j, 0] = broken[j, 0] + certified.acb(0, 1)
    with pytest.raises(ValueError, match="non-real eigenvalue"):
        certified_smallest(broken, certified)


def test_certified_reproduces_float64_where_float64_is_still_valid(certified, zeros):
    """Agreement to six figures at N = 5 and 10, and 8% apart at N = 20.

    The disagreement is the point: an ordinate-slop measurement had predicted
    that row was ~14% ordinate-limited BEFORE any certified value existed, and
    the certified run puts it at 8%. Two estimates of one boundary by different
    means. So this pins where float64 stops rather than asserting it never does.
    """
    from rh_research_engine.experiments.bombieri_finite_model import (
        certified_kernel,
        certified_smallest,
    )

    for count, tolerance in ((5, 1e-5), (10, 1e-5)):
        exact = float(certified_smallest(certified_kernel(count, 0.5, certified), certified).mid())
        rough = float(spectrum(with_quadruple(zeros, count), 0.5)[0][0])
        assert abs(rough - exact) / abs(exact) < tolerance, (count, rough, exact)

    exact = float(certified_smallest(certified_kernel(20, 0.5, certified), certified).mid())
    rough = float(spectrum(with_quadruple(zeros, 20), 0.5)[0][0])
    assert 0.01 < abs(rough - exact) / abs(exact) < 0.3, (rough, exact)


def test_more_digits_resolve_what_sixty_could_not(certified):
    """The determinant is a cost, not a ceiling -- and the radius must MOVE.

    A radius that does not move as precision rises is input-limited, not
    arithmetic-limited. That signature is the same one behind mpmath's `lerchphi`
    defect, whose error is stable across 300-500 digits so that raising precision
    reports convergence on a wrong value. It is also what made this module
    briefly record a permanent size limit that does not exist.

    So this asserts the resolution AND that `certified_kernel` builds its
    literals at the working precision -- if it ever hoists them to import time,
    the second rung stops resolving.
    """
    from rh_research_engine.experiments.bombieri_finite_model import (
        certified_kernel,
        negative_by_descartes,
    )

    certified.ctx.dps = 60
    count, stuck = negative_by_descartes(certified_kernel(20, 2.3, certified), certified)
    assert count is None and stuck == 0

    certified.ctx.dps = 150
    count, stuck = negative_by_descartes(certified_kernel(20, 2.3, certified), certified)
    assert stuck is None, "more digits must resolve it -- see the docstring"
    assert count == 2, count


def test_a_provably_complex_coefficient_indicts_the_assembly(certified):
    """UNRESOLVED and a broken matrix must not share a verdict.

    `P H P = conj(H)` for the conjugation permutation, so the charpoly is real.
    A coefficient whose imaginary part PROVABLY excludes zero is therefore a
    statement about the matrix, and reporting it as UNRESOLVED would send the
    next reader after precision that cannot help.
    """
    from rh_research_engine.experiments.bombieri_finite_model import (
        certified_kernel,
        negative_by_descartes,
    )

    good = certified_kernel(4, 2.3, certified)
    count, stuck = negative_by_descartes(good, certified)
    assert stuck is None and count == 2

    broken = certified_kernel(4, 2.3, certified)
    broken[0, 1] = broken[0, 1] + certified.acb(0, 1)
    with pytest.raises(ValueError, match="assembly is wrong"):
        negative_by_descartes(broken, certified)


def test_the_ratio_sequence_separates_the_two_branches(zeros):
    """Rising toward 1 is convergence; falling is not. No convention needed.

    The retention fraction over the same range of `t` is a smooth crossover --
    0.00, 0.39, 0.46, 0.73, 0.90, 0.95 -- with nothing to read but a line
    somebody draws. The ratios of successive truncations have structure instead,
    and this asserts that structure rather than a threshold.
    """
    from rh_research_engine.experiments.bombieri_finite_model import decay_ratios

    # Deep: the first rung falls hard, and the rest REFUSE rather than divide by
    # a value that is below the float64 floor. The first version of this test
    # asserted three small ratios and failed, which is how the defect surfaced:
    # the 20 -> 40 quotient was a real number over a rounding artifact and read
    # as 1.1e-02 against a certified 4.4e-08.
    deep = decay_ratios(zeros, 0.5)
    assert deep[0] is not None and deep[0] < 1e-3, deep
    assert deep[1:] == [None, None], deep

    for t in (0.9, 1.0, 1.5):
        shallow = decay_ratios(zeros, t)
        assert all(r is not None and 0.5 < r <= 1.0 for r in shallow), (t, shallow)
        # Rising toward 1 is convergence to a non-zero limit.
        assert shallow[-1] > shallow[1], (t, shallow)

    # And where it does speak, it agrees with the certified run: 0.68586 and
    # 0.84990 here against 0.6859 and 0.8499 in ball arithmetic at dps 150.
    middle = decay_ratios(zeros, 0.7)
    assert middle[1] == pytest.approx(0.6859, abs=1e-4), middle
    assert middle[2] == pytest.approx(0.8499, abs=1e-4), middle


def test_the_ratio_sequence_is_not_sold_as_a_proof(zeros):
    """Suzuki Thm 1.3 makes the eigenvalue continuous in `t`, so no finite N
    settles a statement about the limit. The docstring must say so, because the
    table in it is exactly the kind that reads like a located threshold.
    """
    from rh_research_engine.experiments.bombieri_finite_model import decay_ratios

    text = decay_ratios.__doc__
    assert "does not prove" in text.lower()
    assert "Suzuki" in text


def test_the_control_is_reachable_from_the_cli():
    """An experiment with no command is an experiment nobody runs.

    Every other experiment in this package has one; this module went several
    commits without, because `cli.py` held another session's uncommitted work
    and staging it would have swept their files into an unrelated commit -- the
    failure that broke `main` once already.
    """
    from rh_research_engine.cli import experiment_app

    command = next(
        c for c in experiment_app.registered_commands if c.name == "bombieri-finite-model"
    )
    assert command.callback.__doc__
    # The support and the truncation are both exposed: a control run at
    # parameters the caller cannot change is a control nobody can vary.
    names = command.callback.__code__.co_varnames[: command.callback.__code__.co_argcount]
    assert set(names) == {"t", "count"}, names
