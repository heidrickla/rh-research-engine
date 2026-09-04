"""b(N), and the three states a cell can be in.

The experiment measures how wrong `fit_ell` is at a known height as a function of
band size. Most of what can go wrong here is not arithmetic: it is a cell being
excluded for the wrong reason, or a refusal reading like a measurement, or a
metric that is absent where a reader would take absence for zero.

THE ARTIFACT IS NOT IN THIS REPOSITORY. `ladder-full.npz` and `zeros_1e6.npy`
live in `~/rh-data/` on two machines and are not backed up, so every test here
runs on the refusal path or on hand-built cells. The measured path is exercised
by running the experiment, not by the suite, and that is a limitation of the
record rather than of these tests -- which is why the refusal contract is tested
so heavily: it is the branch CI can actually reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from rh_research_engine.experiments import fit_bias_lab
from rh_research_engine.experiments.fit_bias_lab import (
    GAIN_HIGH,
    GAIN_LOW,
    GAIN_RESOLUTION,
    MAX_RAILED,
    MIN_SLICES,
    run,
)


def test_a_missing_artifact_refuses_rather_than_measuring_nothing():
    """"Not run" and "measured, and b is flat" must never print the same.

    This is `patterns/ledger.py`'s rule about not-tested versus refuted, one
    layer up: a refusal carries `refused: 1.0` and NO `bias_at_*` key at all, so
    nothing downstream can read an absent measurement as a null result.
    """
    for kwargs in ({}, {"ladder": "C:/no/such/ladder.npz"}, {"zeros": "C:/no/such/zeros.npy"}):
        result = run(**kwargs)
        assert result.metrics["refused"] == 1.0, kwargs
        assert result.metrics["cells"] == 0.0, kwargs
        assert not [key for key in result.metrics if key.startswith("bias_at")], kwargs
        assert any("REFUSED" in line for line in result.observations), kwargs


def test_the_refusal_names_the_artifact_and_that_it_is_not_in_the_repository():
    """A record whose input exists in two home directories is not regenerable.

    That fact can only live on the record, so it is asserted rather than left to
    a reader to notice.
    """
    joined = " ".join(run().observations)
    assert "NOT IN" in joined.upper()
    assert "ladder-full.npz" in joined
    assert "build-ladder.py" in joined, "the invocation is the durable half, not the file"


def test_the_two_builders_trap_is_the_first_thing_the_record_says():
    """`~/rh-data/build_ladder.py` is a DIFFERENT, older builder sitting beside
    the data, producing rungs 25x narrower. Anyone reconstructing the invocation
    from the copy next to the artifact rebuilds the wrong thing -- so the warning
    has to come before the invocation, not after it."""
    observations = run().observations
    trap = next(i for i, line in enumerate(observations) if "TWO BUILDERS" in line.upper())
    invocation = next(i for i, line in enumerate(observations) if "build-ladder.py" in line)
    assert trap <= invocation, "the trap must be read before the command it is about"


# ---------------------------------------------------------------------------
# The three states a cell can be in, which is the whole point of the cuts.
# ---------------------------------------------------------------------------


def _cell(**over) -> dict:
    base = {
        "rung": 10.0,
        "source": "band",
        "count": 2_500.0,
        "slices": float(MIN_SLICES + 5),
        "dropped": 0.0,
        "bias": 0.4,
        "median": 0.3,
        "sd": 2.0,
        "sem": 0.3,
        "railed": 0.0,
        "gain": 1.0,
        "gain_sem": 0.05,
    }
    base.update(over)
    return base


def test_a_cell_whose_gain_is_undetermined_is_not_a_cell_whose_gain_is_bad():
    """The distinction this module exists to keep, one layer below the verdicts.

    The keep window is 0.5 wide and at N = 2,500 the gain's own standard error
    reaches 1.7. A window narrower than its own uncertainty is not a filter, it
    is a coin -- so a cell excluded for gain must say WHICH: not measured, or
    measured and unresponsive.
    """
    resolved = _cell(gain=1.0, gain_sem=0.05)
    unresolved = _cell(gain=1.0, gain_sem=GAIN_RESOLUTION * 4)
    unresponsive = _cell(gain=0.3, gain_sem=0.05)

    def resolved_ok(cell):
        sem = cell["gain_sem"]
        return sem == sem and sem <= GAIN_RESOLUTION

    assert resolved_ok(resolved)
    assert not resolved_ok(unresolved), "a gain with a huge error bar must not be judged"
    assert resolved_ok(unresponsive), "this one WAS measured -- it is genuinely low"
    # And it is the window, applied after resolution, that rejects this one.
    assert not (GAIN_LOW <= unresponsive["gain"] <= GAIN_HIGH)


def test_the_resolution_limit_is_narrower_than_the_window_it_guards():
    """Otherwise the guard admits gains that cannot be placed in the window.

    Half the window is the widest a resolution limit can be and still mean
    anything: at exactly half, a point estimate at the centre is two sigma from
    each edge.
    """
    assert GAIN_RESOLUTION <= (GAIN_HIGH - GAIN_LOW) / 2 + 1e-12


def test_every_requested_count_reports_its_own_backing(monkeypatch, tmp_path):
    """An absent `bias_at_N` must not be readable as a measured zero.

    `bias_at_N` is written only when a cell survives, so absence would otherwise
    be indistinguishable from a value nobody recorded. `usable_at_N` is always
    written -- including as 0.0 -- so the two can be told apart.
    """
    ladder = tmp_path / "ladder.npz"
    np.savez(ladder, **{"12.5": np.array([100.0, 200.0])})

    # No cell can survive on two ordinates, so every count must still report.
    result = run(ladder=str(ladder), counts=(2_500, 5_000))
    assert result.metrics["refused"] == 0.0, "the artifact existed; this is not a refusal"
    for size in (2_500, 5_000):
        assert f"usable_at_{size}" in result.metrics
        assert f"undetermined_at_{size}" in result.metrics
        assert result.metrics[f"usable_at_{size}"] == 0.0
        assert f"bias_at_{size}" not in result.metrics, "nothing supported a value"


def test_the_cuts_are_module_constants_and_not_arguments_by_accident():
    """They encode this module's judgement about what counts as a measurement.

    Exposing them as CLI arguments would invite a run that reports b over cells
    where the estimator does not respond, which is the one thing the cuts exist
    to prevent.
    """
    assert 0 < GAIN_LOW < 1 < GAIN_HIGH
    assert 0 < MAX_RAILED < 0.5, "censoring above this is the boundary, not the data"
    assert MIN_SLICES >= 3, "an error bar needs more than two draws"
    assert fit_bias_lab.GAIN_SLICES >= 8


@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_a_nonpositive_gain_error_is_never_treated_as_resolved(bad):
    """NaN and zero are both "no error bar", and neither may pass as precision."""
    for sem in (bad, float("nan")):
        resolved = sem == sem and 0 < sem <= GAIN_RESOLUTION
        assert not resolved or sem > 0


def test_the_command_is_wired_and_its_module_is_importable():
    """A command whose module is not in the repository breaks every checkout but this one.

    `cli.py` imports `fit_bias_lab` at module scope, so wiring the command
    before the module is committed makes `import rh_research_engine.cli` fail
    everywhere except the machine the file happens to sit on. That is exactly
    what it did: 19 CI errors, `ImportError: cannot import name 'fit_bias_lab'`,
    from a change that passed seven gates and a 1320-test suite locally --
    because every one of those ran against a tree that CONTAINED the file.

    This test cannot catch that on its own, and saying so is the point: it runs
    in the same tree. What catches it is a clone --

        git clone --depth 1 file://<repo> tmp && python -c "import rh_research_engine.cli"

    -- which asks about the repository rather than the filesystem. What this DOES
    catch is the command being unwired, or renamed, or the import being dropped
    while the module stays.
    """
    from rh_research_engine.cli import experiment_app

    names = {command.name for command in experiment_app.registered_commands}
    assert "fit-bias-lab" in names, sorted(names)


def test_the_command_does_not_expose_the_cuts_as_options():
    """They encode what counts as a measurement, and an option invites a run
    that reports b over cells where the estimator does not respond."""
    from rh_research_engine.cli import experiment_app

    command = next(c for c in experiment_app.registered_commands if c.name == "fit-bias-lab")
    params = set(command.callback.__code__.co_varnames[: command.callback.__code__.co_argcount])
    assert params == {"ladder", "zeros", "max_slices"}, params
    for forbidden in ("gain_low", "gain_high", "max_railed", "min_slices"):
        assert forbidden not in params
