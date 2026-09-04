"""The external comparison, and the refusals around data that is not ours.

The tables are not vendored -- they are somebody else's published data -- so
the comparison itself skips unless a directory is supplied through
`RHRE_ODLYZKO_DIR`. What does NOT skip is everything about how the module
behaves when the data is absent or broken, because that is the branch a
checkout without the files always takes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.symbolic.odlyzko import (
    HIGH_TABLES,
    SOURCE,
    OdlyzkoComparison,
    compare,
    read_table,
)

_DATA = os.environ.get("RHRE_ODLYZKO_DIR")
needs_tables = pytest.mark.skipif(
    not (_DATA and Path(_DATA).exists()),
    reason=f"set RHRE_ODLYZKO_DIR to a directory of tables from {SOURCE}",
)


# --- absent and broken data -----------------------------------------------


def test_absent_tables_raise_and_say_where_to_get_them(tmp_path):
    """No data, no comparison. The URL travels with the refusal.

    A thinner check under the same name would be worse than none: a reader
    seeing `odlyzko-check` pass would take it to mean the external comparison
    ran.
    """
    with pytest.raises(FileNotFoundError) as caught:
        compare(tmp_path)
    message = str(caught.value)
    assert "zeros1" in message
    assert SOURCE in message


def test_a_truncated_table_is_refused_rather_than_measured(tmp_path):
    """A partial download is a broken input, not a small sample.

    Measured as a small sample it would produce a large noise floor and a
    residual inside it -- which reads as "no effect here", the most misleading
    thing a truncated file could say.
    """
    stub = tmp_path / "zeros3"
    stub.write_text("\n".join(str(value / 10) for value in range(50)), encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        read_table(stub)
    assert "too few" in str(caught.value)
    assert SOURCE in str(caught.value)


def test_the_prose_header_is_skipped_not_parsed(tmp_path):
    """Every table opens with a paragraph describing its base and accuracy."""
    table = tmp_path / "zeros3"
    table.write_text(
        "Values of gamma - 267653395647, where gamma runs over the heights\n"
        "of the zeros numbered 10^12 + 1 through 10^12 + 10^4.\n\n"
        + "\n".join(f"     {value / 1000:.10f}" for value in range(2000)),
        encoding="utf-8",
    )
    values = read_table(table)
    assert len(values) == 2000
    assert values[0] == 0.0


def test_the_high_tables_carry_their_base_and_index():
    """The bases are the difference between a spacing and a rounding error.

    `zeros5` holds offsets from 1.37e21 because the ordinates themselves have
    twenty-two digits before the point; a plain decimal would lose the part
    that varies. Spacings are differences of offsets, so the base cancels --
    but it is still needed for the mean density, and a wrong one would scale
    every spacing.
    """
    assert set(HIGH_TABLES) == {"zeros3", "zeros4", "zeros5"}
    for name, (base, label) in HIGH_TABLES.items():
        assert base > 1e11, name
        assert label.startswith("10^")
    assert HIGH_TABLES["zeros5"][0] > HIGH_TABLES["zeros4"][0]


# --- what the record refuses ----------------------------------------------


@pytest.mark.parametrize("confidence", sorted(RIGOROUS, key=str))
def test_the_comparison_cannot_claim_a_rigorous_confidence(confidence):
    """Two finite samples agreeing, about a limit neither reaches."""
    with pytest.raises(ValidationError) as caught:
        OdlyzkoComparison(confidence=confidence)
    assert "limit neither reaches" in str(caught.value)


def test_shape_is_gone_needs_every_index_to_be_consistent_with_none():
    """One index short of zero is not "the shape is gone".

    The verdict is a conjunction, and it has to be: a correction that had died
    at 10^12 and returned at 10^22 would be a much more interesting finding
    than the one being reported, and averaging would hide it.
    """
    gone = OdlyzkoComparison(
        indices=["10^12", "10^22"],
        surviving_fraction=[0.01, 0.14],
        surviving_uncertainty=[0.155, 0.155],
    )
    assert gone.shape_is_gone

    surviving = OdlyzkoComparison(
        indices=["10^12", "10^22"],
        surviving_fraction=[0.01, 0.80],
        surviving_uncertainty=[0.155, 0.155],
    )
    assert not surviving.shape_is_gone

    assert not OdlyzkoComparison().shape_is_gone, "no data is not a verdict"


# --- the comparison itself ------------------------------------------------


@needs_tables
def test_our_ordinates_match_his_to_his_stated_accuracy():
    """The check that rules this engine's zero-finder in or out.

    Held to his stated accuracy rather than to a tolerance chosen afterwards:
    3e-9 is what his file claims, and agreeing to it is the strongest thing
    the comparison can say.
    """
    result = compare(Path(_DATA))
    assert result.ordinate_agreement <= result.stated_accuracy * 1.05
    assert result.residual_correlation > 0.999, (
        "the residual must not depend on whose zeros it is computed from"
    )


@needs_tables
def test_the_low_height_shape_does_not_survive_to_the_high_tables():
    """The finite-height reading, confirmed by data this engine cannot produce.

    Every index consistent with none of the shape remaining. Reported as a
    bound rather than an absence: 10000 zeros resolve the surviving amplitude
    to about a third, and no better.
    """
    result = compare(Path(_DATA))
    assert result.shape_is_gone
    for fraction, uncertainty in zip(
        result.surviving_fraction, result.surviving_uncertainty, strict=True
    ):
        assert abs(fraction) < 2 * uncertainty
    assert max(result.surviving_uncertainty) > 0.1, (
        "the bound is loose, and a test claiming otherwise would overstate it"
    )
    assert result.confidence is Confidence.NUMERICAL
