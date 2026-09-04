"""Test-session wiring.

Nothing here changes what is under test. The one fixture memoises a pure
function whose cost dominates the suite, so that a height computed by one test
is not computed again by the next.
"""

from __future__ import annotations

import numpy as np
import pytest

#: The module that tests the zero finder itself. It must reach the real
#: function, or its tests stop exercising the thing they are named for.
NOT_MEMOISED = {"test_riemann_siegel"}


@pytest.fixture(scope="session")
def _zeros_cache() -> dict:
    return {}


@pytest.fixture(autouse=True)
def _memoise_the_zeros(request, _zeros_cache):
    """`zero_ordinates` is pure and costs 62 s at `T = 2x10^5`. Compute once.

    It is the single largest cost in the suite -- `test_pair_correlation`,
    `test_level_spacing`, `test_spacing_decay`, `test_moments` and
    `test_symbolic` all want the same heights, and `T = 5000` alone is asked
    for eleven times. Nothing about the answer depends on how many times it is
    computed: the function takes a height, returns the ordinates below it, and
    verifies its own result against `ZeroCount` before returning. Measured, it
    takes the suite from 163 s to 134 s under `-n auto`.

    A COPY is handed out each time. Sharing the array itself would be shared
    mutable state between tests wearing the shape of a cache.

    INSTALLED PER TEST, NOT ONCE PER SESSION, and that is the whole design.
    A session-scoped patch would leave `test_riemann_siegel` exercising the
    real finder only because pytest imports test modules during collection,
    before session fixtures run, so its top-level `from ... import
    zero_ordinates` happens to bind the original. That is true today and it is
    an accident of ordering: import the module any later -- lazily, from
    another test, under a different runner -- and the module under test would
    silently be reading a cache. Swapping the attribute around each test
    instead makes the exclusion a fact about the fixture rather than about when
    something was imported.
    """
    from rh_research_engine.symbolic import riemann_siegel

    if request.module.__name__ in NOT_MEMOISED:
        yield
        return

    original = riemann_siegel.zero_ordinates

    def memoised(height, **kwargs):
        key = (float(height), tuple(sorted(kwargs.items())))
        if key not in _zeros_cache:
            _zeros_cache[key] = original(height, **kwargs)
        return _zeros_cache[key].copy()

    memoised.is_test_memo = True
    riemann_siegel.zero_ordinates = memoised
    try:
        yield
    finally:
        riemann_siegel.zero_ordinates = original


def is_memoised(function) -> bool:
    """Whether `function` is the test cache rather than the real thing."""
    return bool(getattr(function, "is_test_memo", False))


__all__ = ["NOT_MEMOISED", "is_memoised", "np"]
