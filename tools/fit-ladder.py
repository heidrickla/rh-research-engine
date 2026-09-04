"""Regress fitted `l` on true `l` across the T < 10^6 bands AND the ladder rungs.

WHY A LADDER AT ALL. `recover_height` reads `l` off the pair correlation band by
band, and the lever it has is the span of `l` those bands cover. Below
`T = 10^6` that span is 8.92..11.98 and the top of it is the noisy end. The
rungs at `l = 12.5, 13.0, 13.5` are built by `build-ladder.py` at heights a full
run cannot reach, which is the whole reason `zeros_in_band` exists.

WHAT IT BUYS, AND IT IS LESS THAN IT LOOKS. The three rungs came back with
sigma 0.61, 0.60 and 1.13 against 0.19 for the best low band, so the weighted
regression discounts them by a factor of about 36. Distant bands are
intrinsically uninformative at a fixed zero count: the model predicted its own
experiment would disappoint and it did. HEIGHT IS THE WRONG AXIS -- more zeros
at moderate `l` buys more per hour than the same hours spent climbing.

Run `build-ladder.py` first; this reads what it wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from rh_research_engine.symbolic import pair_correlation as pc  # noqa: E402
from rh_research_engine.symbolic.height_recovery import (  # noqa: E402
    UNCERTAINTY_CHUNKS,
    UNCERTAINTY_ROTATIONS,
    _weighted_line,
    curve_bank,
    ell_grid,
    fit_ell,
    narrow_ell_bands,
)

EDGES = np.linspace(0, 3, 31)
PRIMES = 10_000


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: fit-ladder.py <zeros.npy> <ladder.npz>")
    zeros_path, ladder_path = Path(sys.argv[1]), Path(sys.argv[2])

    grid = ell_grid()
    bank = curve_bank(EDGES, grid, prime_limit=PRIMES)

    def fit(unfolded: np.ndarray) -> float:
        density = pc.measured_density(unfolded, window=3.0, bins=30).density
        return fit_ell(density, EDGES, grid=grid, prime_limit=PRIMES, bank=bank)

    def sigma(
        unfolded: np.ndarray,
        chunks: int = UNCERTAINTY_CHUNKS,
        rotations: int = UNCERTAINTY_ROTATIONS,
    ) -> float:
        """Averaged over where the cuts fall: one placement carries 25-28%."""
        size = len(unfolded) // chunks
        step = max(1, size // max(1, rotations))
        draws = []
        for turn in range(max(1, rotations)):
            rolled = np.roll(unfolded, turn * step) if turn else unfolded
            values = [fit(rolled[i * size : (i + 1) * size]) for i in range(chunks)]
            draws.append(float(np.std(values, ddof=1) / np.sqrt(chunks)))
        return float(np.mean(draws))

    rows: list[tuple[float, float, float, str]] = []
    for piece in narrow_ell_bands(np.load(zeros_path)):
        unfolded = pc.unfold(piece)
        rows.append(
            (float(np.log(piece / (2 * np.pi)).mean()), fit(unfolded), sigma(unfolded), "T<1e6")
        )
    ladder = np.load(ladder_path)
    for key in ladder.files:
        piece = np.sort(ladder[key])
        unfolded = pc.unfold(piece)
        rows.append(
            (float(np.log(piece / (2 * np.pi)).mean()), fit(unfolded), sigma(unfolded), "ladder")
        )
    rows.sort()

    print(f"{'true l':>8} {'fitted':>8} {'sigma':>7} {'diff':>8}  source")
    for true_ell, fitted, error, source in rows:
        print(f"{true_ell:8.3f} {fitted:8.3f} {error:7.3f} {fitted - true_ell:+8.3f}  {source}")

    true_ell = np.array([r[0] for r in rows])
    fitted = np.array([r[1] for r in rows])
    errors = np.array([r[2] for r in rows])
    errors = np.maximum(errors, errors[errors > 0].min())
    slope, _, error = _weighted_line(true_ell, fitted, errors)

    print()
    print(f"  l spans [{true_ell.min():.2f}, {true_ell.max():.2f}]  ({len(true_ell)} bands)")
    print(
        f"  slope {slope:.3f} +/- {error:.3f}   "
        f"({abs(slope - 1) / error:.1f}s from 1, {abs(slope) / error:.1f}s from 0)"
    )


if __name__ == "__main__":
    main()
