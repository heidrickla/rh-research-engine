"""Build the high rungs of the l ladder, band by band.

`zero_ordinates` cannot reach these heights: its rescan allocates one grid over
every suspect interval at once, which is 11 GB at T = 4e6 and spent seven hours
swapping before it was killed. The ceiling is memory, not precision and not the
count check.

`zeros_in_band` is proportional to the band instead of to everything beneath
it, and each band is still verified against `ZeroCount(high) - ZeroCount(low)`.
So a rung is built as contiguous SUB-BANDS of about twenty thousand zeros:

  * every sub-band's rescan stays small, which is what failed at scale;
  * every sub-band is independently count-checked, so a missed zero is caught
    in the piece that lost it rather than diluted across the rung;
  * a sub-band that refuses can be reported and skipped without losing the
    rung -- but it is recorded, because a rung with a hole in it is not the
    rung it claims to be.

THE RUNGS ARE BUILT TO A WIDTH NOW, NOT TO A COUNT, and that was worth
changing. Built to 200,000 zeros they came out 0.020-0.060 wide in `l` against
the 0.5 cap, and their fitted `l` carried sigma 0.61, 0.60 and 1.13 against 0.19
for the best low band -- so the regression discounted them about 36x and the
whole climb bought +1.5 sigma. The constraint was never the height. It was that
"distant bands are uninformative AT FIXED ZERO COUNT", and the count was fixed
by this file.

Measured inside a single band, holding height nearly constant, `sigma ~ N^-0.399`
with 6% scatter. Filling a rung to the full 0.5 is 8-25x the zeros at the SAME
height, which takes those sigmas to 0.26, 0.21 and 0.32 -- level with the best
band this engine has. The earlier cross-band exponent of -0.293 was confounded:
those bands differ in height as well as count.

IN PARALLEL, because at full width this is millions of zeros per rung. Each
sub-band is an independent `zeros_in_band` call with its own count check, so
they distribute with nothing shared and nothing to merge but a sort.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "src")

from rh_research_engine.symbolic.riemann_siegel import zeros_in_band  # noqa: E402

SUB_BAND = 20_000

#: Full band width in `l`, matching `MAX_BAND_ELL_SPAN` in `height_recovery`.
#: A rung narrower than the bands it is regressed against is a rung throwing
#: away the zeros that would have made it count.
WIDTH = 0.5

WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))


def _one(bounds: tuple[float, float]) -> tuple[np.ndarray, str | None]:
    low, high = bounds
    try:
        return zeros_in_band(low, high), None
    except RuntimeError as failure:
        return np.array([]), f"[{low:.1f},{high:.1f}] refused: {failure}"


def build(ell: float, width: float = WIDTH) -> tuple[np.ndarray, list[str]]:
    """One rung: every zero within `width` of `ell`, in parallel sub-bands."""
    low, high = ell - width / 2, ell + width / 2
    expected = np.exp(ell) * ell * width
    count = max(1, int(expected // SUB_BAND))
    edges = np.linspace(low, high, count + 1)
    bounds = [
        (float(2 * np.pi * np.exp(edges[i])), float(2 * np.pi * np.exp(edges[i + 1])))
        for i in range(count)
    ]
    pieces, notes = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for zeros, note in pool.map(_one, bounds):
            if note:
                notes.append(note)
            elif len(zeros):
                pieces.append(zeros)
    if not pieces:
        return np.array([]), notes
    return np.sort(np.concatenate(pieces)), notes


DEFAULT_RUNGS = (12.5, 13.0, 13.5)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: build-ladder.py <out.npz> [l ...]")
    out = sys.argv[1]
    # The rungs were hardcoded, which is how a set of them ends up chosen once
    # and never revisited. The gap between the T < 10^6 bands (topping out at
    # l = 11.42) and the first rung at 12.52 is over two band widths wide, and
    # filling it is minutes of compute -- but only if the file lets you ask.
    wanted = tuple(float(value) for value in sys.argv[2:]) or DEFAULT_RUNGS
    rungs = {}
    # SAVE AFTER EVERY RUNG. The first version wrote only at the end, a
    # timeout killed it during the last rung, and thirty-eight minutes of
    # completed work went with it. A long computation that cannot be
    # interrupted without loss is a computation nobody will run twice.
    for ell in wanted:
        started = time.time()
        zeros, notes = build(ell)
        elapsed = time.time() - started
        if not len(zeros):
            print(f"  l={ell}: NOTHING -- {notes}", flush=True)
            continue
        got = float(np.log(zeros.mean() / (2 * np.pi)))
        span = float(
            np.log(zeros.max() / (2 * np.pi)) - np.log(zeros.min() / (2 * np.pi))
        )
        rungs[str(ell)] = zeros
        print(
            f"  l={ell:4.1f}: {len(zeros):7d} zeros in {elapsed:6.1f}s  "
            f"mean l {got:.4f}  span {span:.4f}  refusals {len(notes)}",
            flush=True,
        )
        for note in notes:
            print(f"      {note}", flush=True)
        np.savez_compressed(out, **rungs)
        print(f"      saved {len(rungs)} rung(s) so far", flush=True)
    np.savez_compressed(out, **rungs)
    print(f"  wrote {out} with {len(rungs)} rungs", flush=True)


if __name__ == "__main__":
    main()
