# Build 0.2 Notes — Correlation Laboratory

## Added

- FFT autocorrelation engine for centered von-Mangoldt shell data.
- Hardy--Littlewood prime-pair singular-series sieve.
- Actual/model screening decomposition.
- Geometric multi-scale correlation scan.
- Algebraic exponent-to-zero-strip scorer.
- Unit tests for autocorrelation, singular series, shell kernel, and bound translation.

## Benchmark diagnostic

For `q=4` over `X=2,000..20,000`, the localized point-wave energy was small and highly oscillatory, while the leading Hardy--Littlewood model left an approximately scale-stable lower-order counterterm around `O(1)` (about 1.4 in this finite experiment).

This does **not** certify any asymptotic exponent. It suggests that the next experiment should model the lower-order counterterm after universal `A_q log X` screening rather than repeatedly rediscover the leading cancellation.

## Next build

1. Counterterm discovery/regression in an arithmetic symbolic basis.
2. Exact local-window (not point-only) variance runner.
3. Major/minor arc spectral decomposition of the same kernel.
4. Synthetic off-line-zero injection to verify exponent recovery.
5. Candidate-bound promotion gate requiring a symbolic/proof artifact before updating `implied_theta_upper`.
