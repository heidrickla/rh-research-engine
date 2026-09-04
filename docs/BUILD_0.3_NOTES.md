# Build 0.3 Notes — Active Search Layer

Version 0.3 moves the project from measuring one correlation remainder to actively probing its structure.

## Added experiments

### Counterterm discovery
Fits only a deliberately small interpretable basis to the numerical screening remainder:

\[
R_q(X) \approx c_0 + c_1/\log X + c_2/(\log X)^2.
\]

The fit is a conjecture generator. It is never promoted to a theorem automatically.

### Local-window variance
Computes a numerical approximation to

\[
V_q(X;L)=\int_{\log X}^{\log X+L}|W_q(t)|^2\,dt.
\]

For fixed \(q\), subexponential growth is the target spectral criterion.

### Fourier arc diagnostics
Computes the power spectrum of the centered smoothed prime shell and splits it into a low-frequency and high-frequency band. This is exploratory only: a rigorous circle-method major-arc decomposition requires rational arcs and analytic estimates.

### Synthetic off-line-zero injection
Injects the exact Gamma-filter response of a hypothetical zero

\[
\rho=\tfrac12+\eta+i\gamma
\]

and tests recovery of the predicted energy-growth exponent \(2\eta\). This validates the detection pipeline without making claims about actual zeros.

## Research discipline

The engine distinguishes:

1. numerical pattern,
2. conjectured counterterm,
3. symbolically verified identity,
4. rigorous analytic bound,
5. formal proof.

Only levels 4–5 may improve the authoritative zero-free-strip score.
