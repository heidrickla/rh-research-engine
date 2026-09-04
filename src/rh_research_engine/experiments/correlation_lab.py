from __future__ import annotations

from ..core.models import ExperimentResult
from ..math.correlation import shell_correlation_breakdown


def run(X: int = 20_000, q: float = 4.0, h_max: int | None = None) -> ExperimentResult:
    b = shell_correlation_breakdown(X=X, q=q, h_max=h_max)
    return ExperimentResult(
        name="correlation-lab",
        parameters={"X": X, "q": q, "h_max": h_max},
        metrics={
            "X": b.X,
            "q": b.q,
            "n_max": b.n_max,
            "h_max": b.h_max,
            "wave": b.wave,
            "total_energy": b.total_energy,
            "diagonal_energy": b.diagonal_energy,
            "actual_offdiag": b.actual_offdiag,
            "hl_model_offdiag": b.model_offdiag,
            "hl_model_energy": b.hl_model_energy,
            "screening_remainder": b.screening_remainder,
            "diag_coeff_exact": b.diagonal_coefficient,
            "diag_coeff_q_over_4": b.asymptotic_diagonal_coefficient,
        },
        observations=[
            "Diagnostic decomposition of the localized prime shell into diagonal noise, actual off-diagonal covariance, and Hardy--Littlewood model covariance.",
            "No rigorous bound on the screening remainder is inferred from a numerical value.",
        ],
    )
