from __future__ import annotations

from .models import TransformResult


def mellin_exp_power(q: str = "q", s: str = "s") -> TransformResult:
    return TransformResult(transform="mellin", input_expression="exp(-u**q)", output_expression=f"gamma({s}/{q})/{q}", conditions=[f"Re({s}) > 0", f"{q} > 0"], rule_id="MELLIN-EXP-POWER")


def fourier_log_shell(q: str = "q", omega: str = "omega") -> TransformResult:
    return TransformResult(transform="fourier", input_expression="q*exp(q*u)*exp(-exp(q*u))", output_expression=f"gamma(1-I*{omega}/{q})", conditions=[f"{q} > 0"], rule_id="FOURIER-LOG-SHELL")


def laplace_monomial(alpha: str = "alpha", s: str = "s") -> TransformResult:
    return TransformResult(transform="laplace", input_expression=f"t**({alpha}-1)", output_expression=f"gamma({alpha})/{s}**{alpha}", conditions=[f"Re({alpha}) > 0", f"Re({s}) > 0"], rule_id="LAPLACE-MONOMIAL")


REGISTRY = {"mellin-exp-power": mellin_exp_power, "fourier-log-shell": fourier_log_shell, "laplace-monomial": laplace_monomial}
