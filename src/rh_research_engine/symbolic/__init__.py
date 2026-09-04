from .analysis import asymptotic, residue
from .assumptions import extract_assumptions
from .certified import check_certificate_against_expression
from .citations import Citation, SourceKind, knowledge_citation
from .conjecture import minimize_conjecture
from .counterterms import build_counterterm_ansatz, generate_counterterm_basis
from .decompose import search_decompositions
from .equivalence import canonicalize, domain_conditions, equivalent, fingerprint
from .exponents import safe_binomial_decay_to_theta, screening_remainder_to_theta
from .formula_index import FormulaIndex
from .ingest import ingest_file, ingest_text
from .lean import export_polynomial_identity
from .parser import extract_equations, parse_math
from .proof_gap import extract_proof_gaps
from .proof_queue import (
    ProofQueue,
    ProofQueueEntry,
    ProofQueueVerdict,
    build_proof_queue,
)
from .route_matcher import match_route
from .sanity import check_asymptotic, growth_exponent
from .simplify import simplify_with_trace
from .transforms import REGISTRY as TRANSFORM_REGISTRY

__all__ = [
    "Citation",
    "FormulaIndex",
    "ProofQueue",
    "ProofQueueEntry",
    "ProofQueueVerdict",
    "SourceKind",
    "TRANSFORM_REGISTRY",
    "asymptotic",
    "build_counterterm_ansatz",
    "build_proof_queue",
    "canonicalize",
    "domain_conditions",
    "check_asymptotic",
    "check_certificate_against_expression",
    "equivalent",
    "export_polynomial_identity",
    "extract_assumptions",
    "extract_equations",
    "extract_proof_gaps",
    "fingerprint",
    "generate_counterterm_basis",
    "growth_exponent",
    "knowledge_citation",
    "ingest_file",
    "ingest_text",
    "match_route",
    "minimize_conjecture",
    "parse_math",
    "residue",
    "safe_binomial_decay_to_theta",
    "screening_remainder_to_theta",
    "search_decompositions",
    "simplify_with_trace",
]
