from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class KnowledgeIntegrityError(ValueError):
    """Raised when durable memory cannot be trusted as loaded."""


class KnowledgeStatus(StrEnum):
    """Closed vocabulary for durable-memory entries.

    Deliberately absent: ``proved``, ``verified``, ``theorem``. Durable memory
    records what the research program has established *about routes*; it is not
    a place where a claim can acquire proof status by being written down. An
    entry whose status is not in this list is quarantined rather than loaded.
    """

    EXACT = "exact"
    EXACT_ALGEBRA = "exact_algebra"
    EXACT_ALGEBRA_IN_MODEL = "exact_algebra_in_model"
    EXACT_CALCULUS = "exact_calculus"
    EXACT_CONSTRUCTION = "exact_construction"
    EXACT_DISTRIBUTIONAL = "exact_distributional"
    DERIVED_SYMBOLIC = "derived_symbolic"
    DERIVED_SYMBOLIC_NEEDS_EXTERNAL_CHECK = "derived_symbolic_needs_external_check"
    DERIVED_FROM_ABSCISSA = "derived_from_abscissa"
    DERIVED_FROM_STANDARD_EXPONENT_RELATION = "derived_from_standard_exponent_relation"
    ASYMPTOTIC_DERIVED = "asymptotic_derived"
    KNOWN = "known"
    KNOWN_FRAMEWORK = "known_framework"
    KNOWN_EQUIVALENCE_FRAMEWORK = "known_equivalence_framework"
    KNOWN_MODEL_INPUT = "known_model_input"
    KNOWN_OR_STANDARD_CONSEQUENCE = "known_or_standard_consequence"
    CLASSICAL_FAMILY_REPACKAGED = "classical_family_repackaged"
    CONDITIONAL_ON_RH_STANDARD = "conditional_on_RH_standard"
    RESEARCH_TARGET = "research_target"
    FALSE_ROUTE = "false_route"
    GOVERNANCE = "governance"


#: Statuses that mark a route as refuted. These must survive every reload:
#: losing them silently is how a permanently dead route gets rediscovered.
NO_GO_STATUSES = frozenset({KnowledgeStatus.FALSE_ROUTE})


class KnowledgeItem(BaseModel):
    id: str
    title: str
    status: KnowledgeStatus
    domain: str
    statement: str
    formulas: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    notes: str = ""


class QuarantinedItem(BaseModel):
    id: str
    reason: str
    raw: dict


#: Where durable memory lived before the authoritative-state layout.
LEGACY_KNOWLEDGE_PATH = Path("research_state/math_knowledge.json")

#: Where it lives now.
CANONICAL_KNOWLEDGE_PATH = Path("research_state/authoritative/knowledge/math_knowledge.json")


def resolve_knowledge_path(
    *,
    canonical: Path = CANONICAL_KNOWLEDGE_PATH,
    legacy: Path | None = None,
) -> Path:
    """Resolve durable memory to the canonical path.

    The relocation is complete: the legacy copy has been deleted and there is
    no default fallback to it. Passing ``legacy`` explicitly still works, so a
    rollback or a comparison against an archived copy can be performed
    deliberately -- but no ordinary read silently reaches for an old path that
    is supposed to be gone.

    When both are supplied they must be identical. While two copies exist,
    "which one is the research record?" has no defensible answer, and picking
    one silently would let an edit to the abandoned copy vanish or an edit to
    the live copy be shadowed.
    """
    has_canonical = canonical.exists()
    has_legacy = legacy is not None and legacy.exists()
    if not has_canonical and not has_legacy:
        legacy_note = f" nor {legacy}" if legacy is not None else ""
        # Durable memory is the authoritative research record. Returning a
        # non-existent path let every caller degrade to "zero records", which
        # reads as a clean empty knowledge base rather than as a missing one --
        # and an empty base silently satisfies every no-go check.
        raise KnowledgeIntegrityError(
            f"durable memory is missing: {canonical}{legacy_note} does not exist. "
            "Restore it from version control; an absent research record is not an "
            "empty one."
        )
    if has_canonical and has_legacy:
        canonical_hash = hashlib.sha256(canonical.read_bytes()).hexdigest()
        legacy_hash = hashlib.sha256(legacy.read_bytes()).hexdigest()
        if canonical_hash != legacy_hash:
            raise KnowledgeIntegrityError(
                "durable memory exists at two paths with different contents, so "
                "there is no defensible way to choose one:\n"
                f"  {canonical}  sha256 {canonical_hash}\n"
                f"  {legacy}  sha256 {legacy_hash}\n"
                "Reconcile them deliberately, then delete the copy you are not "
                "keeping. During the staged relocation both must stay identical."
            )
        return canonical
    if has_canonical:
        return canonical
    if has_legacy:
        return legacy
    return canonical


class KnowledgeBase:
    def __init__(self, path: Path | None = None, *, allow_missing: bool = False) -> None:
        """Durable memory, read from `path` or resolved to the canonical copy.

        `allow_missing` exists for the one legitimate case -- bootstrapping a
        fresh state directory -- and defaults to False everywhere else. An
        explicit path used to bypass the resolver's fail-closed behaviour and
        return an empty list for a file that was not there, so a caller pointing
        at the wrong path got a clean empty knowledge base instead of an error.
        """
        self.path = resolve_knowledge_path() if path is None else path
        self.allow_missing = allow_missing

    @property
    def seal_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".sha256")

    # -- integrity ---------------------------------------------------------

    def _read_document(self) -> list[dict]:
        """Read and integrity-check the raw document.

        There is no corruption-recovery path here by design. A previous version
        tolerated a trailing ``]``/``}`` suffix via ``raw_decode``, which parses
        the longest valid *prefix* -- so a file truncated at an item boundary
        and then closed loaded as complete, silently dropping every entry after
        the cut, including the ``false_route`` records. Trailing bytes are now a
        hard error, and a sidecar checksum catches truncation that still
        happens to be syntactically valid.
        """
        raw = self.path.read_bytes()
        self._verify_seal(raw)
        text = raw.decode("utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KnowledgeIntegrityError(
                f"{self.path} is not valid JSON: {exc}. Durable memory is never "
                "auto-repaired; restore it from version control."
            ) from exc
        if not isinstance(data, list):
            raise KnowledgeIntegrityError("knowledge document must be a JSON array")
        return data

    def _verify_seal(self, raw: bytes) -> None:
        if not self.seal_path.exists():
            if self.path == CANONICAL_KNOWLEDGE_PATH:
                # The canonical copy is the research record. An absent seal
                # there is not "unsealed yet" -- it is the one state in which
                # truncation and tampering go undetected, so it fails closed.
                raise KnowledgeIntegrityError(
                    f"{self.path} has no seal at {self.seal_path.name}. The "
                    "authoritative copy must be sealed; without it a truncated or "
                    "edited file loads silently. Run `rhre knowledge seal`."
                )
            return
        expected = self.seal_path.read_text(encoding="utf-8").split()[0].strip()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise KnowledgeIntegrityError(
                f"{self.path} does not match its seal ({self.seal_path.name}).\n"
                f"  expected {expected}\n  actual   {actual}\n"
                "The file has been truncated, edited, or line-ending-mangled. "
                "Restore it, or re-seal deliberately with `rhre knowledge seal`."
            )

    def seal(self) -> str:
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.seal_path.write_text(digest + "\n", encoding="utf-8", newline="")
        return digest

    # -- loading -----------------------------------------------------------

    def load(self) -> list[KnowledgeItem]:
        """Load durable memory, refusing anything outside the closed vocabulary."""
        items, quarantined = self.load_with_quarantine()
        if quarantined:
            detail = "; ".join(f"{q.id}: {q.reason}" for q in quarantined[:5])
            raise KnowledgeIntegrityError(
                f"{len(quarantined)} durable-memory item(s) failed validation: {detail}"
            )
        return items

    def load_with_quarantine(self) -> tuple[list[KnowledgeItem], list[QuarantinedItem]]:
        """Load, separating valid items from rejected ones instead of raising.

        Used by reporting tools that need to show *what* is wrong rather than
        stop at the first problem.
        """
        if not self.path.exists():
            if not self.allow_missing:
                raise KnowledgeIntegrityError(
                    f"durable memory is missing at {self.path}. An absent research "
                    "record is not an empty one: returning zero records would read as "
                    "a clean knowledge base and silently satisfy every no-go check. "
                    "Pass allow_missing=True only when bootstrapping fresh state."
                )
            return [], []
        items: list[KnowledgeItem] = []
        quarantined: list[QuarantinedItem] = []
        for raw_item in self._read_document():
            if not isinstance(raw_item, dict):
                quarantined.append(
                    QuarantinedItem(id="<malformed>", reason="entry is not an object", raw={})
                )
                continue
            try:
                items.append(KnowledgeItem.model_validate(raw_item))
            except Exception as exc:
                quarantined.append(
                    QuarantinedItem(
                        id=str(raw_item.get("id", "<no id>")),
                        reason=_short_reason(exc, raw_item),
                        raw=raw_item,
                    )
                )
        return items, quarantined

    def get(self, item_id: str) -> KnowledgeItem | None:
        wanted = item_id.casefold()
        return next((item for item in self.load() if item.id.casefold() == wanted), None)

    def search(self, query: str) -> list[KnowledgeItem]:
        tokens = [token.casefold() for token in query.split() if token.strip()]
        if not tokens:
            return self.load()
        found: list[KnowledgeItem] = []
        for item in self.load():
            haystack = " ".join(
                [
                    item.id,
                    item.title,
                    item.status.value,
                    item.domain,
                    item.statement,
                    " ".join(item.formulas),
                    item.notes,
                ]
            ).casefold()
            if all(token in haystack for token in tokens):
                found.append(item)
        return found

    def semantic_hashes(self) -> dict[str, str]:
        """Per-record content hash, keyed by ID.

        Byte equality proves a *file* was copied faithfully. This proves each
        *record* survived: a reformatting migration will change the file hash
        by design, and this is what still has to hold across it.
        """
        out: dict[str, str] = {}
        for item in self.load():
            payload = json.dumps(
                item.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            out[item.id] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return dict(sorted(out.items()))

    def dependency_edges(self) -> list[tuple[str, str]]:
        """Every (record, dependency) pair, sorted."""
        return sorted(
            (item.id, dependency)
            for item in self.load()
            for dependency in item.dependencies
        )

    def validate_dependencies(self) -> list[str]:
        items = self.load()
        ids = {item.id for item in items}
        errors: list[str] = []
        for item in items:
            for dependency in item.dependencies:
                if dependency not in ids:
                    errors.append(f"{item.id}: unknown dependency {dependency}")
        return errors

    def audit(self) -> list[str]:
        """Every integrity problem, as human-readable lines. Never raises."""
        problems: list[str] = []
        try:
            items, quarantined = self.load_with_quarantine()
        except KnowledgeIntegrityError as exc:
            return [f"integrity: {exc}"]
        for q in quarantined:
            problems.append(f"quarantined {q.id}: {q.reason}")
        ids = {item.id for item in items}
        for item in items:
            for dependency in item.dependencies:
                if dependency not in ids:
                    problems.append(f"{item.id}: unknown dependency {dependency}")
        if not self.seal_path.exists():
            problems.append(
                f"no integrity seal at {self.seal_path.name}; run `rhre knowledge seal`"
            )
        return problems


def _short_reason(exc: Exception, raw_item: dict) -> str:
    status = raw_item.get("status")
    if status is not None and status not in set(KnowledgeStatus):
        return (
            f"status {status!r} is outside the closed vocabulary "
            "(durable memory cannot confer proof status)"
        )
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
