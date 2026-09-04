from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from ..contracts.roles import META_ROLES
from ..core.knowledge import KnowledgeBase, KnowledgeStatus


class RouteMatch(BaseModel):
    knowledge_id: str
    title: str
    status: str
    domain: str
    score: float
    matched_terms: list[str] = Field(default_factory=list)
    action: str

    @property
    def is_no_go(self) -> bool:
        """Whether this match is a refuted route, by role rather than by name.

        The caller used to test `match.status == "false_route"`. That is the
        substring-classification defect in its narrowest form: it is correct
        today only because exactly one status happens to be spelled that way,
        and a second no-go status added later would be read as ordinary prior
        work with nothing failing.
        """
        from ..contracts.mappings import role_from_knowledge_status
        from ..contracts.roles import Role
        from ..core.knowledge import KnowledgeStatus

        return role_from_knowledge_status(KnowledgeStatus(self.status)) is Role.NO_GO


def _terms(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_]+", text.casefold())
    stop = {"the", "a", "an", "of", "and", "or", "to", "for", "is", "in", "with", "by", "on", "that", "this", "as"}
    return {word for word in words if len(word) > 2 and word not in stop}


def _action_for(status) -> str:
    """What to do about a match, from the canonical axes.

    This was `"equivalent" in item.status` -- classification by spelling, the
    same defect that promoted 14 of 21 statuses to rigorous. It also missed
    `conditional_on_RH_standard`, which is RH-equivalent and contains no
    "equivalent" substring, so an RH-conditional route was reported as ordinary
    prior work rather than a reformulation.
    """
    from ..contracts.mappings import (
        KNOWLEDGE_STATUS_RH_EQUIVALENT,
        role_from_knowledge_status,
    )
    from ..contracts.roles import Role

    role = role_from_knowledge_status(status)
    if role is Role.NO_GO:
        return "reject_or_require_new_distinguishing_assumption"
    if status in KNOWLEDGE_STATUS_RH_EQUIVALENT or role is Role.EQUIVALENCE:
        return "classify_as_reformulation_unless_proof_obligation_is_reduced"
    if status is KnowledgeStatus.RESEARCH_TARGET:
        return "link_to_existing_target"
    if role in META_ROLES:
        return "governance_record_not_a_mathematical_route"
    return "review_existing_result_before_claiming_novelty"


def match_route(
    text: str, *, knowledge_path: Path | None = None, limit: int = 8
) -> list[RouteMatch]:
    """Match a proposed route against durable memory.

    `knowledge_path=None` resolves through `KnowledgeBase`, which knows where
    durable memory actually lives. This used to hardcode the pre-relocation
    path, so after the move it silently matched against nothing -- and a route
    matcher that finds no no-go routes reports every dead end as novel.
    """
    wanted = _terms(text)
    matches: list[RouteMatch] = []
    for item in KnowledgeBase(knowledge_path).load():
        haystack = " ".join([item.title, item.statement, item.notes, " ".join(item.formulas)])
        theirs = _terms(haystack)
        shared = wanted & theirs
        if not shared:
            continue
        score = len(shared) / max(1, len(wanted))
        action = _action_for(item.status)
        matches.append(RouteMatch(knowledge_id=item.id, title=item.title, status=item.status, domain=item.domain, score=score, matched_terms=sorted(shared), action=action))
    matches.sort(key=lambda m: (-m.score, m.knowledge_id))
    return matches[:limit]
