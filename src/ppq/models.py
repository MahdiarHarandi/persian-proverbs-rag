"""Small public data model shared by the final pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SearchMode(str, Enum):
    SURFACE = "surface"
    MEANING = "meaning"


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    CLARIFY = "CLARIFY"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    reference: str
    license_tag: str
    snapshot: str
    provenance_status: str
    review_status: str


@dataclass(frozen=True)
class Variant:
    variant_id: str
    text: str
    source_id: str
    source_ref: str = ""


@dataclass(frozen=True)
class FixedExpression:
    expression_id: str
    family_id: str
    expression_type: str
    gloss: str
    canonical_variant_id: str
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Candidate:
    expression_id: str
    family_id: str
    variant_id: str
    score: float


@dataclass(frozen=True)
class Match:
    expression_id: str
    family_id: str
    variant_id: str
    text: str
    gloss: str
    expression_type: str
    source_id: str
    source_name: str
    source_ref: str
    license_tag: str
    provenance_status: str
    review_status: str


@dataclass(frozen=True)
class SearchResult:
    mode: SearchMode
    decision: Decision
    match: Match | None
    alternatives: tuple[Match, ...]
    score: float
    margin: float
    reason: str

    def __post_init__(self) -> None:
        """Reject impossible result states at the public model boundary.

        Retrieval and verification are intentionally allowed to be complicated;
        the object that reaches rendering is not.  Encoding these invariants in
        the immutable public model makes it impossible for a later adapter to
        accidentally expose an ACCEPT without a corpus-resolved match, or to
        attach generated alternatives to an ABSTAIN.
        """
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("search score must be a finite value in [0, 1]")
        if not math.isfinite(self.margin) or not 0.0 <= self.margin <= 1.0:
            raise ValueError("search margin must be a finite value in [0, 1]")
        if not self.reason:
            raise ValueError("search result reason must be non-empty")
        if self.decision is Decision.ACCEPT:
            if self.match is None or self.alternatives:
                raise ValueError("ACCEPT requires exactly one match and no alternatives")
        elif self.decision is Decision.CLARIFY:
            if self.match is not None or not self.alternatives:
                raise ValueError("CLARIFY requires alternatives and no selected match")
        elif self.match is not None or self.alternatives:
            raise ValueError("ABSTAIN cannot contain renderable matches")


class AttestationStatus(str, Enum):
    """Corpus-bounded authenticity status for the attestation task."""

    ATTESTED = "ATTESTED"
    NOT_ATTESTED_IN_CORPUS = "NOT_ATTESTED_IN_CORPUS"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


@dataclass(frozen=True)
class AttestationResult:
    """Result of checking whether one complete expression is corpus-attested.

    ``NOT_ATTESTED_IN_CORPUS`` deliberately does *not* mean that an expression
    is linguistically fake. It only states that the current reviewed corpus
    contains no matching stored variant after orthographic normalization.
    Suggestions, when present, are separate attested records and must never be
    interpreted as evidence that the checked wording itself is authentic.
    """

    status: AttestationStatus
    checked_text: str
    match: Match | None
    suggestions: tuple[Match, ...]
    reason: str
    nearest_score: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.nearest_score) or not 0.0 <= self.nearest_score <= 1.0:
            raise ValueError("nearest_score must be a finite value in [0, 1]")
        if not self.reason:
            raise ValueError("attestation reason must be non-empty")
        if self.status is AttestationStatus.ATTESTED:
            if self.match is None or self.suggestions:
                raise ValueError("ATTESTED requires one match and no suggestions")
        elif self.status is AttestationStatus.NEEDS_CLARIFICATION:
            if self.match is not None or self.suggestions:
                raise ValueError("NEEDS_CLARIFICATION cannot expose matches")
        elif self.match is not None:
            raise ValueError("NOT_ATTESTED_IN_CORPUS cannot contain an attested match")


class TaskType(str, Enum):
    """High-level user task used by the optional automatic router."""

    SURFACE = "surface"
    MEANING = "meaning"
    ATTESTATION = "attestation"


class RouteStatus(str, Enum):
    """Whether the router selected one task or intentionally asked the user."""

    ROUTED = "ROUTED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


@dataclass(frozen=True)
class TaskRoute:
    status: RouteStatus
    task: TaskType | None
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("route reason must be non-empty")
        if self.status is RouteStatus.ROUTED and self.task is None:
            raise ValueError("ROUTED requires a concrete task")
        if self.status is RouteStatus.NEEDS_CLARIFICATION and self.task is not None:
            raise ValueError("NEEDS_CLARIFICATION must not preselect a task")


@dataclass(frozen=True)
class AutoResult:
    """Result of automatic task routing plus exactly one task-level response.

    A clarification route contains neither a search result nor an attestation
    result.  This fail-closed shape prevents an uncertain router from leaking a
    quotation merely because one probe happened to return a candidate.
    """

    route: TaskRoute
    search_result: SearchResult | None = None
    attestation_result: AttestationResult | None = None

    def __post_init__(self) -> None:
        if self.route.status is RouteStatus.NEEDS_CLARIFICATION:
            if self.search_result is not None or self.attestation_result is not None:
                raise ValueError("clarification routes cannot expose probe results")
            return

        if self.route.task is TaskType.ATTESTATION:
            if self.attestation_result is None or self.search_result is not None:
                raise ValueError("attestation route requires only an attestation result")
            return

        if self.search_result is None or self.attestation_result is not None:
            raise ValueError("surface/meaning route requires only a search result")
        expected_mode = (
            SearchMode.SURFACE if self.route.task is TaskType.SURFACE else SearchMode.MEANING
        )
        if self.search_result.mode is not expected_mode:
            raise ValueError("routed task and search-result mode disagree")
