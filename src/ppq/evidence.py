"""Typed evidence ledger used by the hardened PPQ semantic path.

The ledger deliberately separates candidate generation from semantic
verification.  Retrieval scores/ranks are recorded for auditability but are not
authoritative ACCEPT evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .models import Candidate
from .semantic_verifier import SemanticVerification


@dataclass(frozen=True)
class SemanticCandidateEvidence:
    candidate: Candidate
    retrieval_rank: int
    retrieval_source: str
    lexical_rank: int | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    best_dense_view_name: str | None = None
    best_dense_view_text: str | None = None
    verification: SemanticVerification | None = None

    def __post_init__(self) -> None:
        if self.retrieval_rank < 1:
            raise ValueError("retrieval_rank must be >= 1")
        if not self.retrieval_source:
            raise ValueError("retrieval_source must be non-empty")
        if (
            self.verification is not None
            and self.verification.family_id != self.candidate.family_id
        ):
            raise ValueError("verification/candidate family mismatch")

    def with_verification(self, verification: SemanticVerification) -> "SemanticCandidateEvidence":
        if verification.family_id != self.candidate.family_id:
            raise ValueError("verification/candidate family mismatch")
        return replace(self, verification=verification)


@dataclass(frozen=True)
class EvidenceLedger:
    query: str
    query_id: str | None
    semantic_candidates: tuple[SemanticCandidateEvidence, ...] = ()
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("ledger query must be non-empty")
        families = [row.candidate.family_id for row in self.semantic_candidates]
        if len(families) != len(set(families)):
            raise ValueError("semantic ledger candidates must be family-deduplicated")

    def verified(self) -> tuple[SemanticCandidateEvidence, ...]:
        return tuple(row for row in self.semantic_candidates if row.verification is not None)
