"""Conservative corpus-grounded resolver for verified semantic evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Corpus
from .evidence import EvidenceLedger
from .models import Decision, SearchMode, SearchResult
from .semantic_verifier import SemanticVerdict


@dataclass(frozen=True)
class ConservativeResolverPolicy:
    """Small policy surface; no retrieval-score acceptance thresholds exist."""

    max_alternatives: int = 3
    clarify_on_multiple_supported: bool = True

    def __post_init__(self) -> None:
        if self.max_alternatives < 1:
            raise ValueError("max_alternatives must be >= 1")


class ConservativeSemanticResolver:
    """Resolve only independently SUPPORTED corpus candidates.

    Retrieval rank controls ordering only.  It cannot turn an unverified
    candidate into an ACCEPT, regardless of dense/BM25/RRF score.
    """

    def __init__(self, corpus: Corpus, policy: ConservativeResolverPolicy | None = None):
        self.corpus = corpus
        self.policy = policy or ConservativeResolverPolicy()

    def resolve(self, ledger: EvidenceLedger) -> SearchResult:
        if not ledger.semantic_candidates:
            return SearchResult(
                mode=SearchMode.MEANING,
                decision=Decision.ABSTAIN,
                match=None,
                alternatives=(),
                score=0.0,
                margin=0.0,
                reason="no-semantic-candidates",
            )

        supported = [
            row
            for row in ledger.semantic_candidates
            if row.verification is not None
            and row.verification.verdict is SemanticVerdict.SUPPORTED
        ]
        top_score = ledger.semantic_candidates[0].candidate.score
        second_score = (
            ledger.semantic_candidates[1].candidate.score
            if len(ledger.semantic_candidates) > 1
            else 0.0
        )
        diagnostic_margin = max(0.0, min(1.0, top_score - second_score))

        if not supported:
            reason = (
                "semantic-verifier-not-run"
                if not any(row.verification is not None for row in ledger.semantic_candidates)
                else "no-independently-supported-semantic-candidate"
            )
            return SearchResult(
                mode=SearchMode.MEANING,
                decision=Decision.ABSTAIN,
                match=None,
                alternatives=(),
                score=max(0.0, min(1.0, top_score)),
                margin=diagnostic_margin,
                reason=reason,
            )

        if len(supported) > 1 and self.policy.clarify_on_multiple_supported:
            alternatives = tuple(
                self.corpus.resolve_canonical_family(row.candidate.family_id)
                for row in supported[: self.policy.max_alternatives]
            )
            return SearchResult(
                mode=SearchMode.MEANING,
                decision=Decision.CLARIFY,
                match=None,
                alternatives=alternatives,
                score=max(0.0, min(1.0, supported[0].candidate.score)),
                margin=0.0,
                reason="multiple-semantically-supported-candidates",
            )

        selected = supported[0]
        return SearchResult(
            mode=SearchMode.MEANING,
            decision=Decision.ACCEPT,
            match=self.corpus.resolve_canonical_family(selected.candidate.family_id),
            alternatives=(),
            score=max(0.0, min(1.0, selected.candidate.score)),
            margin=diagnostic_margin,
            reason="independent-semantic-verification-passed",
        )
