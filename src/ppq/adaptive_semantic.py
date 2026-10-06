"""Low-risk adaptive semantic retrieval for the integrated PPQ runtime.

BGE/dense evidence remains candidate-only.  Candidates are verified in rank
order and the service stops at the first independently SUPPORTED candidate.
Vetoed/uncertain candidates do not become negative truth claims; the service
tries the next candidate and otherwise abstains.
"""

from __future__ import annotations

from .corpus import Corpus
from .evidence import EvidenceLedger
from .evidence_discovery import SemanticEvidenceDiscovery
from .resolver import ConservativeSemanticResolver
from .semantic_verifier import (
    AbstainingSemanticVerifier,
    SemanticVerdict,
    SemanticVerification,
    SemanticVerifier,
)


class AdaptiveGroundedSemanticService:
    """Verify top candidates sequentially, stopping after the first support."""

    def __init__(
        self,
        corpus: Corpus,
        discovery: SemanticEvidenceDiscovery,
        verifier: SemanticVerifier | None = None,
        *,
        max_verifications: int = 2,
        resolver: ConservativeSemanticResolver | None = None,
    ) -> None:
        if max_verifications < 1:
            raise ValueError("max_verifications must be >= 1")
        self.corpus = corpus
        self.discovery = discovery
        self.verifier = verifier or AbstainingSemanticVerifier()
        self.max_verifications = max_verifications
        self.resolver = resolver or ConservativeSemanticResolver(corpus)

    def collect_evidence(
        self,
        query: str,
        *,
        query_id: str | None = None,
        candidate_k: int = 20,
        alternate_query: str | None = None,
    ) -> EvidenceLedger:
        ledger = self.discovery.discover(
            query,
            query_id=query_id,
            limit=candidate_k,
            alternate_query=alternate_query,
        )
        rows = list(ledger.semantic_candidates)

        for index, row in enumerate(rows[: self.max_verifications]):
            match = self.corpus.resolve_canonical_family(row.candidate.family_id)
            try:
                verification = self.verifier.verify(query, match, query_id=query_id)
                if verification.family_id != row.candidate.family_id:
                    raise ValueError("semantic verifier returned a mismatched family_id")
            except Exception as exc:
                verifier_id = str(getattr(self.verifier, "verifier_id", "semantic-verifier"))
                verification = SemanticVerification(
                    family_id=row.candidate.family_id,
                    verdict=SemanticVerdict.UNCERTAIN,
                    verifier_id=verifier_id or "semantic-verifier",
                    reason=f"verifier-error:{type(exc).__name__}",
                )
            rows[index] = row.with_verification(verification)
            if verification.verdict is SemanticVerdict.SUPPORTED:
                break

        return EvidenceLedger(
            query=ledger.query,
            query_id=ledger.query_id,
            semantic_candidates=tuple(rows),
            metadata={
                **dict(ledger.metadata or {}),
                "verification_strategy": "rank-ordered-stop-on-first-supported",
                "max_verifications": str(self.max_verifications),
            },
        )

    def search(
        self,
        query: str,
        *,
        query_id: str | None = None,
        candidate_k: int = 20,
    ):
        ledger = self.collect_evidence(query, query_id=query_id, candidate_k=candidate_k)
        return self.resolver.resolve(ledger)

    def search_with_hint(
        self,
        query: str,
        semantic_hint: str | None,
        *,
        query_id: str | None = None,
        candidate_k: int = 20,
    ):
        """Candidate-only multi-view search; verification still sees raw query."""
        ledger = self.collect_evidence(
            query,
            query_id=query_id,
            candidate_k=candidate_k,
            alternate_query=semantic_hint,
        )
        return self.resolver.resolve(ledger)

    def search_many(
        self,
        query: str,
        requested_count: int,
        *,
        query_id: str | None = None,
        candidate_k: int = 20,
    ) -> tuple:
        """Return exactly N independently supported corpus matches or none.

        This is a conservative extension for RETRIEVE_EXAMPLES. Retrieval rank
        remains candidate-only; every returned family must pass the same frozen
        semantic verifier. To avoid unbounded model calls, at most 2*N candidates
        are checked (capped by candidate_k and never below max_verifications). If
        the requested number cannot be grounded, the method returns an empty
        tuple so the caller can abstain rather than pad with unverified examples.
        """
        if requested_count < 1 or requested_count > 10:
            raise ValueError("requested_count must be in [1,10]")
        ledger = self.discovery.discover(query, query_id=query_id, limit=candidate_k)
        budget = min(candidate_k, max(self.max_verifications, requested_count * 2))
        supported = []
        for row in ledger.semantic_candidates[:budget]:
            match = self.corpus.resolve_canonical_family(row.candidate.family_id)
            try:
                verification = self.verifier.verify(query, match, query_id=query_id)
                if verification.family_id != row.candidate.family_id:
                    raise ValueError("semantic verifier returned a mismatched family_id")
            except Exception:
                continue
            if verification.verdict is SemanticVerdict.SUPPORTED:
                supported.append(match)
                if len(supported) >= requested_count:
                    return tuple(supported)
        return ()
