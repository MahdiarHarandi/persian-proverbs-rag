"""Bridge the frozen applicability gate into the semantic evidence ledger."""

from __future__ import annotations

from .applicability_gate import ConservativeApplicabilityGate
from .applicability_verifier import ApplicabilityVerdict
from .models import Match
from .semantic_verifier import SemanticVerdict, SemanticVerification


class ApplicabilitySemanticVerifier:
    """SemanticVerifier adapter using only query + stored corpus gloss.

    The adapter receives a corpus-resolved :class:`Match` because that is the
    existing semantic-verifier interface, but forwards only ``candidate.gloss``
    to the applicability gate.  FFE surface text, source metadata and retrieval
    scores never enter the model-side applicability boundary.
    """

    def __init__(self, gate: ConservativeApplicabilityGate) -> None:
        self.gate = gate

    @property
    def verifier_id(self) -> str:
        return self.gate.verifier_id

    def verify(
        self,
        query: str,
        candidate: Match,
        *,
        query_id: str | None = None,
    ) -> SemanticVerification:
        result = self.gate.verify(query, candidate.gloss)
        if result.verdict is ApplicabilityVerdict.SUPPORTED:
            verdict = SemanticVerdict.SUPPORTED
        elif result.verdict is ApplicabilityVerdict.NOT_SUPPORTED:
            verdict = SemanticVerdict.NOT_SUPPORTED
        else:
            verdict = SemanticVerdict.UNCERTAIN
        return SemanticVerification(
            family_id=candidate.family_id,
            verdict=verdict,
            verifier_id=self.verifier_id,
            reason=result.reason,
        )
