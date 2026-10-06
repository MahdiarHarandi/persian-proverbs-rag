"""Conservative two-stage semantic applicability gate.

1. A strict verifier proposes semantic support.
2. A support veto is consulted only after the first stage returns SUPPORTED.
3. A veto result of NOT_SUPPORTED or UNCERTAIN means "do not authorize ACCEPT"; it is not
   promoted to a definitive claim that the candidate is linguistically or
   semantically false.

The gate sees only the user context/need and the reviewed corpus meaning.  FFE
surface text, family identity, retrieval rank/score and human labels are outside
this boundary by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from .applicability_verifier import (
    ApplicabilityVerdict,
    ApplicabilityVerification,
    ApplicabilityVerifier,
)


@dataclass(frozen=True)
class ApplicabilityGateTrace:
    """Audit trace for one acceptance decision without exposing FFE text."""

    primary: ApplicabilityVerification
    veto: ApplicabilityVerification | None
    final: ApplicabilityVerification


class ConservativeApplicabilityGate:
    """Strict-entailment plus support-veto acceptance authority.

    Only ``SUPPORTED -> SUPPORTED`` across both stages authorizes semantic
    acceptance.  A veto never becomes a definitive runtime NOT_SUPPORTED claim;
    it becomes UNCERTAIN / do-not-accept so the resolver may try another
    candidate or abstain.
    """

    def __init__(
        self,
        primary: ApplicabilityVerifier,
        support_veto: ApplicabilityVerifier,
        *,
        verifier_id: str = "conservative-strict-support-veto",
    ) -> None:
        if not verifier_id:
            raise ValueError("verifier_id must be non-empty")
        self.primary = primary
        self.support_veto = support_veto
        self._verifier_id = verifier_id

    @property
    def verifier_id(self) -> str:
        return self._verifier_id

    def verify_with_trace(
        self,
        context: str,
        candidate_meaning: str,
    ) -> ApplicabilityGateTrace:
        try:
            primary = self.primary.verify(context, candidate_meaning)
        except Exception as exc:
            final = ApplicabilityVerification(
                verdict=ApplicabilityVerdict.UNCERTAIN,
                verifier_id=self.verifier_id,
                reason=f"primary-error:{type(exc).__name__}",
            )
            return ApplicabilityGateTrace(
                primary=ApplicabilityVerification(
                    verdict=ApplicabilityVerdict.UNCERTAIN,
                    verifier_id=str(getattr(self.primary, "verifier_id", "primary")) or "primary",
                    reason=f"primary-error:{type(exc).__name__}",
                ),
                veto=None,
                final=final,
            )

        if primary.verdict is ApplicabilityVerdict.NOT_SUPPORTED:
            final = ApplicabilityVerification(
                verdict=ApplicabilityVerdict.NOT_SUPPORTED,
                verifier_id=self.verifier_id,
                reason=f"v2-not-supported:{primary.reason}",
            )
            return ApplicabilityGateTrace(primary=primary, veto=None, final=final)

        if primary.verdict is ApplicabilityVerdict.UNCERTAIN:
            final = ApplicabilityVerification(
                verdict=ApplicabilityVerdict.UNCERTAIN,
                verifier_id=self.verifier_id,
                reason=f"v2-uncertain:{primary.reason}",
            )
            return ApplicabilityGateTrace(primary=primary, veto=None, final=final)

        # core is invoked only for a tentative extended SUPPORTED candidate.
        try:
            veto = self.support_veto.verify(context, candidate_meaning)
        except Exception as exc:
            veto = ApplicabilityVerification(
                verdict=ApplicabilityVerdict.UNCERTAIN,
                verifier_id=str(getattr(self.support_veto, "verifier_id", "support-veto"))
                or "support-veto",
                reason=f"veto-error:{type(exc).__name__}",
            )

        if veto.verdict is ApplicabilityVerdict.SUPPORTED:
            final_verdict = ApplicabilityVerdict.SUPPORTED
            reason = f"v2-supported;v3-kept:{veto.reason}"
        else:
            # The frozen Fresh Shadow supports core as an ACCEPT safety gate,
            # not as definitive negative authority.  A veto therefore blocks
            # ACCEPT but leaves room for a lower-ranked candidate.
            final_verdict = ApplicabilityVerdict.UNCERTAIN
            reason = f"v2-supported;v3-vetoed:{veto.verdict.value}:{veto.reason}"

        final = ApplicabilityVerification(
            verdict=final_verdict,
            verifier_id=self.verifier_id,
            reason=reason,
        )
        return ApplicabilityGateTrace(primary=primary, veto=veto, final=final)

    def verify(self, context: str, candidate_meaning: str) -> ApplicabilityVerification:
        return self.verify_with_trace(context, candidate_meaning).final
