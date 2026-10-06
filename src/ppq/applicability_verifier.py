"""Conservative semantic applicability verification boundary.

This component is intentionally narrower than a general LLM judge.  It answers
one question only: does a reviewed corpus meaning fit the user's context/need?
It has no authority over authenticity, provenance, candidate retrieval, or FFE
surface text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Protocol


class ApplicabilityVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class ApplicabilityVerification:
    verdict: ApplicabilityVerdict
    verifier_id: str
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.verifier_id:
            raise ValueError("verifier_id must be non-empty")


class ApplicabilityVerifier(Protocol):
    @property
    def verifier_id(self) -> str: ...

    def verify(self, context: str, candidate_meaning: str) -> ApplicabilityVerification: ...


class AbstainingApplicabilityVerifier:
    """Fail-closed default used until a model verifier is explicitly installed."""

    verifier_id = "abstain-no-applicability-verifier"

    def verify(self, context: str, candidate_meaning: str) -> ApplicabilityVerification:
        return ApplicabilityVerification(
            verdict=ApplicabilityVerdict.UNCERTAIN,
            verifier_id=self.verifier_id,
            reason="no-independent-applicability-verifier",
        )


def parse_applicability_payload(
    payload: str | Mapping[str, object],
) -> tuple[ApplicabilityVerdict, str]:
    """Strictly parse the closed verifier output.

    Extra keys are rejected so a model cannot smuggle an alternative FFE or a
    confidence score into the future resolver boundary.
    """
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise ValueError("applicability verifier output must be a JSON object")
    allowed = {"verdict", "reason"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"applicability verifier emitted forbidden fields: {sorted(unknown)}")
    if "verdict" not in payload:
        raise ValueError("applicability verifier output is missing verdict")
    verdict = ApplicabilityVerdict(str(payload["verdict"]))
    reason = str(payload.get("reason", ""))
    return verdict, reason


class CallableApplicabilityVerifier:
    """Strict adapter for local/remote verifier experiments.

    The callable receives *only* ``context`` and ``candidate_meaning``.  This
    makes rank/score, gold/non-gold status, family identity and surface wording
    unavailable by construction at this interface.
    """

    def __init__(
        self,
        fn: Callable[[str, str], str | Mapping[str, object]],
        *,
        verifier_id: str,
    ) -> None:
        if not verifier_id:
            raise ValueError("verifier_id must be non-empty")
        self.fn = fn
        self._verifier_id = verifier_id

    @property
    def verifier_id(self) -> str:
        return self._verifier_id

    def verify(self, context: str, candidate_meaning: str) -> ApplicabilityVerification:
        verdict, reason = parse_applicability_payload(self.fn(context, candidate_meaning))
        return ApplicabilityVerification(
            verdict=verdict,
            verifier_id=self.verifier_id,
            reason=reason,
        )
