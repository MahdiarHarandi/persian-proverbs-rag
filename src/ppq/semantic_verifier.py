"""Independent semantic verification boundary for Meaning/Scenario retrieval.

Dense, lexical and hybrid scores are candidate-generation evidence only.  A
candidate may become an ACCEPT only after an independent verifier marks the
(query meaning, stored family gloss) pair as SUPPORTED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from .models import Match
from .semantic_overlay import SemanticAcceptability, SemanticAcceptabilityOverlay


class SemanticVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class SemanticVerification:
    family_id: str
    verdict: SemanticVerdict
    verifier_id: str
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("verification family_id must be non-empty")
        if not self.verifier_id:
            raise ValueError("verifier_id must be non-empty")


class SemanticVerifier(Protocol):
    @property
    def verifier_id(self) -> str: ...

    def verify(
        self,
        query: str,
        candidate: Match,
        *,
        query_id: str | None = None,
    ) -> SemanticVerification: ...


class AbstainingSemanticVerifier:
    """Safe default: no independent evidence means no semantic ACCEPT."""

    verifier_id = "abstain-no-semantic-verifier"

    def verify(
        self, query: str, candidate: Match, *, query_id: str | None = None
    ) -> SemanticVerification:
        return SemanticVerification(
            family_id=candidate.family_id,
            verdict=SemanticVerdict.UNCERTAIN,
            verifier_id=self.verifier_id,
            reason="no-independent-semantic-verifier",
        )


class OverlaySemanticVerifier:
    """Oracle/evaluation verifier backed only by reviewed adjudication labels.

    This is useful for architecture tests and oracle analyses.  It is not a
    production semantic model and intentionally returns UNCERTAIN for unlabeled
    pairs rather than treating them as negatives.
    """

    verifier_id = "human-semantic-overlay"

    def __init__(self, overlay: SemanticAcceptabilityOverlay):
        self.overlay = overlay

    def verify(
        self, query: str, candidate: Match, *, query_id: str | None = None
    ) -> SemanticVerification:
        if not query_id:
            return SemanticVerification(
                family_id=candidate.family_id,
                verdict=SemanticVerdict.UNCERTAIN,
                verifier_id=self.verifier_id,
                reason="query-id-required-for-overlay",
            )
        label = self.overlay.label(query_id, candidate.family_id)
        if label is SemanticAcceptability.ACCEPTABLE:
            verdict = SemanticVerdict.SUPPORTED
        elif label is SemanticAcceptability.NOT_ACCEPTABLE:
            verdict = SemanticVerdict.NOT_SUPPORTED
        else:
            verdict = SemanticVerdict.UNCERTAIN
        return SemanticVerification(
            family_id=candidate.family_id,
            verdict=verdict,
            verifier_id=self.verifier_id,
            reason=("overlay-unlabeled" if label is None else f"overlay:{label.value}"),
        )


class CallableSemanticVerifier:
    """Strict adapter for Gemma/remote verifier experiments.

    The callable receives only the user intent text and the corpus-resolved
    candidate.  It must return a JSON object with exactly one `verdict` field
    and optional `reason`.  No generated alternative FFE is accepted by this
    adapter.
    """

    def __init__(self, fn: Callable[[str, Match], str | dict], verifier_id: str):
        self.fn = fn
        self._verifier_id = verifier_id

    @property
    def verifier_id(self) -> str:
        return self._verifier_id

    def verify(
        self, query: str, candidate: Match, *, query_id: str | None = None
    ) -> SemanticVerification:
        payload = self.fn(query, candidate)
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError("semantic verifier output must be a JSON object")
        allowed = {"verdict", "reason"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"semantic verifier emitted forbidden fields: {sorted(unknown)}")
        if "verdict" not in payload:
            raise ValueError("semantic verifier output is missing verdict")
        verdict = SemanticVerdict(str(payload["verdict"]))
        return SemanticVerification(
            family_id=candidate.family_id,
            verdict=verdict,
            verifier_id=self.verifier_id,
            reason=str(payload.get("reason", "")),
        )
