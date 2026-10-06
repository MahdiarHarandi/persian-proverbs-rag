"""Deterministic global resolver for the integrated PPQ runtime.

This module intentionally contains no model, retrieval, embedding, or threshold
logic.  It only converts already-grounded task results into one small public
runtime state.  Any authentic FFE exposed here must already be a ``Match``
resolved through :class:`ppq.corpus.Corpus`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import AttestationResult, AttestationStatus, Decision, Match, SearchResult
from .planner_types import FFEPlan


class RuntimeDecision(str, Enum):
    """Final non-generative decision of the PPQ grounding layer."""

    GENERAL = "GENERAL"
    ACCEPT = "ACCEPT"
    CLARIFY = "CLARIFY"
    ABSTAIN = "ABSTAIN"
    NOT_ATTESTED_IN_CURRENT_CORPUS = "NOT_ATTESTED_IN_CURRENT_CORPUS"


@dataclass(frozen=True)
class PPQRuntimeResult:
    """One safe result returned before any free-form response composition.

    ``matches`` and ``suggestions`` can contain only corpus-resolved ``Match``
    objects.  The runtime never stores generated FFE strings in this public
    object.  Explanations are read later from ``Match.gloss`` rather than being
    invented by this layer.
    """

    decision: RuntimeDecision
    plan: FFEPlan | None
    matches: tuple[Match, ...] = ()
    suggestions: tuple[Match, ...] = ()
    reason: str = ""
    planner_error: str | None = None

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("runtime result reason must be non-empty")

        if self.decision is RuntimeDecision.ACCEPT:
            if not self.matches:
                raise ValueError("ACCEPT requires at least one corpus-resolved match")
        elif self.matches:
            raise ValueError("only ACCEPT may expose selected matches")

        if self.decision is RuntimeDecision.NOT_ATTESTED_IN_CURRENT_CORPUS:
            # Suggestions are allowed here because they are separate attested
            # corpus records, never evidence that the checked wording is real.
            pass
        elif self.suggestions:
            raise ValueError("suggestions are only valid for NOT_ATTESTED results")

        if self.decision is RuntimeDecision.GENERAL and self.plan is not None:
            if self.plan.requires_ffe:
                raise ValueError("GENERAL cannot carry an FFE-required plan")

    @property
    def grounded(self) -> bool:
        return bool(self.matches)

    @property
    def glosses(self) -> tuple[str, ...]:
        """Return corpus-authored meanings for accepted matches."""
        return tuple(match.gloss for match in self.matches)


class GlobalRuntimeResolver:
    """Pure deterministic mapping from task evidence to runtime decisions."""

    def from_general(self, plan: FFEPlan, *, reason: str = "planner-general") -> PPQRuntimeResult:
        return PPQRuntimeResult(
            decision=RuntimeDecision.GENERAL,
            plan=plan,
            reason=reason,
        )

    def from_clarify(self, plan: FFEPlan | None, *, reason: str) -> PPQRuntimeResult:
        return PPQRuntimeResult(
            decision=RuntimeDecision.CLARIFY,
            plan=plan,
            reason=reason,
        )

    def from_abstain(
        self,
        plan: FFEPlan | None,
        *,
        reason: str,
        planner_error: str | None = None,
    ) -> PPQRuntimeResult:
        return PPQRuntimeResult(
            decision=RuntimeDecision.ABSTAIN,
            plan=plan,
            reason=reason,
            planner_error=planner_error,
        )

    def from_search(self, plan: FFEPlan, result: SearchResult) -> PPQRuntimeResult:
        if result.decision is Decision.ACCEPT:
            assert result.match is not None
            return PPQRuntimeResult(
                decision=RuntimeDecision.ACCEPT,
                plan=plan,
                matches=(result.match,),
                reason=result.reason,
            )
        if result.decision is Decision.CLARIFY:
            # SearchResult alternatives are intentionally not surfaced here.
            # Clarification is a no-quotation state at the global boundary.
            return self.from_clarify(plan, reason=result.reason)
        return self.from_abstain(plan, reason=result.reason)

    def from_attestation(self, plan: FFEPlan, result: AttestationResult) -> PPQRuntimeResult:
        if result.status is AttestationStatus.ATTESTED:
            assert result.match is not None
            return PPQRuntimeResult(
                decision=RuntimeDecision.ACCEPT,
                plan=plan,
                matches=(result.match,),
                reason=result.reason,
            )
        if result.status is AttestationStatus.NEEDS_CLARIFICATION:
            return self.from_clarify(plan, reason=result.reason)
        return PPQRuntimeResult(
            decision=RuntimeDecision.NOT_ATTESTED_IN_CURRENT_CORPUS,
            plan=plan,
            suggestions=result.suggestions,
            reason=result.reason,
        )

    def from_matches(
        self,
        plan: FFEPlan,
        matches: tuple[Match, ...],
        *,
        reason: str,
    ) -> PPQRuntimeResult:
        if not matches:
            return self.from_abstain(plan, reason=reason)
        return PPQRuntimeResult(
            decision=RuntimeDecision.ACCEPT,
            plan=plan,
            matches=matches,
            reason=reason,
        )
