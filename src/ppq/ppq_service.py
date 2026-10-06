"""Single structured entry point for PPQ grounding.

The service integrates planning, surface binding, exact corpus attestation,
semantic verification, multi-expression scope handling, and deterministic
resolution.

It deliberately does *not* generate the final conversational prose.  That is a
later response-composition step.  This service returns only control state and
corpus-resolved ``Match`` objects, preserving the core invariant that authentic
FFE text can enter a response only through ``Corpus.resolve``.
"""

from __future__ import annotations

import re
from typing import Protocol

from .adaptive_semantic import AdaptiveGroundedSemanticService
from .corpus import Corpus
from .models import AttestationStatus, Decision
from .multi_scope import MultiScopePlanner
from .planner_types import FFEAction, FFEPlan, PlanRoute
from .runtime_resolver import GlobalRuntimeResolver, PPQRuntimeResult
from .service import RetrievalService
from .slm import PlannerPrediction
from .surface_binder import SurfaceMentionBinder
from .surface_restore_base import SurfaceRestoreRescue


class PlannerLike(Protocol):
    def plan(self, query: str) -> PlannerPrediction: ...


_EXPLICIT_ALL_MENTIONS = re.compile(
    r"(?:هر\s*دو|هردو|هر\s*کدوم|هر\s*کدام|هر\s+کدام|دوتاشون|دو\s*تاشون)"
)
_EXPLICIT_AMBIGUOUS_ONE = re.compile(
    r"(?:فقط\s+یکی|یکی\s+از\s+این\s+(?:دو|دوتا)).{0,60}"
    r"(?:نمی\s*گم|نمیگم|مشخص\s+نمی\s*کنم|معلوم\s+نمی\s*کنم)"
)
_POST_TARGET_OPERATION_CUE = re.compile(
    r"(?:یعنی\s+چی|منظورش\s+چیه|معن(?:ی|یش)\s*(?:رو|شو)?\s*(?:بگو|بده|چیه)|"
    r"ک[ِی]?\s+به\s+کار\s+می\s*ره)"
)


class PPQService:
    """Integrated non-generative PPQ service with fail-closed dispatch."""

    def __init__(
        self,
        corpus: Corpus,
        planner: PlannerLike,
        surface_service: RetrievalService,
        semantic_service: AdaptiveGroundedSemanticService,
        *,
        multi_scope_planner: MultiScopePlanner | None = None,
        resolver: GlobalRuntimeResolver | None = None,
        max_surface_mentions: int = 10,
        enable_retrieve_examples: bool = False,
        restore_rescue=None,
    ) -> None:
        self.corpus = corpus
        self.planner = planner
        self.surface_service = surface_service
        self.semantic_service = semantic_service
        self.multi_scope_planner = multi_scope_planner
        self.resolver = resolver or GlobalRuntimeResolver()
        self.surface_binder = SurfaceMentionBinder(
            surface_service,
            max_mentions=max_surface_mentions,
        )
        self.enable_retrieve_examples = bool(enable_retrieve_examples)
        self.restore_rescue = restore_rescue or SurfaceRestoreRescue(corpus)

    def process(self, query: str, *, query_id: str | None = None) -> PPQRuntimeResult:
        if not isinstance(query, str) or not query.strip():
            return self.resolver.from_abstain(None, reason="empty-query")

        prediction = self.planner.plan(query)
        if not prediction.valid or prediction.plan is None:
            return self.resolver.from_abstain(
                None,
                reason="planner-invalid-or-unavailable",
                planner_error=prediction.error or "invalid-plan",
            )

        plan = prediction.plan
        if plan.route is PlanRoute.GENERAL:
            return self.resolver.from_general(plan)
        if plan.route is PlanRoute.CLARIFY:
            return self.resolver.from_clarify(plan, reason="planner-requested-clarification")

        if not plan.actions:
            return self.resolver.from_abstain(plan, reason="ffe-plan-without-actions")

        # The v1 typed planner permits only one action, except the explicitly
        # supported dependent ATTEST -> EXPLAIN sequence.
        if plan.actions == (FFEAction.ATTEST, FFEAction.EXPLAIN):
            return self._attest_and_explain(plan)
        if len(plan.actions) != 1:
            return self.resolver.from_abstain(plan, reason="unsupported-multi-action-plan")

        action = plan.actions[0]
        if action is FFEAction.SEARCH_BY_MEANING:
            # Frozen runtime invariant: BGE sees the raw user query.  The
            # planner semantic_query is understanding metadata, not a rewrite
            # authorized to alter candidate retrieval.
            if hasattr(self.semantic_service, "search_with_hint"):
                result = self.semantic_service.search_with_hint(
                    query,
                    plan.semantic_query,
                    query_id=query_id,
                )
            else:
                result = self.semantic_service.search(query, query_id=query_id)
            return self.resolver.from_search(plan, result)

        if action is FFEAction.RESTORE_SURFACE:
            return self._surface_action(query, plan, mode="restore")

        if action is FFEAction.ATTEST:
            return self._surface_action(query, plan, mode="attest")

        if action is FFEAction.EXPLAIN:
            return self._surface_action(query, plan, mode="explain")

        if action is FFEAction.RETRIEVE_EXAMPLES:
            if not self.enable_retrieve_examples:
                return self.resolver.from_abstain(
                    plan, reason="retrieve-examples-not-integrated-runtime"
                )
            if not hasattr(self.semantic_service, "search_many"):
                return self.resolver.from_abstain(
                    plan, reason="retrieve-examples-verifier-unavailable"
                )
            matches = self.semantic_service.search_many(
                query,
                plan.requested_count,
                query_id=query_id,
            )
            if len(matches) != plan.requested_count:
                return self.resolver.from_abstain(
                    plan,
                    reason="insufficient-independently-supported-examples",
                )
            return self.resolver.from_matches(
                plan,
                tuple(matches),
                reason="requested-examples-independently-semantically-supported",
            )

        return self.resolver.from_abstain(plan, reason="unsupported-ffe-action")

    def _attest_and_explain(self, plan: FFEPlan) -> PPQRuntimeResult:
        assert plan.target_span is not None
        attestation = self.surface_service.attest(plan.target_span)
        # Explanation is allowed only if the supplied wording itself is
        # corpus-attested.  A fuzzy suggestion must never silently become a
        # positive authenticity claim in this joint task.
        return self.resolver.from_attestation(plan, attestation)

    def _surface_action(
        self,
        full_query: str,
        plan: FFEPlan,
        *,
        mode: str,
    ) -> PPQRuntimeResult:
        assert plan.target_span is not None

        multi = self._resolve_multi_surface(full_query, plan, mode=mode)
        if multi is not None:
            return multi

        if mode == "restore":
            # When the user explicitly says the supplied wording is only
            # the *beginning* of a longer remembered expression, a short exact
            # family must not eclipse a unique longer corpus continuation.  This
            # This hook exists only for RESTORE; ATTEST never calls it.
            contextual = getattr(self.restore_rescue, "contextual_prefix_completion", None)
            if contextual is not None:
                pref = contextual(plan.target_span, full_query)
                if pref.family_id is not None:
                    match = self.corpus.resolve_canonical_family(pref.family_id)
                    return self.resolver.from_matches(
                        plan,
                        (match,),
                        reason=f"restore-surface-rescue:{pref.reason}",
                    )

            primary = self.surface_service.correct_quote(plan.target_span)
            if primary.decision is Decision.ACCEPT:
                return self.resolver.from_search(plan, primary)
            rescued = self.restore_rescue.resolve(plan.target_span)
            if rescued.family_id is not None:
                match = self.corpus.resolve_canonical_family(rescued.family_id)
                return self.resolver.from_matches(
                    plan,
                    (match,),
                    reason=f"restore-surface-rescue:{rescued.reason}",
                )
            return self.resolver.from_search(plan, primary)

        if mode == "attest":
            return self.resolver.from_attestation(
                plan,
                self.surface_service.attest(plan.target_span),
            )

        # EXPLAIN: exact attestation is preferred, but noisy/incomplete user
        # wording may be restored through the already-frozen Surface verifier.
        attestation = self.surface_service.attest(plan.target_span)
        if attestation.status is AttestationStatus.ATTESTED:
            return self.resolver.from_attestation(plan, attestation)

        restored = self.surface_service.correct_quote(plan.target_span)
        if restored.decision is Decision.ACCEPT and restored.match is not None:
            return self.resolver.from_matches(
                plan,
                (restored.match,),
                reason="surface-restored-for-grounded-explanation",
            )
        if restored.decision is Decision.CLARIFY:
            return self.resolver.from_clarify(plan, reason=restored.reason)
        return self.resolver.from_abstain(plan, reason=restored.reason)

    @staticmethod
    def _dedupe_bindings(bindings):
        """Preserve first mention order while keeping each family only once."""
        seen = set()
        out = []
        for binding in bindings:
            if binding.family_id in seen:
                continue
            seen.add(binding.family_id)
            out.append(binding)
        return out

    def _resolve_multi_surface(
        self,
        full_query: str,
        plan: FFEPlan,
        *,
        mode: str,
    ) -> PPQRuntimeResult | None:
        """Use runtime binder/scope only when >=2 distinct families are grounded."""
        bindings = self.surface_binder.bind(full_query)
        distinct = {binding.family_id for binding in bindings}
        if len(distinct) < 2:
            return None

        normalized_query = full_query.replace("\u200c", " ")
        # Obvious "both/all mentioned expressions" requests do not need a
        # second LLM scope call.  Binder evidence has already grounded each span
        # independently, so selecting all is deterministic and lower-risk.
        if _EXPLICIT_AMBIGUOUS_ONE.search(normalized_query):
            return self.resolver.from_clarify(
                plan,
                reason="explicit-multi-target-ambiguity",
            )
        if _EXPLICIT_ALL_MENTIONS.search(normalized_query):
            selected = self._dedupe_bindings(bindings)
            matches = tuple(
                self.corpus.resolve_canonical_family(binding.family_id) for binding in selected
            )
            if mode == "attest":
                return self.resolver.from_matches(
                    plan, matches, reason="explicit-all-multi-ffe-independent-attestation-passed"
                )
            if mode == "restore":
                return self.resolver.from_matches(
                    plan,
                    matches,
                    reason="explicit-all-multi-ffe-independent-surface-restoration-passed",
                )
            return self.resolver.from_matches(
                plan, matches, reason="explicit-all-multi-ffe-grounded-explanation-ready"
            )

        # Whole-query binding can occasionally pick up an ordinary
        # phrase after the real target (for example the meta phrase "به کار
        # میره"). If the finalized planner target overlaps exactly one grounded
        # binding, respect that local target and continue through the normal
        # single-surface path instead of escalating to multi-scope. Explicit
        # both/all and explicit one-of-two ambiguity have already been handled
        # above, so this cannot silently discard a stated multi-target request.
        target = plan.target_span or ""
        overlapping = [
            binding
            for binding in bindings
            if target and (binding.raw_span in target or target in binding.raw_span)
        ]
        if len(overlapping) == 1:
            # Narrow exception only for a target followed by an operation cue
            # where every extra binding occurs inside/after that meta-language.
            # This fixes spurious bindings such as "به کار میره" without
            # changing the historical fail-closed behavior for genuine
            # multi-expression queries like "معنی A و B رو بگو".
            target_binding = overlapping[0]
            cue = _POST_TARGET_OPERATION_CUE.search(full_query, target_binding.raw_end)
            extras = [b for b in bindings if b is not target_binding]
            if cue is not None and all(b.raw_start >= cue.start() for b in extras):
                return None

        if self.multi_scope_planner is None:
            return self.resolver.from_clarify(
                plan,
                reason="multiple-grounded-ffes-require-target-scope",
            )

        scope = self.multi_scope_planner.plan(
            full_query,
            [binding.as_dict() for binding in bindings],
        )
        if not bool(scope.get("valid")) or scope.get("route") == "CLARIFY":
            return self.resolver.from_clarify(
                plan,
                reason="multi-ffe-scope-ambiguous-or-invalid",
            )

        allowed_spans = set(scope.get("target_spans", []))
        selected = self._dedupe_bindings(
            [binding for binding in bindings if binding.raw_span in allowed_spans]
        )
        if not selected:
            return self.resolver.from_clarify(
                plan,
                reason="multi-ffe-scope-selected-no-grounded-span",
            )

        # Multi-surface matches are already independently grounded by the
        # binder.  Resolve canonical families again at the public boundary.
        matches = tuple(
            self.corpus.resolve_canonical_family(binding.family_id) for binding in selected
        )

        if mode == "attest":
            return self.resolver.from_matches(
                plan,
                matches,
                reason="multi-ffe-independent-attestation-passed",
            )
        if mode == "restore":
            return self.resolver.from_matches(
                plan,
                matches,
                reason="multi-ffe-independent-surface-restoration-passed",
            )
        return self.resolver.from_matches(
            plan,
            matches,
            reason="multi-ffe-grounded-explanation-ready",
        )
