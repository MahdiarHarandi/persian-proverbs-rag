"""Task-level application service for correction, meaning retrieval and attestation."""

from __future__ import annotations

from dataclasses import dataclass

from .attestation import AttestationChecker
from .contracts import CandidateRetriever, SurfaceEvidenceRetriever
from .corpus import Corpus
from .local_surface import LocalSurfaceAligner, LocalSurfaceRanking
from .meaning import MeaningEvidence, meaning_evidence
from .models import (
    AttestationResult,
    AutoResult,
    Candidate,
    Decision,
    RouteStatus,
    SearchMode,
    SearchResult,
    TaskRoute,
    TaskType,
)
from .normalize import normalize_compact, retrieval_tokens
from .query import SurfaceQueryAnalysis, SurfaceQueryAnalyzer
from .request import RequestIntake
from .retrieval import FixedExpressionRetriever, SurfaceEvidence
from .router import TaskRouter


@dataclass(frozen=True)
class DecisionPolicy:
    """Fixed, inspectable thresholds selected on development data."""

    # Historical direct Surface boundary.  Strong exact/partial and high-score
    # fused matches continue to use this path unchanged.
    surface_minimum_score: float = 0.86
    surface_min_margin: float = 0.03

    # Evidence-aware selective-accept boundary. These values were
    # frozen using only the Surface benchmark development split. They require a clear ranking
    # margin plus agreement between independent surface signals; a single weak
    # fuzzy/BM25 score cannot trigger ACCEPT.
    surface_evidence_minimum_score: float = 0.60
    surface_evidence_min_margin: float = 0.10
    surface_evidence_min_compact_chars: int = 8
    surface_lexical_min_tokens: int = 3
    surface_lexical_tfidf_min: float = 0.35
    surface_lexical_coverage_min: float = 0.60
    surface_lexical_bm25_min: float = 0.60
    surface_edit_min_compact_chars: int = 10
    surface_edit_tfidf_min: float = 0.18
    surface_edit_coverage_min: float = 0.58
    surface_edit_fuzzy_min: float = 0.85
    # Low-information additions that may occur in remembered fragments.  Kept
    # in policy rather than hidden inside verification logic so future
    # calibration can replace this list without changing code paths.
    surface_allowed_filler_tokens: tuple[str, ...] = ("فلان", "دیگه", "دیگر")

    # benchmark local-window fallback.  These were selected on the frozen DEV Surface
    # split only; LOCKED_TEST remains untouched until final evaluation.
    local_surface_candidate_pool: int = 10
    local_surface_minimum_score: float = 0.72
    local_surface_min_margin: float = 0.04
    # Guard short generic local windows (e.g. "یکی به") from overriding a
    # stronger family-level candidate. Two-token rescue is retained only for
    # very short candidate FFEs such as "آب در هاون کوبیدن".
    local_surface_min_window_tokens: int = 3
    local_surface_short_window_max_candidate_tokens: int = 4
    # If two high-confidence local alignments occupy separate request spans,
    # treat the request as containing multiple FFEs and fail closed.
    local_surface_multi_alignment_min_score: float = 0.85
    local_surface_multi_alignment_max_overlap: float = 0.20

    meaning_minimum_score: float = 0.30
    meaning_min_margin: float = 0.05
    # A Meaning ACCEPT/CLARIFY must be anchored by at least one
    # non-trivial Persian content token shared with the stored gloss.  This
    # blocks accidental character-overlap matches on out-of-domain questions
    # without pretending to solve low-overlap scenario understanding.
    meaning_require_content_anchor: bool = True
    max_alternatives: int = 3

    def thresholds(self, mode: SearchMode) -> tuple[float, float]:
        if mode is SearchMode.SURFACE:
            return self.surface_minimum_score, self.surface_min_margin
        return self.meaning_minimum_score, self.meaning_min_margin


class RetrievalService:
    """Rank, selectively verify, then resolve only stored corpus records."""

    def __init__(
        self,
        corpus: Corpus,
        policy: DecisionPolicy | None = None,
        retriever: CandidateRetriever | None = None,
        *,
        surface_retriever: SurfaceEvidenceRetriever | None = None,
        meaning_retriever: CandidateRetriever | None = None,
        task_router: TaskRouter | None = None,
        enable_surface_evidence_verification: bool = True,
        enable_local_surface_verification: bool = True,
    ):
        """Build the task service with independently replaceable retrieval paths.

        ``retriever`` is the backwards-compatible single-retriever argument used
        by the shipped lexical ablations.  New integrations should prefer
        ``surface_retriever`` and ``meaning_retriever`` so a semantic Meaning
        backend can be evaluated without changing Surface or Attestation.
        """
        if retriever is not None and (
            surface_retriever is not None or meaning_retriever is not None
        ):
            raise ValueError("use either legacy retriever or task-specific retrievers, not both")

        self.corpus = corpus
        self.policy = policy or DecisionPolicy()
        shared_retriever = retriever or FixedExpressionRetriever(corpus)
        self.surface_retriever = surface_retriever or shared_retriever
        self.meaning_retriever = meaning_retriever or shared_retriever
        # Kept as a compatibility alias for existing experiment code.  New code
        # should use the task-specific attributes above.
        self.retriever = self.surface_retriever
        self.enable_surface_evidence_verification = enable_surface_evidence_verification
        if enable_surface_evidence_verification and not hasattr(
            self.surface_retriever, "surface_evidence"
        ):
            raise TypeError(
                "surface_retriever must provide surface_evidence when evidence "
                "verification is enabled"
            )
        self.surface_query_analyzer = SurfaceQueryAnalyzer(corpus)
        self.request_intake = RequestIntake()
        self.enable_local_surface_verification = (
            enable_local_surface_verification and enable_surface_evidence_verification
        )
        self.local_surface_aligner = LocalSurfaceAligner(corpus, self.surface_retriever)
        self.attestation_checker = AttestationChecker(corpus, self.surface_retriever)
        self.task_router = task_router or TaskRouter()

    def _analyze_surface_query(self, query: str) -> SurfaceQueryAnalysis:
        return self.surface_query_analyzer.analyze(query)

    def correct_quote(
        self,
        query: str,
        *,
        allow_local_surface: bool = True,
    ) -> SearchResult:
        """Task 1: recover/quote one stored expression from surface evidence."""
        return self.search(query, SearchMode.SURFACE, allow_local_surface=allow_local_surface)

    def search_meaning(self, query: str) -> SearchResult:
        """Historical lexical Meaning baseline.

        Candidate-only dense/hybrid retrievers are intentionally rejected by
        :meth:`search`; final semantic retrieval must use
        ``GroundedSemanticService`` so an independent verifier is required
        before ACCEPT.
        """
        return self.search(query, SearchMode.MEANING)

    def attest(self, query: str) -> AttestationResult:
        """Task 3: check complete-surface membership in the reviewed corpus."""
        return self.attestation_checker.check(query)

    def auto(self, query: str) -> AutoResult:
        """Route a natural request to one task without weakening task safety.

        Explicit task wording is dispatched directly.  Otherwise Surface and
        Meaning are evaluated as probes and the router either selects one of
        those already-verified results or asks the user to choose a task.
        Attestation is never inferred from fuzzy similarity; it requires an
        explicit authenticity cue.
        """
        explicit = self.task_router.explicit_route(query)
        if explicit is not None:
            if explicit.status is RouteStatus.NEEDS_CLARIFICATION:
                return AutoResult(route=explicit)
            if explicit.task is TaskType.ATTESTATION:
                return AutoResult(
                    route=explicit,
                    attestation_result=self.attest(query),
                )
            if explicit.task is TaskType.MEANING:
                return AutoResult(
                    route=explicit,
                    search_result=self.search_meaning(query),
                )
            return AutoResult(
                route=explicit,
                search_result=self.correct_quote(query),
            )

        # Local Surface alignment is a verifier, not an intent detector.  For
        # an implicit query the legacy probes remain unchanged until the benchmark
        # Understanding layer decides that Surface evidence is relevant.
        surface = self.correct_quote(query, allow_local_surface=False)
        meaning = self.search_meaning(query)
        route = self.task_router.route_from_probes(query, surface, meaning)
        if route.status is RouteStatus.NEEDS_CLARIFICATION:
            return AutoResult(route=route)
        if route.task is TaskType.SURFACE:
            return AutoResult(route=route, search_result=surface)
        if route.task is TaskType.MEANING:
            return AutoResult(route=route, search_result=meaning)
        # ``route_from_probes`` never selects Attestation, because authenticity
        # is explicit-only.  Keep the fallback fail-closed if that invariant is
        # ever changed accidentally.
        return AutoResult(
            route=TaskRoute(
                RouteStatus.NEEDS_CLARIFICATION,
                None,
                "unexpected-router-task",
            )
        )

    def rank(
        self,
        query: str,
        mode: SearchMode | str,
        limit: int = 10,
    ) -> list[Candidate]:
        search_mode = SearchMode(mode)
        retrieval_query = query
        if search_mode is SearchMode.SURFACE and self.enable_surface_evidence_verification:
            retrieval_query = self._analyze_surface_query(query).retrieval_query
        active_retriever = (
            self.surface_retriever if search_mode is SearchMode.SURFACE else self.meaning_retriever
        )
        return active_retriever.search(retrieval_query, search_mode, limit=limit)

    def search(
        self,
        query: str,
        mode: SearchMode | str,
        *,
        allow_local_surface: bool = True,
    ) -> SearchResult:
        search_mode = SearchMode(mode)
        if search_mode is SearchMode.MEANING and getattr(
            self.meaning_retriever, "candidate_only", False
        ):
            raise RuntimeError(
                "candidate-only semantic retrievers must use GroundedSemanticService; "
                "retrieval score/rank cannot authorize semantic ACCEPT"
            )
        if not query or not query.strip():
            return SearchResult(
                mode=search_mode,
                decision=Decision.ABSTAIN,
                match=None,
                alternatives=(),
                score=0.0,
                margin=0.0,
                reason="empty-query",
            )

        internal_limit = max(self.policy.max_alternatives + 1, 2)
        if (
            search_mode is SearchMode.SURFACE
            and self.enable_local_surface_verification
            and allow_local_surface
        ):
            internal_limit = max(internal_limit, self.policy.local_surface_candidate_pool)
        candidates = self.rank(
            query,
            search_mode,
            limit=internal_limit,
        )
        return self.decide_ranked(
            search_mode,
            candidates,
            query=query,
            allow_local_surface=allow_local_surface,
        )

    def _resolve_family_canonical(self, family_id: str):
        return self.corpus.resolve_canonical_family(family_id)

    def _multiple_expression_result(
        self,
        analysis: SurfaceQueryAnalysis,
        top_score: float,
        margin: float,
    ) -> SearchResult:
        alternatives = tuple(
            self._resolve_family_canonical(family_id)
            for family_id in analysis.contained_family_ids[: self.policy.max_alternatives]
        )
        return SearchResult(
            mode=SearchMode.SURFACE,
            decision=Decision.CLARIFY,
            match=None,
            alternatives=alternatives,
            score=top_score,
            margin=margin,
            reason="multiple-attested-expressions",
        )

    def _surface_evidence_is_sufficient(
        self,
        query: str,
        top: Candidate,
        margin: float,
    ) -> tuple[bool, SurfaceEvidence]:
        evidence = self.surface_retriever.surface_evidence(query, top)
        compact_length = len(normalize_compact(query))
        tokens = retrieval_tokens(query)
        if (
            top.score < self.policy.surface_evidence_minimum_score
            or margin < self.policy.surface_evidence_min_margin
            or compact_length < self.policy.surface_evidence_min_compact_chars
        ):
            return False, evidence

        edit_agreement = (
            compact_length >= self.policy.surface_edit_min_compact_chars
            and evidence.tfidf >= self.policy.surface_edit_tfidf_min
            and evidence.coverage >= self.policy.surface_edit_coverage_min
            and evidence.fuzzy >= self.policy.surface_edit_fuzzy_min
        )
        candidate_text = self.corpus.resolve(top.expression_id, top.variant_id).text
        candidate_tokens = set(retrieval_tokens(candidate_text))
        unsupported_tokens = {token for token in tokens if token not in candidate_tokens}
        # Lexical agreement is safe for sparse/reordered queries only when the
        # query words are actually supported by the stored candidate.  A small
        # explicit filler vocabulary covers the Phase-8 development-set
        # placeholder/addition patterns.  This guard is what prevents a fluent
        # semantic substitution such as "خورشید" for attested "ماه" from
        # being accepted merely because all the surrounding words match.
        allowed_fillers = set(self.policy.surface_allowed_filler_tokens)
        lexical_tokens_supported = not unsupported_tokens or unsupported_tokens <= allowed_fillers
        lexical_agreement = (
            lexical_tokens_supported
            and len(tokens) >= self.policy.surface_lexical_min_tokens
            and evidence.tfidf >= self.policy.surface_lexical_tfidf_min
            and evidence.coverage >= self.policy.surface_lexical_coverage_min
            and evidence.bm25 >= self.policy.surface_lexical_bm25_min
        )
        return edit_agreement or lexical_agreement, evidence

    def _local_surface_ranking(
        self,
        query: str,
        candidates: list[Candidate],
    ) -> LocalSurfaceRanking:
        request = self.request_intake.build(query)
        return self.local_surface_aligner.rank(request, candidates)

    def _local_surface_window_is_specific(self, local_top) -> bool:
        if local_top is None:
            return False
        window_tokens = len(retrieval_tokens(local_top.raw_text))
        if window_tokens >= self.policy.local_surface_min_window_tokens:
            return True
        candidate_text = self.corpus.resolve(
            local_top.candidate.expression_id,
            local_top.candidate.variant_id,
        ).text
        candidate_tokens = len(retrieval_tokens(candidate_text))
        return (
            window_tokens == 2
            and candidate_tokens <= self.policy.local_surface_short_window_max_candidate_tokens
            and local_top.token_support >= 1.0
            and local_top.char_support >= 0.95
        )

    def _local_surface_has_multiple_strong_spans(self, local: LocalSurfaceRanking) -> bool:
        if len(local.alignments) < 2:
            return False
        first, second = local.alignments[:2]
        threshold = self.policy.local_surface_multi_alignment_min_score
        if first.score < threshold or second.score < threshold:
            return False
        overlap_start = max(first.raw_start, second.raw_start)
        overlap_end = min(first.raw_end, second.raw_end)
        overlap = max(0, overlap_end - overlap_start)
        shorter = min(first.raw_end - first.raw_start, second.raw_end - second.raw_start)
        overlap_ratio = overlap / max(1, shorter)
        return overlap_ratio <= self.policy.local_surface_multi_alignment_max_overlap

    def _meaning_evidence(
        self,
        query: str,
        top: Candidate,
    ) -> MeaningEvidence:
        expression = self.corpus.family(top.family_id)
        if expression is None:
            raise KeyError("candidate references an unknown family")
        return meaning_evidence(query, expression.gloss)

    def decide_ranked(
        self,
        mode: SearchMode | str,
        candidates: list[Candidate],
        *,
        query: str | None = None,
        allow_local_surface: bool = True,
    ) -> SearchResult:
        """Apply the decision policy to an already computed ranking.

        Passing ``query`` enables the evidence-aware Surface verifier. Calls
        that omit it retain score/margin-only behavior for controlled ablations.
        """
        search_mode = SearchMode(mode)
        if not candidates:
            return SearchResult(
                mode=search_mode,
                decision=Decision.ABSTAIN,
                match=None,
                alternatives=(),
                score=0.0,
                margin=0.0,
                reason="no-overlap",
            )

        top = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = max(0.0, top.score - second_score)

        if (
            search_mode is SearchMode.MEANING
            and query is not None
            and self.policy.meaning_require_content_anchor
            and top.score >= self.policy.meaning_minimum_score
        ):
            evidence = self._meaning_evidence(query, top)
            if not evidence.has_content_anchor:
                return SearchResult(
                    mode=search_mode,
                    decision=Decision.ABSTAIN,
                    match=None,
                    alternatives=(),
                    score=top.score,
                    margin=margin,
                    reason="insufficient-meaning-content-anchor",
                )

        if (
            search_mode is SearchMode.SURFACE
            and query is not None
            and self.enable_surface_evidence_verification
        ):
            analysis = self._analyze_surface_query(query)
            if analysis.has_multiple_attested_expressions:
                return self._multiple_expression_result(analysis, top.score, margin)

            if (
                top.score >= self.policy.surface_minimum_score
                and margin >= self.policy.surface_min_margin
            ):
                return SearchResult(
                    mode=search_mode,
                    decision=Decision.ACCEPT,
                    match=self.corpus.resolve(top.expression_id, top.variant_id),
                    alternatives=(),
                    score=top.score,
                    margin=margin,
                    reason=(
                        "wrapper-extracted-score-and-margin-passed"
                        if analysis.wrapper_extracted
                        else "score-and-margin-passed"
                    ),
                )

            sufficient, _ = self._surface_evidence_is_sufficient(
                analysis.retrieval_query, top, margin
            )
            if sufficient:
                return SearchResult(
                    mode=search_mode,
                    decision=Decision.ACCEPT,
                    match=self.corpus.resolve(top.expression_id, top.variant_id),
                    alternatives=(),
                    score=top.score,
                    margin=margin,
                    reason="surface-evidence-agreement",
                )

            if self.enable_local_surface_verification and allow_local_surface:
                local = self._local_surface_ranking(query, candidates)
                local_top = local.top
                same_token_count_as_candidate = False
                if local_top is not None:
                    local_candidate_text = self.corpus.resolve(
                        local_top.candidate.expression_id,
                        local_top.candidate.variant_id,
                    ).text
                    same_token_count_as_candidate = len(retrieval_tokens(query)) == len(
                        retrieval_tokens(local_candidate_text)
                    )
                same_count_safe_suffix_wrapper = False
                if local_top is not None and same_token_count_as_candidate:
                    candidate_token_count = len(retrieval_tokens(local_candidate_text))
                    local_token_count = len(retrieval_tokens(local_top.raw_text))
                    trailing_token_count = len(retrieval_tokens(query[local_top.raw_end :]))
                    same_count_safe_suffix_wrapper = (
                        not query[: local_top.raw_start].strip()
                        and trailing_token_count >= 2
                        and candidate_token_count > 0
                        and local_token_count / candidate_token_count < 0.60
                    )
                has_nonlocal_wrapper = (
                    local_top is not None
                    and (not same_token_count_as_candidate or same_count_safe_suffix_wrapper)
                    and bool(
                        query[: local_top.raw_start].strip() or query[local_top.raw_end :].strip()
                    )
                )
                if self._local_surface_has_multiple_strong_spans(local):
                    alternatives = tuple(
                        self.corpus.resolve(
                            alignment.candidate.expression_id,
                            alignment.candidate.variant_id,
                        )
                        for alignment in local.alignments[: self.policy.max_alternatives]
                    )
                    return SearchResult(
                        mode=search_mode,
                        decision=Decision.CLARIFY,
                        match=None,
                        alternatives=alternatives,
                        score=local_top.score if local_top is not None else top.score,
                        margin=local.margin,
                        reason="multiple-local-surface-expressions",
                    )

                if (
                    local_top is not None
                    and has_nonlocal_wrapper
                    and self._local_surface_window_is_specific(local_top)
                    and local_top.score >= self.policy.local_surface_minimum_score
                    and local.margin >= self.policy.local_surface_min_margin
                ):
                    candidate = local_top.candidate
                    return SearchResult(
                        mode=search_mode,
                        decision=Decision.ACCEPT,
                        match=self.corpus.resolve(candidate.expression_id, candidate.variant_id),
                        alternatives=(),
                        score=local_top.score,
                        margin=local.margin,
                        reason="local-surface-evidence-agreement",
                    )

            if top.score >= self.policy.surface_minimum_score:
                alternatives = tuple(
                    self.corpus.resolve(candidate.expression_id, candidate.variant_id)
                    for candidate in candidates[: self.policy.max_alternatives]
                )
                return SearchResult(
                    mode=search_mode,
                    decision=Decision.CLARIFY,
                    match=None,
                    alternatives=alternatives,
                    score=top.score,
                    margin=margin,
                    reason="small-ranking-margin",
                )

            return SearchResult(
                mode=search_mode,
                decision=Decision.ABSTAIN,
                match=None,
                alternatives=(),
                score=top.score,
                margin=margin,
                reason="insufficient-surface-evidence",
            )

        minimum_score, minimum_margin = self.policy.thresholds(search_mode)
        if top.score >= minimum_score and margin >= minimum_margin:
            return SearchResult(
                mode=search_mode,
                decision=Decision.ACCEPT,
                match=self.corpus.resolve(top.expression_id, top.variant_id),
                alternatives=(),
                score=top.score,
                margin=margin,
                reason="score-and-margin-passed",
            )

        if top.score >= minimum_score:
            alternatives = tuple(
                self.corpus.resolve(candidate.expression_id, candidate.variant_id)
                for candidate in candidates[: self.policy.max_alternatives]
            )
            return SearchResult(
                mode=search_mode,
                decision=Decision.CLARIFY,
                match=None,
                alternatives=alternatives,
                score=top.score,
                margin=margin,
                reason="small-ranking-margin",
            )

        return SearchResult(
            mode=search_mode,
            decision=Decision.ABSTAIN,
            match=None,
            alternatives=(),
            score=top.score,
            margin=margin,
            reason="low-score",
        )
