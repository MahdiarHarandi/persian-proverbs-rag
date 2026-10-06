"""Strict corpus attestation for complete Persian fixed expressions.

Attestation is intentionally separated from quote correction.  Correction may
use fuzzy/partial evidence to recover a likely stored expression; attestation
must not.  A wording is ATTESTED only when its complete normalized surface
matches a stored corpus variant.  Near matches can be shown as suggestions,
but they never change a NOT_ATTESTED_IN_CORPUS decision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .contracts import CandidateRetriever
from .corpus import Corpus
from .models import AttestationResult, AttestationStatus, Match, SearchMode
from .normalize import normalize_compact, normalize_retrieval
from .query import AttestationQueryAnalyzer
from .retrieval import FixedExpressionRetriever


@dataclass(frozen=True)
class AttestationPolicy:
    """Small inspectable policy for optional near-match suggestions."""

    max_suggestions: int = 3
    suggestion_min_score: float = 0.45
    min_checked_compact_chars: int = 4


class AttestationChecker:
    """Exact normalized membership checker with clearly separated suggestions."""

    def __init__(
        self,
        corpus: Corpus,
        retriever: CandidateRetriever | None = None,
        policy: AttestationPolicy | None = None,
    ):
        self.corpus = corpus
        self.retriever = retriever or FixedExpressionRetriever(corpus)
        self.policy = policy or AttestationPolicy()
        self.query_analyzer = AttestationQueryAnalyzer()

        retrieval_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
        compact_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for expression, variant in corpus.all_variants():
            pointer = (expression.expression_id, variant.variant_id)
            retrieval_map[normalize_retrieval(variant.text)].append(pointer)
            compact_map[normalize_compact(variant.text)].append(pointer)
        self._retrieval_map = dict(retrieval_map)
        self._compact_map = dict(compact_map)

    def _resolve_first(self, pointers: list[tuple[str, str]]) -> Match:
        """Prefer the earliest stored matching variant deterministically."""
        expression_id, variant_id = pointers[0]
        return self.corpus.resolve(expression_id, variant_id)

    def _suggestions(self, checked_text: str) -> tuple[tuple[Match, ...], float]:
        candidates = self.retriever.search(
            checked_text,
            SearchMode.SURFACE,
            limit=max(self.policy.max_suggestions, 1),
        )
        if not candidates or candidates[0].score < self.policy.suggestion_min_score:
            return (), candidates[0].score if candidates else 0.0
        matches = tuple(
            self.corpus.resolve(candidate.expression_id, candidate.variant_id)
            for candidate in candidates[: self.policy.max_suggestions]
        )
        return matches, candidates[0].score

    def check(self, query: str) -> AttestationResult:
        analysis = self.query_analyzer.analyze(query)
        checked = analysis.checked_text.strip()
        compact = normalize_compact(checked)
        if not compact:
            return AttestationResult(
                status=AttestationStatus.NEEDS_CLARIFICATION,
                checked_text=checked,
                match=None,
                suggestions=(),
                reason="empty-expression",
            )
        if len(compact) < self.policy.min_checked_compact_chars:
            return AttestationResult(
                status=AttestationStatus.NEEDS_CLARIFICATION,
                checked_text=checked,
                match=None,
                suggestions=(),
                reason="expression-too-short",
            )

        retrieval = normalize_retrieval(checked)
        pointers = self._retrieval_map.get(retrieval)
        if pointers:
            return AttestationResult(
                status=AttestationStatus.ATTESTED,
                checked_text=checked,
                match=self._resolve_first(pointers),
                suggestions=(),
                reason=(
                    "wrapper-extracted-normalized-corpus-match"
                    if analysis.wrapper_extracted
                    else "normalized-corpus-match"
                ),
            )

        # Compact equality is only an orthographic fallback (space/ZWNJ
        # differences).  It is still equality over the *entire* expression;
        # substring/partial/fuzzy matches never become attestation evidence.
        pointers = self._compact_map.get(compact)
        if pointers:
            return AttestationResult(
                status=AttestationStatus.ATTESTED,
                checked_text=checked,
                match=self._resolve_first(pointers),
                suggestions=(),
                reason=(
                    "wrapper-extracted-compact-corpus-match"
                    if analysis.wrapper_extracted
                    else "compact-corpus-match"
                ),
            )

        suggestions, nearest_score = self._suggestions(checked)
        return AttestationResult(
            status=AttestationStatus.NOT_ATTESTED_IN_CORPUS,
            checked_text=checked,
            match=None,
            suggestions=suggestions,
            reason=(
                "wrapper-extracted-no-corpus-match"
                if analysis.wrapper_extracted
                else "no-corpus-match"
            ),
            nearest_score=nearest_score,
        )
