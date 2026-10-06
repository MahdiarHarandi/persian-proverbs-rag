"""Candidate-bounded local alignment for wrapped surface requests.

Whole-query verification can reject correct candidates when wrapper language
dilutes the evidence. This module searches within the raw request for the best
local token window for each corpus-backed candidate.

No fixed expression is generated here.  Candidate text is read only through
``Corpus.resolve`` and final rendering remains the service's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .contracts import SurfaceEvidenceRetriever
from .corpus import Corpus
from .models import Candidate
from .normalize import normalize_compact, retrieval_tokens
from .request import RequestView
from .retrieval import SurfaceEvidence


def _token_lcs(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _bigrams(value: str) -> set[str]:
    if not value:
        return set()
    if len(value) < 2:
        return {value}
    return {value[index : index + 2] for index in range(len(value) - 1)}


@dataclass(frozen=True)
class LocalSurfaceAlignment:
    candidate: Candidate
    raw_start: int
    raw_end: int
    raw_text: str
    score: float
    evidence: SurfaceEvidence
    token_support: float
    char_support: float


@dataclass(frozen=True)
class LocalSurfaceRanking:
    alignments: tuple[LocalSurfaceAlignment, ...]

    @property
    def top(self) -> LocalSurfaceAlignment | None:
        return self.alignments[0] if self.alignments else None

    @property
    def margin(self) -> float:
        if not self.alignments:
            return 0.0
        second = self.alignments[1].score if len(self.alignments) > 1 else 0.0
        return max(0.0, self.alignments[0].score - second)


class LocalSurfaceAligner:
    """Re-rank a bounded Surface candidate pool using local request windows."""

    MAX_CANDIDATES = 10
    MAX_EXTRA_WINDOW_TOKENS = 2
    MAX_EVIDENCE_WINDOWS_PER_CANDIDATE = 3
    MIN_WINDOW_TOKENS = 2

    # These weights were selected only on the frozen benchmark DEV Surface split.
    # The cheap stage is for window proposal; the final stage gives slightly
    # greater weight to the existing, independently inspectable Surface
    # evidence so this component remains an evidence verifier, not a new model.
    FINAL_CHEAP_WEIGHT = 0.45
    FINAL_EVIDENCE_WEIGHT = 0.55

    def __init__(self, corpus: Corpus, retriever: SurfaceEvidenceRetriever):
        self._corpus = corpus
        self._retriever = retriever

    @staticmethod
    def _evidence_score(evidence: SurfaceEvidence) -> float:
        # Same bounded noisy-OR weights used by the production fusion ranker.
        return 1.0 - (
            (1.0 - evidence.tfidf)
            * (1.0 - 0.30 * evidence.coverage)
            * (1.0 - 0.35 * evidence.bm25)
            * (1.0 - 0.40 * evidence.fuzzy)
        )

    @staticmethod
    def _cheap_alignment_score(window: str, candidate_text: str) -> tuple[float, float, float]:
        query_tokens = retrieval_tokens(window)
        candidate_tokens = retrieval_tokens(candidate_text)
        query_compact = normalize_compact(window)
        candidate_compact = normalize_compact(candidate_text)
        if not query_compact:
            return 0.0, 0.0, 0.0

        matched_tokens = _token_lcs(query_tokens, candidate_tokens)
        token_support = matched_tokens / max(1, len(query_tokens))
        candidate_token_coverage = matched_tokens / max(1, len(candidate_tokens))

        query_grams = _bigrams(query_compact)
        candidate_grams = _bigrams(candidate_compact)
        matched_grams = query_grams & candidate_grams
        query_char_coverage = len(matched_grams) / max(1, len(query_grams))
        candidate_char_coverage = len(matched_grams) / max(1, len(candidate_grams))

        matcher = SequenceMatcher(None, query_compact, candidate_compact, autojunk=False)
        char_support = sum(block.size for block in matcher.get_matching_blocks()) / max(
            1, len(query_compact)
        )
        length_quality = min(1.0, len(query_compact) / 8.0)

        score = (
            0.30 * token_support
            + 0.15 * candidate_token_coverage
            + 0.25 * query_char_coverage
            + 0.15 * char_support
            + 0.10 * candidate_char_coverage
            + 0.05 * length_quality
        )
        # A six-character compact fragment occurring verbatim inside an
        # attested candidate is strong local evidence, while still requiring a
        # family-level margin at the service policy boundary.
        if len(query_compact) >= 6 and query_compact in candidate_compact and candidate_compact:
            score = max(
                score,
                0.92 + 0.05 * min(1.0, len(query_compact) / len(candidate_compact)),
            )
        return min(1.0, score), token_support, char_support

    def _candidate_alignment(
        self,
        request: RequestView,
        candidate: Candidate,
    ) -> LocalSurfaceAlignment | None:
        if not request.tokens:
            return None
        match = self._corpus.resolve(candidate.expression_id, candidate.variant_id)
        candidate_token_count = len(retrieval_tokens(match.text))
        max_window_tokens = min(
            len(request.tokens),
            max(3, candidate_token_count + self.MAX_EXTRA_WINDOW_TOKENS),
        )
        min_window_tokens = (
            self.MIN_WINDOW_TOKENS if len(request.tokens) >= self.MIN_WINDOW_TOKENS else 1
        )

        proposals: list[tuple[float, int, int, int, str, float, float]] = []
        for window_size in range(min_window_tokens, max_window_tokens + 1):
            for token_index in range(len(request.tokens) - window_size + 1):
                first = request.tokens[token_index]
                last = request.tokens[token_index + window_size - 1]
                raw_start, raw_end = first.raw_start, last.raw_end
                window = request.raw_slice(raw_start, raw_end)
                cheap, token_support, char_support = self._cheap_alignment_score(window, match.text)
                proposals.append(
                    (
                        cheap,
                        raw_end - raw_start,
                        -raw_start,
                        raw_start,
                        window,
                        token_support,
                        char_support,
                    )
                )

        if not proposals:
            return None
        proposals.sort(reverse=True)

        best: LocalSurfaceAlignment | None = None
        for proposal in proposals[: self.MAX_EVIDENCE_WINDOWS_PER_CANDIDATE]:
            cheap, _, _, raw_start, window, token_support, char_support = proposal
            raw_end = raw_start + len(window)
            evidence = self._retriever.surface_evidence(window, candidate)
            evidence_score = self._evidence_score(evidence)
            final_score = min(
                1.0,
                self.FINAL_CHEAP_WEIGHT * cheap + self.FINAL_EVIDENCE_WEIGHT * evidence_score,
            )
            alignment = LocalSurfaceAlignment(
                candidate=candidate,
                raw_start=raw_start,
                raw_end=raw_end,
                raw_text=window,
                score=final_score,
                evidence=evidence,
                token_support=token_support,
                char_support=char_support,
            )
            if best is None or (
                alignment.score,
                alignment.raw_end - alignment.raw_start,
                -alignment.raw_start,
            ) > (
                best.score,
                best.raw_end - best.raw_start,
                -best.raw_start,
            ):
                best = alignment
        return best

    def rank(
        self,
        request: RequestView,
        candidates: list[Candidate],
    ) -> LocalSurfaceRanking:
        alignments = [
            alignment
            for candidate in candidates[: self.MAX_CANDIDATES]
            if (alignment := self._candidate_alignment(request, candidate)) is not None
        ]
        alignments.sort(
            key=lambda item: (
                -item.score,
                -item.candidate.score,
                item.candidate.family_id,
            )
        )
        return LocalSurfaceRanking(alignments=tuple(alignments))
