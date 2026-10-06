"""Evidence-first semantic candidate discovery.

This service ranks corpus families and records retrieval-channel evidence.  It
never returns a user-renderable Match and never makes ACCEPT/CLARIFY decisions.
"""

from __future__ import annotations

from .contracts import CandidateRetriever
from .evidence import EvidenceLedger, SemanticCandidateEvidence
from .models import SearchMode


class SemanticEvidenceDiscovery:
    def __init__(
        self, retriever: CandidateRetriever, *, source_name: str = "semantic-candidate-retriever"
    ):
        self.retriever = retriever
        self.source_name = source_name

    def _search_one(self, query: str, *, limit: int):
        channel_by_family = {}
        if hasattr(self.retriever, "search_with_evidence"):
            candidates, channels = self.retriever.search_with_evidence(
                query, SearchMode.MEANING, limit=limit
            )
            channel_by_family = {row.family_id: row for row in channels}
        else:
            candidates = self.retriever.search(query, SearchMode.MEANING, limit=limit)
        return list(candidates), channel_by_family

    def discover(
        self,
        query: str,
        *,
        query_id: str | None = None,
        limit: int = 20,
        alternate_query: str | None = None,
    ) -> EvidenceLedger:
        """Discover candidates from raw user text plus an optional meaning hint.

        The alternate view is candidate-recall evidence only.  It never replaces
        the raw request and cannot authorize ACCEPT.  When present, candidates
        are round-robin interleaved by rank (raw1, hint1, raw2, hint2, ...) and
        deduplicated by family.  This keeps the previously frozen raw-query path
        while making long/mixed prompts less vulnerable to instruction noise.
        """
        if limit <= 0:
            raise ValueError("limit must be > 0")
        if not query or not query.strip():
            raise ValueError("semantic discovery requires a non-empty query")

        raw_candidates, raw_channels = self._search_one(query, limit=limit)
        use_alt = bool(
            alternate_query and alternate_query.strip() and alternate_query.strip() != query.strip()
        )
        alt_candidates = []
        alt_channels = {}
        if use_alt:
            alt_candidates, alt_channels = self._search_one(alternate_query.strip(), limit=limit)

        ordered: list[tuple[object, object | None, str, int]] = []
        seen: set[str] = set()
        depth = max(len(raw_candidates), len(alt_candidates))
        for i in range(depth):
            for source, candidates, channels in (
                ("raw", raw_candidates, raw_channels),
                ("semantic_hint", alt_candidates, alt_channels),
            ):
                if i >= len(candidates):
                    continue
                candidate = candidates[i]
                if candidate.family_id in seen:
                    continue
                seen.add(candidate.family_id)
                ordered.append((candidate, channels.get(candidate.family_id), source, i + 1))
                if len(ordered) >= limit:
                    break
            if len(ordered) >= limit:
                break

        rows: list[SemanticCandidateEvidence] = []
        for rank, (candidate, channel, view_name, view_rank) in enumerate(ordered, 1):
            dense_score = None if channel is None else getattr(channel, "dense_score", None)
            dense_rank = None if channel is None else getattr(channel, "dense_rank", None)
            if channel is not None and dense_score is not None and dense_rank is None:
                dense_rank = view_rank
            rows.append(
                SemanticCandidateEvidence(
                    candidate=candidate,
                    retrieval_rank=rank,
                    retrieval_source=f"{self.source_name}:{view_name}",
                    lexical_rank=None
                    if channel is None
                    else getattr(channel, "lexical_rank", None),
                    lexical_score=None
                    if channel is None
                    else getattr(channel, "lexical_score", None),
                    dense_rank=dense_rank,
                    dense_score=dense_score,
                    rrf_score=None if channel is None else getattr(channel, "rrf_score", None),
                    best_dense_view_name=(
                        view_name
                        if channel is None
                        else getattr(channel, "best_view_name", view_name)
                    ),
                    best_dense_view_text=None
                    if channel is None
                    else getattr(channel, "best_view_text", None),
                )
            )
        return EvidenceLedger(
            query=query,
            query_id=query_id,
            semantic_candidates=tuple(rows),
            metadata={
                "candidate_count": str(len(rows)),
                "retrieval_views": "raw+semantic_hint" if use_alt else "raw",
            },
        )
