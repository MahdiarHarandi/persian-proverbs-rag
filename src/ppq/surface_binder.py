"""Frozen iterative surface binder promoted from the runtime notebook.

The binder does not decide user intent and does not introduce a new retrieval
model or threshold.  It repeatedly reuses the already-frozen Surface candidate
generator, local aligner and isolated ``correct_quote`` verifier.  Accepted
regions are masked with spaces, preserving offsets in the original request.

This module is deliberately independent of any LLM.  It only proposes
corpus-grounded exact substrings for downstream bounded planning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import Decision, SearchMode
from .service import RetrievalService


@dataclass(frozen=True)
class SurfaceMentionBinding:
    """One independently verified local Surface mention in the user request."""

    binding_index: int
    raw_start: int
    raw_end: int
    raw_span: str
    family_id: str
    expression_id: str
    variant_id: str
    local_score: float
    verification_reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class SurfaceMentionBinder:
    """Iteratively bind corpus-grounded Surface mentions without an LLM."""

    def __init__(self, service: RetrievalService, *, max_mentions: int = 10):
        if max_mentions < 1:
            raise ValueError("max_mentions must be positive")
        self.service = service
        self.max_mentions = max_mentions

    def bind(self, query: str) -> tuple[SurfaceMentionBinding, ...]:
        if not query or not query.strip():
            return ()

        residual = query
        bindings: list[SurfaceMentionBinding] = []

        for _ in range(self.max_mentions):
            candidates = self.service.rank(
                residual,
                SearchMode.SURFACE,
                limit=self.service.policy.local_surface_candidate_pool,
            )
            if not candidates:
                break

            # This is intentionally the same frozen local-alignment path used
            # by the accepted runtime experiment.  No new scoring rule is added.
            local = self.service._local_surface_ranking(residual, candidates)
            alignment = local.top
            if alignment is None:
                break

            if alignment.score < self.service.policy.local_surface_minimum_score:
                break

            start = alignment.raw_start
            end = alignment.raw_end
            raw_span = query[start:end]
            if not raw_span.strip():
                break

            # Independent isolated verification is mandatory.  A local fuzzy
            # window can never become a binding on alignment score alone.
            isolated = self.service.correct_quote(raw_span)
            if isolated.decision is not Decision.ACCEPT or isolated.match is None:
                break

            if isolated.match.family_id != alignment.candidate.family_id:
                break

            bindings.append(
                SurfaceMentionBinding(
                    binding_index=len(bindings) + 1,
                    raw_start=start,
                    raw_end=end,
                    raw_span=raw_span,
                    family_id=isolated.match.family_id,
                    expression_id=isolated.match.expression_id,
                    variant_id=isolated.match.variant_id,
                    local_score=float(alignment.score),
                    verification_reason=isolated.reason,
                )
            )

            # Preserve character positions exactly while removing this region
            # from consideration in subsequent iterations.
            residual = residual[:start] + (" " * (end - start)) + residual[end:]

        bindings.sort(key=lambda item: (item.raw_start, item.raw_end))
        return tuple(bindings)


def bind_surface_mentions(
    query: str,
    service: RetrievalService,
    *,
    max_mentions: int = 10,
) -> list[dict[str, object]]:
    """Notebook-compatible wrapper returning dictionaries in text order."""

    binder = SurfaceMentionBinder(service, max_mentions=max_mentions)
    return [binding.as_dict() for binding in binder.bind(query)]
