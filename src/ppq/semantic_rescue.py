"""runtime raw-first semantic retrieval with bounded hint rescue.

runtime interleaved raw and planner-hint candidates before verification.  With a
verification budget of two this accidentally demoted the already-validated raw
Top-2 path (raw1, hint1 were checked; raw2 often was not).  runtime restores the
frozen raw-query priority and uses the planner meaning hint only as a second
candidate-recall phase after the raw Top-2 fails.

Both phases remain candidate-only.  Every accepted family still passes the same
independent applicability verifier; retrieval rank/score never authorizes ACCEPT.
"""

from __future__ import annotations

from dataclasses import replace

from .corpus import Corpus
from .evidence import EvidenceLedger, SemanticCandidateEvidence
from .evidence_discovery import SemanticEvidenceDiscovery
from .recommendation_policy import RecommendationSuitabilityPolicy
from .resolver import ConservativeSemanticResolver
from .semantic_verifier import (
    AbstainingSemanticVerifier,
    SemanticVerdict,
    SemanticVerification,
    SemanticVerifier,
)


class RawFirstSemanticRescueService:
    """Verify raw Top-K first, then bounded distinct hint candidates as rescue."""

    def __init__(
        self,
        corpus: Corpus,
        discovery: SemanticEvidenceDiscovery,
        verifier: SemanticVerifier | None = None,
        *,
        raw_verifications: int = 2,
        hint_rescue_verifications: int = 2,
        cross_view_rescue_verifications: int = 0,
        cross_view_window: int = 8,
        resolver: ConservativeSemanticResolver | None = None,
        suitability_policy: RecommendationSuitabilityPolicy | None = None,
    ) -> None:
        if raw_verifications < 1:
            raise ValueError("raw_verifications must be >= 1")
        if hint_rescue_verifications < 0:
            raise ValueError("hint_rescue_verifications must be >= 0")
        if cross_view_rescue_verifications < 0:
            raise ValueError("cross_view_rescue_verifications must be >= 0")
        if cross_view_window < 3:
            raise ValueError("cross_view_window must be >= 3")
        self.corpus = corpus
        self.discovery = discovery
        self.verifier = verifier or AbstainingSemanticVerifier()
        self.raw_verifications = raw_verifications
        self.hint_rescue_verifications = hint_rescue_verifications
        self.cross_view_rescue_verifications = cross_view_rescue_verifications
        self.cross_view_window = cross_view_window
        self.resolver = resolver or ConservativeSemanticResolver(corpus)
        self.suitability_policy = suitability_policy or RecommendationSuitabilityPolicy()
        self._trace_by_query_id: dict[str, dict[str, object]] = {}

    def _verify_row(self, query: str, row: SemanticCandidateEvidence, *, query_id: str | None):
        match = self.corpus.resolve_canonical_family(row.candidate.family_id)
        suitability = self.suitability_policy.evaluate(query, match)
        if not suitability.allowed:
            verifier_id = str(getattr(self.verifier, "verifier_id", "semantic-verifier"))
            verification = SemanticVerification(
                family_id=row.candidate.family_id,
                verdict=SemanticVerdict.NOT_SUPPORTED,
                verifier_id=(verifier_id or "semantic-verifier") + "+recommendation-policy",
                reason=suitability.reason,
            )
            return row.with_verification(verification)
        try:
            verification = self.verifier.verify(query, match, query_id=query_id)
            if verification.family_id != row.candidate.family_id:
                raise ValueError("semantic verifier returned a mismatched family_id")
        except Exception as exc:
            verifier_id = str(getattr(self.verifier, "verifier_id", "semantic-verifier"))
            verification = SemanticVerification(
                family_id=row.candidate.family_id,
                verdict=SemanticVerdict.UNCERTAIN,
                verifier_id=verifier_id or "semantic-verifier",
                reason=f"verifier-error:{type(exc).__name__}",
            )
        return row.with_verification(verification)

    @staticmethod
    def _relabeled(
        row: SemanticCandidateEvidence, *, rank: int, source: str
    ) -> SemanticCandidateEvidence:
        return replace(row, retrieval_rank=rank, retrieval_source=source, verification=None)

    def collect_evidence(
        self,
        query: str,
        *,
        semantic_hint: str | None = None,
        query_id: str | None = None,
        candidate_k: int = 20,
    ) -> EvidenceLedger:
        raw = self.discovery.discover(query, query_id=query_id, limit=candidate_k)
        use_hint = bool(
            semantic_hint
            and semantic_hint.strip()
            and semantic_hint.strip() != query.strip()
            and self.hint_rescue_verifications > 0
        )
        hint = (
            self.discovery.discover(semantic_hint.strip(), query_id=query_id, limit=candidate_k)
            if use_hint
            else None
        )

        rows: list[SemanticCandidateEvidence] = []
        seen: set[str] = set()

        # Phase 1: preserve the historically validated raw Top-2 path.
        raw_phase = list(raw.semantic_candidates[: self.raw_verifications])
        for row in raw_phase:
            if row.candidate.family_id in seen:
                continue
            seen.add(row.candidate.family_id)
            rows.append(
                self._relabeled(
                    row, rank=len(rows) + 1, source=f"{row.retrieval_source}:raw-primary"
                )
            )

        supported = False
        for i in range(len(rows)):
            rows[i] = self._verify_row(query, rows[i], query_id=query_id)
            if rows[i].verification and rows[i].verification.verdict is SemanticVerdict.SUPPORTED:
                supported = True
                break

        # Phase 2: only after the raw primary path fails, use distinct planner-
        # hint candidates as bounded recall rescue.  Verification still sees the
        # complete original user query, never the hint.
        hint_checked = 0
        if not supported and hint is not None:
            for row in hint.semantic_candidates:
                if row.candidate.family_id in seen:
                    continue
                seen.add(row.candidate.family_id)
                rescue = self._relabeled(
                    row,
                    rank=len(rows) + 1,
                    source=f"{row.retrieval_source}:semantic-hint-rescue",
                )
                rescue = self._verify_row(query, rescue, query_id=query_id)
                rows.append(rescue)
                hint_checked += 1
                if rescue.verification and rescue.verification.verdict is SemanticVerdict.SUPPORTED:
                    supported = True
                    break
                if hint_checked >= self.hint_rescue_verifications:
                    break

        # Phase 3 (Final runtime, optional): if both primary views failed, verify a
        # tiny number of candidates that independently appear in the bounded
        # raw and semantic-hint windows. Cross-view agreement is only a
        # candidate-selection heuristic; ACCEPT still requires the same frozen
        # semantic verifier and suitability policy.
        cross_checked = 0
        if not supported and hint is not None and self.cross_view_rescue_verifications > 0:
            raw_rank = {
                row.candidate.family_id: i + 1
                for i, row in enumerate(raw.semantic_candidates[: self.cross_view_window])
            }
            hint_rank = {
                row.candidate.family_id: i + 1
                for i, row in enumerate(hint.semantic_candidates[: self.cross_view_window])
            }
            overlap = [fid for fid in raw_rank if fid in hint_rank and fid not in seen]
            overlap.sort(
                key=lambda fid: (raw_rank[fid] + hint_rank[fid], raw_rank[fid], hint_rank[fid])
            )
            raw_by_family = {row.candidate.family_id: row for row in raw.semantic_candidates}
            for fid in overlap:
                row = raw_by_family[fid]
                seen.add(fid)
                rescue = self._relabeled(
                    row,
                    rank=len(rows) + 1,
                    source=f"{row.retrieval_source}:cross-view-consensus-rescue",
                )
                rescue = self._verify_row(query, rescue, query_id=query_id)
                rows.append(rescue)
                cross_checked += 1
                if rescue.verification and rescue.verification.verdict is SemanticVerdict.SUPPORTED:
                    supported = True
                    break
                if cross_checked >= self.cross_view_rescue_verifications:
                    break

        # Keep unverified candidates after the decision horizon for auditability
        # without granting them any authority.
        for row in raw.semantic_candidates[self.raw_verifications :]:
            if row.candidate.family_id in seen or len(rows) >= candidate_k:
                continue
            seen.add(row.candidate.family_id)
            rows.append(
                self._relabeled(
                    row, rank=len(rows) + 1, source=f"{row.retrieval_source}:raw-unverified-tail"
                )
            )
        if hint is not None:
            for row in hint.semantic_candidates:
                if row.candidate.family_id in seen or len(rows) >= candidate_k:
                    continue
                seen.add(row.candidate.family_id)
                rows.append(
                    self._relabeled(
                        row,
                        rank=len(rows) + 1,
                        source=f"{row.retrieval_source}:hint-unverified-tail",
                    )
                )

        if query_id:
            self._trace_by_query_id[query_id] = {
                "strategy": (
                    "raw-top2-then-bounded-hint-then-cross-view-rescue"
                    if self.cross_view_rescue_verifications
                    else "raw-top2-then-bounded-semantic-hint-rescue"
                ),
                "raw_top_families": [
                    r.candidate.family_id for r in raw.semantic_candidates[: self.raw_verifications]
                ],
                "hint_top_families": (
                    []
                    if hint is None
                    else [
                        r.candidate.family_id
                        for r in hint.semantic_candidates[: self.hint_rescue_verifications]
                    ]
                ),
                "cross_view_checked": cross_checked,
                "checked": [
                    {
                        "family_id": r.candidate.family_id,
                        "source": r.retrieval_source,
                        "verdict": (
                            None if r.verification is None else r.verification.verdict.value
                        ),
                        "reason": (None if r.verification is None else r.verification.reason),
                    }
                    for r in rows
                    if r.verification is not None
                ],
                "supported_family_id": next(
                    (
                        r.candidate.family_id
                        for r in rows
                        if r.verification is not None
                        and r.verification.verdict is SemanticVerdict.SUPPORTED
                    ),
                    None,
                ),
            }

        return EvidenceLedger(
            query=query,
            query_id=query_id,
            semantic_candidates=tuple(rows),
            metadata={
                "verification_strategy": (
                    "raw-top2-then-bounded-hint-then-cross-view-rescue"
                    if self.cross_view_rescue_verifications
                    else "raw-top2-then-bounded-semantic-hint-rescue"
                ),
                "raw_verifications": str(self.raw_verifications),
                "hint_rescue_verifications": str(self.hint_rescue_verifications),
                "cross_view_rescue_verifications": str(self.cross_view_rescue_verifications),
                "cross_view_window": str(self.cross_view_window),
                "hint_used": str(use_hint).lower(),
                "hint_checked": str(hint_checked),
                "cross_view_checked": str(cross_checked),
            },
        )

    def trace(self, query_id: str) -> dict[str, object] | None:
        row = self._trace_by_query_id.get(query_id)
        return None if row is None else dict(row)

    def search(self, query: str, *, query_id: str | None = None, candidate_k: int = 20):
        return self.resolver.resolve(
            self.collect_evidence(query, query_id=query_id, candidate_k=candidate_k)
        )

    def search_with_hint(
        self,
        query: str,
        semantic_hint: str | None,
        *,
        query_id: str | None = None,
        candidate_k: int = 20,
    ):
        return self.resolver.resolve(
            self.collect_evidence(
                query,
                semantic_hint=semantic_hint,
                query_id=query_id,
                candidate_k=candidate_k,
            )
        )

    def search_many(
        self,
        query: str,
        requested_count: int,
        *,
        query_id: str | None = None,
        candidate_k: int = 20,
    ) -> tuple:
        if requested_count < 1 or requested_count > 10:
            raise ValueError("requested_count must be in [1,10]")
        ledger = self.discovery.discover(query, query_id=query_id, limit=candidate_k)
        budget = min(candidate_k, max(self.raw_verifications, requested_count * 2))
        supported = []
        for row in ledger.semantic_candidates[:budget]:
            checked = self._verify_row(query, row, query_id=query_id)
            if checked.verification and checked.verification.verdict is SemanticVerdict.SUPPORTED:
                supported.append(self.corpus.resolve_canonical_family(row.candidate.family_id))
                if len(supported) >= requested_count:
                    return tuple(supported)
        return ()
