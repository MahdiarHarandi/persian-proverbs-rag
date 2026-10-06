"""Optional dense Meaning retrieval and rank-level hybrid candidate fusion.

This module is deliberately isolated from Surface correction, Attestation and
rendering. Dense similarity is candidate-generation evidence only; it is never
an authenticity signal and it does not bypass ``Corpus.resolve``.

The production-safe core can therefore run without any embedding dependency.
When a reviewed embedding snapshot is available, :class:`DenseMeaningRetriever`
loads it through a small provider interface.  The project benchmark scripts use
the same interface with Sentence Transformers on Kaggle/Colab.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .corpus import Corpus
from .models import Candidate, SearchMode
from .semantic_docs import (
    SemanticEnrichment,
    SemanticMultiViewProfile,
    SemanticTextView,
    build_semantic_documents,
    build_semantic_vector_views,
)


class EmbeddingProvider(Protocol):
    """Minimal query/document embedding boundary used by dense retrieval."""

    @property
    def model_id(self) -> str: ...

    def encode_queries(self, texts: Sequence[str]): ...

    def encode_documents(self, texts: Sequence[str]): ...


@dataclass(frozen=True)
class DenseSnapshotMetadata:
    model_id: str
    model_revision: str
    corpus_hash: str
    text_view: str
    family_ids: tuple[str, ...]
    dimension: int
    normalized: bool
    embeddings_sha256: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "DenseSnapshotMetadata":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_id=str(payload["model_id"]),
            model_revision=str(payload.get("model_revision", "unknown")),
            corpus_hash=str(payload["corpus_hash"]),
            text_view=str(payload["text_view"]),
            family_ids=tuple(str(item) for item in payload["family_ids"]),
            dimension=int(payload["dimension"]),
            normalized=bool(payload.get("normalized", True)),
            embeddings_sha256=str(payload.get("embeddings_sha256", "")),
        )

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "corpus_hash": self.corpus_hash,
                    "text_view": self.text_view,
                    "family_ids": list(self.family_ids),
                    "dimension": self.dimension,
                    "normalized": self.normalized,
                    "embeddings_sha256": self.embeddings_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class SemanticChannelEvidence:
    family_id: str
    lexical_rank: int | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float = 0.0


class DenseMeaningRetriever:
    """Cosine ranker over one dense vector per reviewed FFE family.

    The retriever accepts either an online ``EmbeddingProvider`` plus corpus
    document embeddings, or a precomputed snapshot.  Scores are clipped to
    ``[0, 1]`` only to remain compatible with the public Candidate type; they
    are retrieval evidence, not probabilities and never authorize ACCEPT.
    """

    # Candidate generation only: legacy score/margin acceptance is forbidden.
    candidate_only = True

    def __init__(
        self,
        corpus: Corpus,
        provider: EmbeddingProvider,
        *,
        text_view: SemanticTextView | str = SemanticTextView.GLOSS_ONLY,
        document_embeddings=None,
    ):
        self.corpus = corpus
        self.provider = provider
        self.text_view = SemanticTextView(text_view)
        provider_metadata = getattr(provider, "metadata", None)
        if provider_metadata is not None and provider_metadata.text_view != self.text_view.value:
            raise ValueError("dense snapshot text_view does not match active retriever view")
        self.documents = build_semantic_documents(corpus)
        self._family_ids = tuple(document.family_id for document in self.documents)
        if document_embeddings is None:
            document_embeddings = provider.encode_documents(
                [document.embedding_text(self.text_view) for document in self.documents]
            )
        self._document_embeddings = self._as_normalized_matrix(document_embeddings)
        if self._document_embeddings.shape[0] != len(self.documents):
            raise ValueError("document embedding count does not match corpus families")

    @staticmethod
    def _numpy():
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional runtime path
            raise RuntimeError("dense semantic retrieval requires numpy") from exc
        return np

    @classmethod
    def _as_normalized_matrix(cls, values):
        np = cls._numpy()
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] <= 0:
            raise ValueError("embeddings must be a non-empty 2D matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("embedding vectors must have non-zero norm")
        return matrix / norms

    def search(
        self,
        query: str,
        mode: SearchMode | str,
        limit: int = 10,
    ) -> list[Candidate]:
        if SearchMode(mode) is not SearchMode.MEANING:
            return []
        if limit <= 0 or not query or not query.strip():
            return []
        np = self._numpy()
        query_vector = self._as_normalized_matrix(self.provider.encode_queries([query]))[0]
        similarities = self._document_embeddings @ query_vector
        size = min(limit, similarities.shape[0])
        if size <= 0:
            return []
        # ``argpartition`` avoids a full O(N log N) sort while deterministic
        # secondary sorting keeps ties reproducible.
        indices = np.argpartition(-similarities, size - 1)[:size]
        indices = sorted(indices.tolist(), key=lambda idx: (-float(similarities[idx]), idx))
        output: list[Candidate] = []
        for index in indices:
            document = self.documents[index]
            score = max(0.0, min(1.0, float(similarities[index])))
            output.append(
                Candidate(
                    expression_id=document.expression_id,
                    family_id=document.family_id,
                    variant_id=document.canonical_variant_id,
                    score=score,
                )
            )
        return output


class PrecomputedEmbeddingProvider:
    """Query encoder paired with a frozen, corpus-hash-checked document matrix."""

    def __init__(
        self,
        query_provider: EmbeddingProvider,
        corpus: Corpus,
        embeddings_path: str | Path,
        metadata_path: str | Path,
    ):
        self.query_provider = query_provider
        self.metadata = DenseSnapshotMetadata.load(metadata_path)
        if self.metadata.corpus_hash != corpus.content_hash:
            raise ValueError("dense snapshot corpus hash does not match active corpus")
        if query_provider.model_id != self.metadata.model_id:
            raise ValueError("query encoder model_id does not match dense snapshot model_id")
        documents = build_semantic_documents(corpus)
        family_ids = tuple(document.family_id for document in documents)
        if family_ids != self.metadata.family_ids:
            raise ValueError("dense snapshot family order does not match active corpus")
        embeddings_file = Path(embeddings_path)
        if self.metadata.embeddings_sha256:
            digest = hashlib.sha256(embeddings_file.read_bytes()).hexdigest()
            if digest != self.metadata.embeddings_sha256:
                raise ValueError("dense snapshot embedding hash does not match metadata")
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("precomputed dense snapshots require numpy") from exc
        self.document_embeddings = np.load(embeddings_file, allow_pickle=False)
        if self.document_embeddings.ndim != 2:
            raise ValueError("dense snapshot must be a 2D matrix")
        if self.document_embeddings.shape != (len(family_ids), self.metadata.dimension):
            raise ValueError("dense snapshot shape disagrees with metadata")

    @property
    def model_id(self) -> str:
        return self.metadata.model_id

    def encode_queries(self, texts: Sequence[str]):
        return self.query_provider.encode_queries(texts)

    def encode_documents(self, texts: Sequence[str]):
        # The caller must pass the active corpus documents in frozen order.
        if len(texts) != self.document_embeddings.shape[0]:
            raise ValueError("snapshot document request count does not match corpus")
        return self.document_embeddings


class HybridMeaningCandidateRetriever:
    """Family-level RRF over lexical and dense candidate generators.

    Raw channel scores/ranks are preserved through ``search_with_evidence``.
    RRF is used only to order a candidate set.  It intentionally does not
    convert rank consensus into an ACCEPT confidence value.
    """

    # Candidate generation only: legacy score/margin acceptance is forbidden.
    candidate_only = True

    def __init__(
        self,
        lexical_retriever,
        dense_retriever,
        *,
        channel_k: int = 20,
        rrf_k: int = 60,
    ):
        if channel_k <= 0 or rrf_k < 0:
            raise ValueError("invalid hybrid retrieval parameters")
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.channel_k = channel_k
        self.rrf_k = rrf_k

    def search_with_evidence(
        self,
        query: str,
        mode: SearchMode | str = SearchMode.MEANING,
        limit: int = 10,
    ) -> tuple[list[Candidate], tuple[SemanticChannelEvidence, ...]]:
        if SearchMode(mode) is not SearchMode.MEANING or limit <= 0:
            return [], ()
        lexical = self.lexical_retriever.search(query, SearchMode.MEANING, limit=self.channel_k)
        dense = self.dense_retriever.search(query, SearchMode.MEANING, limit=self.channel_k)
        by_family: dict[str, dict[str, object]] = {}
        for channel_name, ranking in (("lexical", lexical), ("dense", dense)):
            for rank, candidate in enumerate(ranking, 1):
                row = by_family.setdefault(
                    candidate.family_id,
                    {
                        "candidate": candidate,
                        "rrf": 0.0,
                        "lexical_rank": None,
                        "lexical_score": None,
                        "dense_rank": None,
                        "dense_score": None,
                    },
                )
                # Prefer the corpus pointer from the better-ranked channel only
                # for bookkeeping; both channels point to the same family and
                # final text still resolves through Corpus.resolve.
                if channel_name == "lexical":
                    row["lexical_rank"] = rank
                    row["lexical_score"] = candidate.score
                else:
                    row["dense_rank"] = rank
                    row["dense_score"] = candidate.score
                row["rrf"] = float(row["rrf"]) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(
            by_family.items(),
            key=lambda item: (-float(item[1]["rrf"]), item[0]),
        )
        if not ordered:
            return [], ()
        # Normalize the displayed candidate score only relative to the strongest
        # RRF value.  The raw RRF value remains in evidence and should be used
        # for future calibration instead of treating this as a probability.
        max_rrf = float(ordered[0][1]["rrf"]) or 1.0
        candidates: list[Candidate] = []
        evidence_rows: list[SemanticChannelEvidence] = []
        for family_id, row in ordered[:limit]:
            base = row["candidate"]
            assert isinstance(base, Candidate)
            candidates.append(
                Candidate(
                    expression_id=base.expression_id,
                    family_id=family_id,
                    variant_id=base.variant_id,
                    score=float(row["rrf"]) / max_rrf,
                )
            )
            evidence_rows.append(
                SemanticChannelEvidence(
                    family_id=family_id,
                    lexical_rank=row["lexical_rank"],
                    lexical_score=row["lexical_score"],
                    dense_rank=row["dense_rank"],
                    dense_score=row["dense_score"],
                    rrf_score=float(row["rrf"]),
                )
            )
        return candidates, tuple(evidence_rows)

    def search(
        self,
        query: str,
        mode: SearchMode | str,
        limit: int = 10,
    ) -> list[Candidate]:
        return self.search_with_evidence(query, mode, limit)[0]


@dataclass(frozen=True)
class DenseViewEvidence:
    """Best dense view responsible for one family-level similarity score."""

    family_id: str
    dense_score: float
    best_view_name: str
    best_view_text: str


@dataclass(frozen=True)
class MultiVectorSnapshotMetadata:
    """Integrity metadata for a flattened multi-vector family snapshot."""

    model_id: str
    model_revision: str
    corpus_hash: str
    profile: str
    vector_family_ids: tuple[str, ...]
    vector_view_names: tuple[str, ...]
    dimension: int
    normalized: bool
    embeddings_sha256: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "MultiVectorSnapshotMetadata":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_id=str(payload["model_id"]),
            model_revision=str(payload.get("model_revision", "unknown")),
            corpus_hash=str(payload["corpus_hash"]),
            profile=str(payload["profile"]),
            vector_family_ids=tuple(str(item) for item in payload["vector_family_ids"]),
            vector_view_names=tuple(str(item) for item in payload["vector_view_names"]),
            dimension=int(payload["dimension"]),
            normalized=bool(payload.get("normalized", True)),
            embeddings_sha256=str(payload.get("embeddings_sha256", "")),
        )

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "corpus_hash": self.corpus_hash,
                    "profile": self.profile,
                    "vector_family_ids": list(self.vector_family_ids),
                    "vector_view_names": list(self.vector_view_names),
                    "dimension": self.dimension,
                    "normalized": self.normalized,
                    "embeddings_sha256": self.embeddings_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class MultiVectorDenseMeaningRetriever:
    """Max-over-views dense candidate generator.

    One family may occupy several points in embedding space (gloss, scenario,
    canonical+gloss, and optional reviewed enrichment).  The family score is the
    strongest cosine match across its views.  This improves recall capacity
    without allowing dense evidence to make authenticity or ACCEPT decisions.
    """

    # Candidate generation only: legacy score/margin acceptance is forbidden.
    candidate_only = True

    def __init__(
        self,
        corpus: Corpus,
        provider: EmbeddingProvider,
        *,
        profile: SemanticMultiViewProfile | str = SemanticMultiViewProfile.BASE,
        enrichment: dict[str, SemanticEnrichment] | None = None,
        view_embeddings=None,
    ):
        self.corpus = corpus
        self.provider = provider
        self.profile = SemanticMultiViewProfile(profile)
        self.documents = build_semantic_documents(corpus, enrichment)
        self.views = build_semantic_vector_views(
            corpus, profile=self.profile, enrichment=enrichment
        )
        if not self.views:
            raise ValueError("multi-vector semantic index needs at least one view")
        if view_embeddings is None:
            view_embeddings = provider.encode_documents([row.text for row in self.views])
        self._view_embeddings = DenseMeaningRetriever._as_normalized_matrix(view_embeddings)
        if self._view_embeddings.shape[0] != len(self.views):
            raise ValueError("view embedding count does not match semantic views")
        # Stable family -> flattened view indices mapping.
        self._family_indices: dict[str, list[int]] = {}
        for index, row in enumerate(self.views):
            self._family_indices.setdefault(row.family_id, []).append(index)
        self._documents_by_family = {row.family_id: row for row in self.documents}

    @staticmethod
    def _numpy():
        return DenseMeaningRetriever._numpy()

    def search_with_evidence(
        self,
        query: str,
        mode: SearchMode | str = SearchMode.MEANING,
        limit: int = 10,
    ) -> tuple[list[Candidate], tuple[DenseViewEvidence, ...]]:
        if (
            SearchMode(mode) is not SearchMode.MEANING
            or limit <= 0
            or not query
            or not query.strip()
        ):
            return [], ()
        query_vector = DenseMeaningRetriever._as_normalized_matrix(
            self.provider.encode_queries([query])
        )[0]
        similarities = self._view_embeddings @ query_vector
        family_rows: list[tuple[str, float, int]] = []
        for family_id, indices in self._family_indices.items():
            best_index = max(indices, key=lambda idx: (float(similarities[idx]), -idx))
            family_rows.append((family_id, float(similarities[best_index]), best_index))
        family_rows.sort(key=lambda row: (-row[1], row[0]))
        candidates: list[Candidate] = []
        evidence: list[DenseViewEvidence] = []
        for family_id, raw_score, view_index in family_rows[:limit]:
            document = self._documents_by_family[family_id]
            score = max(0.0, min(1.0, raw_score))
            candidates.append(
                Candidate(
                    expression_id=document.expression_id,
                    family_id=family_id,
                    variant_id=document.canonical_variant_id,
                    score=score,
                )
            )
            view = self.views[view_index]
            evidence.append(
                DenseViewEvidence(
                    family_id=family_id,
                    dense_score=raw_score,
                    best_view_name=view.view_name,
                    best_view_text=view.text,
                )
            )
        return candidates, tuple(evidence)

    def search(
        self,
        query: str,
        mode: SearchMode | str,
        limit: int = 10,
    ) -> list[Candidate]:
        return self.search_with_evidence(query, mode, limit)[0]

    def search_many_with_evidence(
        self,
        queries: Sequence[str],
        mode: SearchMode | str = SearchMode.MEANING,
        limit: int = 10,
    ) -> tuple[list[Candidate], tuple[DenseViewEvidence, ...]]:
        """Max-fuse original query and optional semantic rewrites.

        The original query should always be included by callers.  Rewrites can
        improve recall but cannot authorize ACCEPT; this method only returns
        corpus pointers and retrieval evidence.
        """
        if SearchMode(mode) is not SearchMode.MEANING or limit <= 0:
            return [], ()
        cleaned = tuple(
            dict.fromkeys(str(query).strip() for query in queries if str(query).strip())
        )
        if not cleaned:
            return [], ()
        best: dict[str, tuple[Candidate, DenseViewEvidence]] = {}
        for query in cleaned:
            candidates, evidence = self.search_with_evidence(
                query, SearchMode.MEANING, limit=len(self.documents)
            )
            for candidate, row in zip(candidates, evidence):
                previous = best.get(candidate.family_id)
                if previous is None or row.dense_score > previous[1].dense_score:
                    best[candidate.family_id] = (candidate, row)
        ordered = sorted(best.values(), key=lambda item: (-item[1].dense_score, item[0].family_id))[
            :limit
        ]
        return [item[0] for item in ordered], tuple(item[1] for item in ordered)


class PrecomputedMultiVectorEmbeddingProvider:
    """Query encoder paired with a frozen multi-vector document matrix."""

    def __init__(
        self,
        query_provider: EmbeddingProvider,
        corpus: Corpus,
        embeddings_path: str | Path,
        metadata_path: str | Path,
        *,
        enrichment: dict[str, SemanticEnrichment] | None = None,
    ):
        self.query_provider = query_provider
        self.metadata = MultiVectorSnapshotMetadata.load(metadata_path)
        query_model_id = str(getattr(query_provider, "model_id", ""))
        if query_model_id and query_model_id != self.metadata.model_id:
            raise ValueError("query encoder model_id does not match multi-vector snapshot model_id")
        if self.metadata.corpus_hash != corpus.content_hash:
            raise ValueError("multi-vector snapshot corpus hash does not match active corpus")
        views = build_semantic_vector_views(
            corpus,
            profile=self.metadata.profile,
            enrichment=enrichment,
        )
        family_ids = tuple(row.family_id for row in views)
        view_names = tuple(row.view_name for row in views)
        if (
            family_ids != self.metadata.vector_family_ids
            or view_names != self.metadata.vector_view_names
        ):
            raise ValueError(
                "multi-vector snapshot view order does not match active semantic index"
            )
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("precomputed multi-vector snapshots require numpy") from exc
        embeddings_path = Path(embeddings_path)
        if self.metadata.embeddings_sha256:
            digest = hashlib.sha256(embeddings_path.read_bytes()).hexdigest()
            if digest != self.metadata.embeddings_sha256:
                raise ValueError("multi-vector snapshot embedding hash does not match metadata")
        self.view_embeddings = np.load(embeddings_path, allow_pickle=False)
        if self.view_embeddings.shape != (len(views), self.metadata.dimension):
            raise ValueError("multi-vector snapshot shape disagrees with metadata")

    @property
    def model_id(self) -> str:
        return self.metadata.model_id

    def encode_queries(self, texts: Sequence[str]):
        return self.query_provider.encode_queries(texts)

    def encode_documents(self, texts: Sequence[str]):
        if len(texts) != self.view_embeddings.shape[0]:
            raise ValueError("snapshot view request count does not match semantic index")
        return self.view_embeddings
