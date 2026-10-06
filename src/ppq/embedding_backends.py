"""Optional Sentence-Transformers adapters used by semantic experiments.

The core PPQ package does not import sentence-transformers.  This file is only
loaded by explicit dense-retrieval experiments or deployments that choose an
embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SentenceTransformerModelSpec:
    model_id: str
    query_mode: str = "plain"
    document_mode: str = "plain"
    query_instruction: str = ""
    trust_remote_code: bool = False


MODEL_SPECS: dict[str, SentenceTransformerModelSpec] = {
    "multilingual-e5-base": SentenceTransformerModelSpec(
        model_id="intfloat/multilingual-e5-base",
        query_mode="e5-query",
        document_mode="e5-passage",
    ),
    "bge-m3": SentenceTransformerModelSpec(
        model_id="BAAI/bge-m3",
    ),
    "qwen3-embedding-0.6b": SentenceTransformerModelSpec(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        query_mode="qwen-instruct",
        query_instruction=(
            "Given a Persian user request, retrieve the reviewed Persian fixed-"
            "expression meaning that best matches the intended figurative meaning."
        ),
    ),
    "embeddinggemma-300m": SentenceTransformerModelSpec(
        model_id="google/embeddinggemma-300m",
        query_mode="encode-query",
        document_mode="encode-document",
    ),
}


def resolve_huggingface_revision(model_id: str, revision: str | None = None) -> str:
    """Resolve a mutable Hub revision to an immutable commit SHA."""
    if (
        revision
        and len(revision) >= 12
        and all(ch in "0123456789abcdef" for ch in revision.lower())
    ):
        return revision
    try:
        from huggingface_hub import model_info
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface-hub is required to pin model revisions") from exc
    try:
        info = model_info(model_id, revision=revision or "main")
    except Exception as exc:  # pragma: no cover - network/auth dependent
        raise RuntimeError(
            f"could not resolve immutable revision for {model_id}; pass --revision explicitly"
        ) from exc
    if not info.sha:
        raise RuntimeError(f"Hub returned no commit SHA for {model_id}")
    return str(info.sha)


class SentenceTransformersProvider:
    """Model-aware embedding provider with explicit asymmetric formatting."""

    def __init__(
        self,
        spec: SentenceTransformerModelSpec,
        *,
        device: str | None = None,
        batch_size: int = 32,
        revision: str | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "install sentence-transformers to run neural embedding experiments"
            ) from exc
        kwargs = {"trust_remote_code": spec.trust_remote_code}
        if device:
            kwargs["device"] = device
        if revision:
            kwargs["revision"] = revision
        self.spec = spec
        self.batch_size = batch_size
        self.revision = revision or "main"
        self.model = SentenceTransformer(spec.model_id, **kwargs)

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def _format_queries(self, texts: Sequence[str]) -> list[str]:
        if self.spec.query_mode == "e5-query":
            return [f"query: {text}" for text in texts]
        if self.spec.query_mode == "qwen-instruct":
            instruction = self.spec.query_instruction
            return [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
        return list(texts)

    def _format_documents(self, texts: Sequence[str]) -> list[str]:
        if self.spec.document_mode == "e5-passage":
            return [f"passage: {text}" for text in texts]
        return list(texts)

    def encode_queries(self, texts: Sequence[str]):
        if self.spec.query_mode == "encode-query" and hasattr(self.model, "encode_query"):
            return self.model.encode_query(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return self.model.encode(
            self._format_queries(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_documents(self, texts: Sequence[str]):
        if self.spec.document_mode == "encode-document" and hasattr(self.model, "encode_document"):
            return self.model.encode_document(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return self.model.encode(
            self._format_documents(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
