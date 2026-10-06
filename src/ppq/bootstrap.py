"""Runtime construction for the PPQ application entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .corpus import load_corpus
from .dense_semantic import DenseMeaningRetriever, PrecomputedEmbeddingProvider
from .embedding_backends import MODEL_SPECS, SentenceTransformersProvider
from .gemma_multiscope import GEMMA4_MODEL_ID, GEMMA4_REVISION, GemmaHFJsonGenerator
from .gemma_planner import GemmaFFEPlanBackend
from .runtime import build_assistant_runtime

BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def build_runtime(
    project_root: str | Path,
    embedding_device: str = "cuda:0",
    log: Callable[[str], None] | None = None,
):
    """Build the frozen PPQ runtime used by CLI, console, and web UI."""
    root = Path(project_root)
    emit = log or (lambda _message: None)

    emit("[1/3] Loading corpus...")
    corpus = load_corpus(
        root / "data/corpus/expressions.jsonl",
        root / "data/corpus/sources.jsonl",
    )

    emit(f"[2/3] Loading BGE-M3 query encoder on {embedding_device}...")
    spec = MODEL_SPECS["bge-m3"]
    query_provider = SentenceTransformersProvider(
        spec,
        device=embedding_device,
        batch_size=32,
        revision=BGE_REVISION,
    )
    snapshot = PrecomputedEmbeddingProvider(
        query_provider,
        corpus,
        root / "artifacts/family_embeddings.npy",
        root / "artifacts/family_embeddings.metadata.json",
    )
    retriever = DenseMeaningRetriever(
        corpus,
        snapshot,
        text_view="gloss-only",
        document_embeddings=snapshot.document_embeddings,
    )

    emit("[3/3] Loading Gemma 4 E4B...")
    planner_backend = GemmaFFEPlanBackend.from_pretrained(
        model_id=GEMMA4_MODEL_ID,
        revision=GEMMA4_REVISION,
        device_map="balanced",
        attention="sdpa",
    )
    generator = GemmaHFJsonGenerator(
        processor=planner_backend.processor,
        model=planner_backend.model,
        model_id=GEMMA4_MODEL_ID,
    )

    return build_assistant_runtime(
        corpus,
        retriever,
        generator,
        raw_semantic_verifications=2,
        hint_rescue_verifications=2,
        cross_view_rescue_verifications=1,
        cross_view_window=8,
        enable_multi_scope=True,
        candidate_source_name="bge-m3-gloss-only-snapshot",
    )
