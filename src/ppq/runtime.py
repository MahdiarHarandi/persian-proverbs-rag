"""Final composition of the corpus-grounded assistant runtime."""

from __future__ import annotations

from .applicability_semantic import ApplicabilitySemanticVerifier
from .assistant_runtime import PPQAssistantRuntime
from .contracts import CandidateRetriever
from .corpus import Corpus
from .evidence_discovery import SemanticEvidenceDiscovery
from .multi_scope import MultiScopePlanner, StructuredJsonGenerator
from .planner import Planner
from .ppq_service import PPQService
from .response_composer import SafeResponseComposer
from .semantic_rescue import RawFirstSemanticRescueService
from .service import RetrievalService
from .structured_applicability import build_acceptance_gate
from .surface_binder import SurfaceMentionBinder
from .surface_restore import SurfaceRestore


def build_assistant_runtime(
    corpus: Corpus,
    retriever: CandidateRetriever,
    generator: StructuredJsonGenerator,
    *,
    surface_service: RetrievalService | None = None,
    raw_semantic_verifications: int = 2,
    hint_rescue_verifications: int = 2,
    cross_view_rescue_verifications: int = 1,
    cross_view_window: int = 8,
    enable_multi_scope: bool = True,
    candidate_source_name: str = "bge-m3-gloss-only-candidate",
    composer_max_new_tokens: int = 384,
) -> PPQAssistantRuntime:
    if not getattr(retriever, "candidate_only", False):
        raise ValueError("semantic runtime requires a candidate-only retriever")

    gate = build_acceptance_gate(generator)
    semantic_verifier = ApplicabilitySemanticVerifier(gate)
    discovery = SemanticEvidenceDiscovery(retriever, source_name=candidate_source_name)
    semantic = RawFirstSemanticRescueService(
        corpus,
        discovery,
        semantic_verifier,
        raw_verifications=raw_semantic_verifications,
        hint_rescue_verifications=hint_rescue_verifications,
        cross_view_rescue_verifications=cross_view_rescue_verifications,
        cross_view_window=cross_view_window,
    )

    surface = surface_service or RetrievalService(corpus)
    planner = Planner(generator, SurfaceMentionBinder(surface))
    scope = MultiScopePlanner(generator) if enable_multi_scope else None
    service = PPQService(
        corpus,
        planner,
        surface,
        semantic,
        multi_scope_planner=scope,
        enable_retrieve_examples=True,
        restore_rescue=SurfaceRestore(corpus),
    )
    composer = SafeResponseComposer(
        corpus,
        generator,
        max_new_tokens=composer_max_new_tokens,
        max_attempts=2,
    )
    return PPQAssistantRuntime(service, composer)
