"""Small dependency-free interfaces at PPQ integration boundaries.

Keeping the application service coupled only to these protocols separates
surface correction, semantic retrieval, attestation, and corpus-bound
rendering.

These protocols intentionally expose *candidate identifiers*, never generated
fixed-expression text.  Final display text must still cross ``Corpus.resolve``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .models import Candidate, SearchMode

if TYPE_CHECKING:
    from .retrieval import SurfaceEvidence


class CandidateRetriever(Protocol):
    """Rank corpus-backed candidates for one retrieval mode."""

    def search(
        self,
        query: str,
        mode: SearchMode | str,
        limit: int = 10,
    ) -> list[Candidate]: ...


class SurfaceEvidenceRetriever(CandidateRetriever, Protocol):
    """Surface retriever that can expose verifier-only evidence components."""

    def surface_evidence(
        self,
        query: str,
        candidate: Candidate,
    ) -> "SurfaceEvidence": ...
