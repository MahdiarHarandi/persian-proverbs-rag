"""One-call runtime assistant boundary built from structured grounding + safe composition."""

from __future__ import annotations

from .ppq_service import PPQService
from .response_types import BaseResponseComposer, ComposedResponse


class PPQAssistantRuntime:
    """Run grounding first, then compose only from the resulting safe state."""

    def __init__(self, service: PPQService, composer: BaseResponseComposer) -> None:
        self.service = service
        self.composer = composer

    def handle(self, query: str, *, query_id: str | None = None) -> ComposedResponse:
        runtime = self.service.process(query, query_id=query_id)
        return self.composer.compose(query, runtime)
