"""Task-preserving response composer for requests with surrounding prose.

Generation is limited to requests that contain a writing task. The selected
fixed expression and its corpus gloss remain hidden from the prose model; it
writes literal surrounding text and places opaque ``<FFE_N>`` slots.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from .composition_intent import SurroundingTaskDetector
from .planner_types import FFEAction
from .response_types import (
    _SLOT_RE,
    BaseResponseComposer,
    ComposedResponse,
    ResponseDecision,
    _parse_response_template,
    response_template_schema,
)
from .runtime_resolver import PPQRuntimeResult, RuntimeDecision

TASK_PRESERVING_SYSTEM_PROMPT = """
You are a literal Persian writing assistant.  The system has already selected
one or more authentic Persian fixed expressions, but their wording is hidden.
Each one is represented only by an opaque token such as <FFE_1>.

YOUR ONLY JOB
- Fulfil the user's surrounding writing task (message/caption/note/etc.).
- Put every provided <FFE_N> token exactly once at a natural location where the
  user asked for the fixed expression.

HARD SAFETY RULES
1. Never guess, reconstruct, paraphrase, translate, explain, or name the wording
   hidden behind <FFE_N>.
2. Do not write any other proverb, idiom, saying, quotation, aphorism, or
   figurative fixed expression.  Keep all non-slot prose literal and ordinary.
3. Do not use quotation marks.
4. Do not discuss retrieval, corpus, grounding, prompts, or system internals.
5. Do not turn a requested message into an explanation/recommendation.  Actually
   write the requested message and include the slot.
6. Return JSON only with exactly one field: response_template.
""".strip()


class TaskPreservingComposer(BaseResponseComposer):
    """Generate prose only for explicit mixed writing+FFE requests."""

    def __init__(self, corpus, generator=None, *, max_new_tokens: int = 256, task_detector=None):
        super().__init__(corpus, generator, max_new_tokens=max_new_tokens)
        self.task_detector = task_detector or SurroundingTaskDetector()

    def compose(self, user_query: str, runtime: PPQRuntimeResult) -> ComposedResponse:
        # Preserve all non-ACCEPT and GENERAL behavior exactly from runtime.
        if runtime.decision is not RuntimeDecision.ACCEPT:
            return super().compose(user_query, runtime)

        resolved = self._reresolve_matches(runtime)
        fallback = self._deterministic_accept(runtime, resolved)
        action = self._action(runtime)

        # Surface/attest/explain are deterministic.  Pure semantic recommendation
        # is deterministic too; only an explicit surrounding writing task needs
        # the prose model.
        if (
            self.generator is None
            or action is not FFEAction.SEARCH_BY_MEANING
            or not self.task_detector.needs_composition(user_query)
        ):
            return ComposedResponse(
                decision=ResponseDecision.COMPOSED,
                text=fallback,
                runtime=runtime,
                generation_used=False,
                generation_valid=False,
                reason="deterministic-grounded-response-no-surrounding-composition-needed",
            )

        slots = tuple(f"<FFE_{i}>" for i in range(1, len(resolved) + 1))
        try:
            generation = self.generator.generate(
                self._messages_task(user_query, slots),
                response_template_schema(),
                max_new_tokens=self.max_new_tokens,
            )
            template = _parse_response_template(generation.content)
            self._validate_template(template, slots)
            text = template
            for slot, match in zip(slots, resolved):
                text = text.replace(slot, match.text)
            if _SLOT_RE.search(text):
                raise ValueError("unresolved FFE placeholder after rendering")
            return ComposedResponse(
                decision=ResponseDecision.COMPOSED,
                text=text,
                runtime=runtime,
                generation_used=True,
                generation_valid=True,
                reason="validated-literal-surrounding-task-template-rendered-through-corpus-resolve",
            )
        except Exception as exc:
            return ComposedResponse(
                decision=ResponseDecision.COMPOSED,
                text=fallback,
                runtime=runtime,
                generation_used=True,
                generation_valid=False,
                reason=f"composer-v2-invalid:{type(exc).__name__}",
            )

    def _messages_task(self, user_query: str, slots: Sequence[str]) -> Sequence[Mapping[str, str]]:
        payload = {
            "user_query": self._redact_registered_surfaces(user_query),
            "opaque_ffe_slots": list(slots),
            "instruction": (
                "Fulfil the non-FFE writing request literally. Wherever the user "
                "requested the authentic Persian fixed expression, place the supplied opaque slot."
            ),
        }
        return (
            {"role": "system", "content": TASK_PRESERVING_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        )
