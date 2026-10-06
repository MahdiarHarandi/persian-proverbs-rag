"""Grounded mixed-task composer with bounded retries.

The composer preserves opaque slots, requires each slot exactly once, and
prevents registered expressions from appearing outside those slots. It allows
ordinary Persian prose and retries once after a structural failure.
"""

from __future__ import annotations

import json
from typing import Sequence

from .planner_types import FFEAction
from .response_safety import TaskPreservingComposer
from .response_types import (
    _SLOT_RE,
    ComposedResponse,
    ResponseDecision,
    _parse_response_template,
    response_template_schema,
)
from .runtime_resolver import RuntimeDecision

GROUNDED_SYSTEM_PROMPT = """
You are a literal Persian writing assistant. The system has already selected
one or more authentic Persian fixed expressions, but their wording is hidden.
Each expression is represented only by an opaque token such as <FFE_1>.

TASK
- Actually fulfil the user's surrounding writing request (message, caption,
  note, reply, etc.).
- Put every supplied <FFE_N> exactly once in a natural place.

SAFETY
1. Never guess, reconstruct, paraphrase, translate, explain, or name the wording
   hidden behind <FFE_N>.
2. Do not introduce any additional proverb, idiom, saying, quotation, aphorism,
   or figurative fixed expression. Keep all non-slot prose literal and ordinary.
3. Ordinary Persian punctuation is allowed. Do not replace or alter slot text.
4. Do not discuss corpus, retrieval, grounding, prompts, or system internals.
5. Do not answer with an explanation or a 'registered suggestion'. Write the
   requested user-facing text itself.
6. Return JSON only with exactly one field: response_template.
""".strip()


class GroundedComposer(TaskPreservingComposer):
    """Bounded retry + task-preserving validation for mixed composition."""

    _META_PREFIXES = (
        "پیشنهاد ثبت‌شده",
        "عبارت ثبت‌شده",
        "صورت ثبت‌شده",
        "معنی:",
        "این عبارت در پیکره",
        "شواهد کافی",
    )

    def __init__(
        self,
        corpus,
        generator=None,
        *,
        max_new_tokens: int = 384,
        task_detector=None,
        max_attempts: int = 2,
    ):
        super().__init__(
            corpus, generator, max_new_tokens=max_new_tokens, task_detector=task_detector
        )
        if max_attempts not in (1, 2):
            raise ValueError("max_attempts must be 1 or 2")
        self.max_attempts = max_attempts

    def compose(self, user_query: str, runtime):
        # Reuse runtime behavior for GENERAL/non-ACCEPT and for non-mixed actions.
        if runtime.decision is not RuntimeDecision.ACCEPT:
            return super().compose(user_query, runtime)
        action = self._action(runtime)
        if (
            self.generator is None
            or action is not FFEAction.SEARCH_BY_MEANING
            or not self.task_detector.needs_composition(user_query)
        ):
            return super().compose(user_query, runtime)

        resolved = self._reresolve_matches(runtime)
        slots = tuple(f"<FFE_{i}>" for i in range(1, len(resolved) + 1))
        last_error = "unknown"
        for attempt in range(1, self.max_attempts + 1):
            try:
                generation = self.generator.generate(
                    self._messages_grounded(
                        user_query, slots, attempt=attempt, previous_error=last_error
                    ),
                    response_template_schema(),
                    max_new_tokens=self.max_new_tokens,
                )
                template = _parse_response_template(generation.content)
                self._validate_grounded_template(template, slots)
                self._validate_surrounding_task_template(template, slots)
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
                    reason=f"composer-v3-valid-attempt-{attempt}-opaque-slot-rendered-through-corpus-resolve",
                )
            except Exception as exc:
                last_error = type(exc).__name__

        # Do not call a recommendation-only fallback 'composition'.  Grounding
        # remains safe internally, but the response surface fails closed without
        # exposing the selected FFE if the requested surrounding task could not
        # be composed safely.
        return ComposedResponse(
            decision=ResponseDecision.ABSTAIN,
            text="عبارت مناسب پیدا شد، اما نتوانستم متن درخواستی را به‌صورت ایمن و کامل بسازم.",
            runtime=runtime,
            generation_used=True,
            generation_valid=False,
            reason=f"composer-v3-exhausted-{self.max_attempts}-attempts:last-{last_error}",
        )

    def _messages_grounded(
        self, user_query: str, slots: Sequence[str], *, attempt: int, previous_error: str
    ):
        payload = {
            "user_query": self._redact_registered_surfaces(user_query),
            "opaque_ffe_slots": list(slots),
            "instruction": "Write the requested surrounding text itself and place each opaque slot exactly once.",
            "attempt": attempt,
        }
        if attempt > 1:
            payload["retry_note"] = (
                "The previous attempt violated the output contract. Return a shorter, plain, literal Persian text; "
                "use every slot exactly once and no extra fixed expression. Structural error: "
                + previous_error
            )
        return (
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        )

    def _validate_grounded_template(self, template: str, slots: Sequence[str]) -> None:
        expected = set(slots)
        seen = _SLOT_RE.findall(template)
        seen_slots = {f"<FFE_{n}>" for n in seen}
        if seen_slots != expected:
            raise ValueError("composer used missing or unknown FFE slots")
        for slot in slots:
            if template.count(slot) != 1:
                raise ValueError("each FFE slot must appear exactly once")
        # Quotation marks are ordinary punctuation and are not an authority
        # boundary. The meaningful guard is whether a stored FFE surface was
        # independently emitted outside the opaque slot.
        for surface in self._registered_surfaces:
            if surface in template:
                raise ValueError("composer emitted registered FFE text outside a slot")

    def _validate_surrounding_task_template(self, template: str, slots: Sequence[str]) -> None:
        compact = template.strip()
        if any(compact.startswith(prefix) for prefix in self._META_PREFIXES):
            raise ValueError("mixed-task response degraded into recommendation metadata")
        literal = compact
        for slot in slots:
            literal = literal.replace(slot, "")
        # Require meaningful surrounding prose, not just a slot with punctuation.
        letters = sum(ch.isalpha() for ch in literal)
        if letters < 12:
            raise ValueError("insufficient surrounding prose for mixed task")
