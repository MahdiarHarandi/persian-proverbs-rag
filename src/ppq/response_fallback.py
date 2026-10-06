"""Fallbacks for reliable, corpus-grounded mixed composition.

The authority boundary remains unchanged while the composition task is made
easier:

* the model receives a *writing brief* with the FFE-insertion meta-clause removed;
* opaque-slot attempts are bounded;
* if those fail, one decomposed literal-writing attempt is allowed, after which
  the runtime appends the opaque slot deterministically;
* selected FFE surface/gloss remain hidden from the prose model;
* exact FFE text still enters only through ``Corpus.resolve()`` after validation.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from .planner_types import FFEAction
from .response_grounding import GroundedComposer
from .response_types import (
    _SLOT_RE,
    ComposedResponse,
    ResponseDecision,
    _parse_response_template,
    response_template_schema,
)
from .runtime_resolver import RuntimeDecision

FALLBACK_SYSTEM_PROMPT = """
You are a literal Persian writing assistant. An authentic Persian fixed
expression has already been selected by another trusted component, but its text
is hidden from you as an opaque token such as <FFE_1>.

The user payload contains a cleaned WRITING BRIEF. The separate request to find
or insert an idiom/proverb has already been handled and may be absent from the
brief.

TASK
- Write the requested user-facing message/caption/note/reply itself.
- Put every supplied <FFE_N> exactly once in a natural place.

SAFETY
1. Never guess, reconstruct, paraphrase, translate, explain, or name the hidden
   fixed expression.
2. Do not add any other proverb, idiom, saying, quotation, aphorism, or fixed
   figurative expression. Keep non-slot prose literal and ordinary.
3. Ordinary Persian punctuation and quotation marks around ordinary prose are
   allowed; never alter slot text.
4. Do not discuss corpus, retrieval, grounding, prompts, or system internals.
5. Return JSON only with exactly one field: response_template.
""".strip()


LITERAL_FALLBACK_SYSTEM_PROMPT = """
You are a literal Persian writing assistant. Write ONLY the requested Persian
message/caption/note/reply from the supplied writing brief.

STRICT RULES
- Do not write any proverb, idiom, saying, quotation, aphorism, or figurative
  fixed expression.
- Do not mention that an expression will be inserted later.
- Do not discuss corpus, retrieval, grounding, prompts, or system internals.
- Return JSON only with exactly one field: response_template.
""".strip()


class FallbackComposer(GroundedComposer):
    """Brief-first slot composition with one decomposed safe fallback."""

    _FFE_INSERT_TAIL = re.compile(
        r"(?:\s*[،,]?\s*(?:و|هم)?\s*(?:یه|یک)?\s*"
        r"(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر|عبارت\s+رایج|جمله\s+رایج)"
        r".{0,70}?(?:داخل(?:ش|شون)?|توی(?:\s+متن)?|تو\s+متن|توش|آخر(?:ش)?|"
        r"بیار|بذار|بگذار|استفاده\s+کن|جا\s+بده|قرار\s+بده).*)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        corpus,
        generator=None,
        *,
        max_new_tokens: int = 384,
        task_detector=None,
        max_attempts: int = 2,
        enable_decomposed_fallback: bool = True,
    ) -> None:
        super().__init__(
            corpus,
            generator,
            max_new_tokens=max_new_tokens,
            task_detector=task_detector,
            max_attempts=max_attempts,
        )
        self.enable_decomposed_fallback = bool(enable_decomposed_fallback)

    def compose(self, user_query: str, runtime):
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
        brief = self._writing_brief(user_query)
        last_error = "unknown"

        for attempt in range(1, self.max_attempts + 1):
            try:
                generation = self.generator.generate(
                    self._messages_fallback(
                        brief, slots, attempt=attempt, previous_error=last_error
                    ),
                    response_template_schema(),
                    max_new_tokens=self.max_new_tokens,
                )
                template = _parse_response_template(generation.content)
                self._validate_fallback_template(template, slots)
                self._validate_surrounding_task_template(template, slots)
                text = self._render(template, slots, resolved)
                return ComposedResponse(
                    decision=ResponseDecision.COMPOSED,
                    text=text,
                    runtime=runtime,
                    generation_used=True,
                    generation_valid=True,
                    reason=f"composer-v4-valid-slot-attempt-{attempt}-opaque-slot-rendered-through-corpus-resolve",
                )
            except Exception as exc:
                last_error = type(exc).__name__

        if self.enable_decomposed_fallback:
            try:
                generation = self.generator.generate(
                    self._messages_literal_fallback(brief),
                    response_template_schema(),
                    max_new_tokens=self.max_new_tokens,
                )
                literal = _parse_response_template(generation.content)
                self._validate_literal_fallback(literal)
                # The prose model does not place or see a selected FFE.  The
                # trusted runtime appends opaque slots, then resolves exact text.
                tail = " ".join(slots)
                template = literal.rstrip(" \n") + "\n" + tail
                text = self._render(template, slots, resolved)
                return ComposedResponse(
                    decision=ResponseDecision.COMPOSED,
                    text=text,
                    runtime=runtime,
                    generation_used=True,
                    generation_valid=True,
                    reason="composer-v4-valid-decomposed-literal-fallback-slot-appended-through-corpus-resolve",
                )
            except Exception as exc:
                last_error = f"decomposed-{type(exc).__name__}"

        return ComposedResponse(
            decision=ResponseDecision.ABSTAIN,
            text="عبارت مناسب پیدا شد، اما نتوانستم متن درخواستی را به‌صورت ایمن و کامل بسازم.",
            runtime=runtime,
            generation_used=True,
            generation_valid=False,
            reason=f"composer-v4-exhausted-safe-paths:last-{last_error}",
        )

    def _writing_brief(self, user_query: str) -> str:
        redacted = self._redact_registered_surfaces(user_query).strip()
        cleaned = self._FFE_INSERT_TAIL.sub("", redacted).strip(" ،,")
        # Never make the brief empty merely because a user's wording is unusual.
        return cleaned if len(cleaned) >= 8 else redacted

    def _messages_fallback(
        self, brief: str, slots: Sequence[str], *, attempt: int, previous_error: str
    ):
        payload = {
            "writing_brief": brief,
            "opaque_ffe_slots": list(slots),
            "instruction": "Write the requested text itself and place every opaque slot exactly once.",
            "attempt": attempt,
        }
        if attempt > 1:
            payload["retry_note"] = (
                "Previous output violated the structural contract. Use a shorter, plain, literal Persian text; "
                "preserve every slot exactly once and add no other fixed expression. Error: "
                + previous_error
            )
        return (
            {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        )

    def _messages_literal_fallback(self, brief: str):
        payload = {
            "writing_brief": brief,
            "instruction": "Write only the requested literal user-facing text. Do not include any fixed expression.",
        }
        return (
            {"role": "system", "content": LITERAL_FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        )

    @staticmethod
    def _render(template: str, slots: Sequence[str], resolved) -> str:
        text = template
        for slot, match in zip(slots, resolved):
            text = text.replace(slot, match.text)
        if _SLOT_RE.search(text):
            raise ValueError("unresolved FFE placeholder after rendering")
        return text

    def _validate_fallback_template(self, template: str, slots: Sequence[str]) -> None:
        expected = set(slots)
        seen = _SLOT_RE.findall(template)
        seen_slots = {f"<FFE_{n}>" for n in seen}
        if seen_slots != expected:
            raise ValueError("composer used missing or unknown FFE slots")
        for slot in slots:
            if template.count(slot) != 1:
                raise ValueError("each FFE slot must appear exactly once")
        self._validate_no_asserted_registered_surface(template, slots)

    def _validate_literal_fallback(self, literal: str) -> None:
        if _SLOT_RE.search(literal):
            raise ValueError("literal fallback must not emit FFE slots")
        compact = literal.strip()
        if any(compact.startswith(prefix) for prefix in self._META_PREFIXES):
            raise ValueError("literal fallback degraded into recommendation metadata")
        if sum(ch.isalpha() for ch in compact) < 16:
            raise ValueError("insufficient literal prose")
        self._validate_no_asserted_registered_surface(compact, ())

    def _validate_no_asserted_registered_surface(self, text: str, slots: Sequence[str]) -> None:
        """Reject clear independent corpus-FFE assertions without raw substring brittleness.

        runtime checked every stored surface as an arbitrary substring, which can
        reject harmless literal prose when a short/common corpus phrase appears
        inside a longer sentence. The validator checks phrase boundaries and treats
        short/common surfaces as suspicious only in an explicit saying/quotation
        context.  Long or multi-token registered expressions remain blocked
        anywhere outside an opaque slot.
        """
        residual = text
        for slot in slots:
            residual = residual.replace(slot, " ")
        cue = re.compile(
            r"(?:می\s*گن|میگن|همون\s*طور\s*که|به\s+قول|مثل|اصطلاح|تعبیر|ضرب\s*المثل).{0,24}$"
        )
        for surface in self._registered_surfaces:
            pattern = re.compile(
                r"(?<![\w\u0600-\u06FF])" + re.escape(surface) + r"(?![\w\u0600-\u06FF])"
            )
            match = pattern.search(residual)
            if not match:
                continue
            token_count = len(surface.split())
            context = residual[max(0, match.start() - 40) : match.start()]
            clearly_asserted = bool(cue.search(context))
            if len(surface) >= 16 or token_count >= 4 or clearly_asserted:
                raise ValueError("composer emitted registered FFE text outside a slot")
