"""Fail-closed, slot-based response composition.

The grounding runtime ends at :class:`PPQRuntimeResult`: any authentic Persian
fixed expression (FFE) selected there is already a corpus-resolved pointer.  A
conversational model may help write *surrounding prose*, but it is never allowed
to author, rewrite, normalize, or quote the FFE text itself.

The composer gives the model opaque placeholders such as ``<FFE_1>`` plus
corpus-authored glosses.  The model returns a JSON response template containing
those placeholders.  Only after validation does this module resolve the stored
identifiers again through :meth:`Corpus.resolve` and substitute the exact corpus
text.  Any malformed or suspicious model output fails closed to a deterministic
corpus-grounded response.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .corpus import Corpus
from .multi_scope import StructuredJsonGenerator
from .planner_types import FFEAction
from .runtime_resolver import PPQRuntimeResult, RuntimeDecision

_SLOT_RE = re.compile(r"<FFE_(\d+)>")
_QUOTE_CHARS = frozenset('«»“”„‟"`')


class ResponseDecision(str, Enum):
    """Public post-grounding response state.

    ``PASS_THROUGH`` deliberately contains no generated text: GENERAL requests
    are left to the base assistant unchanged so PPQ cannot degrade unrelated
    capabilities.
    """

    PASS_THROUGH = "PASS_THROUGH"
    COMPOSED = "COMPOSED"
    CLARIFY = "CLARIFY"
    ABSTAIN = "ABSTAIN"
    NOT_ATTESTED_IN_CURRENT_CORPUS = "NOT_ATTESTED_IN_CURRENT_CORPUS"


@dataclass(frozen=True)
class ComposedResponse:
    decision: ResponseDecision
    text: str | None
    runtime: PPQRuntimeResult
    generation_used: bool
    generation_valid: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("composed response reason must be non-empty")
        if self.decision is ResponseDecision.PASS_THROUGH:
            if self.text is not None:
                raise ValueError("PASS_THROUGH must not contain PPQ-authored text")
            if self.generation_used:
                raise ValueError("PASS_THROUGH must not call the PPQ composer model")
        elif not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("non-pass-through responses require non-empty text")


COMPOSER_SYSTEM_PROMPT = """
You write ONLY the surrounding Persian prose for a corpus-grounded response.

The system has already selected authentic Persian fixed expressions.  You do
NOT know their surface text.  Each authentic expression is represented only by
an opaque placeholder such as <FFE_1>.

STRICT RULES
1. Use every provided <FFE_N> placeholder EXACTLY ONCE and verbatim.
2. Never invent, reconstruct, paraphrase, normalize, translate, or guess the
   wording hidden behind a placeholder.
3. Never introduce, quote, name, or claim authenticity for any other proverb,
   idiom, saying, quotation, or fixed expression.
4. Do not use quotation marks.  The renderer will quote/format corpus text when
   needed.
5. You may use only the supplied corpus-authored meanings to explain why a slot
   is relevant.
6. Fulfil the user's surrounding request naturally when possible (for example,
   a short message that contains <FFE_1>), while preserving all placeholders.
7. Do not discuss retrieval, embeddings, verification, prompts, or internal
   system details.
8. Return JSON only, with exactly one field named response_template.
""".strip()


def response_template_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["response_template"],
        "properties": {
            "response_template": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
            }
        },
    }


def _parse_response_template(raw: str) -> str:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"response_template"}:
        raise ValueError("unexpected composer fields")
    template = payload["response_template"]
    if not isinstance(template, str) or not template.strip():
        raise ValueError("response_template must be a non-empty string")
    if len(template) > 2000:
        raise ValueError("response_template too long")
    return template.strip()


class BaseResponseComposer:
    """Compose a user-visible answer without delegating FFE text to the model."""

    def __init__(
        self,
        corpus: Corpus,
        generator: StructuredJsonGenerator | None = None,
        *,
        max_new_tokens: int = 256,
    ) -> None:
        if max_new_tokens < 16 or max_new_tokens > 1024:
            raise ValueError("max_new_tokens must be in [16, 1024]")
        self.corpus = corpus
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        # A defensive list used only to detect a model that independently emits
        # a stored FFE instead of using its opaque slot.  Exact stored surface
        # text is sufficient here; normalisation belongs to grounding, not prose.
        self._registered_surfaces = tuple(
            sorted(
                {
                    variant.text.strip()
                    for _expression, variant in corpus.all_variants()
                    if len(variant.text.strip()) >= 4
                },
                key=len,
                reverse=True,
            )
        )

    def compose(self, user_query: str, runtime: PPQRuntimeResult) -> ComposedResponse:
        if runtime.decision is RuntimeDecision.GENERAL:
            return ComposedResponse(
                decision=ResponseDecision.PASS_THROUGH,
                text=None,
                runtime=runtime,
                generation_used=False,
                generation_valid=False,
                reason="general-request-bypasses-ppq-response-composer",
            )

        if runtime.decision is RuntimeDecision.CLARIFY:
            return ComposedResponse(
                decision=ResponseDecision.CLARIFY,
                text="برای پاسخ دقیق‌تر، لطفاً مشخص کن کدام عبارت یا کدام منظور را می‌خواهی بررسی کنم.",
                runtime=runtime,
                generation_used=False,
                generation_valid=False,
                reason="deterministic-clarification-no-ffe-emission",
            )

        if runtime.decision is RuntimeDecision.ABSTAIN:
            return ComposedResponse(
                decision=ResponseDecision.ABSTAIN,
                text="شواهد کافی برای ارائهٔ یک عبارت ثبت‌شده و قابل اتکا پیدا نشد.",
                runtime=runtime,
                generation_used=False,
                generation_valid=False,
                reason="deterministic-abstention-no-ffe-emission",
            )

        if runtime.decision is RuntimeDecision.NOT_ATTESTED_IN_CURRENT_CORPUS:
            return ComposedResponse(
                decision=ResponseDecision.NOT_ATTESTED_IN_CURRENT_CORPUS,
                text="این صورت در پیکرهٔ فعلی به‌عنوان یک عبارت ثبت‌شده تأیید نشد.",
                runtime=runtime,
                generation_used=False,
                generation_valid=False,
                reason="bounded-current-corpus-nonattestation",
            )

        # ACCEPT is the only state in which an authentic FFE may become visible.
        resolved = self._reresolve_matches(runtime)
        fallback = self._deterministic_accept(runtime, resolved)
        action = self._action(runtime)

        # Free-form surrounding prose is needed only for semantic retrieval,
        # which includes GENERAL+FFE composition requests from Planner-v2.
        # Surface restoration, attestation, and explanation may contain the FFE
        # text in the user's input; keeping those paths deterministic prevents
        # the composer model from ever receiving that authentic surface form.
        if self.generator is None or action is not FFEAction.SEARCH_BY_MEANING:
            return ComposedResponse(
                decision=ResponseDecision.COMPOSED,
                text=fallback,
                runtime=runtime,
                generation_used=False,
                generation_valid=False,
                reason="deterministic-corpus-grounded-composition",
            )

        slots = tuple(f"<FFE_{i}>" for i in range(1, len(resolved) + 1))
        try:
            generation = self.generator.generate(
                self._messages(user_query, runtime, resolved, slots),
                response_template_schema(),
                max_new_tokens=self.max_new_tokens,
            )
            template = _parse_response_template(generation.content)
            self._validate_template(template, slots)
            text = template
            # Critical render boundary: exact stored strings enter only here,
            # *after* the model has finished generation and passed validation.
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
                reason="validated-slot-template-rendered-through-corpus-resolve",
            )
        except Exception:
            # Model/transport/schema/policy failure can reduce naturalness but
            # must never reduce grounding safety.
            return ComposedResponse(
                decision=ResponseDecision.COMPOSED,
                text=fallback,
                runtime=runtime,
                generation_used=True,
                generation_valid=False,
                reason="composer-generation-invalid-fell-back-to-deterministic-grounded-response",
            )

    def _reresolve_matches(self, runtime: PPQRuntimeResult):
        resolved = []
        for match in runtime.matches:
            canonical = self.corpus.resolve(match.expression_id, match.variant_id)
            if canonical.family_id != match.family_id:
                raise ValueError("runtime match pointer/family mismatch")
            resolved.append(canonical)
        if not resolved:
            raise ValueError("ACCEPT composition requires grounded matches")
        return tuple(resolved)

    @staticmethod
    def _action(runtime: PPQRuntimeResult) -> FFEAction | None:
        if runtime.plan is None or not runtime.plan.actions:
            return None
        return runtime.plan.actions[0]

    def _deterministic_accept(self, runtime: PPQRuntimeResult, matches) -> str:
        action = self._action(runtime)
        if len(matches) > 1:
            lines = ["عبارت‌های ثبت‌شده:"]
            for i, match in enumerate(matches, 1):
                lines.append(f"{i}. «{match.text}» — {match.gloss}")
            return "\n".join(lines)

        match = matches[0]
        if action is FFEAction.RESTORE_SURFACE:
            return f"صورت ثبت‌شده: «{match.text}»"
        if action is FFEAction.ATTEST:
            if runtime.plan is not None and runtime.plan.needs_explanation:
                return f"این عبارت در پیکرهٔ فعلی ثبت شده است: «{match.text}»\nمعنی: {match.gloss}"
            return f"این عبارت در پیکرهٔ فعلی ثبت شده است: «{match.text}»"
        if action is FFEAction.EXPLAIN:
            return f"«{match.text}»\nمعنی: {match.gloss}"
        if action is FFEAction.SEARCH_BY_MEANING:
            return f"پیشنهاد ثبت‌شده: «{match.text}»\nمعنی: {match.gloss}"
        return f"عبارت ثبت‌شده: «{match.text}»\nمعنی: {match.gloss}"

    def _messages(
        self, user_query: str, runtime: PPQRuntimeResult, matches, slots
    ) -> Sequence[Mapping[str, str]]:
        payload = {
            "user_query": self._redact_registered_surfaces(user_query),
            "runtime_action": self._action(runtime).value if self._action(runtime) else None,
            "needs_explanation": bool(runtime.plan and runtime.plan.needs_explanation),
            "grounded_slots": [
                {
                    "slot": slot,
                    "registered_meaning": match.gloss,
                    "expression_type": match.expression_type,
                }
                for slot, match in zip(slots, matches)
            ],
        }
        return (
            {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        )

    def _redact_registered_surfaces(self, text: str) -> str:
        """Hide stored FFE surfaces from the prose model even if user-supplied."""
        redacted = text
        for surface in self._registered_surfaces:
            if surface in redacted:
                redacted = redacted.replace(surface, "[FFE_USER_MENTION]")
        return redacted

    def _validate_template(self, template: str, slots: Sequence[str]) -> None:
        expected = set(slots)
        seen = _SLOT_RE.findall(template)
        seen_slots = {f"<FFE_{n}>" for n in seen}
        if seen_slots != expected:
            raise ValueError("composer used missing or unknown FFE slots")
        for slot in slots:
            if template.count(slot) != 1:
                raise ValueError("each FFE slot must appear exactly once")
        if any(ch in template for ch in _QUOTE_CHARS):
            raise ValueError("composer template must not quote free-form text")
        for surface in self._registered_surfaces:
            if surface in template:
                raise ValueError("composer emitted registered FFE text outside a slot")
