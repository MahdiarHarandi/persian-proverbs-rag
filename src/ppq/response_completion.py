"""Deterministic completion for mixed-writing requests.

If all model-based slot or literal attempts fail,
the runtime reuses only the user's redacted writing brief, extracts the requested
message proposition, wraps it in literal Persian prose, and inserts the selected
FFE through the normal Corpus.resolve() render boundary.

The deterministic fallback never sees or invents an FFE string.  It exists to
avoid turning safe grounding into a non-useful abstention when prose generation
is structurally unreliable.
"""

from __future__ import annotations

import re

from .planner_types import FFEAction
from .response_fallback import FallbackComposer
from .response_types import ComposedResponse, ResponseDecision
from .runtime_resolver import RuntimeDecision

_WRITE_TO_CONTENT = re.compile(
    r"(?:بنویس|بساز|آماده\s+کن|درست\s+کن)\s*(?:که\s+)?(?P<content>.+)$",
    re.IGNORECASE,
)


class CompletionComposer(FallbackComposer):
    """Combine bounded model attempts with a corpus-safe deterministic fallback."""

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
        if task_detector is None:
            from .composition_context import CompositionTaskDetector

            task_detector = CompositionTaskDetector()
        super().__init__(
            corpus,
            generator,
            max_new_tokens=max_new_tokens,
            task_detector=task_detector,
            max_attempts=max_attempts,
            enable_decomposed_fallback=enable_decomposed_fallback,
        )

    def compose(self, user_query: str, runtime):
        out = super().compose(user_query, runtime)
        if out.decision is not ResponseDecision.ABSTAIN:
            return out
        if runtime.decision is not RuntimeDecision.ACCEPT:
            return out
        if self._action(runtime) is not FFEAction.SEARCH_BY_MEANING:
            return out
        if not self.task_detector.needs_composition(user_query):
            return out

        try:
            resolved = self._reresolve_matches(runtime)
            slots = tuple(f"<FFE_{i}>" for i in range(1, len(resolved) + 1))
            brief = self._writing_brief(user_query)
            literal = self._deterministic_literal_message(brief)
            self._validate_literal_fallback(literal)
            tail = " ".join(slots)
            template = literal.rstrip(" \n") + "\n" + tail
            text = self._render(template, slots, resolved)
            return ComposedResponse(
                decision=ResponseDecision.COMPOSED,
                text=text,
                runtime=runtime,
                generation_used=out.generation_used,
                # No model generation passed validation; this field remains
                # semantically honest even though the final response is valid.
                generation_valid=False,
                reason="composer-deterministic-brief-fallback-opaque-slot-rendered-through-corpus-resolve",
            )
        except Exception:
            return out

    def _deterministic_literal_message(self, brief: str) -> str:
        compact = re.sub(r"\s+", " ", brief or "").strip(" ،,؛;:")
        match = _WRITE_TO_CONTENT.search(compact)
        content = (match.group("content") if match else compact).strip(" ،,؛;:")
        if sum(ch.isalpha() for ch in content) < 12:
            raise ValueError("insufficient deterministic writing content")

        # Keep the proposition user-authored.  Only the short literal wrapper is
        # supplied by the runtime, so there is no opportunity to invent a second
        # proverb/idiom.
        if re.search(r"(?:محترمانه|رسمی)", compact):
            lead = "با احترام می‌خواستم بگم: "
        elif re.search(r"(?:دوستانه|صمیمی)", compact):
            lead = "یه نکته رو دوستانه می‌گم: "
        elif re.search(r"(?:آروم|آرام)", compact):
            lead = "آروم می‌خواستم بگم: "
        else:
            lead = "می‌خواستم بگم: "

        if content[-1] not in ".!؟?":
            content += "."
        return lead + content
