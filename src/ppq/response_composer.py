"""Safe final composer for mixed writing and fixed-expression requests.

Generic insertion phrases such as ``یک گفته مناسب هم داخل متن بذار`` are
removed from the writing brief without changing fixed-expression authority:

* the selected FFE surface/gloss remain hidden from the prose model;
* authentic text is still rendered only from ``Corpus.resolve()``;
* generic FFE-insertion meta-clauses are removed from the writing brief;
* the final deterministic fallback turns only the remaining user-authored
  proposition into ordinary literal prose, then inserts opaque slots at runtime;
* no new semantic/retrieval authority is introduced.
"""

from __future__ import annotations

import re
from typing import Sequence

from .planner_types import FFEAction
from .response_completion import CompletionComposer
from .response_types import ComposedResponse, ResponseDecision
from .runtime_resolver import RuntimeDecision

# Nouns users commonly employ when they ask for an FFE to be inserted.  We do
# not infer FFE intent from these words here; CompositionTaskDetector + Planner
# have already done that.  This is only a brief-cleaning operation.
_INSERT_NOUN = re.compile(
    r"(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر|گفته|عبارت|جمله)\b",
    re.IGNORECASE,
)
_INSERT_ACTION = re.compile(
    r"(?:داخل(?:\s+(?:پیام|متن|کپشن|نامه|ایمیل))?|توی(?:\s+(?:پیام|متن|کپشن|نامه|ایمیل))?|"
    r"تو\s+(?:پیام|متن|کپشن|نامه|ایمیل)|توش|داخلش|آخرش|بیار|بذار|بگذار|استفاده\s+کن|"
    r"جا\s+بده|قرار\s+بده)",
    re.IGNORECASE,
)
_BACK_CONNECTOR = re.compile(
    r"(?:[،,؛;]?\s*(?:و|هم)\s+(?:(?:یه|یک)\s+)?)$",
    re.IGNORECASE,
)

_WRITE_TO_CONTENT_final = re.compile(
    r"(?:بنویس|بساز|آماده\s+کن|درست\s+کن)\s*(?:که\s+)?(?P<content>.+)$",
    re.IGNORECASE,
)


class SafeResponseComposer(CompletionComposer):
    """Compose responses with meta-clause cleaning and a literal fallback."""

    @classmethod
    def _strip_ffe_insertion_meta(cls, text: str) -> str:
        """Remove only a trailing *instruction to insert* an FFE.

        This deliberately operates from the last FFE-like noun and requires an
        insertion/action cue in the tail.  Therefore ordinary propositions that
        merely discuss a proverb/phrase are not removed.
        """
        compact = re.sub(r"\s+", " ", text or "").strip()
        matches = list(_INSERT_NOUN.finditer(compact))
        if not matches:
            return compact
        for noun in reversed(matches):
            tail = compact[noun.start() :]
            if not _INSERT_ACTION.search(tail):
                continue
            prefix = compact[: noun.start()]
            # Remove the connector/count phrase immediately before the noun,
            # e.g. ``و یک `` in ``... و یک گفته مناسب هم توش بیار``.
            m = _BACK_CONNECTOR.search(prefix)
            cut = m.start() if m else noun.start()
            cleaned = compact[:cut].rstrip(" ،,؛;:")
            return cleaned or compact
        return compact

    def _writing_brief(self, user_query: str) -> str:
        redacted = self._redact_registered_surfaces(user_query).strip()
        cleaned = self._strip_ffe_insertion_meta(redacted).strip(" ،,")
        return cleaned if len(cleaned) >= 8 else redacted

    @staticmethod
    def _proposition_from_brief(brief: str) -> str:
        compact = re.sub(r"\s+", " ", brief or "").strip(" ،,؛;:")
        match = _WRITE_TO_CONTENT_final.search(compact)
        content = (match.group("content") if match else compact).strip(" ،,؛;:")
        # Defensive second pass: the deterministic path must never echo a
        # leftover instruction to insert an FFE.
        content = SafeResponseComposer._strip_ffe_insertion_meta(content).strip(" ،,؛;:")
        if sum(ch.isalpha() for ch in content) < 12:
            raise ValueError("insufficient deterministic writing proposition")
        return content

    def _deterministic_literal_message(self, brief: str) -> str:
        compact = re.sub(r"\s+", " ", brief or "").strip(" ،,؛;:")
        content = self._proposition_from_brief(compact)

        # Keep the semantic proposition user-authored.  These wrappers add tone
        # only; they do not invent facts or another figurative expression.
        if re.search(r"(?:امیدبخش|دلگرم|ناامید)", compact):
            lead = "فقط می‌خوام یادآوری کنم که "
        elif re.search(r"(?:محترمانه|مودبانه|رسمی)", compact):
            lead = "با احترام، به نظرم "
        elif re.search(r"(?:دوستانه|صمیمی)", compact):
            lead = "دوستانه می‌گم، به نظرم "
        elif re.search(r"(?:آروم|آرام)", compact):
            lead = "آروم می‌خوام بگم که "
        else:
            lead = "به نظرم "

        # A common Persian writing brief describes the recipient in third
        # person (``به جای ... از خودش بذاره ...``).  Prefixing ``آدم`` makes
        # it a neutral principle without risky free-form conjugation.
        if content.startswith("به جای") and "از خودش" in content:
            content = "آدم " + content

        if content[-1] not in ".!؟?":
            content += "."
        return lead + content

    def compose(self, user_query: str, runtime):
        # Reuse all safe model-based paths first.
        out = super().compose(user_query, runtime)
        if out.decision is not ResponseDecision.COMPOSED:
            return out
        # The inherited deterministic fallback is safe, but its slot placement is a
        # detached newline.  Rebuild only that specific fallback with the final
        # cleaned brief and an in-message opaque-slot integration.
        if "composer-deterministic-brief-fallback" not in (out.reason or ""):
            return out
        if runtime.decision is not RuntimeDecision.ACCEPT:
            return out
        if self._action(runtime) is not FFEAction.SEARCH_BY_MEANING:
            return out
        if not self.task_detector.needs_composition(user_query):
            return out

        try:
            resolved = self._reresolve_matches(runtime)
            slots: Sequence[str] = tuple(f"<FFE_{i}>" for i in range(1, len(resolved) + 1))
            brief = self._writing_brief(user_query)
            literal = self._deterministic_literal_message(brief)
            self._validate_literal_fallback(literal)
            # Keep the FFE visibly *inside* the finished message.  Slot text is
            # still opaque until the final Corpus.resolve() render boundary.
            stem = literal.rstrip()
            if stem.endswith("."):
                stem = stem[:-1]
            template = stem + "؛ " + " ".join(slots) + "."
            text = self._render(template, slots, resolved)
            return ComposedResponse(
                decision=ResponseDecision.COMPOSED,
                text=text,
                runtime=runtime,
                generation_used=out.generation_used,
                generation_valid=False,
                reason="composer-clean-deterministic-fallback-opaque-slot-rendered-through-corpus-resolve",
            )
        except Exception:
            return out
