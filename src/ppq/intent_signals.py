"""Conservative deterministic intent signals for Persian FFE requests.

This layer is not an authenticity or retrieval authority.  It only flags strong
linguistic evidence that a request likely needs one of the corpus-bound FFE
operations.  The signals are used to challenge obvious planner under-routing and
to make long, punctuation-free and colloquial requests less brittle.

The rules intentionally require an *operation cue* (find/use/explain/check/fix)
and do not route a request merely because words such as «ضرب‌المثل» or «اصطلاح»
appear as a discussion topic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .planner_types import FFEAction


def _norm(text: str) -> str:
    # Keep this local and deliberately light-weight.  We only normalize common
    # orthographic differences needed by routing cues; corpus normalization is
    # a separate boundary.
    return text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه").lower()


@dataclass(frozen=True)
class FFEIntentSignals:
    strong: bool
    suggested_action: FFEAction | None
    reasons: tuple[str, ...]
    # A narrow, explicit negative instruction such as "هیچ ضرب المثل داخلش نیاور"
    # must be able to protect the GENERAL path even if the LLM planner over-routes.
    force_general: bool = False


class FFEIntentSignalAnalyzer:
    """High-precision cues that may challenge a false GENERAL plan.

    These rules are intentionally conservative.  They are not intended to cover
    every natural-language formulation; the LLM planner remains the main intent
    model.  Their role is to catch clear mixed-composition and indirect lookup
    requests even when punctuation/quotation marks are absent.
    """

    _FFE_NOUN = re.compile(
        r"(?:ضرب\s*المثل|مثل(?:\s+واقعی|\s+فارسی)?|اصطلاح(?:\s+فارسی|\s+رایج)?|"
        r"کنایه|تعبیر(?:\s+رایج|\s+فارسی)?|عبارت\s+رایج|جمله\s+رایج)"
    )
    _SURFACE_FFE_NOUN = re.compile(
        r"(?:ضرب\s*المثل|مثل\s+(?:واقعی|فارسی)|اصطلاح\s+(?:فارسی|رایج)|"
        r"کنایه|تعبیر(?:\s+رایج|\s+فارسی)?|عبارت\s+رایج|جمله\s+رایج)"
    )
    _SEARCH_VERB = re.compile(
        r"(?:می\s*خوام|میخوام|می\s*خواهم|میخواهم|بگو|بگید|پیشنهاد|پیدا\s*کن|"
        r"بیار|بذار|بگذار|اضافه\s*کن|استفاده\s*کن|جا\s*بده|قرار\s*بده|"
        r"مناسب|معادل|جای\s+این|برای\s+این|برای\s+این\s+موقعیت)"
    )
    _MEANING_CONTEXT = re.compile(
        r"(?:برای\s+(?:وقتی|کسی|آدم|این|وضع|وضعیت|حالت|موقعیت)|"
        r"درباره\s+(?:اینکه|این|موضوع)|مناسب\s+(?:این|برای)|مفهوم|معنی|منظور|"
        r"که\s+.{4,}|وقتی\s+.{4,})"
    )
    _INDIRECT_SEARCH = re.compile(
        r"(?:چی\s*می\s*گن|چی\s*میگن|چه\s*می\s*گن|چه\s*میگن|"
        r"وقتی.+چی\s*می\s*گن|برای\s+این\s+(?:وضع|وضعیت|حالت|موقعیت).+چی|"
        r"یه\s+چیزی\s+(?:هست|نیست)\s+که\s+می\s*گن|"
        r"چطور\s+(?:عامیانه|رایج|اصطلاحی)\s+بگم|"
        r"یه\s+جمله\s+جا\s*افتاده.+(?:بگو|میخوام|می\s*خوام))"
    )
    _EXAMPLE_REQUEST = re.compile(
        r"(?:(?:یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|چند|\d+)\s*(?:تا\s*)?"
        r"(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر).{0,35}(?:می\s*خوام|میخوام|بگو|بده|مثال|نمونه)|"
        r"(?:ضرب\s*المثل|اصطلاح|تعبیر).{0,30}(?:مثال\s*بزن|نمونه\s*بده))"
    )
    _EXPLAIN = re.compile(
        r"(?:یعنی\s+چی|معن(?:ی|یش)\s*(?:چیه|چیست)|منظورش\s+چیه|"
        r"معن(?:ی|یش)\s*(?:رو|شو)?\s*(?:بگو|بگید|بده|بدین)|"
        r"چه\s+معنی(?:\s*ای)?\s+می\s*ده|توضیح(?:ش)?\s+بده|"
        r"ک[ِی]?\s+به\s+کار\s+می\s*ره)"
    )
    _ATTEST = re.compile(
        r"(?:واقع(?:ی|اً)|اصیل|معتبر|درست|ثبت\s*(?:شده|شدن|شده\s*اند)|وجود\s+داره|"
        r"ضرب\s*المثل\s+هست|اصطلاح\s+هست).{0,35}(?:\?|هست|نیست|یا|؟)|"
        r"(?:در|توی)\s+پیکره.{0,35}ثبت\s*(?:شده|شدن|هست|است)|"
        r"(?:این|اون).{0,40}(?:واقعیه|اصله|معتبره|درسته)"
    )
    _RESTORE = re.compile(
        r"(?:اصلش\s+چی|درستش\s+(?:چی|رو\s+(?:بگو|بده))|شکل\s+(?:درست|معیار|ثبت\s*شده)|"
        r"صورت\s+(?:درست|معیار|ثبت\s*شده)|صورتش\s+رو\s+(?:بگو|بده)|کاملش\s+چی|"
        r"ادامه\s*(?:اش|ش)\s+چی|یادم\s+(?:رفته|نیست)|"
        r"فکر\s+کنم\s+می\s*گن|فکر\s+کنم\s+میگن|"
        r"اشتباه\s+نوشتم|غلط\s+گفتم|تصحیح\s+کن|اصلاح\s+کن)"
    )

    # Explicit negative FFE instructions are a hard non-interference signal.
    # They are intentionally narrow: the user must both name the FFE class and
    # explicitly forbid including/using it.
    _NO_FFE_REQUEST = re.compile(
        r"(?:هیچ\s+)?(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر).{0,28}"
        r"(?:نیاور|نیار|استفاده\s+نکن|به\s*کار\s+نبر|نذار|نباشه|قرار\s+نده)"
        r"|(?:نیاور|نیار|استفاده\s+نکن|به\s*کار\s+نبر|نذار|نباشه).{0,28}"
        r"(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر)"
    )

    # Category-level/meta discussion is not a request to retrieve one concrete
    # expression.  The deterministic rescue layer should therefore not override
    # a GENERAL planner decision merely because words such as "کنایه" and
    # "توضیح" co-occur.  The main LLM planner can still route a genuine target.
    _META_FFE_DISCUSSION = re.compile(
        r"(?:ضرب\s*المثل\s*ها|مثل\s*ها|اصطلاح\s*ها|کنایه\s*ها|تعبیر\s*ها).{0,45}"
        r"(?:ترجمه|فرهنگ|زبان|ادبیات|چرا|نقش|مفهوم|تحلیل|ویژگی)"
        r"|(?:ترجمه|فرهنگ|زبان|ادبیات).{0,45}"
        r"(?:ضرب\s*المثل\s*ها|مثل\s*ها|اصطلاح\s*ها|کنایه\s*ها|تعبیر\s*ها)"
    )

    # Explicit requests to invent fiction must stay out of corpus routing.
    _FICTIONAL = re.compile(
        r"(?:ساختگی|خیالی|اختراع\s+کن|از\s+خودت\s+بساز|جعلی\s+بساز|"
        r"یه\s+مثل\s+جدید\s+بساز)"
    )

    def analyze(self, query: str, *, grounded_surface_count: int = 0) -> FFEIntentSignals:
        if not isinstance(query, str) or not query.strip():
            return FFEIntentSignals(False, None, ())
        text = _norm(query)
        if self._FICTIONAL.search(text):
            return FFEIntentSignals(False, None, ("explicit-fictional-request",))
        if self._NO_FFE_REQUEST.search(text):
            return FFEIntentSignals(False, None, ("explicit-no-ffe-request",), force_general=True)

        reasons: list[str] = []

        if self._EXAMPLE_REQUEST.search(text):
            return FFEIntentSignals(
                True, FFEAction.RETRIEVE_EXAMPLES, ("explicit-example-count-request",)
            )

        # Deterministic rescue for target-bearing operations requires an
        # independently grounded surface mention.  A bare category noun such as
        # "کنایه‌ها" plus "توضیح بده" is meta-discussion, not evidence of one
        # concrete FFE target.  The LLM planner remains free to identify genuinely
        # supplied but not-yet-grounded target text.
        target_context = grounded_surface_count > 0

        if grounded_surface_count == 0 and self._META_FFE_DISCUSSION.search(text):
            return FFEIntentSignals(False, None, ("meta-ffe-discussion",), force_general=True)

        exact_attest = bool(
            re.search(
                r"(?:دقیقاً|دقیقا|همین|به\s+همین\s+صورت).{0,45}(?:ثبت\s*(?:شده|شدن|شده\s*اند)|تایید|تأیید|وجود\s+داره).{0,20}(?:\?|هست|است|؟)",
                text,
            )
        )
        if exact_attest and target_context:
            reasons.append("exact-surface-attestation-cue")
            return FFEIntentSignals(True, FFEAction.ATTEST, tuple(reasons))

        if self._RESTORE.search(text) and target_context:
            reasons.append("surface-restoration-cue")
            return FFEIntentSignals(True, FFEAction.RESTORE_SURFACE, tuple(reasons))

        if self._ATTEST.search(text) and target_context:
            reasons.append("authenticity-cue")
            return FFEIntentSignals(True, FFEAction.ATTEST, tuple(reasons))

        if self._EXPLAIN.search(text) and target_context:
            reasons.append("meaning-explanation-cue")
            return FFEIntentSignals(True, FFEAction.EXPLAIN, tuple(reasons))

        noun = bool(self._FFE_NOUN.search(text))
        search = bool(self._SEARCH_VERB.search(text))
        indirect = bool(self._INDIRECT_SEARCH.search(text))
        context = bool(self._MEANING_CONTEXT.search(text))
        if (noun and search and context) or indirect:
            if noun:
                reasons.append("ffe-noun-plus-request-cue")
            if indirect:
                reasons.append("indirect-conventional-expression-lookup")
            return FFEIntentSignals(True, FFEAction.SEARCH_BY_MEANING, tuple(reasons))

        return FFEIntentSignals(False, None, ())
