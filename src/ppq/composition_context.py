"""Surrounding mixed-writing intent detection.

The runtime/H detector remains unchanged for reproducibility.  This version is
used only by the safe composer and broadens the insertion noun vocabulary to
include generic user words such as «گفته»، «عبارت» and «جمله».  It still does
not decide whether an FFE is needed; Planner must already have selected the
SEARCH_BY_MEANING path.
"""

from __future__ import annotations

import re

from .composition_intent import _norm


class CompositionTaskDetector:
    _WRITING = re.compile(
        r"(?:پیام|متن|کپشن|نامه|ایمیل|جواب|پست|استوری|یادداشت).{0,35}"
        r"(?:بنویس|بساز|آماده\s+کن|درست\s+کن)"
        r"|(?:بنویس|بساز|آماده\s+کن|درست\s+کن).{0,35}"
        r"(?:پیام|متن|کپشن|نامه|ایمیل|جواب|پست|استوری|یادداشت)"
    )
    _INSERT_FFE = re.compile(
        r"(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر|گفته|عبارت|جمله).{0,45}"
        r"(?:داخل|توی|تو\s+ی|آخر|بذار|بگذار|بیار|استفاده\s+کن|جا\s+بده|قرار\s+بده)"
        r"|(?:داخلش|توش|توی\s+متن|آخرش).{0,45}"
        r"(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر|گفته|عبارت|جمله)"
    )

    def needs_composition(self, query: str) -> bool:
        if not isinstance(query, str) or not query.strip():
            return False
        text = _norm(query)
        return bool(self._WRITING.search(text) and self._INSERT_FFE.search(text))
