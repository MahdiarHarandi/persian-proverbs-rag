"""Conservative detection of a surrounding writing task around an FFE request."""

from __future__ import annotations

import re


def _norm(text: str) -> str:
    return text.replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک").lower()


class SurroundingTaskDetector:
    """Return True only for clear write/compose + FFE-insertion requests.

    Pure meaning lookup should stay deterministic and should not pay a second
    prose-generation cost.  The detector is intentionally narrow; the planner is
    still responsible for deciding that an FFE is required.
    """

    _WRITING = re.compile(
        r"(?:پیام|متن|کپشن|نامه|ایمیل|جواب|پست|استوری|یادداشت).{0,35}"
        r"(?:بنویس|بساز|آماده\s+کن|درست\s+کن)"
        r"|(?:بنویس|بساز|آماده\s+کن|درست\s+کن).{0,35}"
        r"(?:پیام|متن|کپشن|نامه|ایمیل|جواب|پست|استوری|یادداشت)"
    )
    _INSERT_FFE = re.compile(
        r"(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر).{0,45}"
        r"(?:داخل|توی|تو\s+ی|آخر|بذار|بگذار|بیار|استفاده\s+کن|جا\s+بده|قرار\s+بده)"
        r"|(?:داخلش|توش|توی\s+متن|آخرش).{0,45}"
        r"(?:ضرب\s*المثل|مثل|اصطلاح|کنایه|تعبیر)"
    )

    def needs_composition(self, query: str) -> bool:
        if not isinstance(query, str) or not query.strip():
            return False
        text = _norm(query)
        return bool(self._WRITING.search(text) and self._INSERT_FFE.search(text))
