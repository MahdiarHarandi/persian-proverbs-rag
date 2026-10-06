"""Small recommendation-suitability guard for semantic FFE suggestions.

Corpus membership proves only that PPQ can resolve a stored expression.  It does
not imply that every stored expression is suitable as a default recommendation
in a neutral user interaction.  This module keeps that distinction explicit.

The policy is deliberately narrow.  It does not affect restore/explain/attest
operations, and it does not blacklist families by ID.  It only blocks strongly
coarse/vulgar surfaces from *semantic recommendation* unless the user explicitly
asks for vulgar/coarse wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_COARSE_TOKENS = frozenset(
    {
        "گه",
        "گوه",
        "کون",
        "پیزی",
        "ماتحت",
        "نریده",
        "رید",
        "ریده",
    }
)
_ALLOW_COARSE = re.compile(
    r"(?:رکیک|زشت|فحش|خیلی\s+عامیانه|کوچه\s*بازاری|تند\s+و\s+زننده|بی\s*ادبانه)"
)
_TOKEN = re.compile(r"[\u0600-\u06ff]+")


@dataclass(frozen=True)
class RecommendationSuitability:
    allowed: bool
    reason: str


class RecommendationSuitabilityPolicy:
    """Reject only unmistakably coarse default recommendations.

    The policy intentionally does not inspect family IDs or retrieval ranks.
    A user who explicitly asks for coarse/vulgar language can still receive a
    corpus-grounded candidate; supplied expressions remain available to the
    restore/explain/attest paths regardless of this policy.
    """

    def evaluate(self, query: str, match) -> RecommendationSuitability:
        query_text = (query or "").replace("\u200c", " ").lower()
        if _ALLOW_COARSE.search(query_text):
            return RecommendationSuitability(True, "explicit-coarse-register-request")
        surface = (getattr(match, "text", "") or "").replace("\u200c", " ").lower()
        tokens = set(_TOKEN.findall(surface))
        hit = sorted(tokens & _COARSE_TOKENS)
        if hit:
            return RecommendationSuitability(
                False,
                "default-recommendation-coarse-register:" + ",".join(hit),
            )
        return RecommendationSuitability(True, "default-recommendation-register-ok")
