"""Conservative lexical verification helpers for Meaning Search.

The meaning retriever intentionally remains lexical in the default production
path.  This module does not try to infer scenarios or generate semantics; it
only checks that an ACCEPT decision has at least one non-trivial lexical anchor
between the user's description and the stored gloss selected by retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import retrieval_tokens

# Query scaffolding, common function words and a few generic temporal/question
# words are not meaningful evidence that a query describes a proverb gloss.
# Keeping this list explicit makes the verifier inspectable and reproducible.
_MEANING_NON_CONTENT_TOKENS = frozenset(
    {
        "آیا",
        "اگر",
        "اما",
        "از",
        "است",
        "استفاده",
        "اصطلاح",
        "اصطلاحی",
        "الان",
        "امروز",
        "این",
        "آن",
        "با",
        "باشد",
        "برای",
        "به",
        "بود",
        "تا",
        "تقریبا",
        "تقریباً",
        "چرا",
        "چطور",
        "چقدر",
        "چیست",
        "چه",
        "چیزی",
        "حالتی",
        "حالا",
        "در",
        "دیروز",
        "را",
        "رو",
        "روی",
        "شد",
        "شدن",
        "ضرب",
        "ضربالمثل",
        "ضرب‌المثل",
        "فردا",
        "کرد",
        "کرده",
        "کردن",
        "کدام",
        "که",
        "کسی",
        "گفته",
        "مناسب",
        "مناسبی",
        "معنا",
        "معنای",
        "معنی",
        "مورد",
        "می",
        "میشه",
        "میشود",
        "میگن",
        "میگوید",
        "می‌شود",
        "می‌گوید",
        "هیچ",
        "هست",
        "هستند",
        "و",
        "وقتی",
        "یک",
        "یا",
        "شود",
        "کند",
        "کار",
        "هم",
    }
)

_ASCII_TOKEN = re.compile(r"^[A-Za-z0-9_+./:-]+$")


def meaning_content_tokens(text: str) -> tuple[str, ...]:
    """Return conservative Persian content tokens for verification.

    ASCII/technical tokens are deliberately not treated as semantic support for
    a Persian proverb gloss, and generic query scaffolding is removed.  This is
    a verification feature only; retrieval itself still sees the full query.
    """
    output: list[str] = []
    seen: set[str] = set()
    for token in retrieval_tokens(text):
        if len(token) <= 1:
            continue
        if token in _MEANING_NON_CONTENT_TOKENS:
            continue
        if _ASCII_TOKEN.fullmatch(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return tuple(output)


@dataclass(frozen=True)
class MeaningEvidence:
    """Inspectible lexical support between a query and one stored gloss."""

    query_content_tokens: tuple[str, ...]
    gloss_content_tokens: tuple[str, ...]
    shared_content_tokens: tuple[str, ...]

    @property
    def has_content_anchor(self) -> bool:
        return bool(self.shared_content_tokens)

    @property
    def query_support_ratio(self) -> float:
        if not self.query_content_tokens:
            return 0.0
        return len(self.shared_content_tokens) / len(self.query_content_tokens)

    @property
    def gloss_support_ratio(self) -> float:
        if not self.gloss_content_tokens:
            return 0.0
        return len(self.shared_content_tokens) / len(self.gloss_content_tokens)


def meaning_evidence(query: str, gloss: str) -> MeaningEvidence:
    query_tokens = meaning_content_tokens(query)
    gloss_tokens = meaning_content_tokens(gloss)
    shared = tuple(sorted(set(query_tokens) & set(gloss_tokens)))
    return MeaningEvidence(
        query_content_tokens=query_tokens,
        gloss_content_tokens=gloss_tokens,
        shared_content_tokens=shared,
    )
