"""Immutable request intake with raw-preserving normalization and offsets.

benchmark treats the user's original text as evidence.  Preprocessing therefore
creates additional matching views instead of overwriting the raw request.  All
span offsets exposed here point into ``raw_text``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import (
    normalize_compact,
    normalize_display,
    normalize_retrieval_mapped,
)


@dataclass(frozen=True)
class RequestToken:
    text: str
    normalized: str
    raw_start: int
    raw_end: int
    normalized_start: int
    normalized_end: int


@dataclass(frozen=True)
class RequestView:
    raw_text: str
    display_text: str
    normalized_text: str
    compact_text: str
    normalized_to_raw: tuple[tuple[int, int], ...]
    tokens: tuple[RequestToken, ...]

    def raw_span_from_normalized(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end < start or end > len(self.normalized_text):
            raise ValueError("invalid normalized span")
        if start == end:
            if not self.normalized_to_raw:
                return (0, 0)
            boundary = (
                self.normalized_to_raw[-1][1]
                if start == len(self.normalized_to_raw)
                else self.normalized_to_raw[start][0]
            )
            return (boundary, boundary)
        return (
            self.normalized_to_raw[start][0],
            self.normalized_to_raw[end - 1][1],
        )

    def raw_slice(self, start: int, end: int) -> str:
        if start < 0 or end < start or end > len(self.raw_text):
            raise ValueError("invalid raw span")
        return self.raw_text[start:end]


class RequestIntake:
    """Build matching-safe views without semantic deletion or rewriting."""

    _TOKEN_PATTERN = re.compile(r"\S+")

    def build(self, text: str) -> RequestView:
        raw = text or ""
        mapped = normalize_retrieval_mapped(raw)
        normalized = mapped.text
        tokens: list[RequestToken] = []
        for match in self._TOKEN_PATTERN.finditer(normalized):
            start, end = match.span()
            raw_start, raw_end = mapped.raw_span(start, end)
            tokens.append(
                RequestToken(
                    text=raw[raw_start:raw_end],
                    normalized=match.group(0),
                    raw_start=raw_start,
                    raw_end=raw_end,
                    normalized_start=start,
                    normalized_end=end,
                )
            )
        return RequestView(
            raw_text=raw,
            display_text=normalize_display(raw),
            normalized_text=normalized,
            compact_text=normalize_compact(raw),
            normalized_to_raw=mapped.raw_spans,
            tokens=tuple(tokens),
        )
