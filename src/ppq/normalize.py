"""Persian text normalization with separate display and retrieval views.

The project intentionally keeps normalization *views* separate:

``display``
    Conservative cleanup suitable for UI/storage.  It must not rewrite a
    quotation into a different wording.

``retrieval``
    A lossy canonical form used by the current character n-gram retriever.
    This is never rendered as the final quotation.

``compact``
    A whitespace-free derivative of the retrieval form.  It is prepared for
    exact/partial matching so that user input such as ``سرگذشت`` and a stored
    spelling such as ``سر گذشت`` can be compared safely.

Keeping these representations explicit prevents a common safety bug: using a
heavily normalized string as if it were an attested quotation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

ZWNJ = "‌"

_DISPLAY_FOLD = {"ي": "ی", "ك": "ک"}
_RETRIEVAL_FOLD = {
    **_DISPLAY_FOLD,
    "ى": "ی",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ة": "ه",
    "ۀ": "ه",
    "ؤ": "و",
    "ئ": "ی",
}
for _digit in range(10):
    _RETRIEVAL_FOLD[chr(0x0660 + _digit)] = str(_digit)
    _RETRIEVAL_FOLD[chr(0x06F0 + _digit)] = str(_digit)

_EXOTIC_SPACES = re.compile(
    "["
    + "".join(
        chr(cp)
        for cp in (
            0x00A0,
            0x202F,
            0x205F,
            0x3000,
            0x200B,
            *range(0x2000, 0x200C),
        )
    )
    + "]"
)


@dataclass(frozen=True)
class NormalizationViews:
    """All matching-safe views of one input string.

    ``original`` is retained only for diagnostics.  ``display`` is the only
    normalized view that is potentially suitable for UI.  ``retrieval`` and
    ``compact`` are matching aids and must never replace corpus text when a
    quotation is rendered.
    """

    original: str
    display: str
    retrieval: str
    compact: str
    tokens: tuple[str, ...]


def normalize_display(text: str) -> str:
    """Return a minimally normalized form suitable for storage/display."""
    if not text:
        return ""
    value = unicodedata.normalize("NFC", text)
    value = "".join(_DISPLAY_FOLD.get(char, char) for char in value)
    value = _EXOTIC_SPACES.sub(" ", value)
    return re.sub(r"[ \t]+", " ", value).strip()


def normalize_retrieval(text: str) -> str:
    """Return an aggressive canonical form used only for matching.

    The behavior remains compatible with the surface retriever: ZWNJ and
    whitespace become normal spaces, punctuation is ignored, Persian/Arabic
    character variants are folded, digits are ASCII-normalized, and
    diacritics/tatweel are removed.
    """
    if not text:
        return ""
    value = unicodedata.normalize("NFC", text)
    value = "".join(_RETRIEVAL_FOLD.get(char, char) for char in value)
    value = value.replace("ـ", "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    output: list[str] = []
    for char in value:
        if char == ZWNJ or char.isspace():
            output.append(" ")
        elif char.isalnum():
            output.append(char.lower())
    return re.sub(r"\s+", " ", "".join(output)).strip()


def normalize_compact(text: str) -> str:
    """Return a whitespace-free retrieval form for partial matching.

    Example: ``آب که از سرگذشت`` and ``آب که از سر گذشت`` both become
    ``ابکهازسرگذشت``. This view is used only for exact or partial matching.
    """
    return "".join(normalize_retrieval(text).split())


def retrieval_tokens(text: str) -> tuple[str, ...]:
    """Return whitespace-tokenized retrieval-normalized terms."""
    value = normalize_retrieval(text)
    return tuple(value.split()) if value else ()


def normalization_views(text: str) -> NormalizationViews:
    """Build all normalized representations once for a user/corpus string."""
    original = text or ""
    retrieval = normalize_retrieval(original)
    return NormalizationViews(
        original=original,
        display=normalize_display(original),
        retrieval=retrieval,
        compact="".join(retrieval.split()),
        tokens=tuple(retrieval.split()) if retrieval else (),
    )


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Return character n-grams from retrieval-normalized text."""
    value = normalize_retrieval(text)
    if not value:
        return []
    if len(value) < n:
        return [value]
    return [value[index : index + n] for index in range(len(value) - n + 1)]


@dataclass(frozen=True)
class MappedNormalization:
    """Retrieval-normalized text with a reversible map to raw offsets.

    ``raw_spans[i]`` gives the half-open raw character interval that produced
    ``text[i]``.  The mapping is diagnostic/localization metadata only; the
    normalized text is never rendered as an authentic fixed expression.
    """

    text: str
    raw_spans: tuple[tuple[int, int], ...]

    def raw_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a half-open normalized interval back to the original string."""
        if start < 0 or end < start or end > len(self.text):
            raise ValueError("invalid normalized span")
        if start == end:
            if not self.raw_spans:
                return (0, 0)
            if start == len(self.raw_spans):
                boundary = self.raw_spans[-1][1]
            else:
                boundary = self.raw_spans[start][0]
            return (boundary, boundary)
        return (self.raw_spans[start][0], self.raw_spans[end - 1][1])


def normalize_retrieval_mapped(text: str) -> MappedNormalization:
    """Return ``normalize_retrieval(text)`` plus raw-character provenance.

    The implementation mirrors :func:`normalize_retrieval` while processing
    Unicode base+combining clusters so NFC composition still maps back to the
    correct raw interval.  Repeated whitespace is collapsed exactly as in the
    legacy normalizer.  This function exists for request-span localization; it
    does not replace the long-standing normalization API used by retrieval.
    """
    raw = text or ""
    if not raw:
        return MappedNormalization(text="", raw_spans=())

    pieces: list[tuple[str, int, int]] = []
    index = 0
    while index < len(raw):
        end = index + 1
        while end < len(raw) and unicodedata.combining(raw[end]):
            end += 1
        cluster = unicodedata.normalize("NFC", raw[index:end])
        for char in cluster:
            folded = _RETRIEVAL_FOLD.get(char, char)
            # Current folds are one-character substitutions, but iterating
            # keeps this safe if a future normalization maps to >1 character.
            for output_char in folded:
                if output_char == "ـ" or unicodedata.combining(output_char):
                    continue
                if (
                    output_char == ZWNJ
                    or output_char.isspace()
                    or _EXOTIC_SPACES.fullmatch(output_char)
                ):
                    pieces.append((" ", index, end))
                elif output_char.isalnum():
                    pieces.append((output_char.lower(), index, end))
        index = end

    collapsed_chars: list[str] = []
    collapsed_spans: list[tuple[int, int]] = []
    for char, raw_start, raw_end in pieces:
        if char == " ":
            if not collapsed_chars:
                continue
            if collapsed_chars[-1] == " ":
                prev_start, _ = collapsed_spans[-1]
                collapsed_spans[-1] = (prev_start, raw_end)
                continue
        collapsed_chars.append(char)
        collapsed_spans.append((raw_start, raw_end))

    if collapsed_chars and collapsed_chars[-1] == " ":
        collapsed_chars.pop()
        collapsed_spans.pop()

    mapped = MappedNormalization(
        text="".join(collapsed_chars),
        raw_spans=tuple(collapsed_spans),
    )
    # Fail loudly if future edits make the mapped path diverge from the
    # production retrieval normalizer.  This is a development invariant and
    # has negligible cost for the short user requests handled here.
    expected = normalize_retrieval(raw)
    if mapped.text != expected:
        raise AssertionError("mapped normalization diverged from normalize_retrieval")
    return mapped
