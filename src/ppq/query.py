"""Small, deterministic query analysis helpers for Surface retrieval.

Query handling remains deliberately narrow and auditable. The analyzer performs
two operations motivated by the development benchmark:

* when a natural-language wrapper contains exactly one quoted proverb fragment,
  retrieve with that quoted fragment instead of diluting it with the wrapper;
* detect when the raw input already contains two or more *attested* corpus
  expressions, so the verifier can CLARIFY instead of silently selecting one.

Neither operation rewrites or generates a proverb.  All output text is still
resolved from the versioned corpus after verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Corpus
from .normalize import normalize_compact

# These cues are intentionally lexical and inspectable.  Quoted content is not
# extracted from arbitrary text unless the surrounding query clearly asks about
# a proverb/quotation or remembering what was said.
_SURFACE_WRAPPER_CUE = re.compile(
    r"(?:ضرب\s*[‌ ]?المثل|(?:^|\s)مثل(?:\s|$)|چی\s*بود|چیست|"
    r"می[‌ ]?گفت|میگه|یادم|جمله|عبارت|گفته)"
)
_QUOTED_SPAN_PATTERNS = (
    re.compile(r"«([^»]+)»"),
    re.compile(r"“([^”]+)”"),
    re.compile(r'"([^"]+)"'),
)


@dataclass(frozen=True)
class SurfaceQueryAnalysis:
    raw_query: str
    retrieval_query: str
    wrapper_extracted: bool
    contained_family_ids: tuple[str, ...]

    @property
    def has_multiple_attested_expressions(self) -> bool:
        return len(self.contained_family_ids) >= 2


class SurfaceQueryAnalyzer:
    """Analyze a raw Surface query without semantic generation or guessing."""

    MIN_QUOTED_COMPACT_CHARS = 6
    MIN_CONTAINED_VARIANT_COMPACT_CHARS = 8

    def __init__(self, corpus: Corpus):
        self._exact_families: dict[str, tuple[str, ...]] = {}
        exact: dict[str, list[str]] = {}
        contained_units: list[tuple[str, str]] = []
        for expression, variant in corpus.all_variants():
            compact = normalize_compact(variant.text)
            if not compact:
                continue
            exact.setdefault(compact, []).append(expression.family_id)
            if len(compact) >= self.MIN_CONTAINED_VARIANT_COMPACT_CHARS:
                contained_units.append((expression.family_id, compact))
        self._exact_families = {
            compact: tuple(dict.fromkeys(families)) for compact, families in exact.items()
        }
        self._contained_units = tuple(contained_units)

    @staticmethod
    def _extract_single_quoted_fragment(query: str) -> str | None:
        if not _SURFACE_WRAPPER_CUE.search(query):
            return None
        spans: list[str] = []
        for pattern in _QUOTED_SPAN_PATTERNS:
            spans.extend(match.strip() for match in pattern.findall(query))
        spans = [
            span
            for span in spans
            if len(normalize_compact(span)) >= SurfaceQueryAnalyzer.MIN_QUOTED_COMPACT_CHARS
        ]
        # Ambiguous multi-quote requests are intentionally left untouched.
        return spans[0] if len(spans) == 1 else None

    def _contained_families(self, raw_query: str) -> tuple[str, ...]:
        compact_query = normalize_compact(raw_query)
        if not compact_query:
            return ()

        # Exact attestation is stronger than reverse containment.  This avoids
        # treating a legitimate long proverb as "multiple" merely because it
        # happens to contain another short attested phrase.
        exact = self._exact_families.get(compact_query)
        if exact:
            return exact[:1]

        first_position: dict[str, int] = {}
        for family_id, compact_variant in self._contained_units:
            position = compact_query.find(compact_variant)
            if position < 0:
                continue
            current = first_position.get(family_id)
            if current is None or position < current:
                first_position[family_id] = position
        return tuple(
            family_id
            for family_id, _ in sorted(first_position.items(), key=lambda item: (item[1], item[0]))
        )

    def analyze(self, query: str) -> SurfaceQueryAnalysis:
        quoted = self._extract_single_quoted_fragment(query)
        return SurfaceQueryAnalysis(
            raw_query=query,
            retrieval_query=quoted if quoted is not None else query,
            wrapper_extracted=quoted is not None,
            contained_family_ids=self._contained_families(query),
        )


_ATTESTATION_CUE = re.compile(
    r"(?:ضرب\s*[‌ ]?المثل|(?:^|\s)مثل(?:\s|$)|واقعی|اصیل|معتبر|وجود\s+دارد|"
    r"درسته|درست\s+است|هست|است)"
)


@dataclass(frozen=True)
class AttestationQueryAnalysis:
    raw_query: str
    checked_text: str
    wrapper_extracted: bool


class AttestationQueryAnalyzer:
    """Extract one explicitly quoted expression from an authenticity question.

    The checker intentionally does not guess an unquoted span from a long
    sentence.  Plain input is treated as the expression itself; natural wrapper
    extraction is only performed when exactly one quoted span is present and
    the surrounding query contains an attestation/proverb cue.
    """

    MIN_QUOTED_COMPACT_CHARS = 4

    @staticmethod
    def _quoted_spans(query: str) -> list[str]:
        spans: list[str] = []
        for pattern in _QUOTED_SPAN_PATTERNS:
            spans.extend(match.strip() for match in pattern.findall(query))
        return [
            span
            for span in spans
            if len(normalize_compact(span)) >= AttestationQueryAnalyzer.MIN_QUOTED_COMPACT_CHARS
        ]

    def analyze(self, query: str) -> AttestationQueryAnalysis:
        raw = query or ""
        spans = self._quoted_spans(raw)
        if len(spans) == 1 and _ATTESTATION_CUE.search(raw):
            return AttestationQueryAnalysis(
                raw_query=raw,
                checked_text=spans[0],
                wrapper_extracted=True,
            )
        return AttestationQueryAnalysis(
            raw_query=raw,
            checked_text=raw.strip(),
            wrapper_extracted=False,
        )
