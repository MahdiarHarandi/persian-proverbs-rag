"""Conservative surface restoration for extended forms.

This module extends the runtime/H RESTORE-only rescue without changing the
attestation boundary.  It is deliberately not used for ATTEST.  The goal is to
recover a registered surface from common Persian colloquial morphology and from
one bounded lexical substitution when the remaining evidence is strong and
unambiguous.

Safety invariants:
- only RESTORE_SURFACE may use this rescue;
- no score here can make an unattested user string authentic;
- the returned family is still rendered only through ``Corpus.resolve``;
- ambiguous close candidates remain rejected.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

# Explicit Persian/Arabic letters used in contemporary Persian.  The previous
# broad ``\u0600-\u06ff`` range also captured punctuation such as Arabic comma,
# which made tokens like ``گیره،`` differ from ``گیره``.
_TOKEN_RE = re.compile(r"[ءآأإؤئابپتثجچحخدذرزژسشصضطظعغفقکكگلمنوهةۀهیيى]+")

_COLLOQUIAL = {
    "زبون": "زبان",
    "دهن": "دهان",
    "ناخون": "ناخن",
    "شونه": "شانه",
    "خودشو": "خود",
    "خودش": "خود",
    "نمیشه": "شدن",
    "نميشه": "شدن",
    "میشه": "شدن",
    "ميشه": "شدن",
    "شه": "شدن",
    "بشه": "شدن",
    "بشود": "شدن",
    "میکنه": "کردن",
    "ميکنه": "کردن",
    "میکنند": "کردن",
    "ميکنند": "کردن",
    "میگه": "گفتن",
    "ميگه": "گفتن",
    "میگن": "گفتن",
    "ميگن": "گفتن",
}

_STOP = frozenset(
    {
        "را",
        "رو",
        "به",
        "از",
        "در",
        "تو",
        "توی",
        "برا",
        "برای",
        "یک",
        "یه",
        "و",
        "که",
        "کسی",
        "خود",
        "این",
        "اون",
        "آن",
    }
)
_LIGHT = frozenset(
    {
        "کردن",
        "شدن",
        "زدن",
        "گفتن",
        "بودن",
        "داشتن",
        "دادن",
        "گرفتن",
        "بردن",
        "آوردن",
    }
)


def _norm(text: str) -> str:
    return (
        (text or "")
        .replace("\u200c", " ")
        .replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
        .replace("ۀ", "ه")
        .replace("ة", "ه")
        .lower()
    )


def _strip_object_clitic(token: str) -> tuple[str, ...]:
    """Return conservative alternative forms for colloquial accusative ``-و``.

    Persian speech often writes ``کتابو`` for ``کتاب را``.  We keep the original
    form and add the stripped form rather than rewriting blindly.  This prevents
    the rescue from treating every final waw as an object marker.
    """
    forms = [token]
    if len(token) >= 5 and token.endswith("و") and not token.endswith(("او", "یو")):
        stripped = token[:-1]
        if len(stripped) >= 3:
            forms.append(stripped)
    return tuple(dict.fromkeys(forms))


def _stem_one(token: str) -> str:
    token = _COLLOQUIAL.get(token, token)

    # Common possessive/object endings.  The colloquial accusative -و is handled
    # separately as an alternative form so lexical nouns ending in waw are not
    # destroyed unconditionally.
    for suf in ("شان", "شون", "مون", "تون", "اش", "ش"):
        if len(token) > len(suf) + 2 and token.endswith(suf):
            token = token[: -len(suf)]
            break
    token = _COLLOQUIAL.get(token, token)

    # Compact Persian verb normalization.  We intentionally keep lexical stems
    # such as ``گیر`` because they can disambiguate two otherwise similar
    # proverbs, while true light verbs are filtered after canonicalization.
    t = token
    t = re.sub(r"^(?:نمی|می|ن)", "", t)
    for suf in ("یدن", "اند", "ید", "یم", "ین", "ند", "م", "ی", "ه"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    roots = {
        "زد": "زدن",
        "شد": "شدن",
        "کرد": "کردن",
        "گفت": "گفتن",
        "برد": "بردن",
        "داد": "دادن",
        "گرفت": "گرفتن",
    }
    return roots.get(t, t)


def _token_signatures(token: str) -> tuple[str, ...]:
    out: list[str] = []
    for form in _strip_object_clitic(token):
        s = _stem_one(form)
        if not s or s in _STOP or s in _LIGHT or len(s) < 2:
            continue
        out.append(s)
    return tuple(dict.fromkeys(out))


def signature(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(_norm(text)):
        forms = _token_signatures(tok)
        if not forms:
            continue
        # Prefer the normalized/clitic-stripped alternative when available;
        # keeping one token per surface position avoids artificially inflating
        # overlap counts.
        out.append(forms[-1])
    return tuple(out)


@dataclass(frozen=True)
class RestoreRescueResult:
    family_id: str | None
    score: float
    margin: float
    reason: str


class FuzzySurfaceRestore:
    """RESTORE-only rescue using morphology plus bounded near-match evidence."""

    def __init__(self, corpus):
        self.corpus = corpus
        self._rows: list[tuple[str, tuple[str, ...]]] = []
        token_families: dict[str, set[str]] = defaultdict(set)
        for expr, variant in corpus.all_variants():
            sig = signature(variant.text)
            if not sig:
                continue
            self._rows.append((expr.family_id, sig))
            for token in set(sig):
                token_families[token].add(expr.family_id)
        self._token_families = {k: frozenset(v) for k, v in token_families.items()}

    def _distinctive_count(self, overlap: Iterable[str]) -> int:
        return sum(1 for t in set(overlap) if len(self._token_families.get(t, ())) <= 2)

    def resolve(self, text: str) -> RestoreRescueResult:
        q = signature(text)
        if not q:
            return RestoreRescueResult(None, 0.0, 0.0, "empty-normalized-surface")
        qset = set(q)
        scored: dict[str, tuple[float, float, float, int, int]] = {}

        for fid, sig in self._rows:
            sset = set(sig)
            overlap = qset & sset
            if not overlap:
                continue
            qcov = len(overlap) / len(qset)
            ccov = len(overlap) / len(sset)
            base = 0.55 * qcov + 0.45 * ccov
            distinctive = self._distinctive_count(overlap)

            # One distinctive anchor can rescue a one-token short form in a
            # RESTORE request, preserving the old conservative behavior.
            unique_anchor = any(len(self._token_families.get(t, ())) == 1 for t in overlap)
            if len(qset) == 1 and unique_anchor and qcov == 1.0:
                base = max(base, 0.94)

            # Long remembered forms often differ by exactly one colloquial or
            # lexical token while the remaining ordered content is highly
            # diagnostic.  Give such cases a bounded confidence lift; this is
            # restoration only, never authenticity attestation.
            near_single_substitution = (
                len(overlap) >= 3
                and min(len(qset), len(sset)) >= 4
                and qcov >= 0.75
                and ccov >= 0.75
                and distinctive >= 2
            )
            score = max(base, 0.90) if near_single_substitution else base
            current = scored.get(fid)
            record = (score, qcov, ccov, len(overlap), distinctive)
            if current is None or record[0] > current[0]:
                scored[fid] = record

        if not scored:
            return RestoreRescueResult(None, 0.0, 0.0, "no-lexical-rescue-candidate")

        ranked = sorted(scored.items(), key=lambda x: (-x[1][0], x[0]))
        fid, (top, qcov, ccov, overlap_n, distinctive) = ranked[0]
        second = ranked[1][1][0] if len(ranked) > 1 else 0.0
        margin = top - second

        if top >= 0.88 and margin >= 0.12:
            reason = (
                "morphology-distinctive-near-match-rescue"
                if top >= 0.90 and overlap_n >= 3 and qcov >= 0.75 and ccov >= 0.75
                else "morphology-and-distinctive-token-rescue"
            )
            return RestoreRescueResult(fid, top, margin, reason)
        return RestoreRescueResult(None, top, margin, "restore-rescue-threshold-not-met")
