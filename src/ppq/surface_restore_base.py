"""Conservative morphology/register rescue for RESTORE_SURFACE only.

The normal Surface verifier remains authoritative.  This rescue runs only after
that verifier abstains on an explicit restoration request.  It never affects
ATTEST, so a close-but-unattested phrase cannot become an authenticity claim.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[\u0600-\u06ff]+")
_COLLOQUIAL = {
    "زبون": "زبان",
    "دهن": "دهان",
    "ناخون": "ناخن",
    "شونه": "شانه",
    "خودشو": "خود",
    "خودش": "خود",
    "نمیشه": "نشدن",
    "نمیبره": "نبردن",
    "میکنه": "کردن",
    "میکنند": "کردن",
    "میگه": "گفتن",
    "میگن": "گفتن",
}
_STOP = frozenset(
    {"را", "رو", "به", "از", "در", "تو", "توی", "برای", "یک", "یه", "و", "که", "کسی", "خود"}
)
_LIGHT = frozenset(
    {"کردن", "شدن", "زدن", "گفتن", "بودن", "داشتن", "دادن", "گرفتن", "بردن", "آوردن"}
)


def _norm(text: str) -> str:
    return (text or "").replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک").lower()


def _stem(token: str) -> str:
    token = _COLLOQUIAL.get(token, token)
    # common possessive/object endings
    for suf in ("شان", "شون", "مون", "تون", "اش", "ش"):
        if len(token) > len(suf) + 2 and token.endswith(suf):
            token = token[: -len(suf)]
            break
    token = _COLLOQUIAL.get(token, token)
    # very small Persian verb stem normalization sufficient for local evidence
    t = token
    t = re.sub(r"^(?:نمی|می|ن)", "", t)
    for suf in ("یدن", "اند", "ید", "یم", "ین", "ند", "م", "ی", "ه"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    # recurring light-verb past forms
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


def signature(text: str) -> tuple[str, ...]:
    out = []
    for tok in _TOKEN_RE.findall(_norm(text)):
        s = _stem(tok)
        if not s or s in _STOP or s in _LIGHT or len(s) < 2:
            continue
        out.append(s)
    return tuple(out)


@dataclass(frozen=True)
class RestoreRescueResult:
    family_id: str | None
    score: float
    margin: float
    reason: str


class SurfaceRestoreRescue:
    """Match an explicit noisy surface to a unique corpus family conservatively."""

    def __init__(self, corpus):
        self.corpus = corpus
        self._rows = []
        token_families = defaultdict(set)
        for expr, variant in corpus.all_variants():
            sig = signature(variant.text)
            if not sig:
                continue
            self._rows.append((expr.family_id, sig))
            for t in set(sig):
                token_families[t].add(expr.family_id)
        self._token_families = {k: frozenset(v) for k, v in token_families.items()}

    def resolve(self, text: str) -> RestoreRescueResult:
        q = signature(text)
        if not q:
            return RestoreRescueResult(None, 0.0, 0.0, "empty-normalized-surface")
        qset = set(q)
        scored = {}
        for fid, sig in self._rows:
            sset = set(sig)
            overlap = qset & sset
            if not overlap:
                continue
            qcov = len(overlap) / len(qset)
            ccov = len(overlap) / len(sset)
            score = 0.55 * qcov + 0.45 * ccov
            # One distinctive lexical anchor can safely rescue short idioms in a
            # RESTORE request when that anchor occurs in only one family.
            unique_anchor = any(len(self._token_families.get(t, ())) == 1 for t in overlap)
            if len(qset) == 1 and unique_anchor and qcov == 1.0:
                score = max(score, 0.94)
            scored[fid] = max(scored.get(fid, 0.0), score)
        if not scored:
            return RestoreRescueResult(None, 0.0, 0.0, "no-lexical-rescue-candidate")
        ranked = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
        fid, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top - second
        if top >= 0.88 and margin >= 0.12:
            return RestoreRescueResult(fid, top, margin, "morphology-and-distinctive-token-rescue")
        return RestoreRescueResult(None, top, margin, "restore-rescue-threshold-not-met")
