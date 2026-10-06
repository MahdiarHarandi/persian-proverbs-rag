"""RESTORE-only surface rescue that leaves attestation unchanged.

The resolver addresses two restoration failure classes:
1. an explicitly partial remembered prefix when both a short expression and a
   longer corpus expression share that prefix;
2. a long colloquial remembered form with one lexical substitution and strong,
   ordered lexical anchors.

No rule in this module is allowed to authenticate the user's supplied wording.
It can only select a corpus family for RESTORE_SURFACE, after which public text
is resolved from the corpus.
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from .surface_restore_fuzzy import (
    FuzzySurfaceRestore,
    RestoreRescueResult,
    signature,
)

_PARTIAL_PREFIX_CUE = re.compile(
    r"(?:بخشی\s+از\s+(?:عبارت|مثل|اصطلاح).{0,40}(?:یادم|شروع)|"
    r"(?:اولش|ابتداش|شروعش).{0,35}(?:یادم|بگو|بده)|"
    r"(?:باهاش|با\s+این).{0,20}شروع\s*می\s*شه|"
    r"(?:ادامه|کامل).{0,25}(?:عبارت|مثل|اصطلاح))",
    re.IGNORECASE,
)


def _is_prefix(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    return bool(short) and len(long) > len(short) and long[: len(short)] == short


def _lcs_len(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    # Tiny sequences (proverb signatures) make a simple dynamic program both
    # deterministic and sufficiently cheap.
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


class SurfaceRestore(FuzzySurfaceRestore):
    """Context-aware prefix completion + conservative one-substitution rescue."""

    def __init__(self, corpus):
        super().__init__(corpus)
        self._family_signatures: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for expr, variant in corpus.all_variants():
            sig = signature(variant.text)
            if sig:
                self._family_signatures[expr.family_id].append(sig)

    def contextual_prefix_completion(self, text: str, full_query: str) -> RestoreRescueResult:
        if not _PARTIAL_PREFIX_CUE.search(full_query or ""):
            return RestoreRescueResult(None, 0.0, 0.0, "no-explicit-partial-prefix-cue")
        q = signature(text)
        if len(q) < 4:
            return RestoreRescueResult(None, 0.0, 0.0, "partial-prefix-too-short")

        candidates: dict[str, int] = {}
        for fid, sigs in self._family_signatures.items():
            for sig in sigs:
                # Require a meaningful continuation, not merely a punctuation or
                # one-token variant of the same short saying.
                if _is_prefix(q, sig) and len(sig) >= len(q) + 3:
                    candidates[fid] = max(candidates.get(fid, 0), len(sig) - len(q))

        if not candidates:
            return RestoreRescueResult(None, 0.0, 0.0, "no-longer-prefix-completion")
        ranked = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
        top_fid, extension = ranked[0]
        # If two unrelated families have the exact same extension strength, fail
        # closed rather than guessing which continuation the user meant.
        if len(ranked) > 1 and ranked[1][1] == extension:
            return RestoreRescueResult(None, 1.0, 0.0, "ambiguous-long-prefix-completion")
        return RestoreRescueResult(top_fid, 1.0, 1.0, "explicit-long-prefix-completion")

    def resolve(self, text: str) -> RestoreRescueResult:
        primary = super().resolve(text)
        if primary.family_id is not None:
            return primary

        q = signature(text)
        if len(q) < 4:
            return primary

        scored: dict[str, tuple[float, int, float]] = {}
        for fid, sigs in self._family_signatures.items():
            best = None
            for sig in sigs:
                if len(sig) < 4 or abs(len(sig) - len(q)) > 2:
                    continue
                lcs = _lcs_len(q, sig)
                qcov = lcs / len(q)
                ccov = lcs / len(sig)
                if lcs < 3 or qcov < 0.75 or ccov < 0.55:
                    continue
                ratio = SequenceMatcher(None, " ".join(q), " ".join(sig)).ratio()
                record = (ratio, lcs, min(qcov, ccov))
                if best is None or record > best:
                    best = record
            if best is not None:
                scored[fid] = best

        if not scored:
            return primary
        ranked = sorted(scored.items(), key=lambda x: (-x[1][0], -x[1][1], x[0]))
        fid, (ratio, lcs, cov) = ranked[0]
        second = ranked[1][1][0] if len(ranked) > 1 else 0.0
        margin = ratio - second

        # This is intentionally stricter than generic fuzzy matching: at least
        # three ordered anchors, high character/token similarity, and a clear
        # family margin are required.  It remains RESTORE-only.
        if ratio >= 0.80 and lcs >= 3 and margin >= 0.18:
            return RestoreRescueResult(
                fid, min(1.0, ratio), min(1.0, margin), "ordered-one-substitution-restore-rescue"
            )
        return RestoreRescueResult(
            None,
            min(1.0, ratio),
            max(0.0, min(1.0, margin)),
            "ordered-restore-rescue-threshold-not-met",
        )
