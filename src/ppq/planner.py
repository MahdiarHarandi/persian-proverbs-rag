"""Planner wrapper for explicit, ungrounded restoration requests.

``CorePlanner`` provides general request understanding. This module adds one
narrow deterministic rescue for an explicit pattern of the form:

    <remembered/noisy wording> ... درست/اصل/صورت ثبت‌شده‌اش را بگو

This rescue exists because a heavily colloquial surface may fail the binder,
which otherwise deprives CorePlanner's deterministic intent signals of target
context.  It does not invent an FFE and does not assert authenticity; it only
extracts an exact substring from the user's own query and routes it to the
RESTORE_SURFACE path, which remains corpus-grounded and fail-closed.
"""

from __future__ import annotations

import re

from .planner_core import CorePlanner
from .planner_types import FFEAction, FFEPlan, PlanRoute
from .slm import PlannerPrediction

_RESTORE_CUE = re.compile(
    r"(?:اصلش|درستش|کاملش|صورت(?:ش)?\s+(?:ثبت\s*شده|درست|معیار|اصلی)|"
    r"شکل(?:ش)?\s+(?:ثبت\s*شده|درست|معیار|اصلی)|نسخه(?:ش)?\s+(?:درست|اصلی))"
    r".{0,30}(?:بگو|بده|چی(?:ه|ست)?)",
    re.IGNORECASE,
)

# Meta-tail that can appear between the remembered surface and the actual
# restore cue.  It is stripped only from the extracted user substring.
_NOISE_TAIL = re.compile(
    r"\s+(?:رو|را)\s+(?:(?:عامیانه|محاوره(?:\u200c|\s)?ای|غلط|اشتباه|ناقص|تقریبی)\s*)?"
    r"(?:(?:نوشتم|گفتم|شنیدم|یادم\s+مونده|یادم\s+هست))?\s*$",
    re.IGNORECASE,
)

_SUPPLIED_FORM_MARKER = re.compile(
    r"(?:رو|را).{0,35}(?:عامیانه|محاوره(?:\u200c|\s)?ای|غلط|اشتباه|ناقص|نوشتم|گفتم|شنیدم|"
    r"اصلش|درستش|صورت|شکل|نسخه)",
    re.IGNORECASE,
)

_PERSIAN_TOKEN = re.compile(r"[ءآأإؤئابتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]+")


def extract_explicit_restore_target(query: str) -> str | None:
    """Extract a conservative exact user substring preceding a restore cue."""
    if not isinstance(query, str) or not query.strip():
        return None
    cue = _RESTORE_CUE.search(query)
    if cue is None:
        return None
    prefix = query[: cue.start()].strip(" ،,:؛;-.\n\t")
    if not prefix or not _SUPPLIED_FORM_MARKER.search(
        prefix + " " + query[cue.start() : cue.end()]
    ):
        return None
    cleaned = _NOISE_TAIL.sub("", prefix).strip(" ،,:؛;-.\n\t")
    if len(_PERSIAN_TOKEN.findall(cleaned)) < 3:
        return None
    # Avoid treating a meta-question about restoration itself as a supplied FFE.
    if re.search(r"^(?:چطور|چگونه|روش|برای\s+اینکه|آیا\s+می\s*شه)", cleaned):
        return None
    return cleaned


class Planner(CorePlanner):
    """CorePlanner plus one explicit RESTORE post-plan rescue."""

    def plan(self, query: str) -> PlannerPrediction:
        pred = super().plan(query)
        if not pred.valid or pred.plan is None:
            return pred
        if pred.plan.route is PlanRoute.FFE_REQUIRED:
            return pred

        target = extract_explicit_restore_target(query)
        if target is None:
            return pred

        # GENERAL/CLARIFY can be safely challenged only for this explicit
        # restoration form.  The target is copied verbatim from the user query.
        plan = FFEPlan(
            route=PlanRoute.FFE_REQUIRED,
            actions=(FFEAction.RESTORE_SURFACE,),
            target_span=target,
            requested_count=1,
            needs_explanation=False,
        )
        return PlannerPrediction(
            plan=plan,
            raw_output=pred.raw_output,
            latency_ms=pred.latency_ms,
            model=pred.model,
            error=None,
        )
