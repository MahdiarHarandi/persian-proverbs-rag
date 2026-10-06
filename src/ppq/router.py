"""Deterministic, fail-closed routing between the three public PPQ tasks.

The router is a convenience layer, not a new retrieval model.  Explicit user
cues are handled first.  When the wording does not name a task, the already
verified Surface and Meaning results are used as *probes* to infer the most
plausible task.  If those probes disagree without a clear winner, the router
asks the user to choose instead of silently running the wrong task.

Attestation is only auto-selected from explicit authenticity language.  A plain
proverb-like string therefore remains a correction/surface request, which
avoids conflating "find/fix this expression" with "prove that it is authentic".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Decision, RouteStatus, SearchResult, TaskRoute, TaskType
from .normalize import retrieval_tokens

_MEANING_CUES = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?:چه|کدام|یه)\s+(?:ضرب\s*[‌ ]?المثل|مثل|مثلی|اصطلاح|اصطلاحی)",
        r"(?:ضرب\s*[‌ ]?المثل|مثل|اصطلاح)\s+(?:برای|مناسب)",
        r"برای\s+(?:این\s+)?(?:معنی|معنا|مفهوم|حالتی)",
        r"(?:معنی|معنا|مفهوم)\s+(?:می[‌ ]?خوام|مناسب|این)",
        r"چه\s+(?:مثلی|اصطلاحی)\s+(?:داریم|میگن|می‌گن|به\s+کار)",
        r"چی\s+می[‌ ]?گن",
        r"مثل\s+معروف\s+می[‌ ]?خوام",
        r"می[‌ ]?خوام\s+که\s+بگه",
    )
)

_SURFACE_CUES = tuple(
    re.compile(pattern)
    for pattern in (
        r"چی\s*بود",
        r"یادم\s+(?:رفته|نمیاد|نمی‌آد|نمیادش)",
        r"کاملش\s+کن",
        r"ادامه(?:ش|اش)",
        r"درستش\s+(?:چیه|چیست)",
        r"صورت\s+درست",
        r"چطور\s+بود",
        r"می[‌ ]?گفت\s*[«“\"]",
        r"اون\s+(?:ضرب\s*[‌ ]?المثل|مثله)",
    )
)

# Attestation cues are deliberately stricter than the extraction cues in
# query.py.  The word "واقعی" can legitimately occur *inside a meaning
# description* (e.g. "جایگاه واقعی خود را فراموش کرده") and must not turn a
# meaning request into an authenticity request.  We therefore require either a
# quoted expression followed by authenticity language, or a direct predicate
# attached to "مثل/ضرب‌المثل/اصطلاح/عبارت".
_QUOTED_AUTHENTICITY = re.compile(
    r"[«“\"][^»”\"]+[»”\"]"
    r".{0,64}(?:واقعی|اصیل|معتبر|ساختگی|جعلی|درسته|درست\s+است)",
    re.S,
)
_QUOTED_FFE_PREDICATE = re.compile(
    r"[«“\"][^»”\"]+[»”\"]"
    r".{0,48}(?:ضرب\s*[‌ ]?المثل|مثل|اصطلاح|عبارت)"
    r".{0,12}(?:است|هست|محسوب\s+می[‌ ]?شود)",
    re.S,
)
_DIRECT_ATTESTATION = re.compile(
    r"(?:این\s+)?(?:ضرب\s*[‌ ]?المثل|مثل|مثلی|اصطلاح(?:ی)?|عبارت)\s+"
    r"(?:واقعیه|واقعی\s+است|اصیله|اصیل\s+است|معتبره|معتبر\s+است|"
    r"ساختگیه|ساختگی\s+است|جعلیه|جعلی\s+است|درسته|درست\s+است|"
    r"وجود\s+داره|وجود\s+دارد)"
)

_ASCII_TECHNICAL_TOKEN = re.compile(r"^[A-Za-z0-9_+./:-]+$")


@dataclass(frozen=True)
class RouterPolicy:
    """Small, interpretable thresholds used only for task selection."""

    max_phrase_tokens: int = 10
    surface_strong_probe_score: float = 0.75
    surface_strong_probe_advantage: float = 0.15
    surface_fragment_max_tokens: int = 6
    surface_fragment_min_score: float = 0.55
    surface_fragment_min_advantage: float = 0.15
    surface_fragment_min_margin: float = 0.08
    surface_fragment_high_score: float = 0.70
    both_accept_surface_min_score: float = 0.60
    weak_meaning_surface_min_score: float = 0.60
    weak_meaning_surface_min_advantage: float = 0.20
    meaning_scaffold_min_tokens: int = 6
    meaning_scaffold_cues: tuple[str, ...] = ("وقتی", "حالتی", "کسی", "معنی", "معنا")
    meaning_clarify_min_score: float = 0.50
    meaning_clarify_min_advantage: float = 0.15


class TaskRouter:
    """Route to Surface, Meaning or Attestation without free-form inference."""

    def __init__(self, policy: RouterPolicy | None = None):
        self.policy = policy or RouterPolicy()

    @staticmethod
    def _matches_any(patterns: tuple[re.Pattern[str], ...], query: str) -> bool:
        return any(pattern.search(query) for pattern in patterns)

    @classmethod
    def explicit_attestation(cls, query: str) -> bool:
        return bool(
            _DIRECT_ATTESTATION.search(query)
            or _QUOTED_AUTHENTICITY.search(query)
            or _QUOTED_FFE_PREDICATE.search(query)
        )

    @classmethod
    def explicit_meaning(cls, query: str) -> bool:
        return cls._matches_any(_MEANING_CUES, query)

    @classmethod
    def explicit_surface(cls, query: str) -> bool:
        return cls._matches_any(_SURFACE_CUES, query)

    @staticmethod
    def _contains_ascii_technical_token(query: str) -> bool:
        return any(_ASCII_TECHNICAL_TOKEN.fullmatch(token) for token in retrieval_tokens(query))

    def explicit_route(self, query: str) -> TaskRoute | None:
        """Return a route only when the user's wording names the task clearly."""
        text = (query or "").strip()
        if not text:
            return TaskRoute(
                RouteStatus.NEEDS_CLARIFICATION,
                None,
                "empty-query",
            )

        attestation = self.explicit_attestation(text)
        meaning = self.explicit_meaning(text)
        surface = self.explicit_surface(text)

        if attestation:
            return TaskRoute(
                RouteStatus.ROUTED,
                TaskType.ATTESTATION,
                "explicit-attestation-cue",
            )
        if surface and not meaning:
            return TaskRoute(
                RouteStatus.ROUTED,
                TaskType.SURFACE,
                "explicit-surface-cue",
            )
        if meaning and not surface:
            return TaskRoute(
                RouteStatus.ROUTED,
                TaskType.MEANING,
                "explicit-meaning-cue",
            )
        if surface and meaning:
            return TaskRoute(
                RouteStatus.NEEDS_CLARIFICATION,
                None,
                "conflicting-explicit-task-cues",
            )
        if self._contains_ascii_technical_token(text):
            return TaskRoute(
                RouteStatus.NEEDS_CLARIFICATION,
                None,
                "technical-out-of-domain-guard",
            )
        return None

    def route_from_probes(
        self,
        query: str,
        surface: SearchResult,
        meaning: SearchResult,
    ) -> TaskRoute:
        """Route an implicit query using already-verified task-level probes."""
        explicit = self.explicit_route(query)
        if explicit is not None:
            return explicit

        tokens = retrieval_tokens(query)
        token_count = len(tokens)
        policy = self.policy

        # Short proverb-like fragments should stay on the Surface path even if
        # accidental character overlap makes Meaning appear confident.  This
        # only selects a task; the Surface verifier may still ABSTAIN.
        if (
            token_count <= policy.max_phrase_tokens
            and surface.score >= policy.surface_strong_probe_score
            and surface.score >= meaning.score + policy.surface_strong_probe_advantage
        ):
            return TaskRoute(
                RouteStatus.ROUTED,
                TaskType.SURFACE,
                "surface-strong-probe",
            )

        if surface.decision is Decision.ACCEPT and meaning.decision is not Decision.ACCEPT:
            return TaskRoute(
                RouteStatus.ROUTED,
                TaskType.SURFACE,
                "surface-only-accept",
            )
        if meaning.decision is Decision.ACCEPT and surface.decision is not Decision.ACCEPT:
            return TaskRoute(
                RouteStatus.ROUTED,
                TaskType.MEANING,
                "meaning-only-accept",
            )
        if surface.decision is Decision.ACCEPT and meaning.decision is Decision.ACCEPT:
            if (
                token_count <= policy.max_phrase_tokens
                and surface.score >= policy.both_accept_surface_min_score
            ):
                return TaskRoute(
                    RouteStatus.ROUTED,
                    TaskType.SURFACE,
                    "surface-phrase-both-accept",
                )
            return TaskRoute(
                RouteStatus.NEEDS_CLARIFICATION,
                None,
                "both-tasks-accept",
            )

        if surface.decision is Decision.CLARIFY and meaning.decision is Decision.ABSTAIN:
            return TaskRoute(
                RouteStatus.ROUTED,
                TaskType.SURFACE,
                "surface-only-clarify",
            )

        if meaning.decision is Decision.CLARIFY and surface.decision is Decision.ABSTAIN:
            natural_meaning_scaffold = token_count >= policy.meaning_scaffold_min_tokens and any(
                cue in query for cue in policy.meaning_scaffold_cues
            )
            strong_meaning_advantage = (
                meaning.score >= policy.meaning_clarify_min_score
                and meaning.score >= surface.score + policy.meaning_clarify_min_advantage
            )
            if natural_meaning_scaffold or strong_meaning_advantage:
                return TaskRoute(
                    RouteStatus.ROUTED,
                    TaskType.MEANING,
                    "meaning-clarify-probe",
                )

            if (
                token_count <= policy.surface_fragment_max_tokens
                and surface.score >= policy.weak_meaning_surface_min_score
                and surface.score >= meaning.score + policy.weak_meaning_surface_min_advantage
            ):
                return TaskRoute(
                    RouteStatus.ROUTED,
                    TaskType.SURFACE,
                    "surface-fragment-over-weak-meaning",
                )
            return TaskRoute(
                RouteStatus.NEEDS_CLARIFICATION,
                None,
                "weak-meaning-clarify",
            )

        if (
            token_count <= policy.surface_fragment_max_tokens
            and surface.score >= policy.surface_fragment_min_score
            and surface.score >= meaning.score + policy.surface_fragment_min_advantage
            and (
                surface.margin >= policy.surface_fragment_min_margin
                or surface.score >= policy.surface_fragment_high_score
            )
        ):
            return TaskRoute(
                RouteStatus.ROUTED,
                TaskType.SURFACE,
                "surface-fragment-probe",
            )

        return TaskRoute(
            RouteStatus.NEEDS_CLARIFICATION,
            None,
            "insufficient-task-evidence",
        )
