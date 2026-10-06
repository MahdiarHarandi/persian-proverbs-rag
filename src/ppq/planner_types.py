"""Typed contract for selective grounding of Persian fixed expressions.

The planner is deliberately model-independent.  A general-purpose SLM may only
select *whether* an authentic Persian fixed expression (FFE) is required and
which corpus-bound operation should run.  No planner field can carry the final
fixed-expression text; public quotation remains exclusive to ``Corpus.resolve``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class PlanRoute(str, Enum):
    """Top-level routing decision for a general-purpose assistant."""

    GENERAL = "GENERAL"
    FFE_REQUIRED = "FFE_REQUIRED"
    CLARIFY = "CLARIFY"


class FFEAction(str, Enum):
    """Allowed corpus-bound fixed-expression operations."""

    SEARCH_BY_MEANING = "SEARCH_BY_MEANING"
    RESTORE_SURFACE = "RESTORE_SURFACE"
    ATTEST = "ATTEST"
    EXPLAIN = "EXPLAIN"
    RETRIEVE_EXAMPLES = "RETRIEVE_EXAMPLES"


_ALLOWED_MULTI_ACTIONS = {
    (FFEAction.ATTEST, FFEAction.EXPLAIN),
}


@dataclass(frozen=True)
class FFEPlan:
    """Validated plan emitted by the general-purpose SLM gate.

    Version-1 execution is intentionally simple: one FFE operation per request,
    with one explicitly supported dependent multi-intent plan
    ``ATTEST -> EXPLAIN``.  More complex multi-intent requests need a future
    step-based plan because a flat plan cannot safely bind different arguments
    to different operations.

    The plan stores only user spans, semantic descriptions and control metadata.
    Authentic FFE text itself can only enter the response later through
    ``Corpus.resolve``.
    """

    route: PlanRoute
    actions: tuple[FFEAction, ...] = ()
    target_span: str | None = None
    semantic_query: str | None = None
    requested_count: int = 1
    needs_explanation: bool = False

    def __post_init__(self) -> None:
        if self.requested_count < 1 or self.requested_count > 10:
            raise ValueError("requested_count must be in [1, 10]")

        if self.route is PlanRoute.FFE_REQUIRED:
            if not self.actions:
                raise ValueError("FFE_REQUIRED requires at least one action")
        elif self.actions:
            raise ValueError("GENERAL/CLARIFY plans cannot contain FFE actions")

        if len(set(self.actions)) != len(self.actions):
            raise ValueError("FFE actions must be unique and ordered")
        if len(self.actions) > 1 and self.actions not in _ALLOWED_MULTI_ACTIONS:
            raise ValueError(
                "v1 planner supports only the dependent multi-intent ATTEST -> EXPLAIN"
            )

        span_actions = {
            FFEAction.RESTORE_SURFACE,
            FFEAction.ATTEST,
            FFEAction.EXPLAIN,
        }
        if any(action in span_actions for action in self.actions):
            if not self.target_span or not self.target_span.strip():
                raise ValueError("surface/attest/explain actions require target_span")

        if FFEAction.SEARCH_BY_MEANING in self.actions:
            if not self.semantic_query or not self.semantic_query.strip():
                raise ValueError("SEARCH_BY_MEANING requires semantic_query")
        elif self.semantic_query is not None:
            raise ValueError("semantic_query is only valid for SEARCH_BY_MEANING")

        if (
            not any(action in span_actions for action in self.actions)
            and self.target_span is not None
        ):
            raise ValueError("target_span is only valid for surface/attest/explain actions")

        if (
            self.requested_count != 1
            and FFEAction.SEARCH_BY_MEANING not in self.actions
            and FFEAction.RETRIEVE_EXAMPLES not in self.actions
        ):
            raise ValueError("requested_count > 1 requires a retrieval action")

        if FFEAction.EXPLAIN in self.actions and not self.needs_explanation:
            raise ValueError("EXPLAIN action requires needs_explanation=true")

        if self.route is not PlanRoute.FFE_REQUIRED:
            if self.target_span is not None or self.semantic_query is not None:
                raise ValueError("GENERAL/CLARIFY plans cannot carry FFE query fields")
            if self.requested_count != 1:
                raise ValueError("GENERAL/CLARIFY plans must keep requested_count=1")
            if self.needs_explanation:
                raise ValueError("GENERAL/CLARIFY plans cannot request FFE explanation")

    def as_dict(self) -> dict[str, Any]:
        """Serialize the validated plan for JSONL experiment artifacts."""
        return {
            "route": self.route.value,
            "actions": [action.value for action in self.actions],
            "target_span": self.target_span,
            "semantic_query": self.semantic_query,
            "requested_count": self.requested_count,
            "needs_explanation": self.needs_explanation,
        }

    @property
    def requires_ffe(self) -> bool:
        return self.route is PlanRoute.FFE_REQUIRED

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FFEPlan":
        """Parse a JSON-like mapping from an SLM and validate it fail-closed."""
        allowed = {
            "route",
            "actions",
            "target_span",
            "semantic_query",
            "requested_count",
            "needs_explanation",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown plan fields: {sorted(unknown)}")

        # For model experiments require the complete public schema.  This avoids
        # silently rewarding models that omit control fields and rely on Python
        # defaults, while hand-authored callers can still construct FFEPlan
        # directly.
        missing = allowed - set(payload)
        if missing:
            raise ValueError(f"missing plan fields: {sorted(missing)}")

        try:
            route = PlanRoute(payload["route"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("plan route is missing or invalid") from exc

        raw_actions = payload["actions"]
        if not isinstance(raw_actions, (list, tuple)):
            raise ValueError("actions must be an array")
        try:
            actions = tuple(FFEAction(value) for value in raw_actions)
        except (ValueError, TypeError) as exc:
            raise ValueError("plan contains an invalid FFE action") from exc

        requested_count = payload["requested_count"]
        if isinstance(requested_count, bool) or not isinstance(requested_count, int):
            raise ValueError("requested_count must be an integer")

        needs_explanation = payload["needs_explanation"]
        if not isinstance(needs_explanation, bool):
            raise ValueError("needs_explanation must be boolean")

        target_span = payload["target_span"]
        semantic_query = payload["semantic_query"]
        if target_span is not None and not isinstance(target_span, str):
            raise ValueError("target_span must be string or null")
        if semantic_query is not None and not isinstance(semantic_query, str):
            raise ValueError("semantic_query must be string or null")

        return cls(
            route=route,
            actions=actions,
            target_span=target_span,
            semantic_query=semantic_query,
            requested_count=requested_count,
            needs_explanation=needs_explanation,
        )


def ffe_plan_json_schema() -> dict[str, Any]:
    """Return the transport-level JSON schema for strict structured output.

    Cross-field semantics (for example, SEARCH_BY_MEANING requiring a semantic
    query) remain enforced by :class:`FFEPlan`; keeping those checks in Python
    makes the contract portable across local servers with different JSON-schema
    feature support.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "route",
            "actions",
            "target_span",
            "semantic_query",
            "requested_count",
            "needs_explanation",
        ],
        "properties": {
            "route": {
                "type": "string",
                "enum": [route.value for route in PlanRoute],
            },
            "actions": {
                "type": "array",
                "minItems": 0,
                "maxItems": 2,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": [action.value for action in FFEAction],
                },
            },
            "target_span": {"type": ["string", "null"]},
            "semantic_query": {"type": ["string", "null"]},
            "requested_count": {"type": "integer", "minimum": 1, "maximum": 10},
            "needs_explanation": {"type": "boolean"},
        },
    }
