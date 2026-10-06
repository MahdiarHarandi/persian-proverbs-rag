"""Bounded target-scope contract for grounded multi-FFE requests.

This is the productionized runtime-F contract.  The model is *not* an
authenticity authority: it can only choose exact spans already proposed and
independently grounded by :mod:`ppq.surface_binder`.

The module is model-agnostic.  A concrete backend only needs to implement the
``StructuredJsonGenerator`` protocol.  Invalid model output fails closed to
``AMBIGUOUS -> CLARIFY`` with no trusted target spans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence


class TargetScope(str, Enum):
    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class MultiScopePlan:
    scope: TargetScope
    target_spans: tuple[str, ...]
    needs_comparison: bool

    def __post_init__(self) -> None:
        n = len(self.target_spans)
        if self.scope is TargetScope.SINGLE:
            if n != 1:
                raise ValueError("SINGLE requires exactly 1 target span")
            if self.needs_comparison:
                raise ValueError("SINGLE cannot request comparison")
        elif self.scope is TargetScope.MULTIPLE:
            if n < 2:
                raise ValueError("MULTIPLE requires at least 2 target spans")
        elif self.scope is TargetScope.AMBIGUOUS:
            # The parser is responsible for discarding model-supplied
            # ambiguity candidates.  This mirrors the frozen runtime-F
            # contract used for Fresh Shadow extended.
            if self.needs_comparison:
                raise ValueError("AMBIGUOUS cannot request comparison")


@dataclass(frozen=True)
class StructuredGeneration:
    content: str
    latency_ms: float
    model: str | None = None


class StructuredJsonGenerator(Protocol):
    """Backend boundary used by :class:`MultiScopePlanner`."""

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int,
    ) -> StructuredGeneration: ...


MULTI_SCOPE_SYSTEM = """
You are a bounded planner for Persian fixed-expression requests.

A deterministic corpus-grounded surface binder has already identified
two or more candidate spans in the user's original message.

Every candidate span is copied from the user's message and has already
been independently grounded to the registered corpus.

Your job is NOT to determine authenticity.
Your job is NOT to generate a proverb or idiom.
Your job is to determine the user's TARGET SCOPE.

TARGET SCOPE:

SINGLE:
Exactly one candidate is intentionally being requested as the active
fixed-expression target. Other candidates may be examples, background,
quoted context, previous discussion, or incidental matches.

MULTIPLE:
Two or more candidates are intentionally being requested together.
For example, the user asks to compare them, explain both, restore both,
check both, or discuss each of them.

AMBIGUOUS:
Multiple candidates are mentioned, but the user explicitly or implicitly
leaves unresolved WHICH candidate is the intended target.
For example, the user says that one of several expressions is intended
but does not identify which one.

Critical distinction:

Mentioning two expressions does NOT automatically mean MULTIPLE.

If the user asks about BOTH → MULTIPLE.

If the user clearly chooses ONE → SINGLE.

If the user intends ONE but does not reveal which one → AMBIGUOUS.

Rules:

1. target_spans may contain ONLY exact strings from candidate_spans.
2. Never invent, reconstruct, normalize, extend, or shorten a span.
3. For SINGLE, return exactly the one intended candidate.
4. For MULTIPLE, return all candidates that are jointly targeted.
5. For AMBIGUOUS, candidate spans may be returned, but they are only
   ambiguity evidence and will NOT be trusted by runtime.
6. needs_comparison=true only if multiple expressions are explicitly
   being compared or differentiated.
7. Return JSON only.
""".strip()


def multi_scope_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["target_scope", "target_spans", "needs_comparison"],
        "properties": {
            "target_scope": {
                "type": "string",
                "enum": [scope.value for scope in TargetScope],
            },
            "target_spans": {
                "type": "array",
                "minItems": 0,
                "maxItems": 10,
                "uniqueItems": True,
                "items": {"type": "string"},
            },
            "needs_comparison": {"type": "boolean"},
        },
    }


def parse_multi_scope_plan(raw: str, allowed_spans: Sequence[str]) -> MultiScopePlan:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "target_scope",
        "target_spans",
        "needs_comparison",
    }:
        raise ValueError("unexpected fields")

    scope = TargetScope(payload["target_scope"])
    spans = payload["target_spans"]
    if not isinstance(spans, list):
        raise ValueError("target_spans must be list")
    if any(not isinstance(span, str) for span in spans):
        raise ValueError("target spans must be strings")
    if len(spans) != len(set(spans)):
        raise ValueError("duplicate target span")

    allowed = set(allowed_spans)
    if any(span not in allowed for span in spans):
        raise ValueError("span outside binder candidates")

    needs_comparison = payload["needs_comparison"]
    if not isinstance(needs_comparison, bool):
        raise ValueError("needs_comparison must be boolean")

    # Fail-closed ambiguity normalization: model-supplied ambiguous candidates
    # are permitted as diagnostic evidence but never become trusted targets.
    if scope is TargetScope.AMBIGUOUS:
        return MultiScopePlan(
            scope=scope,
            target_spans=(),
            needs_comparison=False,
        )

    return MultiScopePlan(
        scope=scope,
        target_spans=tuple(spans),
        needs_comparison=needs_comparison,
    )


def _prompt_messages(query: str, allowed_spans: Sequence[str]) -> list[dict[str, str]]:
    """Return the frozen generic runtime-F prompt; no benchmark FFEs are leaked."""

    examples = [
        (
            {
                "user_query": "«عبارت الف» و «عبارت ب» را با هم مقایسه کن.",
                "candidate_spans": ["عبارت الف", "عبارت ب"],
            },
            {
                "target_scope": "MULTIPLE",
                "target_spans": ["عبارت الف", "عبارت ب"],
                "needs_comparison": True,
            },
        ),
        (
            {
                "user_query": "عبارت الف فقط مثال قبلی بود؛ الان فقط عبارت ب را توضیح بده.",
                "candidate_spans": ["عبارت الف", "عبارت ب"],
            },
            {
                "target_scope": "SINGLE",
                "target_spans": ["عبارت ب"],
                "needs_comparison": False,
            },
        ),
        (
            {
                "user_query": "یکی از عبارت الف یا عبارت ب منظورم است ولی مشخص نکردم کدام.",
                "candidate_spans": ["عبارت الف", "عبارت ب"],
            },
            {
                "target_scope": "AMBIGUOUS",
                "target_spans": ["عبارت الف", "عبارت ب"],
                "needs_comparison": False,
            },
        ),
    ]

    messages: list[dict[str, str]] = [{"role": "system", "content": MULTI_SCOPE_SYSTEM}]
    for user_payload, assistant_payload in examples:
        messages.append(
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    assistant_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {"user_query": query, "candidate_spans": list(allowed_spans)},
                ensure_ascii=False,
            ),
        }
    )
    return messages


class MultiScopePlanner:
    """Bounded target-scope planner using an injected structured generator."""

    def __init__(self, generator: StructuredJsonGenerator):
        self.generator = generator

    def plan(
        self,
        query: str,
        bindings: Sequence[Mapping[str, object]],
        *,
        max_new_tokens: int = 128,
    ) -> dict[str, object]:
        ordered = sorted(
            bindings,
            key=lambda item: (int(item["raw_start"]), int(item["raw_end"])),
        )
        allowed_spans = [str(binding["raw_span"]) for binding in ordered]
        distinct_families = {str(binding["family_id"]) for binding in ordered}

        if len(distinct_families) < 2:
            return {
                "valid": True,
                "scope": TargetScope.SINGLE.value,
                "route": "NOT_MULTI",
                "target_spans": allowed_spans[:1],
                "needs_comparison": False,
                "latency_ms": 0.0,
                "raw_output": "",
                "error": None,
            }

        generation = self.generator.generate(
            _prompt_messages(query, allowed_spans),
            multi_scope_schema(),
            max_new_tokens=max_new_tokens,
        )
        raw = generation.content.strip()

        try:
            plan = parse_multi_scope_plan(raw, allowed_spans)
            route = {
                TargetScope.SINGLE: "NOT_MULTI",
                TargetScope.MULTIPLE: "MULTI_FFE",
                TargetScope.AMBIGUOUS: "CLARIFY",
            }[plan.scope]
            return {
                "valid": True,
                "scope": plan.scope.value,
                "route": route,
                "target_spans": list(plan.target_spans),
                "needs_comparison": plan.needs_comparison,
                "latency_ms": round(float(generation.latency_ms), 1),
                "raw_output": raw,
                "error": None,
            }
        except Exception as exc:  # model output is untrusted input
            return {
                "valid": False,
                "scope": TargetScope.AMBIGUOUS.value,
                "route": "CLARIFY",
                "target_spans": [],
                "needs_comparison": False,
                "latency_ms": round(float(generation.latency_ms), 1),
                "raw_output": raw,
                "error": f"{type(exc).__name__}: {exc}",
            }
