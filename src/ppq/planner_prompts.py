"""Prompt templates for the general-purpose SLM FFE gate.

The prompts teach *tool-routing behaviour*, not Persian fixed-expression
knowledge.  They never provide a gold proverb/idiom as the desired model output.
The model may only emit the typed :class:`ppq.planner_types.FFEPlan` control object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class PromptMode(str, Enum):
    ZERO_SHOT = "zero-shot"
    FEW_SHOT = "few-shot"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


SYSTEM_PROMPT = """You are the routing/planning layer of a GENERAL-PURPOSE Persian assistant.
Your only job is to decide whether fulfilling the current user request requires
an AUTHENTIC Persian fixed expression (FFE): a proverb, idiom, or other
conventional fixed figurative expression.

IMPORTANT DECISION RULE
Ask: "Must the assistant retrieve/validate an authentic fixed expression in
order to fulfil this request correctly?"
Do NOT route to FFE merely because the user mentions proverbs/idioms as a topic.

Routes:
- GENERAL: no authentic FFE must be retrieved or validated. Examples include
  unrelated questions; discussion/history/grammar about proverbs; translating
  or analysing text already supplied by the user when authenticity is irrelevant;
  and requests to creatively invent a deliberately fictional expression.
- FFE_REQUIRED: an authentic FFE must be retrieved, restored, validated,
  explained as an authentic expression, or returned as one or more real examples.
- CLARIFY: essential referenced expression/context is missing, so guessing would
  be unsafe (for example: "همونی که قبلاً گفتم، اصلش چی بود؟" with no expression).

Allowed FFE actions:
- SEARCH_BY_MEANING: user describes a meaning/situation and wants a matching real FFE.
- RESTORE_SURFACE: user gives an incomplete/noisy remembered FFE and wants its authentic form.
- ATTEST: user asks whether a supplied expression is authentic/attested.
- EXPLAIN: user asks the meaning of a supplied expression as a fixed expression.
- RETRIEVE_EXAMPLES: user asks for N authentic FFEs without one target meaning.

Safety / grounding rules:
1. NEVER output, complete, invent, recommend, or quote a Persian FFE yourself.
2. NEVER add a field containing an FFE text. The downstream corpus resolves FFE text.
3. target_span must be copied CHARACTER-FOR-CHARACTER from the USER message; never reconstruct, normalize, or correct it. Preserve spaces, ZWNJ, punctuation, colloquial spelling, typos, repeated words, and negation exactly as written. Never merge or split tokens.
4. semantic_query must only describe the intended meaning/situation in ordinary
   language. It must not propose a proverb/idiom or quote a candidate FFE.
5. In planner v1 use one action per request. The only supported multi-action plan is
   ["ATTEST", "EXPLAIN"] when the user asks both whether one supplied expression
   is authentic and what it means.
6. If FFE_REQUIRED, provide only the minimum fields required for the action.
7. If GENERAL or CLARIFY, actions must be [], and do not add target_span or semantic_query.
8. Return every schema field exactly once. Return JSON ONLY: no markdown,
   explanation, chain of thought, or extra keys.

Exact JSON schema:
{
  "route": "GENERAL" | "FFE_REQUIRED" | "CLARIFY",
  "actions": ["SEARCH_BY_MEANING" | "RESTORE_SURFACE" | "ATTEST" | "EXPLAIN" | "RETRIEVE_EXAMPLES", ...],
  "target_span": string | null,
  "semantic_query": string | null,
  "requested_count": integer from 1 to 10,
  "needs_explanation": boolean
}

For GENERAL/CLARIFY use requested_count=1 and needs_explanation=false.
For EXPLAIN, including ATTEST+EXPLAIN, set needs_explanation=true.
"""


# Hand-authored routing demonstrations.  These examples are intentionally about
# the *decision boundary* and do not reveal benchmark labels/family ids or teach
# a model any target corpus expression.
_FEW_SHOT: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "فرق PUT و POST در HTTP چیه؟",
        {
            "route": "GENERAL",
            "actions": [],
            "target_span": None,
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "چرا ضرب‌المثل‌ها در فرهنگ و ادبیات اهمیت دارند؟",
        {
            "route": "GENERAL",
            "actions": [],
            "target_span": None,
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "یک اصطلاح کاملاً خیالی و ساختگی برای داستانم اختراع کن.",
        {
            "route": "GENERAL",
            "actions": [],
            "target_span": None,
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "یه ضرب‌المثل واقعی برای موقعیتی می‌خوام که آدم بدون تلاش انتظار نتیجه داره.",
        {
            "route": "FFE_REQUIRED",
            "actions": ["SEARCH_BY_MEANING"],
            "target_span": None,
            "semantic_query": "انتظار رسیدن به نتیجه بدون تلاش و زحمت",
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "اون اصطلاحی که یادمه فقط این تیکه‌ش بود «دستش به ...»؛ صورت درستش چی بود؟",
        {
            "route": "FFE_REQUIRED",
            "actions": ["RESTORE_SURFACE"],
            "target_span": "دستش به ...",
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "«سنگ زیر باران نمی‌ماند» واقعاً یک ضرب‌المثل فارسیه؟",
        {
            "route": "FFE_REQUIRED",
            "actions": ["ATTEST"],
            "target_span": "سنگ زیر باران نمی‌ماند",
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "آیا «چراغا با باد در نمیاد» یه اصطلاح واقعیه؟",
        {
            "route": "FFE_REQUIRED",
            "actions": ["ATTEST"],
            "target_span": "چراغا با باد در نمیاد",
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "«دل به دیوار زدن» به‌عنوان اصطلاح یعنی چی؟",
        {
            "route": "FFE_REQUIRED",
            "actions": ["EXPLAIN"],
            "target_span": "دل به دیوار زدن",
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": True,
        },
    ),
    (
        "«سنگ زیر باران نمی‌ماند» واقعاً یک ضرب‌المثل فارسیه؟ اگر هست معنیش رو هم بگو.",
        {
            "route": "FFE_REQUIRED",
            "actions": ["ATTEST", "EXPLAIN"],
            "target_span": "سنگ زیر باران نمی‌ماند",
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": True,
        },
    ),
    (
        "سه ضرب‌المثل واقعی فارسی مثال بزن.",
        {
            "route": "FFE_REQUIRED",
            "actions": ["RETRIEVE_EXAMPLES"],
            "target_span": None,
            "semantic_query": None,
            "requested_count": 3,
            "needs_explanation": False,
        },
    ),
    (
        "همون عبارتی که قبلاً گفتم، اصلش چی بود؟",
        {
            "route": "CLARIFY",
            "actions": [],
            "target_span": None,
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
)


def few_shot_examples() -> tuple[tuple[str, dict[str, object]], ...]:
    """Return immutable hand-authored planner demonstrations."""
    return _FEW_SHOT


def build_planner_messages(
    query: str, mode: PromptMode | str = PromptMode.ZERO_SHOT
) -> list[dict[str, str]]:
    """Build a model-agnostic chat request without exposing benchmark gold data."""
    prompt_mode = PromptMode(mode)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("planner query must be a non-empty string")

    messages: list[ChatMessage] = [ChatMessage("system", SYSTEM_PROMPT)]
    if prompt_mode is PromptMode.FEW_SHOT:
        for user_text, plan in _FEW_SHOT:
            messages.append(ChatMessage("user", user_text))
            messages.append(
                ChatMessage(
                    "assistant",
                    json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
                )
            )
    # /no_think is deliberately not embedded here: it is model-specific.  The
    # adapter can set a transport hint while the semantic prompt stays portable.
    messages.append(ChatMessage("user", query.strip()))
    return [message.as_dict() for message in messages]


def prompt_texts(messages: Iterable[dict[str, str]]) -> str:
    """Stable text representation used only for prompt audits/tests."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
