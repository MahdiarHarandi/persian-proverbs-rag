"""Robust understanding planner for diverse Persian user prompts.

The planner separates intent drafting from surface grounding. The model may
identify the requested operation and may copy a target substring, but a
corpus-grounded SurfaceMentionBinder independently scans the complete user
message before the final FFEPlan is constructed.  This makes quoted delimiters
optional and allows embedded/noisy mentions inside long colloquial text.

The planner still has no authority to output an authentic FFE.  SEARCH_BY_MEANING
returns only a semantic description; final FFE text remains Corpus.resolve-only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .intent_signals import FFEIntentSignalAnalyzer
from .multi_scope import StructuredJsonGenerator
from .planner_types import FFEAction, FFEPlan, PlanRoute
from .slm import PlannerPrediction
from .surface_binder import SurfaceMentionBinder, SurfaceMentionBinding

PLANNER_SYSTEM_PROMPT = r"""
You are the understanding/routing layer of a GENERAL-PURPOSE Persian assistant.
Your job is to determine whether the current request needs an AUTHENTIC Persian
fixed expression (FFE): proverb, idiom, conventional figurative expression, or
other established Persian fixed expression.

IMPORTANT: the user's intended expression or FFE task may be INDIRECT.
- The expression may be embedded inside a long message.
- It may have NO quotation marks, colon, comma, or punctuation around it.
- The message may be colloquial, have typos, omit half-spaces, or contain several
  unrelated clauses.
- The user may never use the words "proverb" or "idiom" and instead ask things
  such as "برای این موقعیت چی میگن؟", "یه جمله جاافتاده برای این حالت میخوام",
  or "چطور رایج بگم که ...".
- A general writing task still needs FFE retrieval when any part asks to include
  a real/conventional Persian expression matching a meaning or situation.

Do NOT route to FFE merely because the request DISCUSSES proverbs/idioms as a
subject, asks about technical terminology, or explicitly asks to INVENT a
fictional expression.

Routes:
- GENERAL: no authentic FFE retrieval/validation is needed.
- FFE_REQUIRED: an authentic FFE must be found/restored/validated/explained.
- CLARIFY: the request clearly refers to an FFE but essential target/context is
  missing and guessing would be unsafe.

Actions:
- SEARCH_BY_MEANING: described meaning/situation -> matching authentic FFE.
- RESTORE_SURFACE: supplied remembered/noisy/incomplete wording -> authentic form.
- ATTEST: asks whether supplied wording is authentic/attested.
- EXPLAIN: asks meaning of a supplied expression as an FFE.
- If the user asks BOTH authenticity and meaning of the same supplied wording, use actions=["ATTEST","EXPLAIN"].
- RETRIEVE_EXAMPLES: asks for N authentic FFEs/examples.

Target rules:
1. target_span is OPTIONAL at this drafting stage. A deterministic corpus binder
   scans the whole user message after you. If you can identify the target, copy
   it CHARACTER-FOR-CHARACTER from the user message. Do not correct it.
2. Never invent/reconstruct a target span. If uncertain, use null.
3. Quotation marks are NOT required for a target to exist.
4. semantic_query is only for SEARCH_BY_MEANING. Describe only the intended
   meaning/situation in ordinary Persian. Never propose an FFE.
5. If a request asks to write a message/caption/story AND include a real fitting
   expression, use FFE_REQUIRED + SEARCH_BY_MEANING.
6. If user explicitly requests a fictional/invented expression, use GENERAL.
7. Return JSON only and every field exactly once.
""".strip()


PLANNER_EXAMPLES: tuple[tuple[str, dict[str, object]], ...] = (
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
        "درباره نقش ضرب المثل ها در فرهنگ ایران یه پاراگراف بنویس",
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
        "برای داستانم یه ضرب المثل کاملا خیالی از خودت بساز",
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
        "دوستم بدون فکر عجله میکنه بعد همه چیز خراب میشه میخوام یه جمله رایج فارسی برای این موقعیت بهش بگم",
        {
            "route": "FFE_REQUIRED",
            "actions": ["SEARCH_BY_MEANING"],
            "target_span": None,
            "semantic_query": "عجله و اقدام بدون فکر باعث خراب شدن کار می شود",
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "یه پیام کوتاه براش بنویس آخرشم یه تعبیر جاافتاده ایرانی بذار که بگه آدم باید روی پای خودش بایسته",
        {
            "route": "FFE_REQUIRED",
            "actions": ["SEARCH_BY_MEANING"],
            "target_span": None,
            "semantic_query": "آدم باید متکی به تلاش و توان خودش باشد",
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "برای این وضع که یکی هی امروز و فردا میکنه چی میگن",
        {
            "route": "FFE_REQUIRED",
            "actions": ["SEARCH_BY_MEANING"],
            "target_span": None,
            "semantic_query": "کسی انجام کار را مدام به تعویق می اندازد",
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "من وسط حرف شنیدم نمونه ی الف ب یعنی چی دقیقا",
        {
            "route": "FFE_REQUIRED",
            "actions": ["EXPLAIN"],
            "target_span": "نمونه ی الف ب",
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": True,
        },
    ),
    (
        "فکر کنم میگن نمونه ی ج د ولی مطمئن نیستم اصلش چی بود",
        {
            "route": "FFE_REQUIRED",
            "actions": ["RESTORE_SURFACE"],
            "target_span": "نمونه ی ج د",
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "این نمونه ی ه و واقعاً اصطلاح فارسیه یا نه",
        {
            "route": "FFE_REQUIRED",
            "actions": ["ATTEST"],
            "target_span": "نمونه ی ه و",
            "semantic_query": None,
            "requested_count": 1,
            "needs_explanation": False,
        },
    ),
    (
        "سه ضرب المثل درباره صبر میخوام",
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
        "همونی که قبلا گفتم اصلش چی بود",
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


def planner_schema() -> dict[str, object]:
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
            "route": {"type": "string", "enum": [r.value for r in PlanRoute]},
            "actions": {
                "type": "array",
                "items": {"type": "string", "enum": [a.value for a in FFEAction]},
            },
            "target_span": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "semantic_query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "requested_count": {"type": "integer"},
            "needs_explanation": {"type": "boolean"},
        },
    }


@dataclass(frozen=True)
class PlannerDraft:
    route: PlanRoute
    actions: tuple[FFEAction, ...]
    target_span: str | None
    semantic_query: str | None
    requested_count: int
    needs_explanation: bool


def _parse_draft(raw: str, query: str) -> PlannerDraft:
    payload = json.loads(raw)
    required = {
        "route",
        "actions",
        "target_span",
        "semantic_query",
        "requested_count",
        "needs_explanation",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("unexpected planner-v3 fields")
    route = PlanRoute(payload["route"])
    raw_actions = payload["actions"]
    if not isinstance(raw_actions, list):
        raise ValueError("actions must be a list")
    actions = tuple(FFEAction(x) for x in raw_actions)
    if len(actions) > 2 or len(set(actions)) != len(actions):
        raise ValueError("invalid action cardinality")
    if len(actions) == 2 and actions != (FFEAction.ATTEST, FFEAction.EXPLAIN):
        raise ValueError("only ATTEST+EXPLAIN multi-action is supported")
    target = payload["target_span"]
    semantic = payload["semantic_query"]
    count = payload["requested_count"]
    explain = payload["needs_explanation"]
    if target is not None:
        if not isinstance(target, str) or not target.strip() or target not in query:
            raise ValueError("target_span must be an exact non-empty user substring or null")
    if semantic is not None and (not isinstance(semantic, str) or not semantic.strip()):
        raise ValueError("semantic_query must be non-empty string or null")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10:
        raise ValueError("requested_count must be integer in [1,10]")
    if not isinstance(explain, bool):
        raise ValueError("needs_explanation must be boolean")
    if route is not PlanRoute.FFE_REQUIRED and actions:
        raise ValueError("GENERAL/CLARIFY cannot carry actions")
    if route is PlanRoute.FFE_REQUIRED and not actions:
        raise ValueError("FFE_REQUIRED requires action")
    if FFEAction.SEARCH_BY_MEANING in actions and semantic is None:
        raise ValueError("SEARCH_BY_MEANING requires semantic_query")
    if FFEAction.SEARCH_BY_MEANING not in actions and semantic is not None:
        raise ValueError("semantic_query only allowed for SEARCH_BY_MEANING")
    return PlannerDraft(route, actions, target, semantic, count, explain)


def _messages(query: str, bindings: tuple[SurfaceMentionBinding, ...]) -> list[dict[str, str]]:
    # The model receives only a count of independently grounded mentions.  Raw
    # text is already present in the user query; family IDs and corpus surfaces
    # are intentionally not revealed at the planner boundary.
    evidence_note = (
        "Deterministic pre-scan note: "
        f"{len(bindings)} corpus-grounded surface mention(s) were detected somewhere in the user message. "
        "This is evidence of a possible supplied expression, not evidence of user intent."
    )
    out = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT + "\n\n" + evidence_note}]
    for user, answer in PLANNER_EXAMPLES:
        out.append({"role": "user", "content": user})
        out.append(
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
            }
        )
    out.append({"role": "user", "content": query.strip()})
    return out


_BOTH_TARGETS = re.compile(r"(?:هر\s*دو|هردو|هر\s*کدوم|هر\s*کدام|دوتاشون|دو\s*تاشون)")
_EXPLICIT_ONE_UNKNOWN = re.compile(
    r"(?:فقط\s+یکی|یکی\s+از\s+این\s+(?:دو|دوتا)).{0,55}(?:نمی\s*گم|نمیگم|مشخص\s+نمی\s*کنم|معلوم\s+نمی\s*کنم)"
)

_EXACT_ATTEST_QUESTION = re.compile(
    r"(?:دقیقاً|دقیقا|همین|به\s+همین\s+صورت).{0,45}(?:ثبت\s*(?:شده|شدن|شده\s*اند)|تایید|تأیید|وجود\s+داره|عبارت\s+فارسی).{0,20}(?:\?|هست|است|؟)"
    r"|(?:ثبت\s*شده|تایید|تأیید).{0,30}(?:دقیقاً|دقیقا|همین\s+صورت)"
)


class CorePlanner:
    """Robust planner with mention-first deterministic target grounding."""

    def __init__(
        self,
        generator: StructuredJsonGenerator,
        binder: SurfaceMentionBinder,
        *,
        max_new_tokens: int = 256,
        signal_analyzer: FFEIntentSignalAnalyzer | None = None,
    ) -> None:
        self.generator = generator
        self.binder = binder
        self.max_new_tokens = max_new_tokens
        self.signals = signal_analyzer or FFEIntentSignalAnalyzer()

    @property
    def model_name(self) -> str:
        return str(getattr(self.generator, "model_id", "structured-generator"))

    @staticmethod
    def _choose_grounded_target(
        draft_target: str | None,
        bindings: tuple[SurfaceMentionBinding, ...],
    ) -> str | None:
        if draft_target is not None:
            # If the model copied a substring that contains exactly one grounded
            # mention, prefer the independently verified local span.  Otherwise
            # retain the exact user substring so unattested ATTEST can still be
            # answered as NOT_ATTESTED rather than disappearing.
            overlapping = [
                b for b in bindings if b.raw_span in draft_target or draft_target in b.raw_span
            ]
            if len(overlapping) == 1:
                return overlapping[0].raw_span
            return draft_target
        if bindings:
            # One binding is the common embedded/no-quote case. With multiple
            # bindings PPQService immediately invokes the bounded multi-scope
            # planner on the complete query before using this placeholder.
            return bindings[0].raw_span
        return None

    def _finalize(
        self,
        query: str,
        draft: PlannerDraft,
        bindings: tuple[SurfaceMentionBinding, ...],
    ) -> FFEPlan:
        signals = self.signals.analyze(query, grounded_surface_count=len(bindings))

        route = draft.route
        actions = draft.actions
        semantic = draft.semantic_query
        count = draft.requested_count
        target = draft.target_span
        distinct_bindings = {b.family_id for b in bindings}

        # Explicit user prohibition has higher authority than an over-eager FFE
        # route.  This protects GENERAL writing/meta tasks such as "هیچ اصطلاحی
        # داخلش نیاور" from corpus intervention.
        if signals.force_general:
            return FFEPlan(route=PlanRoute.GENERAL)

        normalized_query = query.replace("\u200c", " ")
        # A direct exact-form question such as "دقیقاً همین صورت ثبت شده؟" is
        # attestation, not restoration.  This rule is independent of fuzzy
        # binder evidence; ATTEST later preserves the full user-supplied span
        # unless an exact binding exists.
        if _EXACT_ATTEST_QUESTION.search(normalized_query):
            if route is not PlanRoute.GENERAL:
                route = PlanRoute.FFE_REQUIRED
                actions = (FFEAction.ATTEST,)
                semantic = None
                count = 1

        # Conversely, an imperative request to give/correct the registered
        # surface has RESTORE authority even if the draft model returned the
        # joint ATTEST+EXPLAIN action. This is especially important for explicit
        # all/both multi-restore requests.
        if (
            signals.strong
            and signals.suggested_action is FFEAction.RESTORE_SURFACE
            and bindings
            and not _EXACT_ATTEST_QUESTION.search(normalized_query)
        ):
            route = PlanRoute.FFE_REQUIRED
            actions = (FFEAction.RESTORE_SURFACE,)
            semantic = None
            count = 1

        # Deterministic mention-first rescue for an unambiguous multi-surface
        # request.  When >=2 independently grounded expressions are present and
        # the user explicitly asks about both, a CLARIFY draft is unnecessary.
        # Explicitly unresolved "فقط یکی... نمیگم کدوم" requests remain CLARIFY.
        if (
            route is PlanRoute.CLARIFY
            and len(distinct_bindings) >= 2
            and signals.strong
            and signals.suggested_action
            in {FFEAction.RESTORE_SURFACE, FFEAction.ATTEST, FFEAction.EXPLAIN}
            and _BOTH_TARGETS.search(query.replace("\u200c", " "))
            and not _EXPLICIT_ONE_UNKNOWN.search(query.replace("\u200c", " "))
        ):
            route = PlanRoute.FFE_REQUIRED
            actions = (signals.suggested_action,)
            semantic = None
            target = bindings[0].raw_span
            count = 1

        # Conservative under-routing guard. Only high-precision deterministic
        # signals can challenge GENERAL, and only when they imply one action.
        if route is PlanRoute.GENERAL and signals.strong and signals.suggested_action is not None:
            route = PlanRoute.FFE_REQUIRED
            actions = (signals.suggested_action,)
            semantic = (
                query.strip() if signals.suggested_action is FFEAction.SEARCH_BY_MEANING else None
            )
            count = (
                draft.requested_count
                if signals.suggested_action is FFEAction.RETRIEVE_EXAMPLES
                else 1
            )

        # High-confidence deterministic cues may also correct a single-action
        # mismatch. This is intentionally disabled for the preserved joint
        # ATTEST+EXPLAIN plan and for an already-selected RETRIEVE_EXAMPLES plan.
        if (
            route is PlanRoute.FFE_REQUIRED
            and len(actions) == 1
            and signals.strong
            and signals.suggested_action is not None
            and signals.suggested_action is not actions[0]
            and actions[0] is not FFEAction.RETRIEVE_EXAMPLES
        ):
            proposed = signals.suggested_action
            if proposed is FFEAction.SEARCH_BY_MEANING:
                actions = (proposed,)
                semantic = semantic or query.strip()
                target = None
            elif proposed is FFEAction.RETRIEVE_EXAMPLES:
                actions = (proposed,)
                semantic = None
                target = None
            elif target is not None or bindings:
                actions = (proposed,)
                semantic = None

        if route is PlanRoute.GENERAL:
            return FFEPlan(route=PlanRoute.GENERAL)
        if route is PlanRoute.CLARIFY:
            return FFEPlan(route=PlanRoute.CLARIFY)
        assert actions

        if actions == (FFEAction.ATTEST, FFEAction.EXPLAIN):
            # Exact attestation applies to the complete user-supplied wording.
            # Do not shrink a close-but-unattested phrase to a grounded fragment.
            if target is None:
                return FFEPlan(route=PlanRoute.CLARIFY)
            return FFEPlan(
                route=PlanRoute.FFE_REQUIRED,
                actions=actions,
                target_span=target,
                requested_count=1,
                needs_explanation=True,
            )

        action = actions[0]
        if action in {FFEAction.RESTORE_SURFACE, FFEAction.ATTEST, FFEAction.EXPLAIN}:
            distinct = {b.family_id for b in bindings}
            if action is FFEAction.ATTEST:
                # Exact binder evidence is safe to use as the checked span even
                # when the expression is embedded in an unquoted sentence.  This
                # fixes true-attestation cases where the model copies surrounding
                # meta-language into target_span.  Fuzzy/partial bindings are NOT
                # allowed to shrink a close-but-unattested user phrase, preserving
                # the false-positive safety contract.
                exact_bindings = [
                    b for b in bindings if float(getattr(b, "local_score", 0.0)) >= 0.999999
                ]
                if len(exact_bindings) == 1:
                    target = exact_bindings[0].raw_span
                elif target is None and len(distinct) >= 2:
                    target = bindings[0].raw_span
            else:
                target = self._choose_grounded_target(target, bindings)
            if target is None:
                return FFEPlan(route=PlanRoute.CLARIFY)
            return FFEPlan(
                route=PlanRoute.FFE_REQUIRED,
                actions=(action,),
                target_span=target,
                requested_count=1,
                needs_explanation=(action is FFEAction.EXPLAIN),
            )

        if action is FFEAction.SEARCH_BY_MEANING:
            semantic = semantic or query.strip()
            return FFEPlan(
                route=PlanRoute.FFE_REQUIRED,
                actions=(action,),
                semantic_query=semantic,
                requested_count=count,
                needs_explanation=False,
            )

        if action is FFEAction.RETRIEVE_EXAMPLES:
            return FFEPlan(
                route=PlanRoute.FFE_REQUIRED,
                actions=(action,),
                requested_count=count,
                needs_explanation=False,
            )

        raise ValueError("unsupported planner-v3 action")

    def plan(self, query: str) -> PlannerPrediction:
        if not isinstance(query, str) or not query.strip():
            return PlannerPrediction(
                plan=None, raw_output="", latency_ms=0.0, model=self.model_name, error="empty-query"
            )
        bindings = self.binder.bind(query)
        try:
            generation = self.generator.generate(
                _messages(query, bindings),
                planner_schema(),
                max_new_tokens=self.max_new_tokens,
            )
            draft = _parse_draft(generation.content, query)
            plan = self._finalize(query, draft, bindings)
            return PlannerPrediction(
                plan=plan,
                raw_output=generation.content,
                latency_ms=generation.latency_ms,
                model=generation.model,
                error=None,
            )
        except Exception as exc:
            return PlannerPrediction(
                plan=None,
                raw_output=getattr(locals().get("generation", None), "content", ""),
                latency_ms=float(getattr(locals().get("generation", None), "latency_ms", 0.0)),
                model=str(getattr(locals().get("generation", None), "model", self.model_name)),
                error=f"planner-v3:{type(exc).__name__}:{exc}",
            )
