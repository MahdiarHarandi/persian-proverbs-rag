"""Dependency-light adapter for general-purpose SLM planner inference.

No model is enabled by default. The project talks to local/remote model runners
through an OpenAI-compatible HTTP endpoint using only the Python standard
library. This keeps the retrieval baseline importable without transformers,
CUDA, or a vendor SDK and makes model choice an explicit experiment.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

from .planner_prompts import PromptMode, build_planner_messages
from .planner_types import FFEPlan, ffe_plan_json_schema


class SLMError(RuntimeError):
    """Base error for model transport or planner-output failures."""


class SLMTransportError(SLMError):
    """The model endpoint could not return a usable chat completion."""


class SLMPlanParseError(SLMError):
    """The model returned content that is not a valid typed FFE plan."""


class StructuredOutputMode(str, Enum):
    """Portable structured-output levels for OpenAI-compatible servers."""

    NONE = "none"
    JSON_OBJECT = "json-object"
    JSON_SCHEMA = "json-schema"


class StructuredOutputAPI(str, Enum):
    """Wire format used to request structured output from the serving runtime."""

    OPENAI = "openai"
    VLLM = "vllm"


@dataclass(frozen=True)
class ModelCompletion:
    content: str
    latency_ms: float
    model: str


@dataclass(frozen=True)
class PlannerPrediction:
    plan: FFEPlan | None
    raw_output: str
    latency_ms: float
    model: str
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.plan is not None and self.error is None


class ChatBackend(Protocol):
    """Minimal backend contract used by :class:`SLMPlanner`."""

    @property
    def model_name(self) -> str: ...

    def complete(self, messages: Sequence[Mapping[str, str]]) -> ModelCompletion: ...


class CallableChatBackend:
    """Small adapter useful for deterministic tests and local experiments."""

    def __init__(
        self, function: Callable[[Sequence[Mapping[str, str]]], str], model_name: str = "callable"
    ):
        self._function = function
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, messages: Sequence[Mapping[str, str]]) -> ModelCompletion:
        start = time.perf_counter_ns()
        content = self._function(messages)
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        if not isinstance(content, str):
            raise SLMTransportError("callable backend must return a string")
        return ModelCompletion(content=content, latency_ms=latency_ms, model=self.model_name)


_RESERVED_REQUEST_KEYS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "response_format",
    "structured_outputs",
}


def _vllm_transport_schema() -> dict[str, Any]:
    """Return a vLLM/xgrammar-friendly schema for constrained decoding.

    Keep the exact public FFEPlan fields/types/enums/required keys, but omit
    range/cardinality keywords that some xgrammar builds reject or mishandle.
    The canonical :class:`FFEPlan` parser still enforces action uniqueness,
    action-count limits, requested_count bounds, and all cross-field semantics
    after decoding, so this relaxation affects only the wire grammar.
    """
    schema = ffe_plan_json_schema()
    actions = schema["properties"]["actions"]
    for key in ("minItems", "maxItems", "uniqueItems"):
        actions.pop(key, None)
    requested_count = schema["properties"]["requested_count"]
    for key in ("minimum", "maximum"):
        requested_count.pop(key, None)
    return schema


class OpenAICompatibleChatBackend:
    """OpenAI-compatible chat-completions client implemented with stdlib only.

    It is intended for local runners such as vLLM, SGLang, llama.cpp server,
    LM Studio, or any endpoint implementing ``POST /v1/chat/completions``.
    Structured output is optional because local servers differ in support:
    ``json-schema`` is preferred when implemented, ``json-object`` is a portable
    fallback, and ``none`` lets prompt-only adherence be measured separately.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_tokens: int = 256,
        temperature: float = 0.0,
        structured_output: StructuredOutputMode | str = StructuredOutputMode.NONE,
        structured_output_api: StructuredOutputAPI | str = StructuredOutputAPI.OPENAI,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url is required")
        if not model.strip():
            raise ValueError("model is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_tokens < 32 or max_tokens > 2048:
            raise ValueError("max_tokens must be in [32, 2048]")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self._base_url = base_url.rstrip("/")
        self._model_name = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._structured_output = StructuredOutputMode(structured_output)
        self._structured_output_api = StructuredOutputAPI(structured_output_api)
        self._extra_body = dict(extra_body or {})
        reserved = sorted(_RESERVED_REQUEST_KEYS & set(self._extra_body))
        if reserved:
            raise ValueError(
                "extra_body cannot override experiment-critical request fields: "
                + ", ".join(reserved)
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def structured_output(self) -> StructuredOutputMode:
        return self._structured_output

    @property
    def structured_output_api(self) -> StructuredOutputAPI:
        return self._structured_output_api

    @property
    def endpoint(self) -> str:
        if self._base_url.endswith("/v1"):
            return self._base_url + "/chat/completions"
        return self._base_url + "/v1/chat/completions"

    def _payload(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [dict(message) for message in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._structured_output is StructuredOutputMode.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        elif self._structured_output is StructuredOutputMode.JSON_SCHEMA:
            if self._structured_output_api is StructuredOutputAPI.VLLM:
                # vLLM 0.27.x exposes a native structured-output wire format.
                # Keep the same FFEPlan schema while changing only transport syntax.
                payload["structured_outputs"] = {"json": _vllm_transport_schema()}
            else:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ffe_plan",
                        "strict": True,
                        "schema": ffe_plan_json_schema(),
                    },
                }
        # Explicit extra-body values win so a local runner can receive e.g.
        # chat-template kwargs without making ppq depend on that runner.
        payload.update(self._extra_body)
        return payload

    def complete(self, messages: Sequence[Mapping[str, str]]) -> ModelCompletion:
        payload = json.dumps(self._payload(messages), ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=headers, method="POST"
        )
        start = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise SLMTransportError(f"model endpoint HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SLMTransportError(f"model endpoint unavailable: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0

        try:
            response_json = json.loads(raw.decode("utf-8"))
            choices = response_json["choices"]
            content = choices[0]["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise SLMTransportError("chat completion response has an unsupported shape") from exc
        if not isinstance(content, str):
            raise SLMTransportError("chat completion content must be a string")
        return ModelCompletion(content=content, latency_ms=latency_ms, model=self.model_name)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_JSON = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


def _normalize_model_json_text(raw: str) -> str:
    """Remove transport-format wrappers without repairing plan semantics."""
    text = raw.strip()
    text = _THINK_BLOCK.sub("", text).strip()
    match = _FENCED_JSON.fullmatch(text)
    if match:
        text = match.group(1).strip()
    return text


def is_json_object_output(raw: str) -> bool:
    """Return whether model output is one JSON object after transport wrappers.

    This is intentionally weaker than :func:`parse_ffe_plan`.  Runtime smoke
    tests use it only to decide whether a structured-output transport mode is
    operational.  Cross-field planner semantics (for example target-span
    grounding) remain model behaviour and are scored separately as invalid
    plans rather than being misclassified as transport incompatibility.
    """
    if not isinstance(raw, str) or not raw.strip():
        return False
    text = _normalize_model_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, Mapping)


def parse_ffe_plan(raw: str) -> FFEPlan:
    """Parse one exact JSON object and validate it against ``FFEPlan``.

    Deliberately rejected: prose before/after JSON, arrays, unknown/missing keys,
    invalid actions, or attempts to smuggle generated FFE text. The parser strips
    only common model-format wrappers; it never repairs JSON or guesses values.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise SLMPlanParseError("empty model output")
    text = _normalize_model_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SLMPlanParseError("model output is not one valid JSON object") from exc
    if not isinstance(payload, Mapping):
        raise SLMPlanParseError("planner output must be a JSON object")
    try:
        return FFEPlan.from_mapping(payload)
    except (ValueError, TypeError) as exc:
        raise SLMPlanParseError(str(exc)) from exc


def validate_plan_against_query(plan: FFEPlan, query: str) -> None:
    """Enforce user-grounded planner fields after schema validation.

    A surface/attestation/explanation target is evidence supplied by the user,
    not text the SLM may reconstruct. Therefore the planner must copy it verbatim
    from the current request.
    """
    if plan.target_span is not None and plan.target_span not in query:
        raise SLMPlanParseError("target_span is not an exact substring of the user query")


class SLMPlanner:
    """Run one general-purpose model call and return a fail-closed plan."""

    def __init__(
        self, backend: ChatBackend, *, prompt_mode: PromptMode | str = PromptMode.ZERO_SHOT
    ):
        self.backend = backend
        self.prompt_mode = PromptMode(prompt_mode)

    def plan(self, query: str) -> PlannerPrediction:
        messages = build_planner_messages(query, self.prompt_mode)
        try:
            completion = self.backend.complete(messages)
        except SLMError as exc:
            return PlannerPrediction(
                plan=None,
                raw_output="",
                latency_ms=0.0,
                model=self.backend.model_name,
                error=f"transport:{exc}",
            )
        try:
            plan = parse_ffe_plan(completion.content)
            validate_plan_against_query(plan, query)
        except SLMPlanParseError as exc:
            return PlannerPrediction(
                plan=None,
                raw_output=completion.content,
                latency_ms=completion.latency_ms,
                model=completion.model,
                error=f"parse:{exc}",
            )
        return PlannerPrediction(
            plan=plan,
            raw_output=completion.content,
            latency_ms=completion.latency_ms,
            model=completion.model,
        )
