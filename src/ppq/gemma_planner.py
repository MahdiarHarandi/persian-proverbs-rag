"""Optional local Gemma/Hugging Face backend for structured FFE planning.

The adapter mirrors the successful notebook structured-decoding runtime and is
kept separate from the dependency-free planner contract.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .gemma_multiscope import GEMMA4_MODEL_ID, GEMMA4_REVISION
from .planner_types import ffe_plan_json_schema
from .slm import ModelCompletion


def lmfe_ffe_plan_wire_schema() -> dict:
    schema = copy.deepcopy(ffe_plan_json_schema())
    actions = schema["properties"]["actions"]
    for key in ("minItems", "maxItems", "uniqueItems"):
        actions.pop(key, None)
    requested = schema["properties"]["requested_count"]
    for key in ("minimum", "maximum"):
        requested.pop(key, None)
    for field in ("target_span", "semantic_query"):
        schema["properties"][field] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return schema


@dataclass
class GemmaFFEPlanBackend:
    processor: object
    model: object
    model_id: str = GEMMA4_MODEL_ID
    max_new_tokens: int = 256

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = GEMMA4_MODEL_ID,
        revision: str = GEMMA4_REVISION,
        device_map: str = "balanced",
        attention: str = "sdpa",
    ) -> "GemmaFFEPlanBackend":
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Gemma planner runtime requires CUDA")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation=attention,
        )
        model.eval()
        return cls(processor=processor, model=model, model_id=model_id)

    @property
    def model_name(self) -> str:
        return self.model_id

    @staticmethod
    def _sync_cuda(torch_module: object) -> None:
        if torch_module.cuda.is_available():
            for i in range(torch_module.cuda.device_count()):
                torch_module.cuda.synchronize(i)

    def complete(self, messages: Sequence[Mapping[str, str]]) -> ModelCompletion:
        import torch
        import transformers.tokenization_utils as tokenization_utils
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(lmfe_ffe_plan_wire_schema())
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            self.processor.tokenizer,
            parser,
        )
        inputs = self.processor.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        self._sync_cuda(torch)
        start = time.perf_counter_ns()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        self._sync_cuda(torch)
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        content = self.processor.tokenizer.decode(
            output[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        return ModelCompletion(
            content=content,
            latency_ms=latency_ms,
            model=self.model_name,
        )
