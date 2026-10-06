"""Optional Hugging Face Gemma backend for :mod:`ppq.multi_scope`.

Imports of torch/transformers/lm-format-enforcer are intentionally lazy so the
core PPQ package remains dependency-free for deterministic Surface and corpus
operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .multi_scope import StructuredGeneration

GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"
GEMMA4_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"


@dataclass
class GemmaHFJsonGenerator:
    processor: object
    model: object
    model_id: str = GEMMA4_MODEL_ID

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = GEMMA4_MODEL_ID,
        revision: str = GEMMA4_REVISION,
        device_map: str = "balanced",
        attention: str = "sdpa",
    ) -> "GemmaHFJsonGenerator":
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Gemma runtime requires CUDA")

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

    @staticmethod
    def _sync_cuda(torch_module: object) -> None:
        if torch_module.cuda.is_available():
            for device_id in range(torch_module.cuda.device_count()):
                torch_module.cuda.synchronize(device_id)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int,
    ) -> StructuredGeneration:
        import torch
        import transformers.tokenization_utils as tokenization_utils
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        # Compatibility shim used in the validated Kaggle runtime.
        tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase

        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(dict(schema))
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
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        self._sync_cuda(torch)
        latency_ms = (time.perf_counter() - start) * 1000.0

        content = self.processor.tokenizer.decode(
            output[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        return StructuredGeneration(
            content=content,
            latency_ms=latency_ms,
            model=self.model_id,
        )
