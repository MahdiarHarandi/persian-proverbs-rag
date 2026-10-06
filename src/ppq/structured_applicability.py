"""Model-agnostic structured adapters for the acceptance-verifier prompts."""

from __future__ import annotations

from typing import Mapping, Sequence

from .applicability_gate import ConservativeApplicabilityGate
from .applicability_prompts import (
    STRICT_SUPPORT_SYSTEM_PROMPT,
    SUPPORT_VETO_SYSTEM_PROMPT,
    applicability_json_schema,
    render_user_prompt,
)
from .applicability_verifier import (
    ApplicabilityVerification,
    parse_applicability_payload,
)
from .multi_scope import StructuredJsonGenerator


class StructuredApplicabilityVerifier:
    """Run one applicability prompt through a structured JSON generator."""

    def __init__(
        self,
        generator: StructuredJsonGenerator,
        *,
        system_prompt: str,
        verifier_id: str,
        max_new_tokens: int = 96,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if not verifier_id:
            raise ValueError("verifier_id must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        self.generator = generator
        self.system_prompt = system_prompt
        self._verifier_id = verifier_id
        self.max_new_tokens = max_new_tokens

    @property
    def verifier_id(self) -> str:
        return self._verifier_id

    def verify(self, context: str, candidate_meaning: str) -> ApplicabilityVerification:
        messages: Sequence[Mapping[str, str]] = (
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_user_prompt(context, candidate_meaning)},
        )
        generation = self.generator.generate(
            messages,
            applicability_json_schema(),
            max_new_tokens=self.max_new_tokens,
        )
        verdict, reason = parse_applicability_payload(generation.content)
        return ApplicabilityVerification(
            verdict=verdict,
            verifier_id=self.verifier_id,
            reason=reason,
        )


def build_acceptance_gate(
    generator: StructuredJsonGenerator,
) -> ConservativeApplicabilityGate:
    """Build the validated strict-verifier plus support-veto gate."""

    strict_verifier = StructuredApplicabilityVerifier(
        generator,
        system_prompt=STRICT_SUPPORT_SYSTEM_PROMPT,
        verifier_id="strict-entailment",
    )
    support_veto = StructuredApplicabilityVerifier(
        generator,
        system_prompt=SUPPORT_VETO_SYSTEM_PROMPT,
        verifier_id="support-veto",
    )
    return ConservativeApplicabilityGate(strict_verifier, support_veto)
