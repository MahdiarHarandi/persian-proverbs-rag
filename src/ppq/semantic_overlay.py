"""Human adjudication overlay for semantic candidate acceptability.

The frozen benchmark remains immutable.  This module stores *pair-level* human
judgments for (query, candidate-family) pairs without rewriting benchmark gold.
That distinction is essential because "not present in acceptable_family_ids"
is not evidence that a candidate is semantically wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .corpus import Corpus


class SemanticAcceptability(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    NOT_ACCEPTABLE = "NOT_ACCEPTABLE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class SemanticOverlayRow:
    query_id: str
    family_id: str
    label: SemanticAcceptability
    rationale: str = ""
    adjudication_source: str = "human"
    annotator_count: int = 1

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("semantic overlay query_id must be non-empty")
        if not self.family_id.strip():
            raise ValueError("semantic overlay family_id must be non-empty")
        if self.annotator_count < 1:
            raise ValueError("annotator_count must be >= 1")
        if not self.adjudication_source.strip():
            raise ValueError("adjudication_source must be non-empty")


class SemanticAcceptabilityOverlay:
    """Immutable lookup table for blind semantic adjudication results."""

    def __init__(self, rows: Iterable[SemanticOverlayRow], *, corpus: Corpus | None = None):
        table: dict[tuple[str, str], SemanticOverlayRow] = {}
        for row in rows:
            key = (row.query_id, row.family_id)
            if key in table:
                raise ValueError(f"duplicate semantic overlay pair: {key}")
            if corpus is not None and corpus.family(row.family_id) is None:
                raise ValueError(f"semantic overlay references unknown family: {row.family_id}")
            table[key] = row
        self._rows = table

    def get(self, query_id: str, family_id: str) -> SemanticOverlayRow | None:
        return self._rows.get((query_id, family_id))

    def label(self, query_id: str, family_id: str) -> SemanticAcceptability | None:
        row = self.get(query_id, family_id)
        return None if row is None else row.label

    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> tuple[SemanticOverlayRow, ...]:
        return tuple(self._rows.values())

    @classmethod
    def load_jsonl(
        cls,
        path: str | Path,
        *,
        corpus: Corpus | None = None,
    ) -> "SemanticAcceptabilityOverlay":
        rows: list[SemanticOverlayRow] = []
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(file)
        for line_number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file}:{line_number}: invalid JSON") from exc
            allowed = {
                "query_id",
                "family_id",
                "label",
                "rationale",
                "adjudication_source",
                "annotator_count",
            }
            unknown = set(payload) - allowed
            if unknown:
                raise ValueError(f"{file}:{line_number}: unknown fields {sorted(unknown)}")
            missing = {"query_id", "family_id", "label"} - set(payload)
            if missing:
                raise ValueError(f"{file}:{line_number}: missing fields {sorted(missing)}")
            rows.append(
                SemanticOverlayRow(
                    query_id=str(payload["query_id"]),
                    family_id=str(payload["family_id"]),
                    label=SemanticAcceptability(str(payload["label"])),
                    rationale=str(payload.get("rationale", "")),
                    adjudication_source=str(payload.get("adjudication_source", "human")),
                    annotator_count=int(payload.get("annotator_count", 1)),
                )
            )
        return cls(rows, corpus=corpus)
