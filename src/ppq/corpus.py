"""Strict loading and deterministic resolution of the final corpus."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .models import FixedExpression, Match, Source, Variant
from .normalize import normalize_display, normalize_retrieval

SOURCE_KEYS = {
    "source_id",
    "source_name",
    "license_tag",
    "snapshot",
    "ref",
    "provenance_status",
    "review_status",
}
EXPRESSION_KEYS = {
    "expression_id",
    "family_id",
    "expression_type",
    "gloss",
    "canonical_variant_id",
    "variants",
}
VARIANT_KEYS = {"variant_id", "text", "source_id", "source_ref"}


class Corpus:
    def __init__(
        self,
        expressions: list[FixedExpression],
        sources: dict[str, Source],
        content_hash: str,
    ):
        self.expressions = tuple(expressions)
        self.sources = dict(sources)
        self.content_hash = content_hash
        self._expressions = {item.expression_id: item for item in self.expressions}
        self._families = {item.family_id: item for item in self.expressions}
        self._all_variants: tuple[tuple[FixedExpression, Variant], ...] = tuple(
            (expression, variant)
            for expression in self.expressions
            for variant in expression.variants
        )
        self._variants: dict[tuple[str, str], Variant] = {
            (expression.expression_id, variant.variant_id): variant
            for expression, variant in self._all_variants
        }

    def expression(self, expression_id: str) -> FixedExpression | None:
        return self._expressions.get(expression_id)

    def family(self, family_id: str) -> FixedExpression | None:
        """Return one immutable family record by stable family identifier."""
        return self._families.get(family_id)

    def all_variants(self) -> tuple[tuple[FixedExpression, Variant], ...]:
        """Return the stable immutable variant view used by indexes and audits."""
        return self._all_variants

    def resolve_canonical_family(self, family_id: str) -> Match:
        """Resolve a family to its stored canonical variant through one boundary."""
        expression = self._families.get(family_id)
        if expression is None:
            raise KeyError("unknown family pointer")
        return self.resolve(expression.expression_id, expression.canonical_variant_id)

    def resolve(self, expression_id: str, variant_id: str) -> Match:
        """Resolve identifiers to stored text; no query text is emitted."""
        expression = self._expressions.get(expression_id)
        variant = self._variants.get((expression_id, variant_id))
        if expression is None or variant is None:
            raise KeyError("unknown expression/variant pointer")
        source = self.sources[variant.source_id]
        return Match(
            expression_id=expression.expression_id,
            family_id=expression.family_id,
            variant_id=variant.variant_id,
            text=variant.text,
            gloss=expression.gloss,
            expression_type=expression.expression_type,
            source_id=source.source_id,
            source_name=source.name,
            source_ref=variant.source_ref or source.reference,
            license_tag=source.license_tag,
            provenance_status=source.provenance_status,
            review_status=source.review_status,
        )

    @property
    def record_count(self) -> int:
        return len(self.expressions)

    @property
    def variant_count(self) -> int:
        return len(self._variants)

    @property
    def glossed_count(self) -> int:
        return sum(bool(expression.gloss) for expression in self.expressions)

    @property
    def without_gloss_count(self) -> int:
        return self.record_count - self.glossed_count


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_corpus(expressions_path: str | Path, sources_path: str | Path) -> Corpus:
    expressions_file = Path(expressions_path)
    sources_file = Path(sources_path)
    source_rows = _read_jsonl(sources_file)
    expression_rows = _read_jsonl(expressions_file)
    if not source_rows:
        raise ValueError("the corpus needs at least one source")
    if not expression_rows:
        raise ValueError("the corpus needs at least one expression")

    sources: dict[str, Source] = {}
    for row in source_rows:
        if set(row) != SOURCE_KEYS:
            raise ValueError("all source records must use the final uniform schema")
        source_id = str(row["source_id"]).strip()
        if not source_id or source_id in sources:
            raise ValueError(f"invalid or duplicate source_id: {source_id!r}")
        license_tag = str(row["license_tag"]).strip()
        provenance_status = str(row["provenance_status"]).strip()
        review_status = str(row["review_status"]).strip()
        source_name = str(row["source_name"]).strip()
        reference = str(row["ref"]).strip()
        snapshot = str(row["snapshot"]).strip()
        if not all((source_name, reference, snapshot, license_tag, provenance_status)):
            raise ValueError(f"incomplete source metadata for {source_id!r}")
        if review_status != "human-reviewed":
            raise ValueError(f"source {source_id!r} is not marked human-reviewed")
        sources[source_id] = Source(
            source_id=source_id,
            name=source_name,
            reference=reference,
            license_tag=license_tag,
            snapshot=snapshot,
            provenance_status=provenance_status,
            review_status=review_status,
        )

    expressions: list[FixedExpression] = []
    expression_ids: set[str] = set()
    family_ids: set[str] = set()
    variant_ids: set[str] = set()
    normalized_surfaces: set[str] = set()

    for record_number, row in enumerate(expression_rows, 1):
        if set(row) != EXPRESSION_KEYS:
            raise ValueError("all expression records must use the final uniform schema")

        expression_id = str(row["expression_id"]).strip()
        family_id = str(row["family_id"]).strip()
        expression_type = str(row["expression_type"]).strip()
        raw_gloss = str(row["gloss"])
        gloss = normalize_display(raw_gloss)
        canonical_id = str(row["canonical_variant_id"]).strip()

        if not re.fullmatch(r"p\d{4}", expression_id):
            raise ValueError(f"invalid expression_id: {expression_id!r}")
        if expression_id != f"p{record_number:04d}":
            raise ValueError(f"non-sequential expression_id: {expression_id!r}")
        if expression_id in expression_ids:
            raise ValueError(f"duplicate expression_id: {expression_id!r}")
        if not re.fullmatch(r"f\d{4}", family_id):
            raise ValueError(f"invalid family_id: {family_id!r}")
        if family_id != f"f{record_number:04d}":
            raise ValueError(f"non-sequential family_id: {family_id!r}")
        if family_id in family_ids:
            raise ValueError(f"duplicate family_id: {family_id!r}")
        if expression_type != "fixed_expression":
            raise ValueError(f"non-uniform expression_type for {expression_id}")
        if not gloss:
            raise ValueError(f"empty gloss for {expression_id}")
        if "\n" in raw_gloss or "\r" in raw_gloss or not gloss.endswith("."):
            raise ValueError(f"non-uniform gloss style for {expression_id}")
        if not 16 <= len(gloss) <= 110:
            raise ValueError(
                f"gloss length must be between 16 and 110 characters for {expression_id}"
            )

        variants: list[Variant] = []
        for variant_number, variant_row in enumerate(row["variants"], 1):
            if set(variant_row) != VARIANT_KEYS:
                raise ValueError("all variants must use the final uniform schema")
            variant_id = str(variant_row["variant_id"]).strip()
            text = normalize_display(str(variant_row["text"]))
            source_id = str(variant_row["source_id"]).strip()
            if not re.fullmatch(rf"{re.escape(expression_id)}-v[1-9]\d*", variant_id):
                raise ValueError(f"invalid variant_id: {variant_id!r}")
            if variant_id != f"{expression_id}-v{variant_number}":
                raise ValueError(f"non-sequential variant_id: {variant_id!r}")
            if variant_id in variant_ids:
                raise ValueError(f"duplicate variant_id: {variant_id!r}")
            if not text:
                raise ValueError(f"empty text for {variant_id}")
            has_editorial_marker = (
                "\n" in text
                or "\r" in text
                or "//" in text
                or any(marker in text for marker in "()[]{}<>")
                or ("/" in text and " / " not in text)
            )
            if has_editorial_marker:
                raise ValueError(f"editorial marker in surface {variant_id}")
            if source_id not in sources:
                raise ValueError(f"unknown source {source_id!r} for {variant_id}")
            normalized = normalize_retrieval(text)
            if not normalized:
                raise ValueError(f"surface has no retrievable text: {variant_id}")
            if normalized in normalized_surfaces:
                raise ValueError(f"duplicate normalized surface: {text!r}")
            normalized_surfaces.add(normalized)
            variants.append(
                Variant(
                    variant_id=variant_id,
                    text=text,
                    source_id=source_id,
                    source_ref=str(variant_row["source_ref"]).strip(),
                )
            )
            variant_ids.add(variant_id)

        if not variants or canonical_id != f"{expression_id}-v1":
            raise ValueError(f"invalid canonical variant for {expression_id}")

        expressions.append(
            FixedExpression(
                expression_id=expression_id,
                family_id=family_id,
                expression_type=expression_type,
                gloss=gloss,
                canonical_variant_id=canonical_id,
                variants=tuple(variants),
            )
        )
        expression_ids.add(expression_id)
        family_ids.add(family_id)

    digest = hashlib.sha256()
    digest.update(expressions_file.read_bytes())
    digest.update(b"\0ppq-sources\0")
    digest.update(sources_file.read_bytes())
    return Corpus(expressions, sources, digest.hexdigest()[:16])
