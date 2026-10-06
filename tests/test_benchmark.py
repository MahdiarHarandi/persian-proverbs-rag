import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmark/benchmark.jsonl"
MANIFEST = ROOT / "data/benchmark/manifest.json"


def load_jsonl(path: Path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_benchmark_is_sealed_and_consistent():
    cases = load_jsonl(BENCH)
    corpus = load_jsonl(ROOT / "data/corpus/expressions.jsonl")
    family_ids = {row["family_id"] for row in corpus}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert len(cases) == 200
    assert len({row["id"] for row in cases}) == 200
    assert len({row["prompt"] for row in cases}) == 200
    assert manifest["status"] == "SEALED_PRE_INFERENCE"
    assert manifest["n"] == 200
    assert hashlib.sha256(BENCH.read_bytes()).hexdigest() == manifest["sha256"]

    required = {
        "id",
        "category",
        "prompt",
        "expected_response_decision",
        "expected_runtime_decision",
        "expected_route",
        "expected_actions_any",
        "target_family_ids",
        "comparison_scope",
    }
    for row in cases:
        assert required <= set(row)
        assert row["prompt"].strip()
        assert row["comparison_scope"] in {"ppq_vs_base", "routing_only"}
        assert set(row["target_family_ids"]) <= family_ids

    expected_counts = {
        "surface_completion": 25,
        "surface_noisy": 15,
        "surface_wrong_word": 10,
        "attestation_true": 20,
        "attestation_false": 20,
        "explain": 20,
        "search_by_meaning": 25,
        "search_by_scenario": 20,
        "retrieve_multiple": 10,
        "attest_and_explain": 10,
        "general": 10,
        "clarify": 5,
        "mixed_general_ffe": 10,
    }
    assert Counter(row["category"] for row in cases) == expected_counts


def test_false_attestation_surfaces_are_not_in_corpus_even_after_normalization():
    from ppq.normalize import normalize_retrieval

    cases = load_jsonl(BENCH)
    corpus = load_jsonl(ROOT / "data/corpus/expressions.jsonl")
    surfaces = {v["text"] for row in corpus for v in row["variants"]}
    normalized = {normalize_retrieval(s) for s in surfaces}

    false_cases = [row for row in cases if row["category"] == "attestation_false"]
    assert len(false_cases) == 20
    for row in false_cases:
        quoted = row["prompt"].split("«", 1)[1].split("»", 1)[0]
        assert quoted not in surfaces
        assert normalize_retrieval(quoted) not in normalized
