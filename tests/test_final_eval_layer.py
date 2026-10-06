from pathlib import Path

from scripts.evaluation.common import (
    bootstrap_ci,
    distribution_summary,
    extract_ppq_summary,
    locate_final_benchmark,
    to_jsonable,
    validate_final_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_summary_basic():
    s = distribution_summary([1, 2, 3, 4, 5])
    assert s["n"] == 5
    assert s["median"] == 3
    assert s["min"] == 1
    assert s["max"] == 5


def test_bootstrap_is_deterministic():
    a = bootstrap_ci([1, 2, 3, 4, 5], iterations=100, seed=7)
    b = bootstrap_ci([1, 2, 3, 4, 5], iterations=100, seed=7)
    assert a == b


def test_extract_ppq_summary_nested_mapping():
    obj = {
        "runtime": {
            "plan": {"route": "FFE_REQUIRED", "actions": ["RESTORE_SURFACE"]},
            "decision": "ACCEPT",
            "matches": [{"family_id": "f0016"}],
        },
        "text": "ok",
    }
    s = extract_ppq_summary(obj)
    assert s["route"] == "FFE_REQUIRED"
    assert s["actions"] == ["RESTORE_SURFACE"]
    assert s["decision"] == "ACCEPT"
    assert "f0016" in s["family_ids"]


def test_jsonable_plain():
    assert to_jsonable({"x": [1, 2]}) == {"x": [1, 2]}


def test_benchmark_locator_uses_packaged_paths():
    benchmark, manifest = locate_final_benchmark(ROOT)
    assert benchmark == ROOT / "data" / "benchmark" / "benchmark.jsonl"
    assert manifest == ROOT / "data" / "benchmark" / "manifest.json"
    validated = validate_final_benchmark(benchmark, manifest)
    assert validated["n"] == 200
    assert validated["sha256"] == (
        "9a9af819aa020ee71f4ed1b62151972315dda67e1a63ff3ecc9e48211308ffc8"
    )
