import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/benchmark"


def test_result_artifacts_are_consistent():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((RES / "blind_evaluation.json").read_text(encoding="utf-8"))
    benchmark = [
        json.loads(x)
        for x in (ROOT / "data/benchmark/benchmark.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    review = list(csv.DictReader((RES / "blind_review.csv").open(encoding="utf-8-sig", newline="")))
    key = [
        json.loads(x)
        for x in (RES / "blind_key.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    assert summary["n"] == 200 == len(benchmark)
    assert (
        summary["benchmark_sha256"]
        == "9a9af819aa020ee71f4ed1b62151972315dda67e1a63ff3ecc9e48211308ffc8"
    )
    assert summary["ppq_automatic"]["automatic_e2e"] == {
        "correct": 162,
        "total": 200,
        "accuracy": 0.81,
    }
    assert len(review) == len(key) == evaluation["n_blind_pairs"] == 185
    assert [r["id"] for r in review] == [r["id"] for r in key]
    assert evaluation["pairwise"]["PPQ"] == 167
    assert evaluation["pairwise"]["BASE"] == 11
    assert evaluation["pairwise"]["TIE"] == 7


def test_blind_key_matches_benchmark_seed_42():
    benchmark = [
        json.loads(x)
        for x in (ROOT / "data/benchmark/benchmark.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    actual = [
        json.loads(x)
        for x in (RES / "blind_key.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    rng = random.Random(42)
    expected = []
    for row in benchmark:
        if row["comparison_scope"] != "ppq_vs_base":
            continue
        a, b = ("PPQ", "BASE") if rng.random() < 0.5 else ("BASE", "PPQ")
        expected.append({"id": row["id"], "A": a, "B": b})
    assert actual == expected


def test_frozen_result_manifest_hashes():
    manifest = json.loads((RES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "RESULTS_FROZEN"
    for name, expected in manifest["files"].items():
        path = RES / name
        assert path.is_file()
        assert path.stat().st_size == expected["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
