from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(
        description="Join blind A/B quality outcomes with paired final-200 latency."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "evaluation-output" / "performance",
    )
    p.add_argument(
        "--completed-review",
        type=Path,
        default=ROOT / "results" / "benchmark" / "blind_review.csv",
    )
    p.add_argument(
        "--key",
        type=Path,
        default=ROOT / "results" / "benchmark" / "blind_key.jsonl",
    )
    return p.parse_args()


def clean(v):
    return str(v or "").strip().upper()


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def med(xs):
    return statistics.median(xs) if xs else None


def main():
    args = parse_args()
    d = args.input_dir
    key = {r["id"]: r for r in read_jsonl(args.key)}
    paired = {r["query_id"]: r for r in read_jsonl(d / "paired_cases.jsonl")}
    with args.completed_review.open("r", encoding="utf-8-sig", newline="") as f:
        review = list(csv.DictReader(f))

    rows = []
    for r in review:
        qid = r["id"]
        if qid not in key or qid not in paired:
            continue
        k = key[qid]
        p = paired[qid]
        pref = clean(r.get("better_response"))
        preferred = k["A"] if pref == "A" else k["B"] if pref == "B" else pref
        by_system = {}
        for side in ("a", "b"):
            system = k[side.upper()]
            by_system[system] = {
                "task_fulfilled": clean(r.get(f"task_fulfilled_{side}")),
                "authenticity": clean(r.get(f"authentic_{side}")),
                "appropriateness": clean(r.get(f"appropriate_{side}")),
                "fabricated": clean(r.get(f"fabricated_{side}")),
            }
        rows.append(
            {
                "query_id": qid,
                "category": p.get("category"),
                "preferred_system": preferred,
                "base_latency_ms": p.get("base_latency_ms"),
                "ppq_latency_ms": p.get("ppq_latency_ms"),
                "absolute_overhead_ms": p.get("absolute_overhead_ms"),
                "relative_overhead_pct": p.get("relative_overhead_pct"),
                "slowdown_ratio": p.get("slowdown_ratio"),
                "ppq_task_fulfilled": by_system.get("PPQ", {}).get("task_fulfilled"),
                "base_task_fulfilled": by_system.get("BASE", {}).get("task_fulfilled"),
                "ppq_authenticity": by_system.get("PPQ", {}).get("authenticity"),
                "base_authenticity": by_system.get("BASE", {}).get("authenticity"),
                "ppq_appropriateness": by_system.get("PPQ", {}).get("appropriateness"),
                "base_appropriateness": by_system.get("BASE", {}).get("appropriateness"),
                "ppq_fabricated": by_system.get("PPQ", {}).get("fabricated"),
                "base_fabricated": by_system.get("BASE", {}).get("fabricated"),
            }
        )

    groups = defaultdict(list)
    for r in rows:
        groups[r["preferred_system"]].append(r)
    summary = {
        "schema": "ppq-final-200-quality-latency-tradeoff-v1",
        "n": len(rows),
        "by_preference": {},
    }
    for name, rs in sorted(groups.items()):
        ov = [
            float(r["absolute_overhead_ms"])
            for r in rs
            if r.get("absolute_overhead_ms") is not None
        ]
        sl = [float(r["slowdown_ratio"]) for r in rs if r.get("slowdown_ratio") is not None]
        summary["by_preference"][name] = {
            "n": len(rs),
            "median_absolute_overhead_ms": med(ov),
            "median_slowdown_ratio": med(sl),
        }

    decisive = [r for r in rows if r["preferred_system"] in {"PPQ", "BASE"}]
    summary["decisive"] = {
        "n": len(decisive),
        "ppq_wins": sum(r["preferred_system"] == "PPQ" for r in decisive),
        "base_wins": sum(r["preferred_system"] == "BASE" for r in decisive),
    }
    if decisive:
        summary["decisive"]["ppq_win_rate"] = summary["decisive"]["ppq_wins"] / len(decisive)
    try:
        from scipy.stats import binomtest

        if decisive:
            b = binomtest(
                summary["decisive"]["ppq_wins"], len(decisive), 0.5, alternative="greater"
            )
            summary["decisive"]["exact_binomial_pvalue_ppq_gt_half"] = float(b.pvalue)
    except Exception as exc:
        summary["decisive"]["stat_test_note"] = str(exc)

    out_json = d / "quality_latency_tradeoff.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = d / "quality_latency_tradeoff_cases.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved:", out_json)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
