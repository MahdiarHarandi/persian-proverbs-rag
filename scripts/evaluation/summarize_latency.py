from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .common import (
    bootstrap_ci,
    distribution_summary,
    locate_final_benchmark,
    read_jsonl,
    validate_final_benchmark,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze final 200 paired PPQ vs same-Gemma performance results."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "evaluation-output" / "performance",
    )
    p.add_argument("--base", type=Path, default=None)
    p.add_argument("--ppq", type=Path, default=None)
    p.add_argument("--stage-profile", type=Path, default=None)
    return p.parse_args()


def norm_action(v):
    if isinstance(v, list):
        return "+".join(str(x) for x in v) if v else "NONE"
    return str(v or "NONE")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def grouped_summary(paired, key):
    groups = defaultdict(list)
    for r in paired:
        groups[str(r.get(key) or "UNKNOWN")].append(r)
    out = []
    for name in sorted(groups):
        rows = groups[name]
        base = [r["base_latency_ms"] for r in rows]
        ppq = [r["ppq_latency_ms"] for r in rows]
        ov = [r["absolute_overhead_ms"] for r in rows]
        slow = [r["slowdown_ratio"] for r in rows]
        out.append(
            {
                key: name,
                "n": len(rows),
                "base_median_ms": round(statistics.median(base), 3),
                "ppq_median_ms": round(statistics.median(ppq), 3),
                "median_absolute_overhead_ms": round(statistics.median(ov), 3),
                "median_slowdown_ratio": round(statistics.median(slow), 4),
                "base_p95_ms": round(distribution_summary(base)["p95"], 3),
                "ppq_p95_ms": round(distribution_summary(ppq)["p95"], 3),
            }
        )
    return out


def auto_ppq_gold_metrics(paired):
    route_n = action_n = decision_n = family_n = 0
    route_ok = action_ok = decision_ok = family_ok = 0
    by_cat = defaultdict(
        lambda: {
            "n": 0,
            "route_ok": 0,
            "action_ok": 0,
            "decision_ok": 0,
            "family_scored": 0,
            "family_ok": 0,
        }
    )
    for r in paired:
        gold = r["gold"]
        cat = r["category"]
        b = by_cat[cat]
        b["n"] += 1
        eroute = gold.get("expected_route")
        if eroute is not None:
            route_n += 1
            ok = str(r.get("ppq_route")) == str(eroute)
            route_ok += int(ok)
            b["route_ok"] += int(ok)
        eactions = gold.get("expected_actions_any") or []
        if eactions:
            action_n += 1
            actual = set(r.get("ppq_actions") or [])
            ok = any(str(x) in actual for x in eactions)
            action_ok += int(ok)
            b["action_ok"] += int(ok)
        edec = gold.get("expected_response_decision") or gold.get("expected_runtime_decision")
        if edec is not None:
            decision_n += 1
            ok = str(r.get("ppq_decision")) == str(edec) or str(r.get("ppq_decision")) == str(
                gold.get("expected_runtime_decision")
            )
            decision_ok += int(ok)
            b["decision_ok"] += int(ok)
        targets = set(str(x) for x in (gold.get("target_family_ids") or []))
        if targets:
            family_n += 1
            b["family_scored"] += 1
            actual_f = set(str(x) for x in (r.get("ppq_family_ids") or []))
            ok = bool(targets & actual_f)
            family_ok += int(ok)
            b["family_ok"] += int(ok)
    return {
        "route_accuracy": route_ok / route_n if route_n else None,
        "action_accuracy": action_ok / action_n if action_n else None,
        "decision_match_rate": decision_ok / decision_n if decision_n else None,
        "exact_target_family_hit_rate": family_ok / family_n if family_n else None,
        "counts": {
            "route_n": route_n,
            "action_n": action_n,
            "decision_n": decision_n,
            "family_n": family_n,
        },
        "by_category": {
            k: {
                **v,
                "route_rate": v["route_ok"] / v["n"] if v["n"] else None,
                "action_rate": v["action_ok"] / v["n"] if v["n"] else None,
                "decision_rate": v["decision_ok"] / v["n"] if v["n"] else None,
                "family_rate": v["family_ok"] / v["family_scored"] if v["family_scored"] else None,
            }
            for k, v in sorted(by_cat.items())
        },
        "note": "Diagnostic gold-linked metrics from generic output extraction. Keep the project's frozen official evaluator authoritative when available.",
    }


def main():
    args = parse_args()
    d = args.input_dir
    base_path = args.base or d / "base_predictions_timed.jsonl"
    ppq_path = args.ppq or d / "ppq_predictions_timed.jsonl"
    stage_path = args.stage_profile or d / "ppq_stage_profile.jsonl"

    base_rows = read_jsonl(base_path)
    ppq_rows = read_jsonl(ppq_path)
    base = {str(r["query_id"]): r for r in base_rows if r.get("status") == "PASS"}
    ppq = {str(r["query_id"]): r for r in ppq_rows if r.get("status") == "PASS"}
    benchmark, manifest = locate_final_benchmark(ROOT)
    validated = validate_final_benchmark(benchmark, manifest)
    gold = {str(r.get("id") or r.get("query_id")): r for r in validated["rows"]}

    common = sorted(set(base) & set(ppq) & set(gold))
    if not common:
        raise SystemExit("No completed paired BASE/PPQ cases found.")

    paired = []
    for qid in common:
        b = base[qid]
        p = ppq[qid]
        g = gold[qid]
        tb = float(b["latency_ms"])
        tp = float(p["latency_ms"])
        if tb <= 0:
            continue
        row = {
            "query_id": qid,
            "category": g.get("category"),
            "comparison_scope": g.get("comparison_scope"),
            "prompt": g.get("prompt"),
            "base_latency_ms": tb,
            "ppq_latency_ms": tp,
            "absolute_overhead_ms": tp - tb,
            "relative_overhead_pct": ((tp - tb) / tb) * 100.0,
            "slowdown_ratio": tp / tb,
            "base_input_tokens": b.get("input_tokens"),
            "base_output_tokens": b.get("output_tokens"),
            "ppq_route": p.get("route"),
            "ppq_actions": p.get("actions") or [],
            "ppq_action": norm_action(p.get("actions") or []),
            "ppq_decision": p.get("decision"),
            "ppq_family_ids": p.get("family_ids") or [],
            "ppq_generation_used": p.get("generation_used"),
            "base_response": b.get("response"),
            "ppq_response": p.get("response"),
            "gold": g,
        }
        paired.append(row)

    base_times = [r["base_latency_ms"] for r in paired]
    ppq_times = [r["ppq_latency_ms"] for r in paired]
    overheads = [r["absolute_overhead_ms"] for r in paired]
    relative = [r["relative_overhead_pct"] for r in paired]
    slowdowns = [r["slowdown_ratio"] for r in paired]

    summary = {
        "schema": "ppq-final-200-performance-summary-v1",
        "paired_n": len(paired),
        "benchmark_sha256": validated["sha256"],
        "latency_ms": {
            "base": distribution_summary(base_times),
            "ppq": distribution_summary(ppq_times),
            "absolute_overhead": distribution_summary(overheads),
            "relative_overhead_pct": distribution_summary(relative),
            "slowdown_ratio": distribution_summary(slowdowns),
        },
        "bootstrap_95ci": {
            "median_absolute_overhead_ms": bootstrap_ci(overheads),
            "median_relative_overhead_pct": bootstrap_ci(relative),
            "median_slowdown_ratio": bootstrap_ci(slowdowns),
        },
        "automatic_ppq_diagnostic": auto_ppq_gold_metrics(paired),
        "errors": {
            "base": [r for r in base_rows if r.get("status") != "PASS"],
            "ppq": [r for r in ppq_rows if r.get("status") != "PASS"],
        },
    }

    try:
        from scipy.stats import wilcoxon

        w = wilcoxon(ppq_times, base_times, alternative="two-sided", zero_method="wilcox")
        summary["paired_latency_test"] = {
            "test": "Wilcoxon signed-rank",
            "statistic": float(w.statistic),
            "pvalue": float(w.pvalue),
        }
    except Exception as exc:
        summary["paired_latency_test"] = {"test": "not_available", "reason": str(exc)}

    # Stage-profile aggregation if available.
    if stage_path.exists():
        stages = [r for r in read_jsonl(stage_path) if r.get("status") == "PASS"]
        stage_totals = defaultdict(list)
        stage_counts = defaultdict(list)
        for r in stages:
            for k, v in (r.get("timing_ms") or {}).items():
                stage_totals[k].append(float(v))
            for k, v in (r.get("counts") or {}).items():
                stage_counts[k].append(float(v))
        summary["stage_profile"] = {
            "n": len(stages),
            "timing_ms": {k: distribution_summary(v) for k, v in sorted(stage_totals.items())},
            "counts": {k: distribution_summary(v) for k, v in sorted(stage_counts.items())},
        }

    by_category = grouped_summary(paired, "category")
    by_action = grouped_summary(paired, "ppq_action")

    write_json(d / "final_summary.json", summary)
    write_jsonl(
        d / "paired_cases.jsonl", [{k: v for k, v in r.items() if k != "gold"} for r in paired]
    )
    flat_rows = [
        {
            k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
            for k, v in r.items()
            if k != "gold"
        }
        for r in paired
    ]
    write_csv(d / "final_200_latency.csv", flat_rows)
    write_csv(d / "latency_by_category.csv", by_category)
    write_csv(d / "latency_by_action.csv", by_action)

    # Human-readable defense summary.
    md = []
    md.append("# Final 200 — PPQ vs Same-Gemma Performance Summary\n")
    md.append(f"Paired cases: **{len(paired)} / 200**\n")
    md.append("## Headline steady-state latency\n")
    md.append(f"- Base median: **{summary['latency_ms']['base']['median'] / 1000:.3f} s**")
    md.append(f"- PPQ median: **{summary['latency_ms']['ppq']['median'] / 1000:.3f} s**")
    md.append(
        f"- Median absolute overhead: **{summary['latency_ms']['absolute_overhead']['median'] / 1000:.3f} s**"
    )
    md.append(
        f"- Median relative overhead: **{summary['latency_ms']['relative_overhead_pct']['median']:.1f}%**"
    )
    md.append(f"- Median slowdown: **{summary['latency_ms']['slowdown_ratio']['median']:.2f}×**")
    md.append(f"- Base P95: **{summary['latency_ms']['base']['p95'] / 1000:.3f} s**")
    md.append(f"- PPQ P95: **{summary['latency_ms']['ppq']['p95'] / 1000:.3f} s**\n")
    md.append("## Interpretation\n")
    md.append(
        "Use these timing results together with the condition-blind human A/B metrics. Latency is the computational cost paid for corpus grounding, retrieval and verification; do not interpret latency alone as quality.\n"
    )
    (d / "DEFENSE_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    print(
        json.dumps(
            {
                "paired_n": len(paired),
                "base_median_s": round(summary["latency_ms"]["base"]["median"] / 1000, 3),
                "ppq_median_s": round(summary["latency_ms"]["ppq"]["median"] / 1000, 3),
                "median_overhead_s": round(
                    summary["latency_ms"]["absolute_overhead"]["median"] / 1000, 3
                ),
                "median_overhead_pct": round(
                    summary["latency_ms"]["relative_overhead_pct"]["median"], 2
                ),
                "median_slowdown": round(summary["latency_ms"]["slowdown_ratio"]["median"], 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Saved analysis to", d)


if __name__ == "__main__":
    main()
