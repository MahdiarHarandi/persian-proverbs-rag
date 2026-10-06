from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import (
    RuntimeStageProfiler,
    call_runtime_handle,
    discover_ppq_runtime,
    extract_ppq_summary,
    locate_final_benchmark,
    peak_cuda_memory_mb,
    reset_peak_cuda_memory,
    timed_call,
    validate_final_benchmark,
)

ROOT = Path(__file__).resolve().parents[2]


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic PPQ internal stage profiler. Run separately from authoritative latency."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation-output" / "performance" / "ppq_stage_profile.jsonl",
    )
    p.add_argument("--factory", default="ppq.bootstrap:build_runtime")
    p.add_argument("--device-map", default="balanced")
    p.add_argument("--attention", default="sdpa")
    p.add_argument("--embedding-device", choices=["cuda:0", "cpu"], default="cuda:0")
    p.add_argument("--embedding-batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--n",
        type=int,
        default=40,
        help="Balanced diagnostic subset size. Use 200 only if you need full stage profiling.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def balanced_subset(rows, n, seed):
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r.get("category", "UNKNOWN"), []).append(r)
    rng = random.Random(seed)
    for bucket in by_cat.values():
        rng.shuffle(bucket)
    cats = sorted(by_cat)
    out = []
    while len(out) < n and any(by_cat[c] for c in cats):
        for c in cats:
            if by_cat[c] and len(out) < n:
                out.append(by_cat[c].pop())
    return out


def main():
    args = parse_args()
    benchmark, manifest = locate_final_benchmark(ROOT)
    validated = validate_final_benchmark(benchmark, manifest)
    rows = balanced_subset(validated["rows"], min(args.n, 200), args.seed)

    if args.output.exists():
        raise SystemExit(
            f"{args.output} exists; move/delete it before a new diagnostic profile run."
        )

    print("Building unchanged PPQ runtime...")
    runtime, factory_meta = discover_ppq_runtime(args, ROOT)
    profiler = RuntimeStageProfiler(runtime)
    print("Factory:", factory_meta.get("factory"))
    print(
        f"Profiling {len(rows)} balanced cases. These timings are diagnostic, not headline latency."
    )

    for idx, case in enumerate(rows, 1):
        qid = str(case.get("id") or case.get("query_id"))
        prompt = str(case.get("prompt") or case.get("query_text"))
        profiler.reset()
        reset_peak_cuda_memory()
        try:
            result, total_ms = timed_call(call_runtime_handle, runtime, prompt, qid)
            summary = extract_ppq_summary(result)
            snap = profiler.snapshot()
            row = {
                "query_id": qid,
                "category": case.get("category"),
                "prompt": prompt,
                "status": "PASS",
                "total_profiled_ms": round(total_ms, 3),
                "route": summary["route"],
                "actions": summary["actions"],
                "decision": summary["decision"],
                "family_ids": summary["family_ids"],
                "peak_cuda_memory_mb": peak_cuda_memory_mb(),
                **snap,
            }
        except Exception as exc:
            row = {
                "query_id": qid,
                "category": case.get("category"),
                "prompt": prompt,
                "status": "ERROR",
                "error": repr(exc),
            }
        append_jsonl(args.output, row)
        print(f"{idx:03d}/{len(rows)} {qid} {row['status']} {row.get('total_profiled_ms')} ms")

    profiler.restore()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
