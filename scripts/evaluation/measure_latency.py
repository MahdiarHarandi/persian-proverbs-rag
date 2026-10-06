from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .common import (
    DirectGemmaAdapter,
    call_runtime_handle,
    cuda_sync,
    discover_ppq_runtime,
    environment_metadata,
    extract_ppq_summary,
    find_hf_generation_stack,
    locate_final_benchmark,
    peak_cuda_memory_mb,
    project_root,
    reset_peak_cuda_memory,
    safe_status_record,
    timed_call,
    validate_final_benchmark,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "PASS":
            out.add(str(row["query_id"]))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Paired final-200 PPQ vs same-Gemma timing benchmark. Does not alter PPQ core behavior."
    )
    p.add_argument("--benchmark", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation-output" / "performance",
    )
    p.add_argument(
        "--factory",
        default="ppq.bootstrap:build_runtime",
        help="PPQ runtime factory MODULE:FUNCTION. Default matches the final UI/runtime entry point.",
    )
    p.add_argument("--device-map", default="balanced")
    p.add_argument("--attention", default="sdpa")
    p.add_argument("--embedding-device", choices=["cuda:0", "cpu"], default="cuda:0")
    p.add_argument("--embedding-batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--limit", type=int, default=None, help="Debug only. Do not use for final thesis results."
    )
    p.add_argument("--only-system", choices=["base", "ppq", "both"], default="both")
    p.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Debug only; not allowed for final thesis run.",
    )
    return p.parse_args()


def run_ppq(runtime, case):
    qid = str(case.get("id") or case.get("query_id"))
    prompt = str(case.get("prompt") or case.get("query_text"))
    reset_peak_cuda_memory()
    try:
        result, elapsed = timed_call(call_runtime_handle, runtime, prompt, qid)
        summary = extract_ppq_summary(result)
        return {
            "query_id": qid,
            "category": case.get("category"),
            "prompt": prompt,
            "comparison_scope": case.get("comparison_scope"),
            "system": "PPQ",
            "status": "PASS",
            "latency_ms": round(elapsed, 3),
            "peak_cuda_memory_mb": peak_cuda_memory_mb(),
            "route": summary["route"],
            "actions": summary["actions"],
            "decision": summary["decision"],
            "family_ids": summary["family_ids"],
            "response": summary["text"],
            "generation_used": summary["generation_used"],
            "generation_valid": summary["generation_valid"],
            "reason": summary["reason"],
            "raw_result": summary["raw"],
        }
    except Exception as exc:
        return safe_status_record(case, "PPQ", exc)


def run_base(base, case):
    qid = str(case.get("id") or case.get("query_id"))
    prompt = str(case.get("prompt") or case.get("query_text"))
    reset_peak_cuda_memory()
    try:
        output, elapsed = timed_call(base.generate, prompt)
        return {
            "query_id": qid,
            "category": case.get("category"),
            "prompt": prompt,
            "comparison_scope": case.get("comparison_scope"),
            "system": "BASE",
            "status": "PASS",
            "latency_ms": round(elapsed, 3),
            "peak_cuda_memory_mb": peak_cuda_memory_mb(),
            "response": output["text"],
            "input_tokens": output.get("input_tokens"),
            "output_tokens": output.get("output_tokens"),
        }
    except Exception as exc:
        return safe_status_record(case, "BASE", exc)


def main():
    args = parse_args()
    root = project_root()
    benchmark, manifest = locate_final_benchmark(root)
    if args.benchmark:
        benchmark = args.benchmark
    if args.manifest:
        manifest = args.manifest
    validated = validate_final_benchmark(benchmark, manifest, strict_hash=not args.skip_hash_check)
    rows = list(validated["rows"])
    if args.limit is not None:
        rows = rows[: args.limit]
        print("WARNING: --limit is for debugging only; results are not final-benchmark claims.")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    base_path = out / "base_predictions_timed.jsonl"
    ppq_path = out / "ppq_predictions_timed.jsonl"
    meta_path = out / "run_metadata.json"

    if not args.resume:
        for p in (base_path, ppq_path):
            if p.exists():
                raise SystemExit(f"{p} already exists. Use --resume or move/delete the prior run.")

    meta = environment_metadata(root)
    meta.update(
        {
            "schema": "ppq-final-200-paired-performance-v1",
            "benchmark": {k: v for k, v in validated.items() if k != "rows"},
            "configuration": {
                "device_map": args.device_map,
                "attention": args.attention,
                "embedding_device": args.embedding_device,
                "embedding_batch_size": args.embedding_batch_size,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "warmup": args.warmup,
                "condition_order": "per-query randomized with fixed seed",
                "base_condition": "same loaded Gemma, direct original prompt, no PPQ/corpus/retrieval/verifier",
                "ppq_condition": "unchanged final PPQ runtime",
                "headline_latency": "steady-state end-to-end wall-clock with CUDA synchronize before/after call",
            },
        }
    )

    print("[1/4] Building final PPQ runtime once...")
    t0 = time.perf_counter()
    runtime, factory_meta = discover_ppq_runtime(args, root)
    cuda_sync()
    ppq_startup_ms = (time.perf_counter() - t0) * 1000
    meta["ppq_factory"] = factory_meta
    meta["startup"] = {"ppq_runtime_ms": round(ppq_startup_ms, 3)}

    print("[2/4] Locating the exact loaded Gemma stack for direct Base...")
    model, processor, stack_meta = find_hf_generation_stack(runtime)
    base = DirectGemmaAdapter(model, processor, max_new_tokens=args.max_new_tokens)
    meta["same_gemma_stack"] = stack_meta
    write_json(meta_path, meta)

    # Warm-up is intentionally outside the 200 benchmark and not included in results.
    warmups = [
        "فقط در یک جمله بگو امروز چه روزی است؟",
        "فرق یک فایل و یک پوشه چیست؟",
        "یک جمله کوتاه درباره یادگیری بنویس.",
        "عدد دو به علاوه دو چند است؟",
        "یک تعریف کوتاه از شبکه رایانه‌ای بده.",
    ]
    print(f"[3/4] Warm-up: {args.warmup} non-benchmark prompts")
    for i in range(args.warmup):
        prompt = warmups[i % len(warmups)]
        if args.only_system in {"base", "both"}:
            try:
                timed_call(base.generate, prompt)
            except Exception as exc:
                raise RuntimeError(f"Base warm-up failed: {exc}") from exc
        if args.only_system in {"ppq", "both"}:
            try:
                timed_call(call_runtime_handle, runtime, prompt, f"warmup-{i + 1}")
            except Exception as exc:
                raise RuntimeError(f"PPQ warm-up failed: {exc}") from exc

    completed_base = load_completed(base_path) if args.resume else set()
    completed_ppq = load_completed(ppq_path) if args.resume else set()
    rng = random.Random(args.seed)
    orders = {
        str(r.get("id") or r.get("query_id")): ("BASE_FIRST" if rng.random() < 0.5 else "PPQ_FIRST")
        for r in rows
    }
    (out / "condition_order.json").write_text(
        json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[4/4] Running {len(rows)} paired cases")
    for idx, case in enumerate(rows, 1):
        qid = str(case.get("id") or case.get("query_id"))
        order = orders[qid]
        steps = ["BASE", "PPQ"] if order == "BASE_FIRST" else ["PPQ", "BASE"]
        for system in steps:
            if (
                system == "BASE"
                and args.only_system in {"base", "both"}
                and qid not in completed_base
            ):
                rec = run_base(base, case)
                rec["condition_order"] = order
                append_jsonl(base_path, rec)
                completed_base.add(qid) if rec.get("status") == "PASS" else None
                print(
                    f"{idx:03d}/{len(rows)} {qid} BASE {rec.get('status')} {rec.get('latency_ms')} ms"
                )
            elif (
                system == "PPQ" and args.only_system in {"ppq", "both"} and qid not in completed_ppq
            ):
                rec = run_ppq(runtime, case)
                rec["condition_order"] = order
                append_jsonl(ppq_path, rec)
                completed_ppq.add(qid) if rec.get("status") == "PASS" else None
                print(
                    f"{idx:03d}/{len(rows)} {qid} PPQ  {rec.get('status')} {rec.get('latency_ms')} ms"
                )

    meta["completed"] = {
        "base_pass": len(load_completed(base_path)) if base_path.exists() else 0,
        "ppq_pass": len(load_completed(ppq_path)) if ppq_path.exists() else 0,
    }
    write_json(meta_path, meta)
    print("\nDone.")
    print("Base:", base_path)
    print("PPQ :", ppq_path)
    print("Next: python -m scripts.evaluation.summarize_latency")


if __name__ == "__main__":
    main()
