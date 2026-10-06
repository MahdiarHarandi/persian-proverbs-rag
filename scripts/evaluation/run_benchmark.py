#!/usr/bin/env python3
"""Run the sealed PPQ benchmark against PPQ and base Gemma.

The baseline reuses the same loaded Gemma weights as PPQ, avoiding a second
GPU copy.  The benchmark file is hash-checked against its pre-inference seal
before execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from ppq.bootstrap import build_runtime

ROOT = Path(__file__).resolve().parents[2]

BENCHMARK = ROOT / "data/benchmark/benchmark.jsonl"
MANIFEST = ROOT / "data/benchmark/manifest.json"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def verify_seal() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256(BENCHMARK.read_bytes()).hexdigest()
    if manifest.get("status") != "SEALED_PRE_INFERENCE":
        raise RuntimeError("benchmark manifest is not sealed pre-inference")
    if digest != manifest.get("sha256"):
        raise RuntimeError("benchmark.jsonl does not match its sealed SHA-256")
    return manifest


def base_gemma_response(assistant, prompt: str, max_new_tokens: int = 256) -> tuple[str, float]:
    """Generate a plain Gemma answer from only the original user prompt."""
    import torch

    generator = assistant.composer.generator
    processor = generator.processor
    model = generator.model
    messages = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
    elapsed = time.perf_counter() - start
    text = processor.tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()
    del inputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, elapsed


def plan_dict(runtime) -> dict | None:
    return None if runtime.plan is None else runtime.plan.as_dict()


def score_ppq(case: dict, output) -> dict:
    runtime = output.runtime
    plan = runtime.plan
    actual_actions = [a.value for a in plan.actions] if plan else []
    actual_families = [m.family_id for m in runtime.matches]

    decision_match = output.decision.value == case["expected_response_decision"]
    runtime_match = runtime.decision.value == case["expected_runtime_decision"]
    route = plan.route.value if plan else None
    route_match = route == case["expected_route"]

    allowed_actions = case.get("expected_actions_any", [])
    action_match = True if not allowed_actions else bool(set(actual_actions) & set(allowed_actions))

    targets = case.get("target_family_ids", [])
    family_match = None if not targets else bool(set(actual_families) & set(targets))

    parts = [decision_match, runtime_match, route_match, action_match]
    if family_match is not None:
        parts.append(family_match)
    e2e = all(parts)
    return {
        "decision_match": decision_match,
        "runtime_match": runtime_match,
        "route_match": route_match,
        "action_match": action_match,
        "family_match": family_match,
        "automatic_e2e": e2e,
    }


def ratio(items: list[bool]) -> dict:
    if not items:
        return {"correct": 0, "total": 0, "accuracy": None}
    return {"correct": sum(items), "total": len(items), "accuracy": sum(items) / len(items)}


def metric_block(rows: list[dict]) -> dict:
    return {
        "decision": ratio([r["auto_score"]["decision_match"] for r in rows]),
        "runtime": ratio([r["auto_score"]["runtime_match"] for r in rows]),
        "route": ratio([r["auto_score"]["route_match"] for r in rows]),
        "action": ratio([r["auto_score"]["action_match"] for r in rows]),
        "family": ratio(
            [
                r["auto_score"]["family_match"]
                for r in rows
                if r["auto_score"]["family_match"] is not None
            ]
        ),
        "automatic_e2e": ratio([r["auto_score"]["automatic_e2e"] for r in rows]),
    }


def write_blind_review(rows: list[dict], out_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    review_path = out_dir / "blind_review.csv"
    key_path = out_dir / "blind_key.jsonl"
    fields = [
        "id",
        "category",
        "prompt",
        "response_a",
        "response_b",
        "task_fulfilled_a",
        "authentic_a",
        "appropriate_a",
        "fabricated_a",
        "task_fulfilled_b",
        "authentic_b",
        "appropriate_b",
        "fabricated_b",
        "better_response",
        "notes",
    ]
    key_rows = []
    with review_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["comparison_scope"] != "ppq_vs_base":
                continue
            if rng.random() < 0.5:
                a_name, a_text = "PPQ", row.get("ppq_text") or ""
                b_name, b_text = "BASE", row.get("base_text") or ""
            else:
                a_name, a_text = "BASE", row.get("base_text") or ""
                b_name, b_text = "PPQ", row.get("ppq_text") or ""
            writer.writerow(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "response_a": a_text,
                    "response_b": b_text,
                    "task_fulfilled_a": "",
                    "authentic_a": "",
                    "appropriate_a": "",
                    "fabricated_a": "",
                    "task_fulfilled_b": "",
                    "authentic_b": "",
                    "appropriate_b": "",
                    "fabricated_b": "",
                    "better_response": "",
                    "notes": "",
                }
            )
            key_rows.append({"id": row["id"], "A": a_name, "B": b_name})
    key_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in key_rows) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PPQ benchmark vs base Gemma")
    parser.add_argument("--embedding-device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N prompts")
    parser.add_argument("--ppq-only", action="store_true", help="Skip base Gemma generation")
    parser.add_argument("--seed", type=int, default=42, help="Blind A/B randomization seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation-output" / "benchmark",
        help="Write rerun artifacts outside the frozen results directory.",
    )
    args = parser.parse_args()

    manifest = verify_seal()
    cases = load_jsonl(BENCHMARK)
    if len(cases) != manifest["n"]:
        raise RuntimeError("benchmark row count differs from manifest")
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "outputs.jsonl"
    summary_path = out_dir / "summary.json"

    assistant = build_runtime(ROOT, args.embedding_device, log=print)
    rows = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['category']}")
        start = time.perf_counter()
        ppq = assistant.handle(case["prompt"], query_id=case["id"])
        ppq_elapsed = time.perf_counter() - start
        scoring = score_ppq(case, ppq)

        base_text = None
        base_elapsed = None
        base_error = None
        if not args.ppq_only and case["comparison_scope"] == "ppq_vs_base":
            try:
                base_text, base_elapsed = base_gemma_response(assistant, case["prompt"])
            except Exception as exc:
                base_error = f"{type(exc).__name__}: {exc}"

        runtime = ppq.runtime
        row = {
            **case,
            "ppq_decision": ppq.decision.value,
            "ppq_runtime_decision": runtime.decision.value,
            "ppq_plan": plan_dict(runtime),
            "ppq_match_families": [m.family_id for m in runtime.matches],
            "ppq_match_texts": [m.text for m in runtime.matches],
            "ppq_text": ppq.text,
            "ppq_reason": ppq.reason,
            "ppq_runtime_reason": runtime.reason,
            "ppq_planner_error": runtime.planner_error,
            "ppq_elapsed_seconds": round(ppq_elapsed, 4),
            "base_text": base_text,
            "base_elapsed_seconds": None if base_elapsed is None else round(base_elapsed, 4),
            "base_error": base_error,
            "auto_score": scoring,
        }
        rows.append(row)
        with outputs_path.open("a" if index > 1 else "w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    summary = {
        "benchmark": manifest["benchmark_name"],
        "benchmark_sha256": manifest["sha256"],
        "n": len(rows),
        "categories": dict(sorted(Counter(r["category"] for r in rows).items())),
        "ppq_automatic": metric_block(rows),
        "ppq_by_category": {k: metric_block(v) for k, v in sorted(by_category.items())},
        "base_comparison": "blind A/B review" if not args.ppq_only else "not run",
        "review_fields": [
            "task_fulfilled",
            "authentic",
            "appropriate",
            "fabricated",
            "better_response",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not args.ppq_only:
        write_blind_review(rows, out_dir, args.seed)

    print("\nDone")
    print(f"Outputs: {outputs_path}")
    print(f"Summary: {summary_path}")
    if not args.ppq_only:
        print(f"Blind review: {out_dir / 'blind_review.csv'}")
        print(f"Secret key: {out_dir / 'blind_key.jsonl'}")
        print("Do not open the key until the blind review is finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
