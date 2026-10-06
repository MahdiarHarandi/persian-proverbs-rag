from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate thesis-ready diagnostic figures from final-200 results."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "evaluation-output" / "performance",
    )
    p.add_argument(
        "--blind-evaluation",
        type=Path,
        default=ROOT / "results" / "benchmark" / "blind_evaluation.json",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    d = args.input_dir
    out = args.output_dir or d / "figures"
    paired = read_jsonl(d / "paired_cases.jsonl")

    # Paired latency scatter.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    x = [r["base_latency_ms"] / 1000 for r in paired]
    y = [r["ppq_latency_ms"] / 1000 for r in paired]
    ax.scatter(x, y, alpha=0.7)
    lim = max(x + y) if x and y else 1
    ax.plot([0, lim], [0, lim], linestyle="--")
    ax.set_xlabel("Direct Gemma latency (s)")
    ax.set_ylabel("PPQ latency (s)")
    ax.set_title("Paired end-to-end latency: PPQ vs direct Gemma")
    save(fig, out / "paired_latency_scatter.png")

    # Median slowdown by category.
    by_cat = defaultdict(list)
    for r in paired:
        by_cat[r["category"]].append(float(r["slowdown_ratio"]))
    cats = sorted(by_cat, key=lambda c: statistics.median(by_cat[c]))
    vals = [statistics.median(by_cat[c]) for c in cats]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(cats, vals)
    ax.axvline(1.0, linestyle="--")
    ax.set_xlabel("Median slowdown ratio (PPQ / Base)")
    ax.set_title("Latency cost by benchmark category")
    save(fig, out / "slowdown_by_category.png")

    # Boxplot by actual PPQ action.
    by_action = defaultdict(list)
    for r in paired:
        by_action[r.get("ppq_action") or "NONE"].append(float(r["ppq_latency_ms"]) / 1000)
    actions = sorted(by_action)
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.boxplot([by_action[a] for a in actions], tick_labels=actions, vert=True)
    ax.set_ylabel("PPQ end-to-end latency (s)")
    ax.set_title("PPQ latency by actual action")
    ax.tick_params(axis="x", rotation=35)
    save(fig, out / "ppq_latency_by_action.png")

    # Quality metrics if blind review is complete.
    blind = args.blind_evaluation
    if blind.exists():
        ev = json.loads(blind.read_text(encoding="utf-8"))
        labels = ["Task fulfillment", "FFE authenticity", "FFE appropriateness", "Fabrication"]
        ppq = ev["systems"]["PPQ"]
        base = ev["systems"]["BASE"]
        ppq_vals = [
            100 * ppq["task_fulfilled"]["rate"],
            100 * ppq["authentic"]["rate"],
            100 * ppq["appropriate"]["rate"],
            100 * ppq["fabricated"]["rate"],
        ]
        base_vals = [
            100 * base["task_fulfilled"]["rate"],
            100 * base["authentic"]["rate"],
            100 * base["appropriate"]["rate"],
            100 * base["fabricated"]["rate"],
        ]
        pos = list(range(len(labels)))
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        ax.bar([i - width / 2 for i in pos], base_vals, width, label="Direct Gemma")
        ax.bar([i + width / 2 for i in pos], ppq_vals, width, label="PPQ")
        ax.set_xticks(pos, labels, rotation=18)
        ax.set_ylabel("Percent")
        ax.set_title("Condition-blind quality comparison")
        ax.legend()
        save(fig, out / "quality_ppq_vs_base.png")

    # Internal profiler: stage median if available.
    profile = d / "ppq_stage_profile.jsonl"
    if profile.exists():
        rows = [r for r in read_jsonl(profile) if r.get("status") == "PASS"]
        stage = defaultdict(list)
        for r in rows:
            for k, v in (r.get("timing_ms") or {}).items():
                stage[k].append(float(v) / 1000)
        if stage:
            import statistics

            names = sorted(stage, key=lambda k: statistics.median(stage[k]))
            med = [statistics.median(stage[k]) for k in names]
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            ax.barh(names, med)
            ax.set_xlabel("Median cumulative time per request (s)")
            ax.set_title("Diagnostic PPQ stage timing")
            save(fig, out / "ppq_stage_timing.png")
    print("Saved figures to", out)


if __name__ == "__main__":
    main()
