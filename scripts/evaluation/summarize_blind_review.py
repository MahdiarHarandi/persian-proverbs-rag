#!/usr/bin/env python3
"""Evaluate a completed blind A/B review for the benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "evaluation-output" / "benchmark"

YES = {"yes", "y", "1", "true", "pass", "بله"}
NO = {"no", "n", "0", "false", "fail", "خیر"}
NA = {"", "na", "n/a", "-", "none"}


def parse_bool(v: str):
    s = (v or "").strip().lower()
    if s in YES:
        return True
    if s in NO:
        return False
    if s in NA:
        return None
    raise ValueError(f"invalid review value: {v!r}; use YES/NO/NA")


def rate(vals):
    vals = [v for v in vals if v is not None]
    return {
        "positive": sum(vals),
        "total": len(vals),
        "rate": None if not vals else sum(vals) / len(vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, default=DEFAULT_DIR / "blind_review.csv")
    ap.add_argument("--key", type=Path, default=DEFAULT_DIR / "blind_key.jsonl")
    ap.add_argument("--output", type=Path, default=DEFAULT_DIR / "blind_evaluation.json")
    args = ap.parse_args()

    keys = {
        r["id"]: r
        for r in (
            json.loads(x) for x in args.key.read_text(encoding="utf-8").splitlines() if x.strip()
        )
    }
    rows = list(csv.DictReader(args.review.open(encoding="utf-8-sig", newline="")))
    per = defaultdict(lambda: defaultdict(list))
    wins = {"PPQ": 0, "BASE": 0, "TIE": 0}
    n = 0
    for r in rows:
        if r["id"] not in keys:
            raise ValueError(f"missing key for {r['id']}")
        k = keys[r["id"]]
        n += 1
        for side in ("a", "b"):
            system = k[side.upper()]
            per[system]["task_fulfilled"].append(parse_bool(r[f"task_fulfilled_{side}"]))
            per[system]["authentic"].append(parse_bool(r[f"authentic_{side}"]))
            per[system]["appropriate"].append(parse_bool(r[f"appropriate_{side}"]))
            per[system]["fabricated"].append(parse_bool(r[f"fabricated_{side}"]))
        better = (r.get("better_response") or "").strip().upper()
        if better in {"A", "B"}:
            wins[k[better]] += 1
        elif better in {"TIE", "T", "="}:
            wins["TIE"] += 1
        elif better:
            raise ValueError(f"invalid better_response in {r['id']}: {better!r}")

    metrics = {}
    for system, d in per.items():
        metrics[system] = {
            "task_fulfilled": rate(d["task_fulfilled"]),
            "authentic": rate(d["authentic"]),
            "appropriate": rate(d["appropriate"]),
            "fabricated": rate(d["fabricated"]),
        }
    decisive = wins["PPQ"] + wins["BASE"]
    out = {
        "n_blind_pairs": n,
        "systems": metrics,
        "pairwise": {
            **wins,
            "decisive": decisive,
            "ppq_win_rate_decisive": None if not decisive else wins["PPQ"] / decisive,
        },
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
