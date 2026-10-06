#!/usr/bin/env python3
"""Load PPQ once and accept multiple prompts from a terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppq.bootstrap import build_runtime

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive PPQ console")
    parser.add_argument("--embedding-device", choices=["cuda:0", "cpu"], default="cuda:0")
    args = parser.parse_args()

    assistant = build_runtime(ROOT, args.embedding_device, log=print)
    print("\n✅ PPQ READY")
    print("پرامپت فارسی وارد کن. برای خروج بنویس: /exit")

    counter = 0
    while True:
        try:
            query = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in {"/exit", "exit", "/quit", "quit"}:
            break

        counter += 1
        output = assistant.handle(query, query_id=f"interactive-{counter:04d}")
        print(
            json.dumps(
                {
                    "decision": output.decision.value,
                    "text": output.text,
                    "generation_used": output.generation_used,
                    "generation_valid": output.generation_valid,
                    "reason": output.reason,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
