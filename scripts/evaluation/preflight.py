from __future__ import annotations

import json
from pathlib import Path

from .common import environment_metadata, locate_final_benchmark, validate_final_benchmark

ROOT = Path(__file__).resolve().parents[2]


def main():
    benchmark, manifest = locate_final_benchmark(ROOT)
    val = validate_final_benchmark(benchmark, manifest)
    checks = {
        "benchmark": {
            "status": "PASS",
            "path": str(benchmark),
            "n": val["n"],
            "sha256": val["sha256"],
        },
        "runtime_package": (ROOT / "src" / "ppq").is_dir(),
        "corpus": (ROOT / "data" / "corpus" / "expressions.jsonl").is_file(),
        "embeddings": (ROOT / "artifacts" / "family_embeddings.npy").is_file(),
        "web_ui": (ROOT / "scripts" / "ui.py").is_file(),
    }
    meta = environment_metadata(ROOT)
    print(json.dumps({"checks": checks, "environment": meta}, ensure_ascii=False, indent=2))
    if not all(value is True for key, value in checks.items() if key != "benchmark"):
        raise SystemExit("FAIL: run this from the project root with src/ppq present.")
    if not meta.get("cuda", {}).get("available"):
        print(
            "WARNING: CUDA is not available. Final real-model benchmark should run on the same GPU condition as the thesis baseline."
        )
    print("\nFinal benchmark preflight passed.")


if __name__ == "__main__":
    main()
