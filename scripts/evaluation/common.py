from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

FINAL_BENCHMARK_SHA256 = "9a9af819aa020ee71f4ed1b62151972315dda67e1a63ff3ecc9e48211308ffc8"
GEMMA_MODEL_ID = "google/gemma-4-E4B-it"
GEMMA_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
BGE_MODEL_ID = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8"
    )


def locate_final_benchmark(root: Path | None = None) -> tuple[Path, Path | None]:
    root = root or project_root()
    benchmark = root / "data" / "benchmark" / "benchmark.jsonl"
    manifest = root / "data" / "benchmark" / "manifest.json"
    if not benchmark.is_file():
        raise FileNotFoundError(f"Benchmark not found: {benchmark}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest}")
    return benchmark, manifest


def validate_final_benchmark(
    path: Path, manifest_path: Path | None = None, strict_hash: bool = True
) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 200:
        raise ValueError(f"Final benchmark must contain exactly 200 rows; found {len(rows)}")
    ids = [str(r.get("id") or r.get("query_id") or "") for r in rows]
    if any(not x for x in ids):
        raise ValueError("Every final benchmark row must have id/query_id")
    if len(set(ids)) != 200:
        raise ValueError("Final benchmark ids are not unique")
    digest = sha256_file(path)
    manifest: dict[str, Any] | None = None
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_n = int(manifest.get("n", 200))
        if expected_n != len(rows):
            raise ValueError(f"Manifest n={expected_n} but benchmark has {len(rows)} rows")
        expected_hash = str(manifest.get("sha256") or "")
        if expected_hash and expected_hash != digest:
            raise ValueError(f"Benchmark hash mismatch: manifest={expected_hash}, actual={digest}")
    elif strict_hash and digest != FINAL_BENCHMARK_SHA256:
        raise ValueError(
            "Final benchmark SHA256 does not match the frozen thesis benchmark. "
            f"Expected {FINAL_BENCHMARK_SHA256}, got {digest}."
        )
    return {
        "path": str(path),
        "sha256": digest,
        "n": len(rows),
        "manifest": manifest,
        "rows": rows,
    }


def cuda_sync() -> None:
    """Synchronize every visible CUDA device.

    The final PPQ runtime shards Gemma across two T4 GPUs, so synchronizing
    only the current/default device can under-report wall-clock latency.
    """
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
    except Exception:
        pass


def reset_peak_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
    except Exception:
        pass


def peak_cuda_memory_mb() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                out[f"cuda:{i}"] = round(torch.cuda.max_memory_allocated(i) / (1024**2), 3)
    except Exception:
        pass
    return out


def timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    cuda_sync()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    cuda_sync()
    return result, (time.perf_counter() - t0) * 1000.0


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    if _depth > 14:
        return repr(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj):
        return {k: to_jsonable(v, _depth + 1) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x, _depth + 1) for x in obj]
    for method in ("as_dict", "to_dict", "model_dump", "dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                val = fn()
                if val is not obj:
                    return to_jsonable(val, _depth + 1)
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        safe: dict[str, Any] = {}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            try:
                safe[k] = to_jsonable(v, _depth + 1)
            except Exception:
                safe[k] = repr(v)
        if safe:
            return safe
    return repr(obj)


def _walk_values(obj: Any, max_depth: int = 8) -> Iterable[tuple[str, Any]]:
    seen: set[int] = set()
    stack: list[tuple[str, Any, int]] = [("", obj, 0)]
    while stack:
        path, value, depth = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        yield path, value
        if depth >= max_depth:
            continue
        if isinstance(value, Mapping):
            for k, v in value.items():
                stack.append((f"{path}.{k}" if path else str(k), v, depth + 1))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value[:64]):
                stack.append((f"{path}[{i}]", v, depth + 1))
        elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes, Path)):
            try:
                items = vars(value).items()
            except Exception:
                continue
            for k, v in items:
                if (
                    k.startswith("__")
                    or inspect.ismodule(v)
                    or inspect.isfunction(v)
                    or inspect.ismethod(v)
                ):
                    continue
                stack.append((f"{path}.{k}" if path else k, v, depth + 1))


def recursive_find_key(obj: Any, aliases: Sequence[str]) -> Any:
    aliases_l = {a.lower() for a in aliases}
    for _path, value in _walk_values(obj):
        if isinstance(value, Mapping):
            for k, v in value.items():
                if str(k).lower() in aliases_l:
                    return v
        elif hasattr(value, "__dict__"):
            for k, v in vars(value).items():
                if k.lower() in aliases_l:
                    return v
    return None


def extract_ppq_summary(result: Any) -> dict[str, Any]:
    j = to_jsonable(result)
    route = recursive_find_key(j, ["route"])
    actions = recursive_find_key(j, ["actions", "action"])
    decision = recursive_find_key(j, ["response_decision", "decision", "runtime_decision"])
    family_ids = recursive_find_key(j, ["family_ids", "selected_family_ids", "resolved_family_ids"])
    family_id = recursive_find_key(j, ["family_id", "selected_family_id", "resolved_family_id"])
    text = recursive_find_key(j, ["text", "response_text", "answer", "output_text"])
    generation_used = recursive_find_key(j, ["generation_used", "model_generation_used"])
    generation_valid = recursive_find_key(j, ["generation_valid", "model_generation_valid"])
    reason = recursive_find_key(j, ["reason"])

    if actions is None:
        actions_out: list[str] = []
    elif isinstance(actions, (list, tuple, set)):
        actions_out = [str(getattr(x, "value", x)) for x in actions]
    else:
        actions_out = [str(getattr(actions, "value", actions))]

    fams: list[str] = []
    if isinstance(family_ids, (list, tuple, set)):
        fams.extend(str(x) for x in family_ids if x)
    elif family_ids:
        fams.append(str(family_ids))
    if family_id and str(family_id) not in fams:
        fams.append(str(family_id))

    return {
        "route": None if route is None else str(getattr(route, "value", route)),
        "actions": actions_out,
        "decision": None if decision is None else str(getattr(decision, "value", decision)),
        "family_ids": fams,
        "text": None if text is None else str(text),
        "generation_used": generation_used,
        "generation_valid": generation_valid,
        "reason": reason,
        "raw": j,
    }


def percentile(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def distribution_summary(xs: Sequence[float]) -> dict[str, float | int | None]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p75": percentile(vals, 0.75),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "min": min(vals),
        "max": max(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.median,
    iterations: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int | None]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not vals:
        return {"n": 0, "estimate": None, "low": None, "high": None}
    rng = random.Random(seed)
    n = len(vals)
    estimates = []
    for _ in range(iterations):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        estimates.append(float(statistic(sample)))
    estimates.sort()
    low = percentile(estimates, alpha / 2)
    high = percentile(estimates, 1 - alpha / 2)
    return {"n": n, "estimate": float(statistic(vals)), "low": low, "high": high}


def environment_metadata(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    data: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "project_root": str(root),
        "models": {
            "gemma_model_id": GEMMA_MODEL_ID,
            "gemma_revision": GEMMA_REVISION,
            "bge_model_id": BGE_MODEL_ID,
            "bge_revision": BGE_REVISION,
        },
        "packages": {},
        "cuda": {},
        "git": {},
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        },
        "nvidia_smi": None,
    }
    for pkg in [
        "torch",
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "lmformatenforcer",
        "accelerate",
        "numpy",
    ]:
        try:
            mod = importlib.import_module(pkg)
            data["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            data["packages"][pkg] = f"unavailable:{type(exc).__name__}"
    try:
        import torch

        data["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_mb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**2), 2
                    ),
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        data["cuda"] = {"error": repr(exc)}
    try:
        smi = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()
        )
        data["nvidia_smi"] = [line.strip() for line in smi if line.strip()]
    except Exception:
        data["nvidia_smi"] = None

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = (
            subprocess.call(["git", "diff", "--quiet"], cwd=root, stderr=subprocess.DEVNULL) != 0
        )
        data["git"] = {"commit": commit, "dirty": dirty}
    except Exception:
        data["git"] = {"commit": None, "dirty": None}
    return data


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_factory(fn: Callable[..., Any], args: argparse.Namespace, root: Path) -> Any:
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    for name, p in sig.parameters.items():
        lname = name.lower()
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if lname in {"args", "namespace", "cli_args"}:
            value = args
        elif lname in {"device_map", "devicemap"}:
            value = getattr(args, "device_map", "balanced")
        elif lname in {"attention", "attn", "attn_implementation", "attention_implementation"}:
            value = getattr(args, "attention", "sdpa")
        elif lname in {"embedding_device", "embed_device"}:
            value = getattr(args, "embedding_device", "cuda:0")
        elif lname in {"embedding_batch_size", "embed_batch_size"}:
            value = getattr(args, "embedding_batch_size", 32)
        elif lname in {"root", "project_root", "base_dir"}:
            value = root
        elif lname in {"max_new_tokens", "max_tokens"}:
            value = getattr(args, "max_new_tokens", 256)
        elif lname in {"model_id", "model_name"}:
            value = GEMMA_MODEL_ID
        elif lname in {"revision", "model_revision"}:
            value = GEMMA_REVISION
        elif p.default is not inspect._empty:
            continue
        else:
            raise TypeError(f"Unsupported required factory parameter: {name}")
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            kwargs[name] = value
    return fn(*positional, **kwargs)


def discover_ppq_runtime(
    args: argparse.Namespace, root: Path | None = None
) -> tuple[Any, dict[str, Any]]:
    root = root or project_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))

    factory = getattr(args, "factory", None)
    if not factory or ":" not in factory:
        raise ValueError("--factory must be MODULE:FUNCTION or /path/to/file.py:FUNCTION")
    module_reference, function_name = factory.rsplit(":", 1)
    if module_reference.endswith(".py") or "/" in module_reference:
        module = _load_module_from_file("_ppq_explicit_factory", Path(module_reference))
    else:
        module = importlib.import_module(module_reference)
    function = getattr(module, function_name)
    runtime = _call_factory(function, args, root)
    if not callable(getattr(runtime, "handle", None)):
        raise TypeError(f"Factory {factory} returned an object without .handle()")
    return runtime, {"factory": factory, "mode": "explicit"}


def call_runtime_handle(runtime: Any, prompt: str, query_id: str | None = None) -> Any:
    handle = getattr(runtime, "handle")
    attempts = [
        lambda: handle(query=prompt, query_id=query_id),
        lambda: handle(query=prompt),
        lambda: handle(prompt, query_id=query_id),
        lambda: handle(prompt),
    ]
    last: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last = exc
            continue
    assert last is not None
    raise last


def find_hf_generation_stack(runtime: Any) -> tuple[Any, Any, dict[str, Any]]:
    model_candidates: list[tuple[int, str, Any]] = []
    proc_candidates: list[tuple[int, str, Any]] = []
    for path, value in _walk_values(runtime, max_depth=10):
        if callable(getattr(value, "generate", None)):
            score = 0
            cls = type(value).__name__.lower()
            p = path.lower()
            if "model" in p:
                score += 30
            if "gemma" in cls or "gemma" in p:
                score += 40
            if hasattr(value, "parameters"):
                score += 30
            model_candidates.append((score, path, value))
        if callable(getattr(value, "apply_chat_template", None)):
            score = 0
            p = path.lower()
            cls = type(value).__name__.lower()
            if "processor" in p:
                score += 40
            if "tokenizer" in p:
                score += 30
            if "processor" in cls:
                score += 20
            proc_candidates.append((score, path, value))
    if not model_candidates or not proc_candidates:
        raise RuntimeError(
            "Could not locate the loaded Hugging Face Gemma model/processor inside PPQ runtime. "
            "Use --base-separate-model only if necessary (requires extra VRAM)."
        )
    model_score, model_path, model = sorted(model_candidates, key=lambda x: -x[0])[0]
    proc_score, proc_path, processor = sorted(proc_candidates, key=lambda x: -x[0])[0]
    return (
        model,
        processor,
        {
            "model_path": model_path,
            "processor_path": proc_path,
            "model_score": model_score,
            "processor_score": proc_score,
        },
    )


def _first_model_device(model: Any):
    try:
        for p in model.parameters():
            if p.device.type != "meta":
                return p.device
    except Exception:
        pass
    return getattr(model, "device", "cuda:0")


class DirectGemmaAdapter:
    """Direct same-Gemma no-corpus/no-RAG baseline using the PPQ runtime's loaded HF model."""

    def __init__(self, model: Any, processor: Any, max_new_tokens: int = 256):
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)

    def _template(self, prompt: str):
        variants = [
            [{"role": "user", "content": prompt}],
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        ]
        last: Exception | None = None
        for messages in variants:
            for kw in [
                {
                    "add_generation_prompt": True,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                    "enable_thinking": False,
                },
                {
                    "add_generation_prompt": True,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                },
            ]:
                try:
                    return self.processor.apply_chat_template(messages, **kw)
                except Exception as exc:
                    last = exc
        assert last is not None
        raise last

    def generate(self, prompt: str) -> dict[str, Any]:
        import torch

        inputs = self._template(prompt)
        if not isinstance(inputs, Mapping):
            inputs = {"input_ids": inputs}
        device = _first_model_device(self.model)
        moved = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        input_ids = moved.get("input_ids")
        input_tokens = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else None
        gen_kwargs = {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
        }
        tok = getattr(self.processor, "tokenizer", self.processor)
        for name in ["eos_token_id", "pad_token_id"]:
            val = getattr(tok, name, None)
            if val is not None:
                gen_kwargs[name] = val
        with torch.inference_mode():
            output_ids = self.model.generate(**moved, **gen_kwargs)
        if input_tokens is not None:
            new_ids = output_ids[:, input_tokens:]
        else:
            new_ids = output_ids
        decoder = getattr(self.processor, "batch_decode", None) or getattr(tok, "batch_decode")
        text = decoder(new_ids, skip_special_tokens=True)[0].strip()
        output_tokens = int(new_ids.shape[-1]) if hasattr(new_ids, "shape") else None
        return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens}


class RuntimeStageProfiler:
    """Diagnostic monkey-patch profiler. Use in a separate diagnostic run, not headline latency."""

    RULES = [
        ("binder", re.compile(r"binder", re.I), {"bind"}),
        ("planner", re.compile(r"planner", re.I), {"plan"}),
        (
            "bge_encode",
            re.compile(r"embedding|sentence.*transformer|bge|provider", re.I),
            {"encode_queries", "encode_query"},
        ),
        ("semantic_verifier", re.compile(r"verifier|applicability|gate", re.I), {"verify"}),
        (
            "surface_retrieval",
            re.compile(r"retriev|surface", re.I),
            {"rank", "search", "correct_quote"},
        ),
        ("composer", re.compile(r"composer", re.I), {"compose"}),
        ("corpus_resolve", re.compile(r"corpus", re.I), {"resolve", "resolve_canonical_family"}),
        ("service", re.compile(r"service", re.I), {"process"}),
    ]

    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.records: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)
        self._patched: list[tuple[Any, str, Any]] = []
        self._patch_graph()

    def reset(self) -> None:
        self.records.clear()
        self.counts.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "timing_ms": {k: round(v, 3) for k, v in sorted(self.records.items())},
            "counts": {k: int(v) for k, v in sorted(self.counts.items())},
        }

    def restore(self) -> None:
        for obj, name, original in reversed(self._patched):
            try:
                setattr(obj, name, original)
            except Exception:
                pass
        self._patched.clear()

    def _patch_graph(self) -> None:
        patched_keys: set[tuple[int, str]] = set()
        for path, obj in _walk_values(self.runtime, max_depth=9):
            identity = f"{path} {type(obj).__name__} {type(obj).__module__}"
            for stage, pat, methods in self.RULES:
                if not pat.search(identity):
                    continue
                for method_name in methods:
                    key = (id(obj), method_name)
                    if key in patched_keys:
                        continue
                    original = getattr(obj, method_name, None)
                    if not callable(original):
                        continue

                    def make_wrapper(bound_original, stage_name):
                        def wrapper(*a, **kw):
                            cuda_sync()
                            t0 = time.perf_counter()
                            try:
                                return bound_original(*a, **kw)
                            finally:
                                cuda_sync()
                                self.records[stage_name] += (time.perf_counter() - t0) * 1000.0
                                self.counts[stage_name] += 1

                        return wrapper

                    try:
                        setattr(obj, method_name, make_wrapper(original, stage))
                    except Exception:
                        continue
                    self._patched.append((obj, method_name, original))
                    patched_keys.add(key)


def safe_status_record(
    case: Mapping[str, Any], system: str, exc: Exception, latency_ms: float | None = None
) -> dict[str, Any]:
    return {
        "query_id": str(case.get("id") or case.get("query_id")),
        "category": case.get("category"),
        "prompt": case.get("prompt") or case.get("query_text"),
        "system": system,
        "status": "ERROR",
        "latency_ms": latency_ms,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
