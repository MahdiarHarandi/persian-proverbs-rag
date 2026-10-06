import hashlib
import json
from pathlib import Path

import numpy as np

from ppq.corpus import load_corpus
from ppq.surface_restore import SurfaceRestore

ROOT = Path(__file__).resolve().parents[1]


def test_corpus():
    corpus = load_corpus(
        ROOT / "data/corpus/expressions.jsonl",
        ROOT / "data/corpus/sources.jsonl",
    )
    assert corpus.record_count == 1820
    assert corpus.variant_count == 2067
    assert corpus.without_gloss_count == 0


def test_runtime_assets():
    corpus = load_corpus(
        ROOT / "data/corpus/expressions.jsonl",
        ROOT / "data/corpus/sources.jsonl",
    )
    assert SurfaceRestore(corpus) is not None
    emb = ROOT / "artifacts/family_embeddings.npy"
    meta = ROOT / "artifacts/family_embeddings.metadata.json"
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    matrix = np.load(emb, allow_pickle=False, mmap_mode="r")
    assert matrix.shape == (corpus.record_count, metadata["dimension"])
    assert matrix.dtype == np.float32
    assert metadata["family_ids"] == [row.family_id for row in corpus.expressions]
    assert hashlib.sha256(emb.read_bytes()).hexdigest() == metadata["embeddings_sha256"]


def test_benchmark_smoke():
    rows = [
        json.loads(x)
        for x in (ROOT / "data/benchmark/benchmark.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    assert len(rows) == 200
    assert len({r["id"] for r in rows}) == 200


def test_every_variant_references_a_known_source():
    expressions = [
        json.loads(line)
        for line in (ROOT / "data/corpus/expressions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    sources = {
        row["source_id"]
        for row in (
            json.loads(line)
            for line in (ROOT / "data/corpus/sources.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    assert sources
    assert all(
        variant["source_id"] in sources for row in expressions for variant in row["variants"]
    )


def test_delivery_has_no_local_runtime_artifacts():
    assert not (ROOT / ".gradio").exists()
    assert not any(path.suffix == ".pem" for path in ROOT.rglob("*"))
