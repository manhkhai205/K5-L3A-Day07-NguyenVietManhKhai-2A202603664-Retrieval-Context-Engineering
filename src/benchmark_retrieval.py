"""Benchmark multiple retrieval methods and context configurations on Day 7 dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "starter"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline import hit as is_evidence_hit, validate  # noqa: E402
from retrievers import (  # noqa: E402
    BM25Retriever,
    DenseQdrantRetriever,
    HybridRRFRetriever,
    QueryEmbeddingService,
    TfidfRetriever,
    build_neighbor_lookup,
    infer_filters_from_query,
    pack_context_smart,
)


CONFIG_SPECS: list[dict[str, Any]] = [
    {
        "name": "baseline_tfidf_b800",
        "label": "Baseline (TF-IDF, top_k=3, no filter, budget=800)",
        "method": "tfidf",
        "top_k": 3,
        "filter_mode": "none",
        "budget": 800,
        "neighbor_expand": False,
        "hypothesis": "Baseline lexical TF-IDF từ starter với context budget 800 ký tự mặc định.",
    },
    {
        "name": "method2_bm25_b800",
        "label": "Phương pháp 2 (BM25 + domain codes + provided filter, budget=800)",
        "method": "bm25",
        "top_k": 3,
        "filter_mode": "provided",
        "budget": 800,
        "neighbor_expand": False,
        "hypothesis": "BM25 chuẩn hóa độ dài chunk và giữ mã hành động NIST (MS-2.5, MG-3.1) kết hợp filter được cung cấp.",
    },
    {
        "name": "method3_dense_qdrant_b800",
        "label": "Phương pháp 3 (Dense Qdrant Day 6 + provided filter, budget=800)",
        "method": "dense_qdrant",
        "top_k": 3,
        "filter_mode": "provided",
        "budget": 800,
        "neighbor_expand": False,
        "hypothesis": "Dense retrieval dùng gemini-embedding-2 (768d) trên Qdrant Day 6 giúp bắt ngữ nghĩa câu hỏi diễn đạt lại tốt hơn lexical.",
    },
    {
        "name": "method4_hybrid_rrf_b800",
        "label": "Phương pháp 4 (Hybrid Dense+BM25 RRF + provided filter, budget=800)",
        "method": "hybrid_rrf",
        "top_k": 3,
        "filter_mode": "provided",
        "budget": 800,
        "neighbor_expand": False,
        "hypothesis": "Kết hợp Dense Qdrant và BM25 bằng Reciprocal Rank Fusion (RRF, k=60) để vừa bắt ngữ nghĩa vừa bắt từ khóa/mã mục hiếm.",
    },
    {
        "name": "improve1_dense_qdrant_b2400",
        "label": "Cải thiện 1 (Dense Qdrant + provided filter + Context Budget 2400)",
        "method": "dense_qdrant",
        "top_k": 3,
        "filter_mode": "provided",
        "budget": 2400,
        "neighbor_expand": False,
        "hypothesis": "Tăng context budget từ 800 lên 2400 ký tự để khắc phục lỗi chunk Chiến lược B (dài 650-900 ký tự + header) bị loại khỏi context.",
    },
    {
        "name": "improve2_hybrid_rrf_b2400",
        "label": "Cải thiện 2 (Hybrid RRF + provided filter + Context Budget 2400)",
        "method": "hybrid_rrf",
        "top_k": 3,
        "filter_mode": "provided",
        "budget": 2400,
        "neighbor_expand": False,
        "hypothesis": "Kết hợp Hybrid RRF với Context Budget 2400 để tối ưu đồng thời Hit@1, Evidence Hit@3 và Context Evidence Rate.",
    },
    {
        "name": "improve3_hybrid_rrf_inferred_b3200",
        "label": "Cải thiện 3 (Hybrid RRF + Inferred Filter + Neighbor Expand, budget=3200)",
        "method": "hybrid_rrf",
        "top_k": 3,
        "filter_mode": "inferred",
        "budget": 3200,
        "neighbor_expand": True,
        "hypothesis": "Suy luận thêm filter year/section tất định từ câu hỏi (ví dụ câu q19 thuộc 600-1) và ghép thêm chunk lân cận cùng trang sau top-3 với budget 3200 ký tự.",
    },
]


def resolve_filter(q: dict[str, Any], filter_mode: str) -> dict[str, Any] | None:
    if filter_mode == "none":
        return None
    if filter_mode == "provided":
        return q.get("filter")
    if filter_mode == "inferred":
        return infer_filters_from_query(q["question"], q.get("filter"))
    raise ValueError(f"Unsupported filter_mode: {filter_mode!r}")


def run_config(
    spec: dict[str, Any],
    retriever: Any,
    questions: list[dict[str, Any]],
    neighbor_lookup: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    top_k = int(spec["top_k"])
    budget = int(spec["budget"])
    filter_mode = spec["filter_mode"]
    use_neighbors = bool(spec.get("neighbor_expand", False))

    for q in questions:
        applied_filter = resolve_filter(q, filter_mode)
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=applied_filter, top_k=top_k)
        context, context_ids = pack_context_smart(
            results,
            budget=budget,
            rerank_by_score=False,
            neighbor_lookup=neighbor_lookup if use_neighbors else None,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        marked = [{**c, "hit": is_evidence_hit(c, q["gold"])} for c in results]
        top3_marked = marked[:3]
        rr = 0.0
        for rank_idx, item in enumerate(top3_marked, 1):
            if item["hit"]:
                rr = 1.0 / rank_idx
                break

        # Check whether any chunk packed in final context (including neighbor-expanded chunks) contains gold evidence
        packed_chunks_with_hit = any(
            is_evidence_hit(c, q["gold"]) and c["id"] in context_ids for c in results
        )
        if not packed_chunks_with_hit and use_neighbors:
            # Also check expanded neighbor chunks in context_ids
            for cid in context_ids:
                for ch in neighbor_lookup.values():
                    if ch["id"] == cid and is_evidence_hit(ch, q["gold"]):
                        packed_chunks_with_hit = True
                        break

        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "config_name": spec["name"],
                "strategy": spec["method"],
                "top_k": top_k,
                "filter_mode": filter_mode,
                "applied_filter": applied_filter,
                "budget_chars": budget,
                "neighbor_expand": use_neighbors,
                "gold": q["gold"],
                "results": marked,
                "top3_results": top3_marked,
                "context": context,
                "context_ids": context_ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 3),
                "hit1": bool(top3_marked and top3_marked[0]["hit"]),
                "hit3": any(c["hit"] for c in top3_marked),
                "mrr3": round(rr, 4),
                "context_hit": packed_chunks_with_hit,
            }
        )

    n = len(traces)
    hit1_count = sum(t["hit1"] for t in traces)
    hit3_count = sum(t["hit3"] for t in traces)
    ctx_count = sum(t["context_hit"] for t in traces)
    summary = {
        "config_name": spec["name"],
        "label": spec["label"],
        "strategy": spec["method"],
        "split": questions[0]["split"] if questions else "",
        "top_k": top_k,
        "filter_mode": filter_mode,
        "budget_chars": budget,
        "neighbor_expand": use_neighbors,
        "hypothesis": spec["hypothesis"],
        "n": n,
        "hit1": hit1_count,
        "hit1_rate": round(hit1_count / n, 4) if n else 0.0,
        "hit3": hit3_count,
        "hit3_rate": round(hit3_count / n, 4) if n else 0.0,
        "mrr3": round(statistics.mean(t["mrr3"] for t in traces), 4) if n else 0.0,
        "context_hit": ctx_count,
        "context_evidence_rate": round(ctx_count / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 3) if n else 0.0,
        "total_ms": round(sum(t["elapsed_ms"] for t in traces), 2),
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Day 7 retrieval methods and configurations")
    parser.add_argument("--chunks", type=Path, default=REPO_ROOT / "data" / "chunks.json")
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "data" / "questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional subset of config names to run (defaults to all CONFIG_SPECS)",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "retrieval")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected_questions = [q for q in all_questions if q.get("split") == args.split]
    validate(chunks, selected_questions)

    neighbor_lookup = build_neighbor_lookup(chunks)
    tfidf_ret = TfidfRetriever(chunks)
    bm25_ret = BM25Retriever(chunks)
    embedder = QueryEmbeddingService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("EMBEDDING_MODEL", "gemini-embedding-2"),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
        cache_path=REPO_ROOT / "results" / "cache" / "query_embeddings.json",
    )
    dense_ret = DenseQdrantRetriever(
        embedder=embedder,
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        collection_name=os.getenv("QDRANT_COLLECTION", "day6_nist_chunks"),
        qdrant_strategy=os.getenv("QDRANT_STRATEGY", "B"),
    )
    hybrid_ret = HybridRRFRetriever(dense=dense_ret, lexical=bm25_ret, candidate_k=15, rrf_k=60)

    retrievers = {
        "tfidf": tfidf_ret,
        "bm25": bm25_ret,
        "dense_qdrant": dense_ret,
        "hybrid_rrf": hybrid_ret,
    }

    specs = CONFIG_SPECS
    if args.configs:
        wanted = set(args.configs)
        specs = [s for s in CONFIG_SPECS if s["name"] in wanted]

    summaries: list[dict[str, Any]] = []
    for spec in specs:
        ret = retrievers[spec["method"]]
        summary, traces = run_config(spec, ret, selected_questions, neighbor_lookup)
        cfg_dir = args.output_dir / f"{spec['name']}_{args.split}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (cfg_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(
            f"[{spec['name']} | {args.split}] "
            f"Hit@1={summary['hit1']}/{summary['n']} ({summary['hit1_rate']:.2%}) | "
            f"Hit@3={summary['hit3']}/{summary['n']} ({summary['hit3_rate']:.2%}) | "
            f"ContextRate={summary['context_hit']}/{summary['n']} ({summary['context_evidence_rate']:.2%}) | "
            f"MRR@3={summary['mrr3']:.4f} | median_ms={summary['median_ms']}ms"
        )

    comparison_out = {
        "split": args.split,
        "num_questions": len(selected_questions),
        "embedding_api_calls_new": embedder.api_calls,
        "embedding_cache_hits": embedder.cache_hits,
        "summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comp_file = args.output_dir / f"{args.split}_comparison.json"
    comp_file.write_text(json.dumps(comparison_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
