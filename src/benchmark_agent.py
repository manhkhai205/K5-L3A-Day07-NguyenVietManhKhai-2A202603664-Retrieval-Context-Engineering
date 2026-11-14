"""Run and benchmark real-LLM Agentic RAG configurations on Day 7 cases (dev and test)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("starter", "src"):
    if str(REPO_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / sub))

from agent_runner import GeminiController, run_agent_smart  # noqa: E402
from agentic import run_agent as starter_run_agent  # noqa: E402
from baseline import hit as is_evidence_hit  # noqa: E402
from retrievers import (  # noqa: E402
    BM25Retriever,
    DenseQdrantRetriever,
    HybridRRFRetriever,
    QueryEmbeddingService,
    TfidfRetriever,
    build_neighbor_lookup,
)


AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "agent_baseline_tfidf_b800": {
        "label": "Agent Baseline (starter loop + TF-IDF, top_k=3, no filter, budget=800, max_searches=3)",
        "loop": "starter",
        "method": "tfidf",
        "top_k": 3,
        "filter_mode": "none",
        "budget": 800,
        "max_searches": 3,
        "neighbor_expand": False,
    },
    "agent_improve1_dense_b2400": {
        "label": "Agent Cải thiện 1 (Dense Qdrant + provided filter + budget=2400, max_searches=3)",
        "loop": "smart",
        "method": "dense_qdrant",
        "top_k": 3,
        "filter_mode": "provided",
        "budget": 2400,
        "max_searches": 3,
        "neighbor_expand": False,
    },
    "agent_improve2_hybrid_smart_b3200": {
        "label": "Agent Cải thiện 2 / Đã chọn (Hybrid RRF + inferred filter + neighbor expand + budget=3200, max_searches=3)",
        "loop": "smart",
        "method": "hybrid_rrf",
        "top_k": 3,
        "filter_mode": "inferred",
        "budget": 3200,
        "max_searches": 3,
        "neighbor_expand": True,
    },
}


def compute_context_hit(
    final_ids: list[str],
    gold: list[dict[str, Any]] | None,
    chunk_by_id: dict[str, dict[str, Any]],
) -> bool:
    if not gold:
        return False
    for cid in final_ids:
        ch = chunk_by_id.get(cid)
        if ch and is_evidence_hit(ch, gold):
            return True
    return False


def run_agent_config(
    config_name: str,
    spec: dict[str, Any],
    cases: list[dict[str, Any]],
    retriever: Any,
    chunk_by_id: dict[str, dict[str, Any]],
    neighbor_lookup: dict[tuple[str, int, int], dict[str, Any]],
    output_dir: Path,
    model_name: str,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.json"
    existing_by_id: dict[str, dict[str, Any]] = {}
    if resume and traces_path.exists():
        loaded = json.loads(traces_path.read_text(encoding="utf-8"))
        existing_by_id = {t["query_id"]: t for t in loaded if t.get("status") in ("answered", "abstain")}

    traces: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        qid = case["id"]
        if qid in existing_by_id:
            traces.append(existing_by_id[qid])
            t = existing_by_id[qid]
            print(
                f"  [{config_name}] ({idx}/{len(cases)}) {qid}: [cached] status={t['status']} "
                f"searches={t['num_searches']} cites={t['citations']}"
            )
            continue

        controller = GeminiController(
            model_name=model_name,
            api_key=os.getenv("GEMINI_API_KEY"),
            max_searches=int(spec["max_searches"]),
        )
        t0 = time.perf_counter()
        try:
            if spec["loop"] == "starter":
                res = starter_run_agent(
                    case["question"],
                    controller,
                    retriever,
                    filters=None,
                    max_searches=int(spec["max_searches"]),
                    top_k=int(spec["top_k"]),
                    budget=int(spec["budget"]),
                )
            else:
                res = run_agent_smart(
                    case["question"],
                    controller,
                    retriever,
                    filters=case.get("filter"),
                    filter_mode=spec["filter_mode"],
                    max_searches=int(spec["max_searches"]),
                    top_k=int(spec["top_k"]),
                    budget=int(spec["budget"]),
                    neighbor_lookup=neighbor_lookup if spec.get("neighbor_expand") else None,
                )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            res = {
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [{"event": "error", "error": str(exc)}],
            }
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        search_events = [e for e in res.get("events", []) if e.get("event") == "search"]
        final_ctx_ids = search_events[-1].get("context_ids", []) if search_events else []
        final_ctx = search_events[-1].get("context", "") if search_events else ""
        ctx_hit = compute_context_hit(final_ctx_ids, case.get("gold"), chunk_by_id) if case["answerable"] else None

        trace_entry = {
            "query_id": qid,
            "split": case["split"],
            "question": case["question"],
            "answerable": case["answerable"],
            "config_name": config_name,
            "model": model_name,
            "status": res["status"],
            "answer": res.get("answer", ""),
            "citations": res.get("citations", []),
            "num_searches": len(search_events),
            "final_context_ids": final_ctx_ids,
            "final_context_chars": len(final_ctx),
            "context_evidence_hit": ctx_hit,
            "elapsed_ms": round(elapsed_ms, 2),
            "llm_calls": controller.llm_calls,
            "prompt_tokens": controller.prompt_tokens,
            "completion_tokens": controller.completion_tokens,
            "events": res.get("events", []),
        }
        traces.append(trace_entry)
        traces_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"  [{config_name}] ({idx}/{len(cases)}) {qid} (ans={case['answerable']}): "
            f"status={trace_entry['status']} | searches={trace_entry['num_searches']} | "
            f"ctx_hit={ctx_hit} | cites={trace_entry['citations']} | {trace_entry['elapsed_ms']:.0f}ms"
        )

    ans_traces = [t for t in traces if t["answerable"] and t["status"] != "error"]
    unans_traces = [t for t in traces if not t["answerable"] and t["status"] != "error"]
    search_dist = dict(sorted(Counter(t["num_searches"] for t in traces).items()))

    run_stats = {
        "config_name": config_name,
        "label": spec["label"],
        "split": cases[0]["split"] if cases else "",
        "model": model_name,
        "total_cases": len(cases),
        "answerable_cases": len(ans_traces),
        "unanswerable_cases": len(unans_traces),
        "provider_errors": sum(1 for t in traces if t["status"] == "error"),
        "tool_use_count": sum(1 for t in traces if t["num_searches"] > 0),
        "search_count_distribution": search_dist,
        "avg_searches_per_case": round(statistics.mean(t["num_searches"] for t in traces), 2) if traces else 0.0,
        "context_evidence_hit_count": sum(1 for t in ans_traces if t["context_evidence_hit"]),
        "context_evidence_rate": (
            round(sum(1 for t in ans_traces if t["context_evidence_hit"]) / len(ans_traces), 4)
            if ans_traces
            else 0.0
        ),
        "answered_on_answerable": sum(1 for t in ans_traces if t["status"] == "answered"),
        "false_abstention_on_answerable": sum(1 for t in ans_traces if t["status"] == "abstain"),
        "abstained_on_unanswerable": sum(1 for t in unans_traces if t["status"] == "abstain"),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1) if traces else 0.0,
        "total_sec": round(sum(t["elapsed_ms"] for t in traces) / 1000.0, 2),
        "total_llm_calls": sum(t["llm_calls"] for t in traces),
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "total_completion_tokens": sum(t["completion_tokens"] for t in traces),
        "service_cost_usd": 0.0,
    }
    (output_dir / "run_stats.json").write_text(
        json.dumps(run_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 7 Agentic RAG configurations")
    parser.add_argument("--chunks", type=Path, default=REPO_ROOT / "data" / "chunks.json")
    parser.add_argument("--cases", type=Path, default=REPO_ROOT / "data" / "cases.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=list(AGENT_CONFIGS.keys()),
        help="Agent configurations to run",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "agent")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    model_name = os.getenv("AGENT_MODEL", "gemini-3.8-flash")
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    chunk_by_id = {c["id"]: c for c in chunks}
    neighbor_lookup = build_neighbor_lookup(chunks)

    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    selected_cases = [c for c in all_cases if c.get("split") == args.split]

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

    all_stats = []
    for cfg_name in args.configs:
        spec = AGENT_CONFIGS[cfg_name]
        ret = retrievers[spec["method"]]
        cfg_out = args.output_dir / f"{cfg_name}_{args.split}"
        print(f"\n=== Running {cfg_name} on split={args.split} ({len(selected_cases)} cases) ===")
        stats = run_agent_config(
            cfg_name,
            spec,
            selected_cases,
            ret,
            chunk_by_id,
            neighbor_lookup,
            cfg_out,
            model_name=model_name,
            resume=not args.no_resume,
        )
        all_stats.append(stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
