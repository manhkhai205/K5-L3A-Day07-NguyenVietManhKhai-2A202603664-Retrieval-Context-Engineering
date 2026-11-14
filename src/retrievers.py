"""Retrieval methods and context packing utilities for Day 7 (TF-IDF, BM25, Dense Qdrant, Hybrid RRF)."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import unicodedata
import uuid

from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "starter"))

from baseline import TfidfIndex, pack_context as starter_pack_context  # noqa: E402

ALLOWED_FILTER_KEYS = {"document_id", "source_id", "source", "page", "section", "year"}


def tokenize_domain(text: str) -> list[str]:
    """Tokenize text preserving both standard words and hyphenated NIST action codes (e.g., ms-2.5-003)."""
    norm = unicodedata.normalize("NFC", text).casefold()
    words = re.findall(r"\w+", norm)
    # Also extract hyphenated action codes or section numbers like ms-2.5-003, mg-3.1-001, 100-1, 600-1
    codes = re.findall(r"\b[a-z]{2}-\d+\.\d+-\d+\b|\b\d{3}-\d\b", norm)
    return words + codes


def matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check whether an in-memory chunk dict satisfies metadata filters."""
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for key, val in filters.items():
        if key not in ALLOWED_FILTER_KEYS:
            raise ValueError(f"Unsupported filter key: {key!r}")
        if key in ("source_id", "document_id"):
            actual = chunk.get("source_id")
        elif key in ("source", "page"):
            actual = chunk.get(key)
        else:
            actual = meta.get(key)
        if actual != val:
            return False
    return True


def infer_filters_from_query(question: str, provided_filter: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Deterministically infer safe metadata filters ('year', 'section') from explicit query mentions only."""
    inferred: dict[str, Any] = dict(provided_filter) if provided_filter else {}
    q_norm = question.strip()
    q_lower = q_norm.lower()

    # Infer year only when explicitly grounded in document title/version in the question
    if "year" not in inferred:
        mentions_100 = bool(re.search(r"\b100-1\b|\bai rmf 1\.0\b|\b2023\b", q_lower))
        mentions_600 = bool(
            re.search(r"\b600-1\b|\bgenerative ai profile\b|\bgai\b|\bgenerative ai\b|\b2024\b", q_lower)
        )
        if mentions_100 and not mentions_600:
            inferred["year"] = 2023
        elif mentions_600 and not mentions_100:
            inferred["year"] = 2024

    # Infer section only when uppercase Core function or explicit section name is stated in the question
    if "section" not in inferred:
        sec_matches: list[str] = []
        if re.search(r"\bGOVERN\b", q_norm):
            sec_matches.append("govern")
        if re.search(r"\bMAP\b", q_norm):
            sec_matches.append("map")
        if re.search(r"\bMEASURE\b", q_norm):
            sec_matches.append("measure")
        if re.search(r"\bMANAGE\b", q_norm):
            sec_matches.append("manage")
        if re.search(r"\bappendix\b", q_lower):
            sec_matches.append("appendix")
        if len(sec_matches) == 1:
            inferred["section"] = sec_matches[0]

    return inferred or None


class TfidfRetriever:
    """Method 1: Standard TF-IDF lexical retriever wrapping starter/baseline.py."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self._index = TfidfIndex(chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        if not filters:
            raw = self._index.search(query, top_k=top_k)
        else:
            # Filter candidate pool first, then score
            filtered_Pool = [c for c in self.chunks if matches_filter(c, filters)]
            if not filtered_Pool:
                return []
            raw = TfidfIndex(filtered_Pool).search(query, top_k=top_k)
        return [
            {
                **r,
                "rank": idx + 1,
                "score": round(float(r["score"]), 6),
                "strategy": "tfidf",
            }
            for idx, r in enumerate(raw[:top_k])
        ]


class BM25Retriever:
    """Method 2: Okapi BM25 lexical retriever with domain-code tokenization and metadata filtering."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [Counter(tokenize_domain(c["text"])) for c in chunks]
        self.doc_lens = [sum(cnt.values()) for cnt in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        df = Counter(term for cnt in self.doc_tokens for term in cnt)
        n_docs = len(chunks)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        q_terms = Counter(tokenize_domain(query))
        results: list[dict[str, Any]] = []
        for chunk, t_counts, dl in zip(self.chunks, self.doc_tokens, self.doc_lens):
            if not matches_filter(chunk, filters):
                continue
            score = 0.0
            for term, qf in q_terms.items():
                tf = t_counts.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += idf * ((tf * (self.k1 + 1.0)) / denom) * qf
            if score > 0:
                results.append({**chunk, "score": round(score, 6), "strategy": "bm25"})

        ranked = sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]
        return [{**r, "rank": idx + 1} for idx, r in enumerate(ranked)]


class QueryEmbeddingService:
    """Embed queries with Day 6's gemini-embedding-2 (768-d) with optional disk cache for reproducibility."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-embedding-2",
        dimensions: int = 768,
        cache_path: Path | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for Dense Qdrant retrieval.")
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = genai.Client(api_key=self.api_key)
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.api_calls = 0
        self.cache_hits = 0

    def embed_query(self, query: str) -> list[float]:
        key = f"{self.model_name}:{self.dimensions}:{query.strip()}"
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        resp = self.client.models.embed_content(
            model=self.model_name,
            contents=query.strip(),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.dimensions,
            ),
        )
        vec = [float(v) for v in resp.embeddings[0].values]
        if len(vec) != self.dimensions:
            raise ValueError(f"Query vector dimension mismatch: got {len(vec)}, expected {self.dimensions}")
        self.api_calls += 1
        self.cache[key] = vec
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        return vec


class DenseQdrantRetriever:
    """Method 3 (Mandatory): Dense retrieval on Day 6 Qdrant vector store with gemini-embedding-2."""

    def __init__(
        self,
        embedder: QueryEmbeddingService,
        qdrant_url: str = "http://127.0.0.1:6333",
        collection_name: str = "day6_nist_chunks",
        qdrant_strategy: str = "B",
        local_path: Path | None = None,
        chunks_fallback_path: Path | None = None,
    ) -> None:
        self.embedder = embedder
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.qdrant_strategy = qdrant_strategy
        self.local_path = local_path or (REPO_ROOT / "results" / "cache" / "qdrant_local")
        self.chunks_fallback_path = chunks_fallback_path or (REPO_ROOT / "data" / "chunks.json")
        self.client = self._connect_or_init_local()

    def _connect_or_init_local(self) -> QdrantClient:
        try:
            remote = QdrantClient(url=self.qdrant_url, timeout=3.0, check_compatibility=False)
            remote.get_collection(self.collection_name)
            return remote
        except Exception:
            self.local_path.mkdir(parents=True, exist_ok=True)
            local_client = QdrantClient(path=str(self.local_path))
            existing = {c.name for c in local_client.get_collections().collections}
            need_populate = self.collection_name not in existing
            if not need_populate:
                cnt = local_client.count(collection_name=self.collection_name).count
                need_populate = cnt == 0
            if need_populate:
                self._populate_local_qdrant(local_client)
            return local_client

    def _populate_local_qdrant(self, client: QdrantClient) -> None:
        chunks = json.loads(self.chunks_fallback_path.read_text(encoding="utf-8"))
        existing = {c.name for c in client.get_collections().collections}
        if self.collection_name not in existing:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
        doc_cache_path = REPO_ROOT / "results" / "cache" / "doc_embeddings_b.json"
        doc_vecs: dict[str, list[float]] = {}
        if doc_cache_path.exists():
            doc_vecs = json.loads(doc_cache_path.read_text(encoding="utf-8"))

        missing_chunks = [c for c in chunks if c["id"] not in doc_vecs]
        if missing_chunks:
            print(f"Embedding {len(missing_chunks)} Day 6 chunks with {self.embedder.model_name} for local Qdrant...")

            def _embed_one(ch: dict[str, Any]) -> tuple[str, list[float]]:
                for attempt in range(4):
                    try:
                        resp = self.embedder.client.models.embed_content(
                            model=self.embedder.model_name,
                            contents=ch["text"],
                            config=types.EmbedContentConfig(
                                task_type="RETRIEVAL_DOCUMENT",
                                output_dimensionality=self.embedder.dimensions,
                            ),
                        )
                        return ch["id"], [float(v) for v in resp.embeddings[0].values]
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(1.0 * (attempt + 1))
                raise RuntimeError("Unreachable")

            with ThreadPoolExecutor(max_workers=6) as pool:
                for cid, vec in pool.map(_embed_one, missing_chunks):
                    doc_vecs[cid] = vec
            doc_cache_path.parent.mkdir(parents=True, exist_ok=True)
            doc_cache_path.write_text(json.dumps(doc_vecs), encoding="utf-8")

        points = []
        for ch in chunks:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.qdrant_strategy}:{ch['id']}"))
            payload = {
                "strategy": self.qdrant_strategy,
                "chunk_id": ch["id"],
                "document_id": ch["source_id"],
                "source": ch["source"],
                "page": int(ch["page"]),
                "text": ch["text"],
                **ch.get("metadata", {}),
            }
            points.append(models.PointStruct(id=point_id, vector=doc_vecs[ch["id"]], payload=payload))

        for start in range(0, len(points), 100):
            client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + 100],
                wait=True,
            )

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        q_vec = self.embedder.embed_query(query)
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="strategy",
                match=models.MatchValue(value=self.qdrant_strategy),
            )
        ]
        if filters:
            for k, v in filters.items():
                if k not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported metadata filter key: {k!r}")
                qdrant_key = "document_id" if k == "source_id" else k
                conditions.append(
                    models.FieldCondition(
                        key=qdrant_key,
                        match=models.MatchValue(value=v),
                    )
                )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=models.Filter(must=conditions),
            limit=top_k,
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for idx, pt in enumerate(response.points, 1):
            payload = pt.payload or {}
            meta = {
                k: v
                for k, v in payload.items()
                if k not in {"strategy", "chunk_id", "document_id", "source", "page", "text"}
            }
            results.append(
                {
                    "id": payload["chunk_id"],
                    "source_id": payload["document_id"],
                    "source": payload["source"],
                    "page": int(payload["page"]),
                    "text": payload["text"],
                    "metadata": meta,
                    "score": round(float(pt.score), 6),
                    "rank": idx,
                    "strategy": "dense_qdrant",
                }
            )
        return results


class HybridRRFRetriever:
    """Method 4: Hybrid retrieval combining Dense Qdrant + Lexical BM25 via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense: DenseQdrantRetriever,
        lexical: BM25Retriever,
        *,
        candidate_k: int = 15,
        rrf_k: int = 60,
        dense_weight: float = 1.35,
        lexical_weight: float = 1.0,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        k_cand = max(self.candidate_k, top_k * 3)
        dense_hits = self.dense.search(query, filters=filters, top_k=k_cand)
        lex_hits = self.lexical.search(query, filters=filters, top_k=k_cand)

        by_id: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        lex_ranks: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        lex_scores: dict[str, float] = {}

        for rank, item in enumerate(dense_hits, 1):
            cid = item["id"]
            by_id[cid] = item
            dense_ranks[cid] = rank
            dense_scores[cid] = item["score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank)

        for rank, item in enumerate(lex_hits, 1):
            cid = item["id"]
            if cid not in by_id:
                by_id[cid] = item
            lex_ranks[cid] = rank
            lex_scores[cid] = item["score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.lexical_weight / (self.rrf_k + rank)

        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (
                -rrf_scores[cid],
                -dense_scores.get(cid, 0.0),
                -lex_scores.get(cid, 0.0),
                cid,
            ),
        )[:top_k]

        results: list[dict[str, Any]] = []
        for idx, cid in enumerate(sorted_ids, 1):
            base = dict(by_id[cid])
            meta = dict(base.get("metadata", {}))
            meta["rrf_detail"] = {
                "dense_rank": dense_ranks.get(cid),
                "dense_score": dense_scores.get(cid),
                "lexical_rank": lex_ranks.get(cid),
                "lexical_score": lex_scores.get(cid),
            }
            results.append(
                {
                    **base,
                    "metadata": meta,
                    "score": round(rrf_scores[cid], 6),
                    "rank": idx,
                    "strategy": "hybrid_rrf",
                }
            )
        return results


def pack_context_smart(
    candidates: list[dict[str, Any]],
    *,
    budget: int = 2400,
    rerank_by_score: bool = True,
    neighbor_lookup: dict[tuple[str, int, int], dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """Pack unique chunks into context budget, optionally sorting across multi-turn searches and adding neighbors."""
    if not candidates or budget <= 0:
        return "", []

    # Deduplicate keeping highest score (or latest search order when rerank_by_score=False)
    if rerank_by_score:
        dedup: dict[str, dict[str, Any]] = {}
        for idx, c in enumerate(candidates):
            cid = c["id"]
            # Prefer higher score; if from different strategies, normalize rank preference
            if cid not in dedup or float(c.get("score", 0.0)) > float(dedup[cid].get("score", 0.0)):
                dedup[cid] = {**c, "_order": idx}
        ordered = list(dedup.values())
    else:
        ordered = candidates

    # Optional neighbor expansion: append adjacent chunks of rank-1 AFTER top-k hits so they never crowd out rank 2-3
    if neighbor_lookup and ordered:
        top1 = ordered[0]
        m = re.search(r":p(\d+):c(\d+)$", top1["id"])
        if m:
            pg = int(m.group(1))
            c_num = int(m.group(2))
            neighbors: list[dict[str, Any]] = []
            for adj_num in (c_num - 1, c_num + 1):
                adj = neighbor_lookup.get((top1["source_id"], pg, adj_num))
                if adj:
                    neighbors.append({**adj, "score": round(float(top1.get("score", 0.0)) * 0.9, 6)})
            ordered = list(ordered) + neighbors

    return starter_pack_context(ordered, budget=budget)


def build_neighbor_lookup(chunks: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Index chunks by (source_id, page, chunk_index) for fast same-page neighbor expansion."""
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for c in chunks:
        m = re.search(r":p(\d+):c(\d+)$", c["id"])
        if m:
            lookup[(c["source_id"], int(m.group(1)), int(m.group(2)))] = c
    return lookup
