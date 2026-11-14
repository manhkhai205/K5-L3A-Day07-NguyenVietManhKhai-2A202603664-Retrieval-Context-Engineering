import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("starter", "src"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from retrievers import (  # noqa: E402
    BM25Retriever,
    HybridRRFRetriever,
    build_neighbor_lookup,
    infer_filters_from_query,
    pack_context_smart,
)


class FakeDense:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, *, filters=None, top_k=3):
        return [
            {**c, "score": round(0.9 - idx * 0.1, 4), "rank": idx + 1, "strategy": "dense_qdrant"}
            for idx, c in enumerate(self.chunks[:top_k])
        ]


class RetrieverUnitTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "nist_ai_100_1:B:p12:c001",
                "source_id": "nist_ai_100_1",
                "source": "https://doi.org/10.6028/NIST.AI.100-1",
                "page": 12,
                "text": "While the AI RMF can be used to prioritize risk, it does not prescribe risk tolerance.",
                "metadata": {"section": "foundational_information", "year": 2023},
            },
            {
                "id": "nist_ai_100_1:B:p12:c002",
                "source_id": "nist_ai_100_1",
                "source": "https://doi.org/10.6028/NIST.AI.100-1",
                "page": 12,
                "text": "To the extent that challenges for specifying AI risk tolerances remain unresolved.",
                "metadata": {"section": "foundational_information", "year": 2023},
            },
            {
                "id": "nist_ai_600_1:B:p35:c002",
                "source_id": "nist_ai_600_1",
                "source": "https://doi.org/10.6028/NIST.AI.600-1",
                "page": 35,
                "text": "MS-2.5-003 Review and verify sources and citations in GAI system outputs.",
                "metadata": {"section": "measure", "year": 2024},
            },
        ]

    def test_bm25_matches_domain_code_and_filter(self):
        bm25 = BM25Retriever(self.chunks)
        hits = bm25.search("MS-2.5-003 citations", filters={"section": "measure"}, top_k=2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "nist_ai_600_1:B:p35:c002")

    def test_hybrid_rrf_combines_dense_and_lexical(self):
        bm25 = BM25Retriever(self.chunks)
        dense = FakeDense(list(reversed(self.chunks)))
        hybrid = HybridRRFRetriever(dense=dense, lexical=bm25, candidate_k=3, rrf_k=60)
        hits = hybrid.search("MS-2.5-003 citations", top_k=2)
        self.assertEqual(hits[0]["id"], "nist_ai_600_1:B:p35:c002")
        self.assertIn("rrf_detail", hits[0]["metadata"])

    def test_infer_filters_from_query(self):
        f1 = infer_filters_from_query("Under the MEASURE function in NIST AI 600-1, what is verified?")
        self.assertEqual(f1, {"year": 2024, "section": "measure"})
        f2 = infer_filters_from_query("In the 2023 AI RMF 1.0 Appendix, which systems need oversight?")
        self.assertEqual(f2, {"year": 2023, "section": "appendix"})

    def test_neighbor_expansion_packs_adjacent_chunk(self):
        lookup = build_neighbor_lookup(self.chunks)
        ctx, ids = pack_context_smart(
            [self.chunks[1]],
            budget=1000,
            rerank_by_score=False,
            neighbor_lookup=lookup,
        )
        self.assertEqual(ids, ["nist_ai_100_1:B:p12:c002", "nist_ai_100_1:B:p12:c001"])
        self.assertIn("does not prescribe risk tolerance", ctx)


if __name__ == "__main__":
    unittest.main()
