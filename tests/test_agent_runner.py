import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("starter", "src"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from agent_runner import run_agent_smart  # noqa: E402
from retrievers import build_neighbor_lookup  # noqa: E402


class ScriptedSmartController:
    def __init__(self, actions):
        self.actions = list(actions)
        self.seen_history = []

    def decide(self, question, history):
        self.seen_history.append(history)
        return self.actions.pop(0)


class FakeMapRetriever:
    def __init__(self, by_query):
        self.by_query = by_query

    def search(self, query, *, filters=None, top_k=3):
        return self.by_query.get(query, [])[:top_k]


class SmartAgentRunnerTests(unittest.TestCase):
    def test_multi_turn_interleaving_keeps_second_turn_chunk_in_tight_budget(self):
        c1 = {
            "id": "doc:B:p01:c001",
            "source_id": "doc",
            "source": "src://doc",
            "page": 1,
            "text": "A" * 120,
            "metadata": {},
            "score": 0.9,
        }
        c2 = {
            "id": "doc:B:p02:c001",
            "source_id": "doc",
            "source": "src://doc",
            "page": 2,
            "text": "B" * 120,
            "metadata": {},
            "score": 0.95,
        }
        ret = FakeMapRetriever({"q1": [c1], "q2": [c2]})
        ctrl = ScriptedSmartController(
            [
                {"action": "search", "query": "q1", "thought": "first"},
                {"action": "search", "query": "q2", "thought": "second"},
                {"action": "answer", "answer": "Found in [doc:B:p02:c001].", "citations": ["doc:B:p02:c001"]},
            ]
        )
        # Budget only fits 1 chunk (~150 chars); latest search top-1 (c2) is prioritized first
        out = run_agent_smart("question?", ctrl, ret, budget=180, max_searches=3, top_k=1)
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["doc:B:p02:c001"])

    def test_neighbor_expanded_chunk_can_be_cited(self):
        chunks = [
            {
                "id": "nist_ai_100_1:B:p12:c001",
                "source_id": "nist_ai_100_1",
                "source": "src://100-1",
                "page": 12,
                "text": "While the AI RMF can be used to prioritize risk, it does not prescribe risk tolerance.",
                "metadata": {"year": 2023},
                "score": 0.8,
            },
            {
                "id": "nist_ai_100_1:B:p12:c002",
                "source_id": "nist_ai_100_1",
                "source": "src://100-1",
                "page": 12,
                "text": "Challenges for specifying AI risk tolerances remain unresolved.",
                "metadata": {"year": 2023},
                "score": 0.85,
            },
        ]
        lookup = build_neighbor_lookup(chunks)
        ret = FakeMapRetriever({"risk tolerance": [chunks[1]]})
        ctrl = ScriptedSmartController(
            [
                {"action": "search", "query": "risk tolerance"},
                {
                    "action": "answer",
                    "answer": "It does not prescribe risk tolerance [nist_ai_100_1:B:p12:c001].",
                    "citations": ["nist_ai_100_1:B:p12:c001"],
                },
            ]
        )
        out = run_agent_smart("Does AI RMF prescribe risk tolerance?", ctrl, ret, budget=1000, neighbor_lookup=lookup)
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["nist_ai_100_1:B:p12:c001"])


if __name__ == "__main__":
    unittest.main()
