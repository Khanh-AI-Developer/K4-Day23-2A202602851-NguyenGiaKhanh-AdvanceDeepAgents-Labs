import unittest
from unittest.mock import MagicMock
from agents import build_subagents, build_lead_agent, LEAD_PROMPT, RESEARCHER_PROMPT, CHECKER_PROMPT


class TestAgents(unittest.TestCase):
    def test_prompts_contain_required_contracts(self):
        # Lead prompt must mention write_todos, finalize_citations, check_citations, sources.json
        self.assertIn("write_todos", LEAD_PROMPT)
        self.assertIn("finalize_citations.py", LEAD_PROMPT)
        self.assertIn("check_citations.py", LEAD_PROMPT)
        self.assertIn("sources.json", LEAD_PROMPT)
        self.assertIn("report.md", LEAD_PROMPT)

        # Researcher prompt must mention untrusted data and notes
        self.assertIn("untrusted", RESEARCHER_PROMPT.lower())
        self.assertIn("notes", RESEARCHER_PROMPT.lower())

        # Checker prompt must mention SUPPORTED
        self.assertIn("SUPPORTED", CHECKER_PROMPT)

    def test_build_subagents(self):
        subs = build_subagents()
        self.assertEqual(len(subs), 2)
        names = {s["name"] for s in subs}
        self.assertEqual(names, {"researcher", "citation-checker"})

        researcher = next(s for s in subs if s["name"] == "researcher")
        self.assertEqual(len(researcher["tools"]), 5)
        self.assertIn("middleware", researcher)

        checker = next(s for s in subs if s["name"] == "citation-checker")
        self.assertEqual(len(checker["tools"]), 1)
        self.assertIn("middleware", checker)

    def test_build_lead_agent(self):
        from deepagents.backends.state import StateBackend
        from model import make_model
        backend = StateBackend()
        model = make_model()
        agent = build_lead_agent(backend, model)
        self.assertIsNotNone(agent)


if __name__ == "__main__":
    unittest.main()
