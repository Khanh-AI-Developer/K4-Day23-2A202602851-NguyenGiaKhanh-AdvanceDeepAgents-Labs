import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from research import slugify, summarize, save_outputs, build_prompt


class TestResearch(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Survey about world model"), "survey-about-world-model")
        self.assertEqual(slugify("../../etc/passwd"), "etc-passwd")
        self.assertEqual(slugify(""), "topic")
        self.assertEqual(slugify("   "), "topic")
        # Long string truncation
        long_topic = "a" * 100
        self.assertEqual(len(slugify(long_topic)), 60)

    def test_build_prompt(self):
        p = build_prompt("survey about world model")
        self.assertIn("survey about world model", p)
        self.assertIn("write_todos", p)

    def test_summarize(self):
        # Mock messages with tool_calls and usage
        mock_msg1 = MagicMock()
        mock_msg1.tool_calls = [{"name": "write_todos"}, {"name": "task"}]
        mock_msg1.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

        mock_msg2 = MagicMock()
        mock_msg2.tool_calls = [{"name": "task"}, {"name": "execute"}]
        mock_msg2.usage_metadata = {"input_tokens": 120, "output_tokens": 60}

        res = summarize([mock_msg1, mock_msg2], 12.345, "test-model")
        self.assertEqual(res["model"], "test-model")
        self.assertEqual(res["elapsed_s"], 12.3)
        self.assertEqual(res["subagent_calls"], 2)
        self.assertEqual(res["tool_calls"]["task"], 2)
        self.assertEqual(res["tool_calls"]["write_todos"], 1)
        self.assertEqual(res["tool_calls"]["execute"], 1)
        self.assertEqual(res["tokens"]["input"], 220)
        self.assertEqual(res["tokens"]["output"], 110)

    def test_save_outputs_failure_writes_nothing(self):
        mock_backend = MagicMock()
        # Mock download returning None for report
        from agents import REPORT_PATH, SOURCES_PATH
        with self.assertRaises(RuntimeError):
            save_outputs(mock_backend, "bad topic", [], 1.0, "model")


if __name__ == "__main__":
    unittest.main()
