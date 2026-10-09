import unittest
from check_citations import check


class TestCheckCitations(unittest.TestCase):
    def setUp(self):
        self.valid_sources = [
            {"n": 1, "url": "https://arxiv.org/abs/2401.00001", "title": "Paper 1", "source": "arxiv"},
            {"n": 2, "url": "https://huggingface.co/papers/2401.00002", "title": "Paper 2", "source": "hf-daily"},
        ]
        self.valid_report = (
            "# Survey on AI\n\n"
            "This is based on [1] and also [2].\n"
            "Here is a code example `[3]` which should not count.\n"
            "Here is a link [1](https://example.com) which should not count as citation.\n\n"
            "## References\n"
            "[1] Paper 1. arxiv. https://arxiv.org/abs/2401.00001 (2024)\n"
            "[2] Paper 2. hf-daily. https://huggingface.co/papers/2401.00002 (2024)\n"
        )

    def test_valid_report(self):
        problems = check(self.valid_report, self.valid_sources)
        self.assertEqual(problems, [])

    def test_empty_sources(self):
        problems = check(self.valid_report, [])
        self.assertTrue(any("no sources" in p.lower() or "empty" in p.lower() for p in problems))

    def test_invalid_source_schema(self):
        bad_sources = [
            {"n": "one", "url": "https://arxiv.org/abs/1"},
            {"n": 2, "url": "ftp://bad-url.com"},
            {"n": 3, "url": "https://arxiv.org/abs/1"},  # duplicate url with #1
        ]
        problems = check(self.valid_report, bad_sources)
        self.assertTrue(len(problems) >= 3)

    def test_missing_references_heading(self):
        report = "This cites [1]. But has no references heading."
        problems = check(report, self.valid_sources)
        self.assertTrue(any("references" in p.lower() for p in problems))

    def test_uncited_source(self):
        # Only cites [1], forgets [2]
        report = (
            "# Survey\n\nOnly cites [1].\n\n"
            "## References\n"
            "[1] Paper 1. https://arxiv.org/abs/2401.00001\n"
            "[2] Paper 2. https://huggingface.co/papers/2401.00002\n"
        )
        problems = check(report, self.valid_sources)
        self.assertTrue(any("never cited" in p.lower() or "not cited" in p.lower() for p in problems))

    def test_missing_source_for_citation(self):
        # Cites [3] which is not in sources
        report = (
            "# Survey\n\nCites [1], [2], and [3].\n\n"
            "## References\n"
            "[1] Paper 1. https://arxiv.org/abs/2401.00001\n"
            "[2] Paper 2. https://huggingface.co/papers/2401.00002\n"
        )
        problems = check(report, self.valid_sources)
        self.assertTrue(any("[3]" in p for p in problems))

    def test_grouped_references_line_or_wrong_url(self):
        # Line 1 bundling two URLs
        report = (
            "# Survey\n\nCites [1] and [2].\n\n"
            "## References\n"
            "[1] Paper 1. https://arxiv.org/abs/2401.00001 and https://extra.com\n"
            "[2] Paper 2. https://wrong-url.com\n"
        )
        problems = check(report, self.valid_sources)
        self.assertTrue(len(problems) >= 2)

    def test_compatibility_with_finalize_citations(self):
        from finalize_citations import finalize
        draft_body = (
            "# World Models Survey\n\n"
            "World models learn environment dynamics [1]. Recent advances explore multimodal representations [2].\n"
        )
        sources = [
            {"n": 1, "url": "https://arxiv.org/abs/1803.10122", "title": "World Models", "source": "arxiv", "date": "2018"},
            {"n": 2, "url": "https://huggingface.co/papers/2401.00002", "title": "HF Paper", "source": "hf-daily", "date": "2024"},
            {"n": 3, "url": "https://example.com/unused", "title": "Unused", "source": "web"},
        ]
        final_report, final_sources, finalize_problems = finalize(draft_body, sources)
        self.assertEqual(finalize_problems, [])
        check_problems = check(final_report, final_sources)
        self.assertEqual(check_problems, [])


if __name__ == "__main__":
    unittest.main()
