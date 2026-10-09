import json
import unittest
from unittest.mock import MagicMock, patch
import httpx
from tools import RetryableError, with_retry, arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch


class TestTools(unittest.TestCase):
    def test_with_retry_success(self):
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RetryableError("temporary glitch")
            return "ok"

        with patch("time.sleep") as mock_sleep:
            res = with_retry(fn, attempts=5, base=0.1, cap=1.0)
            self.assertEqual(res, "ok")
            self.assertEqual(calls, 3)
            self.assertEqual(mock_sleep.call_count, 2)

    def test_with_retry_respects_retry_after(self):
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RetryableError("slow down", retry_after=7.5)
            return "ok"

        with patch("time.sleep") as mock_sleep:
            res = with_retry(fn, attempts=3)
            self.assertEqual(res, "ok")
            mock_sleep.assert_called_with(7.5)

    def test_with_retry_gives_up_on_last_attempt(self):
        def fn():
            raise RetryableError("permanent failure")

        with patch("time.sleep") as mock_sleep:
            with self.assertRaises(RetryableError):
                with_retry(fn, attempts=3, base=0.1)
            # Should have slept twice (attempts 0 and 1), but NOT on attempt 2
            self.assertEqual(mock_sleep.call_count, 2)

    def test_with_retry_does_not_retry_unretryable(self):
        def fn():
            raise ValueError("bad parameter")

        with patch("time.sleep") as mock_sleep:
            with self.assertRaises(ValueError):
                with_retry(fn, attempts=3)
            mock_sleep.assert_not_called()

    def test_arxiv_empty_query(self):
        res = arxiv_search.invoke({"query": "??? !!!"})
        self.assertEqual(res, "NO RESULTS")

    def test_arxiv_query_sanitization(self):
        # Should not raise exception and should return valid response string
        res = arxiv_search.invoke({"query": "world model: survey (2024)", "max_results": 1})
        self.assertIsInstance(res, str)
        # It's either JSON array or "NO RESULTS" or "ERROR"
        if not res.startswith("ERROR") and res != "NO RESULTS":
            data = json.loads(res)
            self.assertIsInstance(data, list)
            if data:
                self.assertIn("url", data[0])
                self.assertTrue(data[0]["url"].startswith("https://arxiv.org/abs/"))

    def test_exa_key_redaction(self):
        with patch.dict("os.environ", {"EXA_API_KEY": "secret-test-key-12345"}):
            with patch("httpx.post", side_effect=httpx.ConnectError("Failed connecting with secret-test-key-12345")):
                res = web_search.invoke({"query": "test"})
                self.assertNotIn("secret-test-key-12345", res)
                self.assertTrue(res.startswith("ERROR:"))


if __name__ == "__main__":
    unittest.main()
