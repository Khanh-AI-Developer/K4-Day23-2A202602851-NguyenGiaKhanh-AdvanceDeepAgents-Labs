"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _clean_error(exc: Exception) -> str:
    msg = f"ERROR: {type(exc).__name__}: {exc}"
    key = os.getenv("EXA_API_KEY", "").strip()
    if key and key in msg:
        msg = msg.replace(key, "[REDACTED]")
    msg = re.sub(r"exaApiKey=[^&\s'\"]+", "exaApiKey=[REDACTED]", msg)
    return msg


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.HTTPStatusError, httpx.TransportError) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code not in RETRYABLE_STATUS_CODES:
                raise
            if attempt == attempts - 1:
                raise
            delay = None
            if isinstance(e, RetryableError) and e.retry_after is not None:
                try:
                    delay = float(e.retry_after)
                except (ValueError, TypeError):
                    pass
            elif isinstance(e, httpx.HTTPStatusError) and "retry-after" in e.response.headers:
                try:
                    delay = float(e.response.headers["retry-after"])
                except (ValueError, TypeError):
                    pass
            if delay is None:
                exponential = base * (2 ** attempt)
                jitter = random.uniform(0.1, 0.5)
                delay = min(cap, exponential + jitter)
            else:
                delay = min(cap, delay)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = re.findall(r"[\w\-]+", query)
        if not terms:
            return "NO RESULTS"

        # Rate limiting: minimum 3 seconds between arXiv calls
        elapsed = time.time() - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _LAST_ARXIV_CALL = time.time()

        max_res = max(1, min(int(max_results), 30))
        search_query = "all:" + " AND all:".join(terms)
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_res,
        }

        def _fetch():
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(ARXIV_URL, params=params)
                resp.raise_for_status()
                return resp.text

        xml_text = with_retry(_fetch, attempts=5, base=2.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue
            raw_id = id_elem.text.strip()
            paper_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id.split("/")[-1]
            clean_id = re.sub(r"v\d+$", "", paper_id)
            url = f"https://arxiv.org/abs/{clean_id}"

            pub_elem = entry.find("atom:published", ns)
            published = (pub_elem.text.strip() if pub_elem is not None and pub_elem.text else "")[:10]

            title_elem = entry.find("atom:title", ns)
            title = " ".join((title_elem.text or "").split()) if title_elem is not None else "Untitled"

            sum_elem = entry.find("atom:summary", ns)
            summary = " ".join((sum_elem.text or "").split())[:600] if sum_elem is not None else ""

            records.append({
                "id": clean_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _clean_error(exc)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        lim = max(1, min(int(limit), 100))
        params = {"limit": lim}
        if date:
            params["date"] = date

        def _fetch():
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(HF_DAILY_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        data = with_retry(_fetch, attempts=4, base=1.0, cap=30.0)
        if not isinstance(data, list):
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue

            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            summary = " ".join(str(paper.get("summary") or item.get("summary") or "").split())[:600]
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(paper.get("upvotes") or 0)
            github = paper.get("githubRepo")
            stars = paper.get("githubStars")

            records.append({
                "id": str(paper_id),
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if keyword:
            kw = keyword.lower()
            records = [r for r in records if kw in (r["title"] + " " + r["summary"]).lower()]

        records.sort(key=lambda r: r.get("upvotes") or 0, reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _clean_error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        lim = max(1, min(int(limit), 50))
        params = {"q": query, "limit": lim}

        def _fetch():
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(HF_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        data = with_retry(_fetch, attempts=4, base=1.0, cap=30.0)
        if not isinstance(data, list):
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue

            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            # Prefer ai_summary over summary
            raw_summary = paper.get("ai_summary") or paper.get("summary") or item.get("ai_summary") or item.get("summary") or ""
            summary = " ".join(str(raw_summary).split())[:600]
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(paper.get("upvotes") or 0)
            github = paper.get("githubRepo")
            stars = paper.get("githubStars")

            records.append({
                "id": str(paper_id),
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _clean_error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    exa_key = os.getenv("EXA_API_KEY", "").strip()
    url = f"{EXA_URL}?exaApiKey={exa_key}" if exa_key else EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if exa_key:
        headers["x-api-key"] = exa_key
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _post():
        resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        text = resp.text
        data = None
        for line in text.splitlines():
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    break
                except Exception:
                    pass
        if data is None:
            try:
                data = resp.json()
            except Exception:
                pass
        if not data or not isinstance(data, dict):
            raise RuntimeError(f"invalid response from Exa: {text[:200]}")

        if "error" in data:
            err = str(data["error"])
            if "rate limit" in err.lower() or "429" in err:
                raise RetryableError(f"Exa rate limited: {err}", retry_after=20.0)
            raise RuntimeError(f"Exa error: {err}")

        result = data.get("result", {})
        meta = result.get("_meta", {})
        meta_str = json.dumps(meta).lower()
        if "rate limit" in meta_str or meta.get("rate_limited") or meta.get("ratelimit"):
            raise RetryableError("Exa rate limited (_meta flag)", retry_after=20.0)

        contents = result.get("content", [])
        text_parts = [c.get("text", "") for c in contents if isinstance(c, dict) and c.get("type") == "text"]
        full_text = "\n".join(text_parts).strip()
        if "rate limit" in full_text.lower() and len(full_text) < 300:
            raise RetryableError(f"Exa rate limit message: {full_text}", retry_after=20.0)
        return full_text

    return with_retry(_post, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        obj = objective.strip() or f"find comprehensive research papers, surveys, and technical blogs about {query}"
        num_res = max(1, min(int(num_results), 20))
        arguments = {
            "query": query,
            "objective": obj,
            "numResults": num_res,
        }
        text = _call_exa_mcp("web_search_exa", arguments)
        if not text:
            return "NO RESULTS"
        return text
    except Exception as exc:
        return _clean_error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        arguments = {"urls": [url]}
        text = _call_exa_mcp("web_fetch_exa", arguments)
        if not text:
            return "NO RESULTS"
        if len(text) > 12000:
            text = text[:12000] + "\n... [truncated]"
        return text
    except Exception as exc:
        return _clean_error(exc)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
