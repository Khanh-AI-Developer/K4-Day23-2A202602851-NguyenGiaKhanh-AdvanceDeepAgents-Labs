"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_MD_LINK = re.compile(r"\[\d+\]\([^\)]+\)")
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_REF_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")
_URL = re.compile(r"https?://[^\s)\]>]+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources or not isinstance(sources, list):
        return ["no sources in sources.json"]

    seen_urls = set()
    sources_by_n = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry is not a dict: {entry!r}")
            continue
        n = entry.get("n")
        if not isinstance(n, int):
            problems.append(f"source n must be an int, got {n!r}")
        url = str(entry.get("url") or "")
        if not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] url must start with http:// or https://, got {url!r}")
        if url in seen_urls:
            problems.append(f"duplicate url in sources.json: {url}")
        else:
            seen_urls.add(url)
        if isinstance(n, int):
            sources_by_n[n] = entry

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' heading in report")
        return problems

    body = report_text[: matches[-1].start()]
    refs_text = report_text[matches[-1].end() :]

    # Extract cited numbers from body only
    cited = set()
    segments = _CODE.split(body)
    for i, seg in enumerate(segments):
        if i % 2:  # inside code block / span
            continue
        clean_seg = _MD_LINK.sub(" ", seg)
        for m in _GROUP.finditer(clean_seg):
            for num in _group_numbers(m.group(1)):
                cited.add(num)

    source_nums = set(sources_by_n.keys())
    for num in sorted(cited):
        if num not in source_nums:
            problems.append(f"[{num}] is cited in the text but missing from sources.json")
    for num in sorted(source_nums):
        if num not in cited:
            problems.append(f"source [{num}] never cited in report body")

    ref_lines = [line.strip() for line in refs_text.splitlines() if line.strip()]
    seen_ref_nums = []
    for line in ref_lines:
        m = _REF_LINE.match(line)
        if not m:
            problems.append(f"reference line does not start with '[n]': {line}")
            continue
        ref_n = int(m.group(1))
        seen_ref_nums.append(ref_n)
        urls = _URL.findall(line)
        if len(urls) != 1:
            problems.append(f"reference line [{ref_n}] must contain exactly ONE URL (found {len(urls)}): {line}")
        elif ref_n in sources_by_n:
            expected_url = sources_by_n[ref_n].get("url")
            # Strip trailing punctuation that might attach to URL
            found_url = urls[0].rstrip(".,;)")
            if found_url != expected_url:
                problems.append(f"reference line [{ref_n}] url {found_url!r} does not match sources.json url {expected_url!r}")

    for num in sorted(source_nums):
        count = seen_ref_nums.count(num)
        if count == 0:
            problems.append(f"missing reference line for source [{num}]")
        elif count > 1:
            problems.append(f"duplicate reference line for source [{num}] ({count} times)")

    for ref_n in seen_ref_nums:
        if ref_n not in source_nums:
            problems.append(f"reference line [{ref_n}] does not correspond to any source in sources.json")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
