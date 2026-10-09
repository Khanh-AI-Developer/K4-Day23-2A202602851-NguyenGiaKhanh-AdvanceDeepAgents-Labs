"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic or not isinstance(topic, str):
        return "topic"
    s = topic.strip().lower()
    s = re.sub(r"[^\w]+", "-", s).strip("-")
    s = s[:60].rstrip("-")
    return s if s else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct an in-depth academic research survey on the topic: '{topic}'.\n\n"
        "Requirements (execute step-by-step):\n"
        "1. Plan with `write_todos` and split into N >= 3 independent sub-questions.\n"
        "2. Delegate each sub-question to `researcher` subagents using `task(subagent_type='researcher', description=...)` in parallel. "
        "Instruct each researcher to write its notes to /tmp/work/research/notes/<NN>-<slug>.md.\n"
        "3. Read the generated note files and merge the sources into `/tmp/work/research/sources.json` using the `write_file` tool. "
        "Verify that at least 3 distinct source families (among arxiv, hf-daily, hf-search, web) are present.\n"
        "4. Write the comprehensive survey body to `/tmp/work/report/report.md` using the `write_file` tool (following REPORT_TEMPLATE.md). Do NOT write ## References.\n"
        "5. Execute `python3 /tmp/work/research/finalize_citations.py` using the `execute` tool to generate references.\n"
        "6. Execute `python3 /tmp/work/research/check_citations.py` using the `execute` tool. If any errors are reported, fix them until it prints 'OK'.\n"
        "7. Call `task(subagent_type='citation-checker', description=...)` to spot check key claims.\n"
        "8. Provide a final summary when done."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_counts = Counter()
    total_input = 0
    total_output = 0
    subagent_calls = 0

    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if calls is None and isinstance(msg, dict):
            calls = msg.get("tool_calls")
        if calls and isinstance(calls, list):
            for c in calls:
                name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
                if name:
                    tool_counts[name] += 1
                    if name == "task":
                        subagent_calls += 1

        usage = getattr(msg, "usage_metadata", None)
        if usage is None and isinstance(msg, dict):
            usage = msg.get("usage_metadata")
        if usage and isinstance(usage, dict):
            total_input += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            total_output += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": total_input,
            "output": total_output,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("Report is missing or empty in sandbox")
    if not sources_bytes:
        raise RuntimeError("sources.json is missing in sandbox")

    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}")

    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json is empty or not a list")

    summary = summarize(messages, elapsed, model_name)
    source_families = sorted({
        str(s.get("source", "")).strip()
        for s in sources
        if isinstance(s, dict) and s.get("source")
    })

    meta = dict(
        summary,
        topic=topic,
        n_sources=len(sources),
        source_families=source_families,
    )

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)

    report_file = reports_dir / f"{slug}.md"
    sources_file = reports_dir / f"{slug}.sources.json"
    meta_file = reports_dir / f"{slug}.meta.json"

    report_file.write_bytes(report_bytes)
    sources_file.write_bytes(sources_bytes)
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return report_file


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = (topic or "").strip()
    if not topic:
        print("Usage: python research.py <topic>", file=sys.stderr)
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "model")
    start = time.monotonic()

    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })
        agent = build_lead_agent(backend, model)
        print(f"[*] Starting research agent for topic: {topic!r}", flush=True)
        messages = []
        for event in agent.stream(
            {"messages": [{"role": "user", "content": build_prompt(topic)}]},
            config={"recursion_limit": 1000},
            stream_mode="values",
        ):
            if isinstance(event, dict) and "messages" in event and event["messages"]:
                messages = event["messages"]
                latest = messages[-1]
                calls = getattr(latest, "tool_calls", None)
                if calls:
                    for tc in calls:
                        print(f"-> [Lead Tool Call] {tc['name']}: {str(tc.get('args', ''))[:150]}", flush=True)
                else:
                    msg_type = getattr(latest, "type", "msg")
                    content_str = str(getattr(latest, "content", ""))
                    if msg_type == "tool":
                        print(f"<- [Tool Output] {getattr(latest, 'name', 'tool')}: {content_str[:150]}...", flush=True)
                    elif content_str:
                        print(f"[{msg_type}] {content_str[:150]}...", flush=True)

        elapsed = time.monotonic() - start
        try:
            out_path = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"[+] Report saved to: {out_path}", flush=True)
            return 0
        except RuntimeError as exc:
            print(f"[-] FAILED: {exc}", file=sys.stderr, flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
