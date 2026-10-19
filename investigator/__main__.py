"""CLI: run one incident investigation and print the report plus token/cost breakdown.

    uv run python -m investigator <scenario-id>                 # LLM agents, preset from INVESTIGATOR_PRESET (default budget)
    uv run python -m investigator <scenario-id> --preset balanced
    uv run python -m investigator <scenario-id> --stub          # offline, no LLM calls
    uv run python -m investigator <scenario-id> --system baseline   # single-agent baseline (same tools/budgets)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from investigator.runner import RunResult
from investigator.scenario import load_scenario, scenario_ids
from investigator.schemas import InvestigationConfig, Usage
from investigator.systems import SYSTEMS, model_labels, resolve_preset, result_to_dict, run_system

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def _usage_row(name: str, u: Usage) -> str:
    return (f"{name:<16} {u.llm_calls:>5} {u.input_tokens:>9} {u.cache_read_tokens:>8} {u.output_tokens:>8} "
            f"{u.reasoning_tokens:>8} {u.tool_calls:>5} {u.cost_usd:>10.4f}" + ("  *unpriced" if u.unpriced_calls else ""))


def print_result(result: RunResult) -> None:
    report = result.report
    print(f"\n=== {report.incident_title}")
    print(f"outcome: {report.outcome} | confidence: {report.confidence} | stop: {report.stop_reason}")
    if report.root_cause:
        print(f"root cause: {report.root_cause}")
    print(f"\n{report.summary}\n")
    for h in report.hypotheses[:5]:
        print(f"  #{h.rank} [{h.status}/{h.confidence}] {h.statement}  (+{h.supporting_evidence} -{h.contradicting_evidence})")
    if report.corrective_actions:
        print("\nactions:")
        for a in report.corrective_actions:
            print(f"  {a.rank}. [{a.category}] {a.action}")
    if report.missing_data:
        print("\nmissing data:", *report.missing_data, sep="\n  - ")
    if report.gaps:
        print(f"\ngaps: {len(report.gaps)} (tool failures / data quality / unchecked)")

    header = f"{'':<16} {'calls':>5} {'input':>9} {'cache_rd':>8} {'output':>8} {'thinking':>8} {'tools':>5} {'cost_usd':>10}"
    print("\nusage by agent\n" + header)
    for name, u in sorted(result.by_agent.items()):
        print(_usage_row(name, u))
    if result.by_model:
        print("\nusage by model\n" + header)
        for name, u in sorted(result.by_model.items()):
            print(_usage_row(name, u))
    print("\n" + _usage_row("TOTAL", result.usage))
    print(f"latency: {result.latency_s:.1f}s | tool calls executed: {result.tool_calls}")


def save(result: RunResult, scenario_id: str, label: str) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{scenario_id}-{label}.json"
    path.write_text(json.dumps({"scenario": scenario_id, "preset": label, **result_to_dict(result)}, indent=2, default=str))
    return path


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="investigator")
    parser.add_argument("scenario", choices=scenario_ids())
    parser.add_argument("--preset", choices=["budget", "balanced", "quality"], default=None)
    parser.add_argument("--stub", action="store_true", help="offline run with deterministic stub workers")
    parser.add_argument("--system", choices=SYSTEMS, default="multi_agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    preset = resolve_preset(args.preset, args.stub)
    if labels := model_labels(args.system, preset):
        print("models:", labels)
    result = run_system(load_scenario(args.scenario), args.system, preset, InvestigationConfig())
    print_result(result)
    print(f"\nsaved: {save(result, args.scenario, f'{args.system}-{preset}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
