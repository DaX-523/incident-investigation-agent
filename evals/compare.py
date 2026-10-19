"""Compare the multi-agent system with the single-agent baseline on the incident dataset.

    uv run python -m evals.compare --stub                         # offline dry run, no LLM calls
    uv run python -m evals.compare --preset budget                # every scenario x both systems
    uv run python -m evals.compare --scenarios inc-001 inc-002 --systems baseline
    uv run python -m evals.compare --out evals/results/<dir>      # resume: finished runs are skipped
    uv run python -m evals.compare --rescore evals/results/<dir>  # re-score saved runs, no model calls

Each run is saved under <out>/runs/ as soon as it finishes; results.json and results.md are
rewritten from the saved runs. Manual decisions on "review" verdicts go in <out>/reviews.yaml as
`inc-001/baseline: correct` and are applied on the next --rescore.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from dataset.schema import AnswerKey
from dataset.validate import ANSWER_KEYS_DIR, load_answer_key
from evals.scoring import Score, score_run
from investigator.scenario import load_scenario, scenario_ids
from investigator.systems import SYSTEMS, model_labels, resolve_preset, result_to_dict, run_system

RESULTS_DIR = Path(__file__).resolve().parent / "results"
log = logging.getLogger("evals.compare")


def run_one(scenario_id: str, system: str, preset: str, runs_dir: Path) -> Path:
    path = runs_dir / f"{scenario_id}-{system}.json"
    record: dict[str, Any] = {"scenario": scenario_id, "system": system, "preset": preset}
    try:
        record |= result_to_dict(run_system(load_scenario(scenario_id), system, preset))
    except Exception as exc:  # one broken run must not sink the sweep
        record |= {"report": None, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def load_reviews(out: Path) -> dict[str, str]:
    path = out / "reviews.yaml"
    return yaml.safe_load(path.read_text()) or {} if path.exists() else {}


def score_dir(out: Path) -> list[Score]:
    keys: dict[str, AnswerKey] = {}
    reviews = load_reviews(out)
    scores = []
    for path in sorted((out / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        sid, system = run["scenario"], run["system"]
        keys.setdefault(sid, load_answer_key(ANSWER_KEYS_DIR / f"{sid}.yaml"))
        score = score_run(run, keys[sid], system)
        if (decision := reviews.get(f"{sid}/{system}")) in ("correct", "wrong") and score.verdict == "review":
            score.verdict = decision
        scores.append(score)
    return scores


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), 3) if present else None


def aggregate(scores: list[Score]) -> dict[str, dict[str, Any]]:
    out = {}
    for system in SYSTEMS:
        s = [x for x in scores if x.system == system]
        if not s:
            continue
        n = len(s)
        correct = sum(x.verdict == "correct" for x in s)
        out[system] = {
            "runs": n,
            "root_cause_accuracy": round(correct / n, 3),
            "root_cause_accuracy_incl_review": round((correct + sum(x.verdict == "review" for x in s)) / n, 3),
            "needs_review": sum(x.verdict == "review" for x in s),
            "errors": sum(x.verdict == "error" for x in s),
            "outcome_correct": round(sum(x.outcome_ok for x in s) / n, 3),
            "claims_grounded": _mean([x.claims_grounded for x in s]),
            "citations_valid": _mean([x.citations_valid for x in s]),
            "key_evidence_retrieved": _mean([x.key_evidence_retrieved for x in s]),
            "key_evidence_cited": _mean([x.key_evidence_cited for x in s]),
            "factors_mentioned": _mean([x.factors_mentioned for x in s]),
            "fell_for_red_herring": sum(bool(x.red_herrings_matched) for x in s),
            "security_flagged": f"{sum(bool(x.security_flagged) for x in s)}/{sum(x.security_expected for x in s)}",
            "tool_calls_mean": _mean([x.tool_calls for x in s]),
            "llm_calls_mean": _mean([x.llm_calls for x in s]),
            "latency_s_mean": _mean([x.latency_s for x in s]),
            "latency_s_p50": round(statistics.median(x.latency_s for x in s), 1),
            "input_tokens_total": sum(x.input_tokens for x in s),
            "output_tokens_total": sum(x.output_tokens for x in s),
            "tokens_mean": round(sum(x.input_tokens + x.output_tokens for x in s) / n),
            "cost_usd_total": round(sum(x.cost_usd for x in s), 4),
            "cost_usd_mean": round(sum(x.cost_usd for x in s) / n, 4),
        }
    return out


def by_tag(scores: list[Score]) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    for tag in sorted({x.tag for x in scores}):
        for system in SYSTEMS:
            s = [x for x in scores if x.tag == tag and x.system == system]
            if s:
                table.setdefault(tag, {})[system] = f"{sum(x.verdict == 'correct' for x in s)}/{len(s)}"
    return table


_ICON = {"correct": "✅", "review": "🟡", "wrong": "❌", "error": "💥"}


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.2f}" if v < 10 else f"{v:.1f}"
    return str(v)


def render_markdown(meta: dict[str, Any], agg: dict, tags: dict, scores: list[Score]) -> str:
    systems = [s for s in SYSTEMS if s in agg]
    lines = [f"# Multi-agent vs single-agent baseline ({meta['preset']})", "",
             f"Generated {meta['generated_at']}. Scenarios: {meta['scenarios']}. "
             "Scored deterministically against `dataset/answer_keys/` (see `evals/scoring.py`).", ""]
    for system, labels in meta["models"].items():
        if labels:
            lines.append(f"- **{system}** models: " + ", ".join(f"{r}={m}" for r, m in labels.items()))
    lines += ["", "## Summary", "", "| metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for metric in next(iter(agg.values())):
        lines.append(f"| {metric} | " + " | ".join(_fmt(agg[s][metric]) for s in systems) + " |")

    lines += ["", "## Root-cause accuracy by scenario type", "", "| type | " + " | ".join(systems) + " |",
              "|---|" + "---|" * len(systems)]
    lines += [f"| {tag} | " + " | ".join(row.get(s, "–") for s in systems) + " |" for tag, row in tags.items()]

    lines += ["", "## Per scenario", "",
              "| scenario | type | expected | system | verdict | outcome/conf | key ev. cited | grounded | tools | latency s | tokens | cost $ |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(scores, key=lambda x: (x.scenario, SYSTEMS.index(x.system))):
        lines.append(
            f"| {x.scenario} | {x.tag} | {x.expected_outcome}/{x.expected_confidence} | {x.system} | "
            f"{_ICON[x.verdict]} {x.verdict} | {x.outcome or '–'}/{x.confidence or '–'} | {_fmt(x.key_evidence_cited)} | "
            f"{_fmt(x.claims_grounded)} | {x.tool_calls} | {x.latency_s:.1f} | {x.input_tokens + x.output_tokens} | {x.cost_usd:.4f} |")

    flagged = [x for x in scores if x.verdict in ("review", "error") or x.red_herrings_matched]
    if flagged:
        lines += ["", "## Needs a human look", ""]
        for x in flagged:
            lines.append(f"- **{x.scenario}/{x.system}** ({x.verdict}, alias match {x.match_score}): "
                         f"{x.error or x.root_cause or '(no root cause)'}")
            if x.red_herrings_matched:
                lines.append(f"  - matches red herring: {x.red_herrings_matched}")
    lines += ["", "## Metric definitions", "",
              "- **root_cause_accuracy**: share of runs whose root cause (or rank-1 hypothesis) matches an answer-key "
              "alias; for expected-inconclusive scenarios, share that reported inconclusive. `review` = partial lexical "
              "match, decided manually in `reviews.yaml`.",
              "- **claims_grounded**: share of supported/refuted hypotheses and corrective actions citing at least one "
              "evidence id that traces to an executed tool call. **citations_valid**: share of all cited ids that do.",
              "- **key_evidence_retrieved / cited**: share of the answer key's required evidence the system fetched with "
              "a successful tool call / cited in its report.",
              "- **factors_mentioned**: share of must-mention contributing factors found in the report (lexical, approximate).",
              "- Tokens include cache reads/writes; cost uses `investigator/llm/pricing.py`. Latency is wall clock per run."]
    return "\n".join(lines) + "\n"


def write_results(out: Path, meta: dict[str, Any]) -> tuple[dict, list[Score]]:
    scores = score_dir(out)
    agg, tags = aggregate(scores), by_tag(scores)
    (out / "results.json").write_text(json.dumps(
        {"meta": meta, "aggregate": agg, "by_tag": tags, "scores": [asdict(s) for s in scores]}, indent=2))
    (out / "results.md").write_text(render_markdown(meta, agg, tags, scores))
    return agg, scores


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="evals.compare")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--scenarios", nargs="+", choices=scenario_ids(), default=scenario_ids())
    parser.add_argument("--preset", choices=["budget", "balanced", "quality"], default=None)
    parser.add_argument("--stub", action="store_true", help="deterministic stub workers, no LLM calls")
    parser.add_argument("--parallel", type=int, default=1, help="concurrent runs (inflates per-run latency)")
    parser.add_argument("--out", type=Path, help="results directory; existing runs in it are skipped")
    parser.add_argument("--rescore", type=Path, metavar="DIR", help="re-score saved runs in DIR; no model calls")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.rescore:
        out = args.rescore
        meta = json.loads((out / "results.json").read_text())["meta"] if (out / "results.json").exists() else {}
    else:
        preset = resolve_preset(args.preset, args.stub)
        out = args.out or RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{preset}"
        runs_dir = out / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(sid, system) for sid in args.scenarios for system in args.systems
                if not (runs_dir / f"{sid}-{system}.json").exists()]
        log.info("%s: %d runs to do (%d already saved), preset=%s", out, len(jobs),
                 len(args.scenarios) * len(args.systems) - len(jobs), preset)
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            futures = {pool.submit(run_one, sid, system, preset, runs_dir): (sid, system) for sid, system in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                run = json.loads(future.result().read_text())
                cost = (run.get("usage") or {}).get("cost_usd", 0.0)
                log.info("[%d/%d] %s/%s %s %.1fs $%.4f", i, len(jobs), *futures[future],
                         run.get("error") or (run["report"] or {}).get("outcome"), run.get("latency_s", 0.0), cost)
        meta = {"preset": preset, "parallel": args.parallel,
                "models": {s: model_labels(s, preset) for s in args.systems}}

    meta |= {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "scenarios": len({p.name.rsplit("-", 1)[0] for p in (out / "runs").glob("*.json")})}
    meta.setdefault("preset", "?")
    meta.setdefault("models", {})
    agg, _ = write_results(out, meta)
    for system, row in agg.items():
        log.info("%s: accuracy %.0f%% (+%d review), grounded %s, tools %.1f, latency %.1fs, $%.4f total",
                 system, 100 * row["root_cause_accuracy"], row["needs_review"], row["claims_grounded"],
                 row["tool_calls_mean"], row["latency_s_mean"], row["cost_usd_total"])
    log.info("wrote %s", out / "results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
