"""Compare the multi-agent system with the single-agent baseline on the incident dataset.

    uv run python -m evals.compare --stub                         # offline dry run, no LLM calls
    uv run python -m evals.compare --preset budget                # every scenario x both systems
    uv run python -m evals.compare --scenarios inc-001 inc-002 --systems baseline
    uv run python -m evals.compare --out evals/results/<dir>      # resume: finished runs are skipped
    uv run python -m evals.compare --rescore evals/results/<dir>  # re-score saved runs (judge calls only)
    uv run python -m evals.compare --rescore evals/results/<dir> --judge none   # deterministic only

Each run is saved under <out>/runs/ as soon as it finishes; results.json and results.md are
rewritten from the saved runs. An LLM judge (evals/judge.py, JUDGE_MODEL) grades root cause, red
herrings, must-not items and factors; its verdicts are cached in <out>/judgements/ per judge model.
Manual decisions on "review" verdicts go in <out>/reviews.yaml as `inc-001/baseline: correct` and are
applied on the next --rescore.
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
from evals.judge import Judge, JudgeVerdict, judge_label
from evals.scoring import Score, score_run
from investigator.report import InvestigationReport
from investigator.scenario import load_scenario, scenario_ids
from investigator.llm.config import PROVIDER_MODELS, resolve_provider
from investigator.systems import SYSTEMS, model_labels, resolve_preset, result_to_dict, run_system

RESULTS_DIR = Path(__file__).resolve().parent / "results"
log = logging.getLogger("evals.compare")


def run_one(scenario_id: str, system: str, preset: str, runs_dir: Path, provider: str | None = None) -> Path:
    path = runs_dir / f"{scenario_id}-{system}.json"
    record: dict[str, Any] = {"scenario": scenario_id, "system": system, "preset": preset, "provider": provider}
    try:
        record |= result_to_dict(run_system(load_scenario(scenario_id), system, preset, provider=provider))
    except Exception as exc:  # one broken run must not sink the sweep
        record |= {"report": None, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def load_reviews(out: Path) -> dict[str, str]:
    path = out / "reviews.yaml"
    return yaml.safe_load(path.read_text()) or {} if path.exists() else {}


def _judgements(out: Path, runs: dict[Path, dict], keys: dict[str, AnswerKey], judge: Judge | None,
                parallel: int) -> dict[Path, JudgeVerdict | str]:
    """Judge verdict (or an error string) per run, reusing cached verdicts from the same judge model."""
    if judge is None:
        return {}
    cache_dir = out / "judgements"
    cache_dir.mkdir(exist_ok=True)
    found: dict[Path, JudgeVerdict | str] = {}
    todo = []
    for path, run in runs.items():
        cached = cache_dir / path.name
        if cached.exists() and (data := json.loads(cached.read_text())).get("model") == judge.ref:
            found[path] = JudgeVerdict.model_validate(data["verdict"])
        elif run.get("report"):
            todo.append(path)

    def grade(path: Path) -> JudgeVerdict | str:
        run = runs[path]
        try:
            verdict, cost = judge(InvestigationReport.model_validate(run["report"]), keys[run["scenario"]])
        except Exception as exc:  # a judge failure falls back to lexical scoring for that run
            return f"{type(exc).__name__}: {exc}"
        (cache_dir / path.name).write_text(json.dumps(
            {"model": judge.ref, "cost_usd": cost, "verdict": verdict.model_dump()}, indent=2))
        return verdict

    if todo:
        log.info("judging %d runs with %s", len(todo), judge.ref)
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            found.update(zip(todo, pool.map(grade, todo)))
    return found


def score_dir(out: Path, judge: Judge | None = None, parallel: int = 4) -> list[Score]:
    keys: dict[str, AnswerKey] = {}
    reviews = load_reviews(out)
    runs = {path: json.loads(path.read_text()) for path in sorted((out / "runs").glob("*.json"))}
    for run in runs.values():
        keys.setdefault(run["scenario"], load_answer_key(ANSWER_KEYS_DIR / f"{run['scenario']}.yaml"))
    judged = _judgements(out, runs, keys, judge, parallel)
    scores = []
    for path, run in runs.items():
        sid, system = run["scenario"], run["system"]
        verdict = judged.get(path)
        score = score_run(run, keys[sid], system, verdict if isinstance(verdict, JudgeVerdict) else None,
                          judge.ref if judge else None)
        if isinstance(verdict, str):
            score.judge_error = verdict
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
        judged = [x for x in s if x.judge_root_cause is not None]
        out[system] = {
            "runs": n,
            "root_cause_accuracy": round(correct / n, 3),
            "root_cause_accuracy_lexical": round(sum(x.lexical_verdict == "correct" for x in s) / n, 3),
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
            "must_not_violations": sum(bool(x.must_not_violated) for x in s) if judged else None,
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
    setup = meta["preset"] + (f", {meta['provider']}" if meta.get("provider") else "")
    lines = [f"# Multi-agent vs single-agent baseline ({setup})", "",
             f"Generated {meta['generated_at']}. Scenarios: {meta['scenarios']}. "
             "Scored against `dataset/answer_keys/`: deterministic checks in `evals/scoring.py`"
             + (f"; root cause, red herrings, must-not and factors graded by LLM judge `{meta['judge']}` "
                "(`evals/judge.py`)." if meta.get("judge") else "; no LLM judge (lexical root-cause matching)."),
             ""]
    if not all(x.evidence_verified for x in scores):
        lines += ["Some runs were saved without tool data, so their key evidence is checked by tool and "
                  "arguments only (`evidence_verified: false` in results.json).", ""]
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

    flagged = [x for x in scores if x.verdict != "correct" or x.red_herrings_matched or x.must_not_violated
               or x.judge_error or (x.judge_root_cause and x.lexical_verdict != x.verdict)]
    if flagged:
        lines += ["", "## Not correct, or worth a look", ""]
        for x in flagged:
            lines.append(f"- **{x.scenario}/{x.system}** ({x.verdict}; lexical {x.lexical_verdict}, alias match "
                         f"{x.match_score}): {x.error or x.root_cause or '(no root cause)'}")
            if x.judge_reason:
                lines.append(f"  - judge: {x.judge_reason}")
            if x.red_herrings_matched:
                lines.append(f"  - adopted red herring: {x.red_herrings_matched}")
            if x.must_not_violated:
                lines.append(f"  - violated must-not: {x.must_not_violated}")
            if x.judge_error:
                lines.append(f"  - judge failed, lexical verdict used: {x.judge_error}")
    lines += ["", "## Metric definitions", "",
              "- **root_cause_accuracy**: share of runs whose root cause the judge grades as the answer key's root "
              "cause (not merely mentioned as an amplifier or trigger); for expected-inconclusive scenarios, share that "
              "reported inconclusive. `review` = judge \"partial\" (or, without a judge, a partial alias match), "
              "decided manually in `reviews.yaml`. **root_cause_accuracy_lexical**: the same by alias word overlap only.",
              "- **fell_for_red_herring / must_not_violations**: runs that adopt a red herring as cause or primary fix / "
              "do something the key forbids (judge).",
              "- **claims_grounded**: share of supported/refuted hypotheses and corrective actions citing at least one "
              "evidence id that traces to an executed tool call. **citations_valid**: share of all cited ids that do.",
              "- **key_evidence_retrieved / cited**: share of the answer key's required evidence the system fetched (a "
              "successful call whose returned data contains the evidence) / cited in its report.",
              "- **factors_mentioned**: share of must-mention contributing factors the report states (judge; lexical "
              "word overlap without a judge).",
              "- Tokens include cache reads/writes; cost uses `investigator/llm/pricing.py`. Latency is wall clock per run."]
    return "\n".join(lines) + "\n"


def write_results(out: Path, meta: dict[str, Any], judge: Judge | None = None) -> tuple[dict, list[Score]]:
    scores = score_dir(out, judge)
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
    parser.add_argument("--provider", choices=sorted(PROVIDER_MODELS), default=None,
                        help="model provider (default: INVESTIGATOR_PROVIDER, then anthropic)")
    parser.add_argument("--stub", action="store_true", help="deterministic stub workers, no LLM calls")
    parser.add_argument("--parallel", type=int, default=1, help="concurrent runs (inflates per-run latency)")
    parser.add_argument("--out", type=Path, help="results directory; existing runs in it are skipped")
    parser.add_argument("--rescore", type=Path, metavar="DIR", help="re-score saved runs in DIR (judge calls only)")
    parser.add_argument("--judge", default=None, metavar="PROVIDER:MODEL|none",
                        help="LLM judge for root cause (default: JUDGE_MODEL or %(prog)s's default; none with --stub)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.rescore:
        out = args.rescore
        meta = json.loads((out / "results.json").read_text())["meta"] if (out / "results.json").exists() else {}
    else:
        preset = resolve_preset(args.preset, args.stub)
        provider = None if args.stub else resolve_provider(args.provider)
        out = args.out or RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{preset}"
        runs_dir = out / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(sid, system) for sid in args.scenarios for system in args.systems
                if not (runs_dir / f"{sid}-{system}.json").exists()]
        log.info("%s: %d runs to do (%d already saved), preset=%s, provider=%s", out, len(jobs),
                 len(args.scenarios) * len(args.systems) - len(jobs), preset, provider)
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            futures = {pool.submit(run_one, sid, system, preset, runs_dir, provider): (sid, system)
                       for sid, system in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                run = json.loads(future.result().read_text())
                cost = (run.get("usage") or {}).get("cost_usd", 0.0)
                log.info("[%d/%d] %s/%s %s %.1fs $%.4f", i, len(jobs), *futures[future],
                         run.get("error") or (run["report"] or {}).get("outcome"), run.get("latency_s", 0.0), cost)
        meta = {"preset": preset, "provider": provider, "parallel": args.parallel,
                "models": {s: model_labels(s, preset, provider) for s in args.systems}}

    meta |= {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "scenarios": len({p.name.rsplit("-", 1)[0] for p in (out / "runs").glob("*.json")})}
    meta.setdefault("preset", "?")
    meta.setdefault("models", {})
    ref = args.judge or ("none" if args.stub else judge_label())
    judge = None
    if ref.lower() != "none":
        try:
            judge = Judge(ref)
        except Exception as exc:  # e.g. no API key for the judge's provider: score without it, say so
            log.warning("LLM judge %s unavailable (%s: %s); falling back to lexical root-cause scoring",
                        ref, type(exc).__name__, exc)
    meta["judge"] = judge.ref if judge else None
    agg, _ = write_results(out, meta, judge)
    for system, row in agg.items():
        log.info("%s: accuracy %.0f%% (+%d review), grounded %s, tools %.1f, latency %.1fs, $%.4f total",
                 system, 100 * row["root_cause_accuracy"], row["needs_review"], row["claims_grounded"],
                 row["tool_calls_mean"], row["latency_s_mean"], row["cost_usd_total"])
    log.info("wrote %s", out / "results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
