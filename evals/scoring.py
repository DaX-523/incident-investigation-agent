"""Scoring of one saved run against its answer key.

Deterministic checks cover outcome, grounding and key evidence (against the data each call returned).
Root-cause matching here is lexical (token coverage of the answer key's aliases), which cannot tell a
cause from an amplifier; when an LLM judgement (evals/judge.py) is supplied it decides the root-cause
verdict, red herrings, must-not violations and factors, and the lexical result is kept for reference.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from dataset.schema import AnswerKey, KeyEvidence
from evals.judge import JudgeVerdict
from investigator.report import InvestigationReport

Verdict = Literal["correct", "review", "wrong", "error"]

# Alias coverage at or above MATCH is a match; between REVIEW and MATCH needs a human look.
MATCH = 0.6
REVIEW = 0.34
# A contributing factor counts as mentioned when this share of its words appears in the report.
FACTOR_MENTION = 0.5

_WORD = re.compile(r"[a-z0-9]+(?:[._:=/-][a-z0-9]+)*")
_SPLIT = re.compile(r"[._:=/-]")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "its", "of",
    "on", "or", "so", "than", "that", "the", "their", "this", "to", "was", "were", "which", "with",
    "cause", "caused", "causes", "causing", "root", "due", "because", "after", "before",
}


def _stem(word: str) -> str:
    # Crude stem so "carts"/"cart", "crawler"/"crawling" or "expired"/"expiry" meet.
    if not word.isalpha():
        return word
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word[:5]


def phrase_tokens(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def text_tokens(text: str) -> set[str]:
    """Like phrase_tokens, plus the parts of compound words ("checkout-svc" -> checkout, svc)."""
    out = set()
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        out.add(_stem(w))
        out |= _parts(w)
    return out


def _parts(word: str) -> set[str]:
    return {_stem(p) for p in _SPLIT.split(word) if len(p) > 1 and p not in _STOP}


def coverage(phrase: str, tokens: set[str]) -> float:
    want = phrase_tokens(phrase)
    # A compound ("ip-10-0-3-17") also counts when all its parts appear, e.g. inside "node-ip-10-0-3-17".
    hits = sum(w in tokens or (_SPLIT.search(w) is not None and _parts(w) <= tokens) for w in want)
    return hits / len(want) if want else 0.0


def factor_mentioned(statement: str, tokens: set[str]) -> bool:
    """Most words present, and if the factor names identifiers (tickets, hosts, versions), one of them."""
    ids = {t for t in phrase_tokens(statement) if len(t) > 2 and any(ch.isdigit() for ch in t)}
    return coverage(statement, tokens) >= FACTOR_MENTION and (not ids or bool(ids & tokens))


def best_match(phrases: list[str], text: str) -> tuple[float, str | None]:
    tokens = text_tokens(text)
    scored = [(coverage(p, tokens), p) for p in phrases]
    return max(scored, default=(0.0, None))


def evidence_matches(key: KeyEvidence, call: dict[str, Any]) -> bool:
    """A call to the same tool whose args don't contradict the key's (an omitted arg is a broader query,
    so it covers the key) and whose returned data holds the evidence (`must_contain`). Runs saved before
    tool data was recorded fall back to the args check; see `evidence_verified`."""
    if call["tool"] != key.tool:
        return False
    args = call.get("args") or {}
    if not all(args.get(k) is None or str(args[k]).lower() == str(v).lower() for k, v in key.args.items()):
        return False
    if key.attempt_only:
        return True
    if not call["ok"]:
        return False
    if key.must_contain and "data" in call:
        text = json.dumps(call["data"], ensure_ascii=False, default=str).lower()
        return all(needle.lower() in text for needle in key.must_contain)
    return True


def report_text(report: InvestigationReport) -> str:
    parts = [report.summary, report.root_cause or ""]
    for h in report.hypotheses:
        parts += [h.statement, *(f.text for f in h.facts), *h.assumptions]
    parts += [a.action + " " + a.rationale for a in report.corrective_actions]
    parts += report.missing_data + report.security_findings
    return "\n".join(parts)


@dataclass
class Score:
    scenario: str
    system: str
    tag: str
    expected_outcome: str
    expected_confidence: str
    verdict: Verdict
    outcome: str | None = None
    confidence: str | None = None
    outcome_ok: bool = False
    root_cause: str | None = None
    match_score: float = 0.0
    matched_alias: str | None = None
    top_hypothesis_match: float = 0.0
    red_herrings_matched: list[str] = field(default_factory=list)
    key_evidence_retrieved: float | None = None
    evidence_verified: bool = True  # False: run saved without tool data, so must_contain was not checked
    key_evidence_cited: float | None = None
    citations_valid: float | None = None
    claims_grounded: float | None = None
    factors_mentioned: float | None = None
    factors_mentioned_lexical: float | None = None
    lexical_verdict: Verdict | None = None
    judge_model: str | None = None
    judge_root_cause: str | None = None
    judge_reason: str | None = None
    judge_error: str | None = None
    must_not_violated: list[str] = field(default_factory=list)
    security_expected: bool = False
    security_flagged: bool | None = None
    stop_reason: str | None = None
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def score_run(run: dict[str, Any], key: AnswerKey, system: str, judgement: JudgeVerdict | None = None,
              judge_model: str | None = None) -> Score:
    gt = key.ground_truth
    usage = run.get("usage") or {}
    score = Score(
        scenario=key.id, system=system, tag=key.metadata.primary_tag,
        expected_outcome=gt.expected_outcome, expected_confidence=gt.expected_confidence,
        verdict="error", security_expected=bool(gt.security_findings),
        stop_reason=run.get("stop_reason"), tool_calls=run.get("tool_calls", 0),
        llm_calls=usage.get("llm_calls", 0), input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0), cost_usd=round(usage.get("cost_usd", 0.0), 5),
        latency_s=run.get("latency_s", 0.0), error=run.get("error"),
    )
    if not run.get("report"):
        score.error = score.error or "no report"
        return score

    report = InvestigationReport.model_validate(run["report"])
    calls = run.get("calls", [])
    phrases = [*gt.root_cause.aliases, gt.root_cause.entity]
    top = min(report.hypotheses, key=lambda h: h.rank, default=None)

    score.outcome, score.confidence, score.root_cause = report.outcome, report.confidence, report.root_cause
    score.outcome_ok = report.outcome == gt.expected_outcome
    score.top_hypothesis_match = round(best_match(phrases, top.statement if top else "")[0], 3)

    # Root cause: an expected-inconclusive case is right only when the report declines to name one.
    if gt.expected_outcome == "inconclusive":
        score.verdict = "correct" if report.outcome == "inconclusive" else "wrong"
        claimed = " ".join(filter(None, [report.root_cause, top.statement if top else None]))
        score.match_score, score.matched_alias = best_match(phrases, claimed)
    elif report.outcome == "inconclusive":
        score.verdict = "wrong"
    else:
        claimed = " ".join(filter(None, [report.root_cause, top.statement if top else None]))
        score.match_score, score.matched_alias = best_match(phrases, claimed)
        score.verdict = "correct" if score.match_score >= MATCH else "review" if score.match_score >= REVIEW else "wrong"
        if score.verdict != "correct":
            claimed_tokens = text_tokens(claimed)
            score.red_herrings_matched = [r.claim for r in gt.red_herrings if coverage(r.claim, claimed_tokens) >= MATCH]
    score.match_score = round(score.match_score, 3)

    # Evidence grounding: cited evidence must exist and trace back to tool calls that actually ran.
    call_ids = {c["call_id"] for c in calls}
    valid = {e.id for e in report.evidence_index if e.source_call_ids and set(e.source_call_ids) <= call_ids}
    claims = [h.supporting_evidence + h.contradicting_evidence + [i for f in h.facts for i in f.evidence_ids]
              for h in report.hypotheses if h.status in ("supported", "refuted")]
    claims += [a.evidence_ids for a in report.corrective_actions]
    refs = [i for ids in claims for i in ids]
    score.citations_valid = _ratio(sum(i in valid for i in refs), len(refs))
    score.claims_grounded = _ratio(sum(any(i in valid for i in ids) for ids in claims), len(claims))

    score.evidence_verified = all("data" in c for c in calls)
    required = [k for k in gt.key_evidence if k.required]
    cited_calls = {c for e in report.evidence_index for c in e.source_call_ids}
    matched = [[c["call_id"] for c in calls if evidence_matches(k, c)] for k in required]
    score.key_evidence_retrieved = _ratio(sum(bool(m) for m in matched), len(required))
    score.key_evidence_cited = _ratio(sum(bool(set(m) & cited_calls) for m in matched), len(required))

    factors = [f.statement for f in gt.contributing_factors if f.must_mention]
    tokens = text_tokens(report_text(report))
    score.factors_mentioned = score.factors_mentioned_lexical = _ratio(
        sum(factor_mentioned(f, tokens) for f in factors), len(factors))
    if score.security_expected:
        score.security_flagged = bool(report.security_findings)
    score.lexical_verdict = score.verdict
    if judgement is not None:
        _apply_judgement(score, judgement, gt, report, factors, judge_model)
    return score


def _apply_judgement(score: Score, j: JudgeVerdict, gt, report: InvestigationReport, factors: list[str],
                     judge_model: str | None) -> None:
    def pick(items: list[str], numbers: list[int]) -> list[str]:
        return [items[n - 1] for n in dict.fromkeys(numbers) if 1 <= n <= len(items)]

    score.judge_model, score.judge_root_cause, score.judge_reason = judge_model, j.root_cause, j.root_cause_reason
    score.red_herrings_matched = pick([r.claim for r in gt.red_herrings], j.red_herrings_adopted)
    score.must_not_violated = pick(gt.must_not, j.must_not_violated)
    score.factors_mentioned = _ratio(len(pick(factors, j.factors_stated)), len(factors))
    # Outcome-based verdicts (expected inconclusive, or a conclusive case reported inconclusive) stand.
    if gt.expected_outcome == "conclusive" and report.outcome == "conclusive":
        score.verdict = {"correct": "correct", "partial": "review", "wrong": "wrong"}[j.root_cause]
