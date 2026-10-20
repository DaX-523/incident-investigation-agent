"""LLM grader for what lexical matching cannot decide.

Word overlap cannot tell whether a report presents the answer key's root cause as *the* cause or only
mentions it as an amplifier or trigger, whether it adopts a red herring, or whether it actually states a
contributing factor rather than naming a related metric. The grader reads the answer key, so it lives in
evals/ and never runs during an investigation. Pick the model with JUDGE_MODEL (provider:model); use one
judge for every run in a comparison so both systems are graded the same way.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from dataset.schema import AnswerKey
from investigator.llm.metering import usage_from_message
from investigator.report import InvestigationReport

DEFAULT_JUDGE = "openai:gpt-5.4-mini"
# Bump when PROMPT or JudgeVerdict changes, so cached verdicts from an older rubric are re-graded.
JUDGE_VERSION = 3
MAX_RESULT_CHARS = 2_500
MAX_RESULTS_CHARS = 150_000


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_cause: Literal["correct", "partial", "wrong"]
    root_cause_reason: str = Field(description="One or two sentences: what the report names as the cause and why that grade.")
    red_herrings_adopted: list[int] = Field(description="Numbers of the red herrings the report adopts.")
    must_not_violated: list[int] = Field(description="Numbers of the must-not items the report violates.")
    factors_stated: list[int] = Field(description="Numbers of the must-mention factors the report states.")
    unsupported_claims: list[str] = Field(
        [], description="Specific factual claims the report presents as observed that no tool result supports.")


PROMPT = """\
You grade an incident-investigation report against the answer key written by the scenario's author.
Grade only what the report concludes and recommends (root cause, summary, hypotheses, actions). Facts
that appear in its evidence but are not part of its conclusions earn no credit. Wording may differ
from the key; judge the substance.

root_cause:
- correct: the report's root cause (or its rank-1 hypothesis if no root cause is stated) is the key's
  root cause: the same originating change or component and the same mechanism.
- partial: it names the right component and mechanism but misses the specific originating change,
  or it names the key's cause only as co-primary with a different cause.
- wrong: it names a different primary cause; or it attributes the right kind of mechanism to a
  different component (above all one the key lists as a red herring); or it mentions the key's
  cause only as a contributing factor, amplifier, trigger, symptom or side effect; or it makes no
  root-cause claim.

red_herrings_adopted: the report presents the claim as the root cause or recommends acting on it as
its primary fix. Mentioning a claim and rejecting it is not adopting it.
must_not_violated: the report does what the item forbids, in its root cause or its top-ranked actions.
factors_stated: the report states the factor's substance. Citing a related metric or log line without
stating the factor does not count.
unsupported_claims: compare the report with the INCIDENT REPORT and the TOOL RESULTS, which together are
everything the investigation was given or observed. List each specific factual claim (a value, setting, default, event or behaviour) that the
report presents as observed but that no tool result shows. Claims the report labels as assumptions or
hypotheses are fine. Quote the claim briefly. Return an empty list when tool results are not given.

Use the item numbers shown. Return empty lists when nothing applies."""


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{n}. {text}" for n, text in enumerate(items, start=1)) or "(none)"


def render_key(key: AnswerKey) -> str:
    gt = key.ground_truth
    return "\n\n".join([
        f"Root cause: {gt.root_cause.statement}\nEntity: {gt.root_cause.entity}\n"
        f"Also acceptable wording: {'; '.join(gt.root_cause.aliases)}",
        "Red herrings:\n" + _numbered([r.claim for r in gt.red_herrings]),
        "Must not:\n" + _numbered(gt.must_not),
        "Must-mention contributing factors:\n" + _numbered([f.statement for f in gt.contributing_factors if f.must_mention]),
    ])


def render_report(report: InvestigationReport) -> str:
    hyps = "\n".join(f"{h.rank}. [{h.status}, {h.confidence}] {h.statement}" for h in report.hypotheses)
    actions = "\n".join(f"{a.rank}. [{a.category}] {a.action}" for a in report.corrective_actions)
    return "\n\n".join([
        f"Outcome: {report.outcome} (confidence {report.confidence})",
        f"Root cause: {report.root_cause or '(none stated)'}",
        f"Summary: {report.summary}",
        f"Ranked hypotheses:\n{hyps or '(none)'}",
        f"Corrective actions:\n{actions or '(none)'}",
        "Missing data:\n" + ("\n".join(f"- {m}" for m in report.missing_data) or "(none)"),
    ])


def render_calls(calls: list[dict[str, Any]] | None) -> str:
    if not calls:
        return "(not provided)"
    budget = max(300, min(MAX_RESULT_CHARS, MAX_RESULTS_CHARS // max(1, len(calls))))
    lines = []
    for c in calls:
        head = f"[{c.get('call_id')}] {c.get('tool')} {json.dumps(c.get('args') or {}, ensure_ascii=False)}"
        if not c.get("ok"):
            lines.append(f"{head} -> failed ({c.get('error')})")
            continue
        body = json.dumps(c.get("data"), ensure_ascii=False, default=str) if "data" in c else "(result not recorded)"
        lines.append(f"{head} -> {body[:budget]}{' …' if len(body) > budget else ''}")
    return "\n".join(lines)


def judge_label(env: dict[str, str] | None = None) -> str:
    return (env if env is not None else os.environ).get("JUDGE_MODEL", DEFAULT_JUDGE)


class Judge:
    def __init__(self, ref: str, model=None):
        self.ref = ref
        if model is None:
            from investigator.llm.config import ModelSpec
            from investigator.llm.factory import build_chat_model

            model = build_chat_model(ModelSpec.parse(ref, "medium", max_tokens=4000))
        self._runnable = model.with_structured_output(JudgeVerdict, method="json_schema", include_raw=True)

    def __call__(self, report: InvestigationReport, key: AnswerKey, calls: list[dict[str, Any]] | None = None,
                 incident: str | None = None) -> tuple[JudgeVerdict, float]:
        """The verdict and what the judge call cost in USD (0.0 if the model is unpriced)."""
        messages = [SystemMessage(PROMPT),
                    HumanMessage(f"ANSWER KEY\n\n{render_key(key)}\n\n---\n\nREPORT\n\n{render_report(report)}"
                                 f"\n\n---\n\nINCIDENT REPORT (given to the investigator)\n\n{incident or '(not provided)'}"
                                 f"\n\n---\n\nTOOL RESULTS\n\n{render_calls(calls)}")]
        out = self._runnable.invoke(messages)
        if out.get("parsed") is None:
            raise ValueError(f"judge output could not be parsed: {out.get('parsing_error')}")
        parsed = out["parsed"]
        verdict = parsed if isinstance(parsed, JudgeVerdict) else JudgeVerdict.model_validate(parsed)
        _, usage = usage_from_message(out["raw"], self.ref.partition(":")[2])
        return verdict, usage.cost_usd
