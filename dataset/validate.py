"""Validate scenarios + answer keys and print a coverage matrix.

Usage: uv run python -m dataset.validate
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import yaml
from pydantic import ValidationError

from dataset.schema import AnswerKey, Scenario

DATASET_DIR = Path(__file__).parent
SCENARIOS_DIR = DATASET_DIR / "scenarios"
ANSWER_KEYS_DIR = DATASET_DIR / "answer_keys"

# Words that reveal a scenario's trap if they appear in agent-visible data.
HINT_WORDS = ("red herring", "ground truth", "root cause is", "misleading", "skewed", "stale", "answer key")


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(path.read_text()))


def load_answer_key(path: Path) -> AnswerKey:
    return AnswerKey.model_validate(yaml.safe_load(path.read_text()))


def scenario_paths() -> list[Path]:
    return sorted(SCENARIOS_DIR.glob("*.yaml"))


def check_scenario(path: Path, s: Scenario) -> list[str]:
    errs: list[str] = []
    env, w = s.environment, s.environment.window

    if s.id != path.stem:
        errs.append(f"id {s.id!r} != filename {path.stem!r}")
    if w.start >= w.end:
        errs.append("window start >= end")

    services = {svc.name for svc in env.topology.services}
    for svc in env.topology.services:
        for dep in svc.depends_on:
            if dep not in services:
                errs.append(f"topology: {svc.name} depends on unknown {dep}")

    def in_window(ts, what):
        if not (w.start <= ts <= w.end):
            errs.append(f"{what} at {ts.isoformat()} outside window")

    for e in env.logs:
        in_window(e.ts, f"log[{e.service}]")
        if e.service not in services:
            errs.append(f"log service {e.service!r} not in topology")
    for t in env.traces:
        in_window(t.start, f"trace {t.trace_id}")
        for sp in t.spans:
            if sp.service not in services:
                errs.append(f"trace {t.trace_id}: span service {sp.service!r} not in topology")
    for a in env.alerts:
        in_window(a.ts, f"alert {a.name}")
    for c in env.changes:
        if c.ts > w.end:
            errs.append(f"change {c.summary[:40]!r} after window end")
    for p in env.past_incidents:
        if p.date >= w.start:
            errs.append(f"past incident {p.id} is not before the window")
    for m in env.metrics:
        if m.service not in services:
            errs.append(f"metric service {m.service!r} not in topology")
        if m.start < w.start:
            errs.append(f"metric {m.service}/{m.name} starts before window")
        end = m.start + timedelta(seconds=m.step_seconds * (len(m.values) - 1))
        if end > w.end:
            errs.append(f"metric {m.service}/{m.name} {m.labels} ends after window ({end.isoformat()})")
    if not (w.start <= s.incident_report.reported_at <= w.end):
        errs.append("incident reported_at outside window")

    # Agent-visible content must not hint at the answer. Tool-fault notes are
    # harness-only, so they are excluded from this check.
    visible = s.model_dump(mode="json", exclude={"tool_faults"})
    text = yaml.safe_dump(visible).lower()
    for word in HINT_WORDS:
        if word in text:
            errs.append(f"agent-visible data contains hint word {word!r}")
    return errs


def check_answer_key(s: Scenario, k: AnswerKey) -> list[str]:
    errs: list[str] = []
    gt = k.ground_truth
    if k.id != s.id or k.metadata.id != s.id:
        errs.append("answer key id mismatch")
    if k.metadata.primary_tag not in k.metadata.tags:
        errs.append("primary_tag missing from tags")
    if k.metadata.provenance == "real_reconstruction" and not k.metadata.confirmed_facts:
        errs.append("real_reconstruction needs confirmed_facts")

    ev_ids = [e.id for e in gt.key_evidence]
    if len(ev_ids) != len(set(ev_ids)):
        errs.append("duplicate key_evidence ids")
    if not any(e.required for e in gt.key_evidence):
        errs.append("no required key evidence")
    for rh in gt.red_herrings:
        for ref in rh.disproved_by:
            if ref not in ev_ids:
                errs.append(f"red herring {rh.claim[:40]!r} cites unknown evidence {ref!r}")
    ranks = sorted(a.rank for a in gt.corrective_actions)
    if ranks != list(range(1, len(ranks) + 1)):
        errs.append(f"corrective action ranks not 1..n: {ranks}")

    report_names = {r.name for r in s.environment.external_reports}
    for e in gt.key_evidence:
        if e.tool == "get_external_report" and e.args.get("name") not in report_names:
            errs.append(f"evidence {e.id} references missing external report {e.args.get('name')!r}")
    return errs


def main() -> int:
    failed = False
    rows: list[tuple[Scenario, AnswerKey]] = []
    key_stems = {p.stem for p in ANSWER_KEYS_DIR.glob("*.yaml")}
    for path in scenario_paths():
        try:
            s = load_scenario(path)
            k = load_answer_key(ANSWER_KEYS_DIR / path.name)
        except (ValidationError, yaml.YAMLError, FileNotFoundError) as e:
            print(f"FAIL {path.name}\n{e}\n")
            failed = True
            continue
        errs = check_scenario(path, s) + check_answer_key(s, k)
        if errs:
            failed = True
            print(f"FAIL {path.name}")
            for e in errs:
                print(f"  - {e}")
        rows.append((s, k))
        key_stems.discard(path.stem)
    for orphan in sorted(key_stems):
        failed = True
        print(f"FAIL answer key {orphan}.yaml has no scenario")

    print(f"\n{len(rows)} scenario/answer-key pairs\n")
    print(f"{'id':<9} {'primary':<26} {'lvl':<4} {'outcome':<13} {'conf':<7} {'faults':<7} {'history':<8} title")
    for s, k in rows:
        m, gt = k.metadata, k.ground_truth
        print(
            f"{s.id:<9} {m.primary_tag:<26} {m.difficulty:<4} {gt.expected_outcome:<13} "
            f"{gt.expected_confidence:<7} {len(s.tool_faults):<7} {len(s.environment.past_incidents):<8} {m.title}"
        )
    tag_counts = Counter(t for _, k in rows for t in k.metadata.tags)
    print("\ntag coverage:", dict(sorted(tag_counts.items())))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
