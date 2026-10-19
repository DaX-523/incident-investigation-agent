from pathlib import Path

import pytest

from dataset.validate import (
    ANSWER_KEYS_DIR,
    check_answer_key,
    check_scenario,
    load_answer_key,
    load_scenario,
    scenario_paths,
)

REPO = Path(__file__).resolve().parents[1]
# Only these locations may read answer keys (graders / dataset tooling / tests).
ANSWER_KEY_READERS = ("dataset", "evals", "tests")
EXCLUDED_DIRS = {".venv", ".git", "__pycache__"}

PATHS = scenario_paths()


def test_minimum_incident_count():
    assert len(PATHS) >= 10


@pytest.mark.parametrize("path", PATHS, ids=lambda p: p.stem)
def test_scenario_and_answer_key_valid(path):
    scenario = load_scenario(path)
    key = load_answer_key(ANSWER_KEYS_DIR / path.name)
    assert check_scenario(path, scenario) + check_answer_key(scenario, key) == []


def test_required_categories_covered():
    tags = {t for p in PATHS for t in load_answer_key(ANSWER_KEYS_DIR / p.name).metadata.tags}
    assert {"misleading_evidence", "conflicting_observability", "tool_failure"} <= tags


def test_agent_code_cannot_reach_answer_keys():
    offenders = []
    for py in REPO.rglob("*.py"):
        rel = py.relative_to(REPO)
        if EXCLUDED_DIRS & set(rel.parts) or rel.parts[0] in ANSWER_KEY_READERS:
            continue
        text = py.read_text()
        if "answer_keys" in text or "AnswerKey" in text or "load_answer_key" in text:
            offenders.append(str(rel))
    assert offenders == [], f"non-eval code references answer keys: {offenders}"


def test_agent_code_has_no_scenario_ids():
    """No hardcoded per-case logic: scenario ids must not appear in agent/tool code."""
    ids = [p.stem for p in PATHS]
    offenders = []
    for py in REPO.rglob("*.py"):
        rel = py.relative_to(REPO)
        if EXCLUDED_DIRS & set(rel.parts) or rel.parts[0] in ANSWER_KEY_READERS:
            continue
        text = py.read_text()
        offenders += [f"{rel}: {i}" for i in ids if i in text]
    assert offenders == []
