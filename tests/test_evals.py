import json

import pytest

from dataset.validate import ANSWER_KEYS_DIR, load_answer_key
from evals import compare
from evals.scoring import best_match, evidence_matches, factor_mentioned, score_run, text_tokens


def key(sid):
    return load_answer_key(ANSWER_KEYS_DIR / f"{sid}.yaml")


def report(outcome="conclusive", root_cause=None, statement="something", evidence=None, cited=("E1",), status="supported"):
    evidence = evidence if evidence is not None else [{"id": "E1", "summary": "s", "tools": ["get_changes"],
                                                        "source_call_ids": ["call-0001"]}]
    return {
        "incident_title": "t", "outcome": outcome, "summary": "s", "root_cause": root_cause,
        "confidence": "high" if outcome == "conclusive" else "low",
        "hypotheses": [{"rank": 1, "hypothesis_id": "H1", "statement": statement, "status": status,
                        "confidence": "high", "supporting_evidence": list(cited)}],
        "corrective_actions": [{"rank": 1, "category": "mitigate", "action": "roll back", "rationale": "r",
                                "evidence_ids": list(cited)}],
        "evidence_index": evidence,
    }


def run(rep, calls=None, **kw):
    calls = calls if calls is not None else [{"call_id": "call-0001", "tool": "get_changes", "args": {}, "ok": True}]
    return {"report": rep, "calls": calls, "tool_calls": len(calls), "latency_s": 1.0,
            "usage": {"llm_calls": 2, "input_tokens": 100, "output_tokens": 10, "cost_usd": 0.01}, **kw}


def test_alias_match_tolerates_rephrasing():
    aliases = ["legacy v1 carts missing currency", "checkout-svc v2.14.0 deploy"]
    assert best_match(aliases, "Deploying checkout-svc v2.14.0 broke checkout")[0] >= 0.6
    assert best_match(aliases, "payments-svc timeouts at the PSP")[0] == 0.0


def test_compound_words_match_their_parts():
    assert {"check", "svc", "checkout-svc"} <= text_tokens("checkout-svc")
    assert best_match(["bad node ip-10-0-3-17"], "node-ip-10-0-3-17 is degraded")[0] >= 0.6


def test_correct_root_cause_scores_correct():
    s = score_run(run(report(root_cause="checkout-svc v2.14.0 deploy broke legacy v1 carts missing currency")),
                  key("inc-001"), "baseline")
    assert s.verdict == "correct" and s.outcome_ok and s.claims_grounded == 1.0 and s.citations_valid == 1.0


def test_red_herring_is_wrong_and_flagged():
    s = score_run(run(report(root_cause="The catalog-svc deploy caused the failures.")), key("inc-001"), "baseline")
    assert s.verdict == "wrong" and s.red_herrings_matched == ["The catalog-svc deploy caused the failures."]


def test_partial_match_needs_review():
    s = score_run(run(report(root_cause="legacy carts broke")), key("inc-001"), "baseline")
    assert s.verdict == "review"


def test_inconclusive_when_conclusive_expected_is_wrong():
    s = score_run(run(report(outcome="inconclusive", statement="checkout-svc v2.14.0 deploy", status="open")),
                  key("inc-001"), "baseline")
    assert s.verdict == "wrong" and s.top_hypothesis_match >= 0.6


def test_expected_inconclusive_rewards_declining_to_guess():
    k = key("inc-008")
    assert k.ground_truth.expected_outcome == "inconclusive"
    assert score_run(run(report(outcome="inconclusive", status="open")), k, "baseline").verdict == "correct"
    assert score_run(run(report(root_cause="Kafka broker 2 problem")), k, "baseline").verdict == "wrong"


def test_citations_to_unexecuted_calls_are_not_grounded():
    rep = report(root_cause="checkout-svc v2.14.0 deploy", evidence=[
        {"id": "E1", "summary": "s", "tools": ["get_changes"], "source_call_ids": ["call-0001"]},
        {"id": "E2", "summary": "s", "tools": ["get_changes"], "source_call_ids": ["call-0099"]}], cited=("E2",))
    s = score_run(run(rep), key("inc-001"), "baseline")
    assert s.citations_valid == 0.0 and s.claims_grounded == 0.0


def test_key_evidence_matching():
    ke = next(k for k in key("inc-001").ground_truth.key_evidence if k.args == {"service": "checkout-svc"})
    assert evidence_matches(ke, {"tool": ke.tool, "args": {"service": "checkout-svc", "start": "x"}, "ok": True})
    assert evidence_matches(ke, {"tool": ke.tool, "args": {}, "ok": True})  # broader query covers it
    assert not evidence_matches(ke, {"tool": ke.tool, "args": {"service": "payments-svc"}, "ok": True})
    assert not evidence_matches(ke, {"tool": ke.tool, "args": {"service": "checkout-svc"}, "ok": False})


def test_key_evidence_retrieved_vs_cited():
    calls = [{"call_id": "call-0001", "tool": "get_changes", "args": {}, "ok": True},
             {"call_id": "call-0002", "tool": "get_traces", "args": {}, "ok": True}]
    s = score_run(run(report(root_cause="checkout-svc v2.14.0 deploy"), calls), key("inc-001"), "baseline")
    required = sum(k.required for k in key("inc-001").ground_truth.key_evidence)
    assert s.key_evidence_retrieved == round(2 / required, 3) and s.key_evidence_cited == round(1 / required, 3)


def test_factor_with_identifier_needs_the_identifier():
    factor = "v1->v2 cart schema migration (CART-212) still incomplete, so old data shapes remain in production."
    vague = text_tokens("v1 cart schema migration still incomplete; old data shapes remain in production")
    assert not factor_mentioned(factor, vague)
    assert factor_mentioned(factor, vague | text_tokens("CART-212"))


def test_failed_run_scores_as_error():
    s = score_run({"report": None, "error": "TimeoutError: x"}, key("inc-001"), "multi_agent")
    assert s.verdict == "error" and s.error == "TimeoutError: x"


def test_compare_end_to_end_with_stubs_resume_and_rescore(tmp_path, monkeypatch):
    out = tmp_path / "res"
    args = ["--stub", "--scenarios", "inc-001", "inc-008", "--out", str(out)]
    assert compare.main(args) == 0
    results = json.loads((out / "results.json").read_text())
    assert {s["system"] for s in results["scores"]} == {"multi_agent", "baseline"} and len(results["scores"]) == 4
    assert set(results["aggregate"]) == {"multi_agent", "baseline"}
    assert "## Summary" in (out / "results.md").read_text()

    def boom(*a, **k):
        raise AssertionError("finished runs must not be re-run")

    monkeypatch.setattr(compare, "run_system", boom)
    assert compare.main(args) == 0  # resume: nothing left to run

    # A manual review decision is applied on rescore.
    path = out / "runs" / "inc-001-baseline.json"
    saved = json.loads(path.read_text())
    saved["report"] = report(root_cause="legacy carts broke")
    path.write_text(json.dumps(saved))
    (out / "reviews.yaml").write_text("inc-001/baseline: correct\n")
    assert compare.main(["--rescore", str(out)]) == 0
    rescored = {(s["scenario"], s["system"]): s for s in json.loads((out / "results.json").read_text())["scores"]}
    assert rescored[("inc-001", "baseline")]["verdict"] == "correct"


def test_crashing_run_is_recorded_not_raised(tmp_path, monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(compare, "run_system", crash)
    assert compare.main(["--stub", "--scenarios", "inc-001", "--systems", "baseline", "--out", str(tmp_path)]) == 0
    score = json.loads((tmp_path / "results.json").read_text())["scores"][0]
    assert score["verdict"] == "error" and "model exploded" in score["error"]


@pytest.mark.parametrize("sid", ["inc-001", "inc-011"])
def test_security_expectation_tracked(sid):
    s = score_run(run(report(root_cause="x")), key(sid), "baseline")
    assert s.security_expected == bool(key(sid).ground_truth.security_findings)
