import json

import pytest

from dataset.validate import ANSWER_KEYS_DIR, load_answer_key
from evals import compare
from evals.judge import Judge, JudgeVerdict
from evals.scoring import best_match, evidence_matches, factor_mentioned, score_run, text_tokens
from investigator.scenario import load_scenario
from investigator.tools.backend import ScenarioBackend
from investigator.tools.registry import TOOL_SPECS
from tests.llm_fakes import FakeModel


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
    assert compare.main(["--rescore", str(out), "--judge", "none"]) == 0
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


# --- evidence checked against returned data, judge ----------------------------------------------------


def test_filtered_query_that_could_not_return_the_evidence_does_not_count():
    k = next(e for e in key("inc-003").ground_truth.key_evidence if e.id == "ev_changes")
    narrow = {"call_id": "c1", "tool": "get_changes", "args": {"service": "api-gateway"}, "ok": True,
              "data": {"changes": [{"summary": "TLS cipher suite list updated"}]}}
    broad = {**narrow, "args": {}, "data": {"changes": [{"summary": "pdp_cache_include_reviews: 0% -> 100%"}]}}
    legacy = {k_: v for k_, v in narrow.items() if k_ != "data"}  # run saved before tool data was recorded
    assert not evidence_matches(k, narrow) and evidence_matches(k, broad) and evidence_matches(k, legacy)
    assert score_run(run(report(), calls=[legacy]), key("inc-003"), "baseline").evidence_verified is False


def test_attempt_only_evidence_counts_a_failed_call():
    k = next(e for e in key("inc-008").ground_truth.key_evidence if e.attempt_only)
    assert evidence_matches(k, {"call_id": "c1", "tool": k.tool, "args": {}, "ok": False})


@pytest.mark.parametrize("sid", [p.stem for p in ANSWER_KEYS_DIR.glob("*.yaml")])
def test_key_evidence_args_can_return_the_must_contain_strings(sid):
    backend = ScenarioBackend(load_scenario(sid).environment)
    for e in key(sid).ground_truth.key_evidence:
        if e.must_contain:
            spec = TOOL_SPECS[e.tool]
            out = getattr(backend, e.tool)(spec.input_model.model_validate(e.args))
            text = json.dumps(out.model_dump(mode="json"), ensure_ascii=False).lower()
            assert all(n.lower() in text for n in e.must_contain), (sid, e.id)


def verdict(root="wrong", rh=(), mn=(), factors=()):
    return JudgeVerdict(root_cause=root, root_cause_reason="r", red_herrings_adopted=list(rh),
                        must_not_violated=list(mn), factors_stated=list(factors))


def test_judge_overrides_lexical_root_cause_and_maps_items():
    k = key("inc-001")
    rep = report(root_cause="checkout-svc v2.14.0 deploy broke legacy v1 carts missing currency")
    assert score_run(run(rep), k, "baseline").verdict == "correct"  # lexical
    s = score_run(run(rep), k, "baseline", verdict("wrong", rh=[1, 99], mn=[1], factors=[1]), "fake:judge")
    assert s.verdict == "wrong" and s.lexical_verdict == "correct" and s.judge_model == "fake:judge"
    assert s.red_herrings_matched == [k.ground_truth.red_herrings[0].claim]  # out-of-range numbers ignored
    assert s.must_not_violated == [k.ground_truth.must_not[0]]
    assert score_run(run(rep), k, "baseline", verdict("partial")).verdict == "review"


def test_judge_does_not_override_outcome_based_verdicts():
    rep = report(outcome="inconclusive")
    assert score_run(run(rep), key("inc-008"), "baseline", verdict("wrong")).verdict == "correct"
    assert score_run(run(rep), key("inc-001"), "baseline", verdict("correct")).verdict == "wrong"


def test_unsupported_claims_are_reported_and_the_judge_sees_tool_results(tmp_path):
    seen = []

    def structured(schema, messages):
        seen.append(messages[-1].content)
        return JudgeVerdict(root_cause="wrong", root_cause_reason="r", red_herrings_adopted=[], must_not_violated=[],
                            factors_stated=[], unsupported_claims=["sysctl timeout lowered to 30s"])

    judge = Judge("fake:judge", model=FakeModel(structured=structured))
    calls = [{"call_id": "call-0001", "tool": "get_changes", "args": {}, "ok": True,
              "data": {"changes": [{"summary": "checkout-svc v2.14.0"}]}},
             {"call_id": "call-0002", "tool": "query_metrics", "args": {"service": "x"}, "ok": False, "error": "not_found"}]
    v, _ = judge(compare.InvestigationReport.model_validate(report(root_cause="r")), key("inc-001"), calls,
                 "Checkout failing\nCustomers see an error")
    assert '[call-0001] get_changes {} -> {"changes"' in seen[0] and "[call-0002] query_metrics" in seen[0]
    assert "Customers see an error" in seen[0]  # the incident report is a source too
    assert "failed (not_found)" in seen[0]
    s = score_run(run(report(root_cause="r"), calls=calls), key("inc-001"), "baseline", v)
    assert s.unsupported_claims == ["sysctl timeout lowered to 30s"]


def test_judge_cache_is_invalidated_when_the_rubric_version_changes(tmp_path):
    calls = []

    def structured(schema, messages):
        calls.append(1)
        return verdict("correct")

    judge = Judge("fake:judge", model=FakeModel(structured=structured))
    out = tmp_path / "res"
    (out / "runs").mkdir(parents=True)
    (out / "runs" / "inc-001-baseline.json").write_text(
        json.dumps({"scenario": "inc-001", "system": "baseline", **run(report(root_cause="x"))}))
    compare.score_dir(out, judge)
    cached = out / "judgements" / "inc-001-baseline.json"
    cached.write_text(cached.read_text().replace('"version": ', '"version": -1, "old": '))
    compare.score_dir(out, judge)
    assert len(calls) == 2


def test_judge_prompt_carries_key_and_report_and_results_are_cached(tmp_path):
    seen = []

    def structured(schema, messages):
        seen.append(messages[-1].content)
        return verdict("correct")

    judge = Judge("fake:judge", model=FakeModel(structured=structured))
    out = tmp_path / "res"
    (out / "runs").mkdir(parents=True)
    rep = report(root_cause="checkout-svc v2.14.0 broke v1 carts")
    (out / "runs" / "inc-001-baseline.json").write_text(json.dumps({"scenario": "inc-001", "system": "baseline", **run(rep)}))
    [s] = compare.score_dir(out, judge)
    assert s.verdict == "correct" and s.judge_root_cause == "correct" and len(seen) == 1
    assert key("inc-001").ground_truth.root_cause.statement in seen[0] and "checkout-svc v2.14.0 broke v1 carts" in seen[0]
    compare.score_dir(out, judge)
    assert len(seen) == 1  # cached per judge model
