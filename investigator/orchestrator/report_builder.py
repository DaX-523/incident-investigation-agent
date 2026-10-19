"""Builds the final report from orchestrator state; the optional narrative only adds prose and actions.

Used as-is when the reporting agent fails, so a report is always produced.
"""

from __future__ import annotations

from investigator.orchestrator.manager import ranked_hypotheses
from investigator.report import (
    Claim,
    EvidenceIndexEntry,
    InvestigationReport,
    ReportHypothesis,
    ReportNarrative,
)
from investigator.schemas import InvestigationState


def build_report(state: InvestigationState, narrative: ReportNarrative | None = None) -> InvestigationReport:
    evidence = {e.id: e for e in state["evidence"]}
    stance: dict[str, dict[str, list[str]]] = {}
    for e in state["evidence"]:
        for link in e.links:
            bucket = stance.setdefault(link.hypothesis_id, {"supports": [], "contradicts": []})
            if link.stance in bucket:
                bucket[link.stance].append(e.id)

    hypotheses = []
    for rank, h in enumerate(ranked_hypotheses(state), start=1):
        linked = stance.get(h.id, {"supports": [], "contradicts": []})
        supporting = list(dict.fromkeys(h.supporting + linked["supports"]))
        contradicting = list(dict.fromkeys(h.contradicting + linked["contradicts"]))
        hypotheses.append(ReportHypothesis(
            rank=rank,
            hypothesis_id=h.id,
            statement=h.statement,
            status=h.status,
            confidence=h.confidence or "low",
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            facts=[Claim(text=evidence[i].summary, evidence_ids=[i]) for i in supporting + contradicting if i in evidence],
            assumptions=h.assumptions,
        ))

    top = hypotheses[0] if hypotheses else None
    conclusive = (
        state.get("stop_reason") == "accepted"
        and top is not None
        and top.status == "supported"
        and top.confidence in ("high", "medium")
    )
    narrative = narrative or ReportNarrative(summary=_default_summary(state, top, conclusive))

    actions = [
        a.model_copy(update={"evidence_ids": [i for i in a.evidence_ids if i in evidence]})
        for a in sorted(narrative.corrective_actions, key=lambda a: a.rank)
    ]
    actions = [a.model_copy(update={"rank": n}) for n, a in enumerate(actions, start=1)]

    verdicts = state.get("verdicts") or []
    return InvestigationReport(
        incident_title=str(state["incident"].get("title", "incident")),
        outcome="conclusive" if conclusive else "inconclusive",
        summary=narrative.summary,
        root_cause=(narrative.root_cause or top.statement) if conclusive and top else None,
        confidence=top.confidence if conclusive and top else "low",
        hypotheses=hypotheses,
        corrective_actions=actions,
        missing_data=narrative.missing_data,
        gaps=list(dict.fromkeys(g.description for g in state["gaps"])),
        security_findings=narrative.security_findings,
        questions_for_human=verdicts[-1].questions_for_human if verdicts else [],
        evidence_index=[
            EvidenceIndexEntry(id=e.id, summary=e.summary, tools=e.tools, source_call_ids=e.source_call_ids)
            for e in state["evidence"]
        ],
        stop_reason=state.get("stop_reason"),
    )


def _default_summary(state: InvestigationState, top: ReportHypothesis | None, conclusive: bool) -> str:
    reason = state.get("stop_reason") or "unknown"
    if top is None:
        return f"Investigation stopped ({reason}) before any hypothesis could be ranked."
    lead = "Most likely cause" if conclusive else "Leading hypothesis (not confirmed)"
    return f"{lead}: {top.statement}. Investigation stopped: {reason}. Evidence items: {len(state['evidence'])}."

