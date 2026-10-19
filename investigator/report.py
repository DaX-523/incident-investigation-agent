"""Final investigation report. Shared by the multi-agent graph and the baseline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from investigator.schemas import Confidence, StopReason


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(_Model):
    text: str
    evidence_ids: list[str] = Field([], description="Facts must cite at least one evidence id.")


class ReportHypothesis(_Model):
    rank: int
    hypothesis_id: str
    statement: str
    status: Literal["supported", "refuted", "inconclusive", "open"]
    confidence: Confidence
    supporting_evidence: list[str] = []
    contradicting_evidence: list[str] = []
    facts: list[Claim] = Field([], description="Directly observed in tool output; each cites evidence.")
    assumptions: list[str] = Field([], description="Inferences not directly observed.")


class CorrectiveActionOut(_Model):
    rank: int
    category: Literal["mitigate", "fix", "prevent", "detect"]
    action: str
    rationale: str
    evidence_ids: list[str] = []


class EvidenceIndexEntry(_Model):
    id: str
    summary: str
    tools: list[str]
    source_call_ids: list[str]


class InvestigationReport(_Model):
    incident_title: str
    outcome: Literal["conclusive", "inconclusive"]
    summary: str
    root_cause: str | None = Field(None, description="Only when the outcome is conclusive.")
    confidence: Confidence
    hypotheses: list[ReportHypothesis]
    corrective_actions: list[CorrectiveActionOut]
    missing_data: list[str] = []
    gaps: list[str] = Field([], description="Tool failures, data-quality problems and unchecked areas.")
    security_findings: list[str] = []
    questions_for_human: list[str] = []
    evidence_index: list[EvidenceIndexEntry] = []
    stop_reason: StopReason | None = None


class ReportNarrative(_Model):
    """What the reporting agent writes. Structure and evidence come from state, not from the LLM."""

    summary: str
    root_cause: str | None = None
    corrective_actions: list[CorrectiveActionOut] = []
    missing_data: list[str] = []
    security_findings: list[str] = []
