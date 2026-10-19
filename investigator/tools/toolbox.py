"""Task-scoped view of the harness handed to a worker.

Enforces the per-task tool-call allowance and records every call itself, so an
Outcome's `tool_calls` come from what actually ran, not from what the agent claims.
"""

from __future__ import annotations

import threading
from typing import Any

from investigator.schemas import ToolCallRecord
from investigator.tools.harness import ToolHarness
from investigator.tools.models import ToolError, ToolResult
from investigator.tools.registry import TOOL_SPECS


class ToolBox:
    def __init__(self, harness: ToolHarness, allowance: int):
        self._harness = harness
        self.allowance = max(0, allowance)
        self._results: list[ToolResult] = []
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        return self.allowance - len(self._results)

    @staticmethod
    def tool_names() -> list[str]:
        return list(TOOL_SPECS)

    def call(self, tool: str, args: dict[str, Any] | None = None) -> ToolResult:
        with self._lock:
            if len(self._results) >= self.allowance:
                return ToolResult(
                    call_id="budget",
                    tool=tool,
                    args=dict(args or {}),
                    ok=False,
                    error=ToolError(kind="budget_exhausted", retryable=False,
                                    message=f"tool-call allowance for this task ({self.allowance}) is used up"),
                    attempts=0,
                    latency_ms=0.0,
                )
        result = self._harness.call(tool, args)
        with self._lock:
            self._results.append(result)
        return result

    def records(self) -> list[ToolCallRecord]:
        with self._lock:
            results = list(self._results)
        return [
            ToolCallRecord(
                call_id=r.call_id,
                tool=r.tool,
                args=r.args,
                ok=r.ok,
                error_kind=r.error.kind if r.error else None,
                error_message=r.error.message if r.error else None,
                quality_notes=r.quality.notes if r.quality else [],
            )
            for r in results
        ]
