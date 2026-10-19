"""Applies a scenario's `tool_faults` to tool invocations.

Each fault keeps a trigger counter so `fails_for_attempts: N` fails the first N
matching invocations and then lets calls through (a flaky backend that recovers).
"""

from __future__ import annotations

import re
import threading
from typing import Any

from dataset.schema import ToolFault


class FaultInjector:
    def __init__(self, faults: list[ToolFault]):
        self._faults = list(faults)
        self._triggered = [0] * len(self._faults)
        self._lock = threading.Lock()

    @staticmethod
    def _matches(fault: ToolFault, tool: str, args: dict[str, Any]) -> bool:
        if fault.tool != tool:
            return False
        for key, expected in fault.match.items():
            if key == "query_regex":
                if not re.search(expected, str(args.get("query") or ""), re.IGNORECASE):
                    return False
            elif args.get(key) != expected:
                return False
        return True

    def take(self, tool: str, args: dict[str, Any]) -> ToolFault | None:
        """Return the active fault for this invocation (and count it), if any."""
        with self._lock:
            for i, fault in enumerate(self._faults):
                if not self._matches(fault, tool, args):
                    continue
                if fault.fails_for_attempts is not None and self._triggered[i] >= fault.fails_for_attempts:
                    continue
                self._triggered[i] += 1
                return fault
        return None
