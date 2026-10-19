"""Single entry point for every tool call: validation, timeout, bounded retries, data-quality checks.

Every call returns a `ToolResult`; tool failures never raise to the caller.
"""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from langsmith import traceable
from pydantic import BaseModel, ValidationError

from dataset.schema import Scenario, ToolFault
from investigator.tools.backend import ScenarioBackend, validate_range
from investigator.tools.errors import (
    ToolFailure,
    ToolInvalidRequestError,
    ToolTimeoutError,
    ToolUnavailableError,
)
from investigator.tools.faults import FaultInjector
from investigator.tools.models import DataQuality, TimeRange, ToolError, ToolResult
from investigator.tools.registry import TOOL_SPECS, ToolSpec


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class RetrySettings:
    max_retries: int = 2
    timeout_s: float = 10.0
    backoff_base_s: float = 0.2
    backoff_cap_s: float = 2.0

    @classmethod
    def from_env(cls) -> RetrySettings:
        return cls(
            max_retries=int(_env_float("TOOL_MAX_RETRIES", cls.max_retries)),
            timeout_s=_env_float("TOOL_TIMEOUT_S", cls.timeout_s),
            backoff_base_s=_env_float("TOOL_BACKOFF_BASE_S", cls.backoff_base_s),
            backoff_cap_s=_env_float("TOOL_BACKOFF_CAP_S", cls.backoff_cap_s),
        )

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """Exponential backoff with equal jitter; `attempt` is 0-based for the failed attempt."""
        delay = min(self.backoff_cap_s, self.backoff_base_s * (2**attempt))
        return delay / 2 + rng.uniform(0, delay / 2)


class ToolHarness:
    def __init__(
        self,
        scenario: Scenario,
        settings: RetrySettings | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        max_workers: int = 8,
    ):
        self.scenario = scenario
        self.settings = settings or RetrySettings.from_env()
        self._backend = ScenarioBackend(scenario.environment)
        self._faults = FaultInjector(scenario.tool_faults)
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")
        self._lock = threading.Lock()
        self._calls: list[ToolResult] = []

    # --- public --------------------------------------------------------------

    @property
    def calls(self) -> list[ToolResult]:
        with self._lock:
            return list(self._calls)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> ToolHarness:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def call(self, tool: str, args: dict[str, Any] | None = None) -> ToolResult:
        # Shows up as a "tool" run in LangSmith traces (no-op unless tracing is enabled).
        return self._traced_call(tool, args, langsmith_extra={"name": tool})

    @traceable(run_type="tool", name="tool_call")
    def _traced_call(self, tool: str, args: dict[str, Any] | None = None) -> ToolResult:
        args = dict(args or {})
        started = time.monotonic()
        spec = TOOL_SPECS.get(tool)
        if spec is None:
            return self._record(tool, args, started, attempts=0,
                                error=ToolInvalidRequestError(f"unknown tool {tool!r}; available: {sorted(TOOL_SPECS)}"))
        try:
            inp = spec.input_model.model_validate(args)
            validate_range(getattr(inp, "start", None), getattr(inp, "end", None))
        except ValidationError as exc:
            return self._record(tool, args, started, attempts=0,
                                error=ToolInvalidRequestError(_short_validation_message(exc)))
        except ToolFailure as exc:
            return self._record(tool, args, started, attempts=0, error=exc)

        normalized_args = inp.model_dump(mode="json", exclude_none=True)
        error: ToolFailure | None = None
        output: BaseModel | None = None
        attempts = 0
        for attempt in range(self.settings.max_retries + 1):
            attempts += 1
            fault = self._faults.take(tool, normalized_args)
            future = self._pool.submit(self._execute, spec, inp, fault)
            try:
                output, error = future.result(timeout=self.settings.timeout_s), None
                break
            except FutureTimeout:
                error = ToolTimeoutError(f"no response within {self.settings.timeout_s:g}s")
            except ToolFailure as exc:
                error = exc
            except Exception as exc:  # backend bug: report, don't retry
                error = ToolFailure(f"internal error: {type(exc).__name__}: {exc}")
            if not error.retryable or attempt == self.settings.max_retries:
                break
            self._sleep(self.settings.backoff(attempt, self._rng))

        if error is not None:
            return self._record(tool, normalized_args, started, attempts=attempts, error=error)
        quality = self._quality(spec, inp, output)
        return self._record(tool, normalized_args, started, attempts=attempts,
                            data=output.model_dump(mode="json"), quality=quality)

    # --- internals -----------------------------------------------------------

    def _execute(self, spec: ToolSpec, inp: BaseModel, fault: ToolFault | None) -> BaseModel:
        if fault is not None and fault.mode == "timeout":
            raise ToolTimeoutError(fault.error_message or "request timed out")
        if fault is not None and fault.mode == "error":
            raise ToolUnavailableError(fault.error_message or "backend unavailable (503)")

        if fault is not None and fault.mode == "stale" and spec.name == "query_metrics":
            # A cache serving its last samples: substitute data, not clipped to the requested range.
            output = self._backend.query_metrics(inp, series_source=fault.substitute_metrics, clip_to_range=False)
        else:
            output = getattr(self._backend, spec.name)(inp)

        if fault is not None and fault.mode == "empty":
            output = output.model_copy(update={spec.result_field: []})
        elif fault is not None and fault.mode == "truncated" and fault.keep_first_n is not None:
            items = getattr(output, spec.result_field)
            output = output.model_copy(update={spec.result_field: items[: fault.keep_first_n]})
        return output

    def _requested_range(self, spec: ToolSpec, inp: BaseModel) -> TimeRange | None:
        if not hasattr(inp, "end"):
            return None
        start, end = getattr(inp, "start", None), getattr(inp, "end", None)
        window = self.scenario.environment.window
        if spec.window_bounded:
            return TimeRange(start=start or window.start, end=end or window.end)
        return TimeRange(start=start, end=end or window.end)

    def _quality(self, spec: ToolSpec, inp: BaseModel, output: BaseModel) -> DataQuality:
        items = getattr(output, spec.result_field)
        timestamps, null_points = _timestamps(spec.name, items)
        requested = self._requested_range(spec, inp)
        returned = TimeRange(start=min(timestamps), end=max(timestamps)) if timestamps else None
        q = DataQuality(requested_range=requested, returned_range=returned, result_count=len(items),
                        empty=not items, null_points=null_points)
        notes = []
        if q.empty:
            notes.append("no results returned")
        if requested and returned and requested.start and requested.end:
            if returned.end < requested.start or returned.start > requested.end:
                q.out_of_requested_range = True
                notes.append(
                    f"returned data spans {returned.start.isoformat()}..{returned.end.isoformat()}, "
                    f"outside the requested {requested.start.isoformat()}..{requested.end.isoformat()}"
                )
        if null_points:
            notes.append(f"{null_points} metric points have no value (gaps)")
        limit = getattr(inp, "limit", None)
        if limit is not None and len(items) >= limit:
            q.at_limit = True
            notes.append(f"result count reached the limit ({limit}); more results may exist")
        q.notes = notes
        return q

    def _record(self, tool, args, started, *, attempts, data=None, quality=None, error: ToolFailure | None = None) -> ToolResult:
        with self._lock:
            call_id = f"call-{len(self._calls) + 1:04d}"
            result = ToolResult(
                call_id=call_id,
                tool=tool,
                args=args,
                ok=error is None,
                data=data,
                error=None if error is None else ToolError(kind=error.kind, message=str(error), retryable=error.retryable),
                quality=quality,
                attempts=attempts,
                latency_ms=round((time.monotonic() - started) * 1000, 2),
            )
            self._calls.append(result)
        return result


def _timestamps(tool: str, items: list[Any]) -> tuple[list[datetime], int]:
    stamps: list[datetime] = []
    nulls = 0
    for item in items:
        if tool == "query_metrics":
            for p in item.points:
                if p.value is None:
                    nulls += 1
                else:
                    stamps.append(p.ts)
        else:
            ts = getattr(item, "ts", None) or getattr(item, "start", None) or getattr(item, "date", None)
            if isinstance(ts, datetime):
                stamps.append(ts)
    return stamps, nulls


def _short_validation_message(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(x) for x in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()[:5]]
    return "invalid arguments: " + "; ".join(parts)
