"""Duck-typed fake chat models for offline tests of the LLM agents."""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

from langchain_core.messages import AIMessage, ToolMessage

_ids = itertools.count(1)


def ai(content: str = "", tool_calls: list[dict] | None = None, model: str = "claude-haiku-5-5",
       input_tokens: int = 1000, output_tokens: int = 200, cache_read: int = 0, stop_reason: str = "end_turn") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": c["name"], "args": c.get("args", {}), "id": f"tc{next(_ids)}"} for c in tool_calls or []],
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        },
        response_metadata={"model_name": model, "stop_reason": stop_reason},
    )


def last_call_ids(messages) -> list[str]:
    """call_ids from the tool results seen so far (what a real model would read)."""
    ids = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.content.startswith("{"):
            ids.append(json.loads(m.content.split("... [investigator truncated")[0] if "truncated" in m.content else m.content)["call_id"])
    return ids


class FakeModel:
    """`script(messages)` returns the next AIMessage; `structured(schema, messages)` returns a parsed object."""

    def __init__(self, script: Callable[[list], AIMessage] | None = None,
                 structured: Callable[[type, list], Any] | None = None, model: str = "claude-haiku-5-5"):
        self.script = script
        self.structured_fn = structured
        self.model = model
        self.calls: list[list] = []
        self.bound_tools: list[dict] = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.script(messages)

    def with_structured_output(self, schema, method=None, include_raw=False):
        fake = self

        class _Runnable:
            def invoke(self, messages):
                fake.calls.append(list(messages))
                parsed = fake.structured_fn(schema, messages)
                if isinstance(parsed, Exception):
                    return {"raw": ai(model=fake.model), "parsed": None, "parsing_error": parsed}
                raw = parsed if isinstance(parsed, AIMessage) else ai(model=fake.model, output_tokens=300)
                return {"raw": raw, "parsed": None if isinstance(parsed, AIMessage) else parsed, "parsing_error": None}

        return _Runnable()
