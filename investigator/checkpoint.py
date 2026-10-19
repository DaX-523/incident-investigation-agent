"""Checkpointers with an explicit allowlist of our state types (strict msgpack deserialization).

Only types defined in our own state/payload modules may be revived from a checkpoint.
"""

from __future__ import annotations

import inspect
from types import ModuleType

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

import dataset.schema
import investigator.orchestrator.context
import investigator.report
import investigator.schemas
import investigator.tools.models

_STATE_MODULES: tuple[ModuleType, ...] = (
    investigator.schemas,
    investigator.report,
    investigator.orchestrator.context,
    investigator.tools.models,
    dataset.schema,
)


def allowed_types() -> list[tuple[str, str]]:
    allowed = set()
    for module in _STATE_MODULES:
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__:
                allowed.add((module.__name__, name))
    return sorted(allowed)


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=allowed_types())


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=serializer())
