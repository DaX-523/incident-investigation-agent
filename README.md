# Incident Investigator

Multi-agent incident root-cause investigation (LangGraph), compared against a single-agent baseline.

## Setup

```bash
uv sync
cp .env.example .env        # add ANTHROPIC_API_KEY and OPENAI_API_KEY
uv run pytest               # offline tests, no API keys needed
```

## Run

```bash
uv run python -m investigator inc-001 --stub              # offline demo, no API calls
uv run python -m investigator inc-001                     # multi-agent investigation
uv run python -m investigator inc-001 --system baseline   # single-agent baseline
uv run python -m evals.compare --preset budget --parallel 4   # both systems on all 16 incidents
uv run langgraph dev                                      # LangGraph Studio
```

Final evaluation configuration (`EVALUATION.md`):

```bash
MODEL_REASONING=openai:o4-mini MODEL_CRITIC=openai:o4-mini MODEL_BASELINE=anthropic:claude-haiku-5-5 \
  uv run python -m evals.compare --preset budget --parallel 4
```

## Docs

[Architecture](artifacts/architecture.md) · [Evaluation](EVALUATION.md) · [Technical note](TECHNICAL_NOTE.md) · [Dataset](dataset/README.md)

## External dependencies and incomplete functionality

- Anthropic and OpenAI APIs (not needed for tests or `--stub`); LangSmith is optional (tracing, Studio).
- Tools are simulated over the scenario files; no real observability systems are queried.
- The planned LangSmith/agentevals evaluator was not built; `evals/compare.py` runs the comparison locally.
