# Incident Investigator

Multi-agent incident root-cause investigation (LangGraph), compared against a single-agent baseline.

## Setup

```bash
uv sync
cp .env.example .env        # add ANTHROPIC_API_KEY and OPENAI_API_KEY
uv run pytest               # offline tests, no API keys needed
```

Use your own API keys in `.env` (never committed). Without keys you can still run the tests, any command
with `--stub`, Studio in `"mode": "stub"`, and re-score the committed runs:
`uv run python -m evals.compare --rescore evals/results/final --judge none`.

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

## Run from LangSmith Studio (no frontend needed)

```bash
uv run langgraph dev        # opens LangSmith Studio in the browser (free LangSmith login)
```

In Studio, pick the `investigator` (or `baseline`) graph, set the run context (assistant settings),
leave the input as `{}` and submit. The graph runs live, node by node, with the report in the final state:

```json
{"scenario": "inc-003", "mode": "llm", "preset": "budget", "provider": "anthropic", "investigation_mode": "eval"}
```

`"mode": "stub"` runs offline without API keys. Studio reads `.env`, so adding
`MODEL_REASONING=openai:o4-mini` and `MODEL_CRITIC=openai:o4-mini` there gives the evaluated configuration. `"investigation_mode": "interactive"` pauses for human review
when the critic asks a question. To see CLI and evaluation runs as traces in LangSmith as well, set
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`.

## Docs

[Architecture](artifacts/architecture.md) ([diagram](artifacts/architecture.png)) · [Evaluation](EVALUATION.md) · [Technical note](TECHNICAL_NOTE.md) · [Dataset](dataset/README.md)

## External dependencies and incomplete functionality

- Anthropic and OpenAI APIs (not needed for tests or `--stub`); LangSmith is optional (tracing, Studio).
- Tools are simulated over the scenario files; no real observability systems are queried.
- The planned LangSmith/agentevals evaluator was not built; `evals/compare.py` runs the comparison locally.
