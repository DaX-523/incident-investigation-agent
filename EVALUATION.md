# Evaluation

**Configuration.** Multi-agent: Claude Haiku 5.5 specialists (planner, observability, dependency, change
history, reporter) with an `o4-mini` reasoning agent and critic. Baseline: one Claude Haiku 5.5 agent with
all tools. Same tool harness and faults, budgets, report schema and domain guidance; only the orchestration
differs. 16 scenarios, one run each. Raw runs, judge verdicts and per-scenario tables: `evals/results/final/`.

**Scoring.** Deterministic checks against `dataset/answer_keys/` (key evidence must appear in the data a call
actually returned; every claim must cite an executed call) plus an LLM judge (`gpt-5.4-mini`, reads the
answer key) for root cause vs. amplifier, red herrings, forbidden actions, factors and unsupported claims.
Re-run the scoring with `uv run python -m evals.compare --rescore evals/results/final`.

## Results

| Metric | Multi-agent | Baseline |
|---|---|---|
| Correct conclusive vs. inconclusive call | **16/16** | 16/16 |
| Root cause correct | 13/16 (+2 partial) | 15/16 (+1 partial) |
| Premature-closure scenarios (decisive evidence in later sources) | **3/3** | 2/3 |
| Claims citing an executed tool call | **1.00** | 0.99 |
| Red herrings adopted / forbidden actions | 0 / 0 | 0 / 0 |
| Runs stating a fact no tool result supports | 9 | 10 |
| Prompt injection flagged | 1/1 | 1/1 |
| Key evidence retrieved and cited | 0.88 | 0.93 |
| Tool calls / tokens per run | 29 / 152K | 19 / 128K |
| Latency / cost per run | 118 s / $0.073 | 55 s / $0.010 |

**Strengths of the multi-agent system.** Every conclusive-or-inconclusive decision is right, including the
tool-failure case where declining to guess is correct. It solves all three premature-closure scenarios,
where a convincing explanation sits in the first sources checked and the overturning evidence is in traces
or infrastructure events; the evidence gate and the mandatory change sweep make it look there. Every claim
is grounded in an executed tool call, and it never adopted a red herring or recommended a forbidden action.

**Where it falls short.** Two root causes were graded partial or wrong: inc-005 (named the faulty node but
not the degraded disk behind it), inc-012 (crawler traffic found, originating change not named). It costs about 7x the baseline and takes
about twice as long.

**Critic model.** The critic decides when evidence suffices, so its model matters most. With a Haiku critic
the system often declared correct investigations inconclusive; a reasoning model in the reasoner and critic
seats removed those errors.

## Matched-model check on hard cases

`evals/results/openai-3/`: both systems on OpenAI models (`gpt-5.4` specialists, `o3` reasoner, critic and
baseline) on three hard scenarios. The single `o3` agent makes one tool call per turn into one growing
context and hit the 400K-token budget on 2 of 3 runs; the multi-agent system kept each specialist's context
short, used fewer tokens per run (234K vs 346K) and cited all key evidence (1.00 vs 0.61), with grounding
1.00 vs 0.85.
