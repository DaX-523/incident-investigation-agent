# Technical note

## Key decisions

- **Orchestrator as the single writer.** An LLM planner proposes hypotheses and tasks; a deterministic
  manager owns shared state. Agents return one typed `Outcome` each, applied in task order, so parallel
  fan-out (`Send`) is race-free.
- **Evidence is grounded by construction.** Every tool call gets a `call_id`; evidence citing a call the
  agent did not make is dropped. Reports separate facts (cited) from assumptions.
- **Structure where models are unreliable, judgment where it matters.** Code guarantees coverage: every
  specialist runs, an unfiltered change sweep always happens, and an evidence gate blocks the report until
  the top hypotheses have support or contradiction. The critic, an LLM, decides when the evidence suffices.
- **Every loop is bounded.** Limits on critic rounds, tool calls, tokens, wall clock, per-task tool calls,
  retries and worker attempts. A limit triggers one closing pass (rank what was gathered, the critic
  decides) rather than a blind stop; a failing tool or agent becomes a recorded gap.
- **Fair baseline.** Same harness, tools, faults, budgets, metering, report schema and domain guidance;
  only the orchestration differs.
- **Honest evaluation.** Answer keys are isolated from agent code (enforced by a test). Key evidence counts
  only if the returned data contains it. An LLM judge with the answer key grades cause versus amplifier,
  red herrings, forbidden actions and claims no tool result supports.

## Results

Final configuration: Haiku specialists with an `o4-mini` reasoner and critic, against a single Haiku agent,
16 scenarios ([EVALUATION.md](EVALUATION.md)). The multi-agent system made the right conclusive or
inconclusive call in 16/16, solved all three premature-closure cases (baseline 2/3), grounded every claim
and never adopted a red herring. Root cause: 13/16 + 2 partial vs 15/16 + 1, at about 7x the cost.

## Trade-offs

- The multi-agent system costs about 7x and takes twice as long. Its gains are coverage, grounding and
  staying on track when a single agent's context fills up (an `o3` baseline hit the token cap on 2 of 3
  hard cases), not raw accuracy on easy cases.
- The critic's model matters most: a small model there over-declares inconclusive, so the critic and
  reasoner run on a reasoning model while cheap models gather.
- Splitting the brief's Dependency Agent in two (dependencies, changes) keeps toolsets short, at the cost
  of one more agent.

## Limitations

- Simulated tools over 16 synthetic scenarios (one reconstructed); results are single runs, so one- or
  two-scenario differences are noise.
- The LLM judge can misread (each verdict's reason is listed for review); lexical scoring is kept for
  reference only.
- Several fixes came from single cases; they are generic, and full sweeps guard against over-fitting.

## Improvements

- Push reports from mechanism to originating change (the source of the remaining partial answers).
- Repeat runs per scenario and report variance; harder, noisier scenarios with larger telemetry.
- Cross-family critic (supported via per-role overrides) to reduce correlated errors.
- LangSmith/agentevals integration for tracked experiments.
