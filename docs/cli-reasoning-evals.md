# CLI Reasoning Evals

`scripts/eval_cli_reasoning.py` runs a compact live eval for the same `MiraAgent`
used by the `mira` CLI.

The goal is not to claim a full benchmark. It is a fast, transparent readiness
check for whether the CLI agent can:

- parse the public CLI commands;
- choose appropriate structure-analysis tools for local PDB fixtures;
- prefer local `pdb_path` inputs rather than network downloads;
- execute required tools successfully;
- synthesize answers that mention expected structural evidence.

## Run

```bash
uv run --extra dev python scripts/eval_cli_reasoning.py \
  --provider minimax \
  --output eval-results/cli_reasoning_minimax.json
```

For a no-cost parser smoke check:

```bash
uv run --extra dev python scripts/eval_cli_reasoning.py --skip-live
```

The live eval uses `MINIMAX_API_KEY` by default. It can also use another provider:

```bash
uv run --extra dev python scripts/eval_cli_reasoning.py \
  --provider openai \
  --model gpt-4o-mini
```

## Cases

- `local_stability_reasoning`: asks the agent to assess ubiquitin stability from
  local SASA and B-factor evidence.
- `interface_evidence_reasoning`: asks the agent to reason over a small two-chain
  local complex and inspect residue-level contacts.
- `quality_triage_reasoning`: asks the agent to combine Ramachandran geometry and
  B-factor evidence.
- `batch_folder_plan_reasoning`: asks for a reusable batch plan over local files
  and checks that `$PDB_PATH`/`pdb_path` is used.

## Scoring

Each live case receives a score in `[0, 1]`.

- 35% required tool selection
- 25% successful execution of required tools
- 15% local `pdb_path` grounding
- 15% expected evidence terms in the final answer
- 10% no failed tool calls

The batch-plan case is scored on required tool coverage and local-path grounding.
Scores above `0.75` are treated as pass for this quick eval.

Use the JSON output as the cited artifact. The report includes command-smoke
results, per-case tools, failed tool calls, token counts, scores, and final-answer
excerpts.
