# MIRA CLI Reasoning Eval

This folder documents the lightweight CLI-agent eval used for the CS 153 demo
and project report. The eval focuses on planning and tool-selection quality:
given a structural biology question, does the CLI agent choose an executable set
of MIRA tools that would gather the right evidence?

The harness reads `benchmark_candidates.jsonl`, infers a task family for each
structure, and scores the generated plan against a rubric of expected tool
groups. It can run in two modes:

- `offline`: deterministic heuristic baseline. Use this to verify the harness,
  output format, and plots without spending model credits.
- `live`: calls the configured provider through `MiraAgent.create_plan`. Use this
  for the real model number in the report.

Run the offline scaled eval:

```bash
venv/bin/python scripts/run_cli_reasoning_eval.py --mode offline --limit 72
```

Run a live MiniMax eval:

```bash
venv/bin/python scripts/run_cli_reasoning_eval.py --mode live --provider minimax --limit 36
```

Outputs are written to a timestamped folder under `eval-results/`:

- `summary.json`: aggregate pass rate, recall, precision, and latency.
- `results.jsonl`: one scored record per task, including the generated plan.
- `results.csv`: spreadsheet-friendly task metrics.
- `report.md`: short human-readable eval summary.
- `plots/*.svg`: publication/demo-ready plots.

This is not a full wet-lab or docking benchmark. It is a fast reasoning eval for
the agent layer, intended to catch wrong tool choice, invalid tool arguments,
and weak task decomposition before running expensive structure-design jobs.
