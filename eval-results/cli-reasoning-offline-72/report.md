# MIRA CLI Reasoning Eval

- Mode: `offline`
- Tasks: `72`
- Pass rate: `100.0%`
- Mean tool recall: `100.0%`
- Mean tool precision: `100.0%`
- Schema-valid plans: `100.0%`
- Mean latency: `0.00s`

## By Category

| Category | n | Pass | Recall | Precision | Schema valid | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active_site | 16 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00s |
| allostery | 22 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00s |
| design | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00s |
| homology | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00s |
| interface | 24 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00s |
| stability | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00s |

## First Misses

No failed tasks at the configured threshold.

## Interpretation

This eval measures planning and tool-use readiness, not final biological correctness. It is meant to show whether the CLI agent can decompose common structure-reasoning questions into executable MIRA tool calls before running expensive analysis or design jobs.
