# MIRA CLI Reasoning Eval

- Mode: `live`
- Tasks: `36`
- Pass rate: `91.7%`
- Mean tool recall: `84.2%`
- Mean tool precision: `82.1%`
- Schema-valid plans: `97.2%`
- Mean latency: `10.94s`

## By Category

| Category | n | Pass | Recall | Precision | Schema valid | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active_site | 10 | 100.0% | 92.5% | 81.0% | 100.0% | 13.76s |
| allostery | 8 | 87.5% | 84.4% | 74.5% | 100.0% | 14.19s |
| design | 2 | 100.0% | 87.5% | 70.2% | 100.0% | 9.37s |
| homology | 4 | 100.0% | 75.0% | 91.7% | 100.0% | 9.73s |
| interface | 8 | 75.0% | 71.9% | 87.1% | 87.5% | 5.22s |
| stability | 4 | 100.0% | 95.0% | 86.7% | 100.0% | 10.80s |

## First Misses

- `interface_1BRS_02` recall `75.0%`, schema valid `False`, tools `load_structure, compute_interface, analyze_interface_energies, get_secondary_structure`
  Schema errors: `step 2 analyze_interface_energies extra args: ['pdb_id']`
- `interface_1YCR_02` recall `50.0%`, schema valid `True`, tools `load_structure, compute_interface, list_residues`
- `allostery_107L_02` recall `50.0%`, schema valid `True`, tools `load_structure, compute_normal_modes, compute_cross_correlations, compute_perturbation_response, predict_hinge_regions, get_residue_contacts`

## Interpretation

This eval measures planning and tool-use readiness, not final biological correctness. It is meant to show whether the CLI agent can decompose common structure-reasoning questions into executable MIRA tool calls before running expensive analysis or design jobs.
