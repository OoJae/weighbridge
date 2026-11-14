# Adapter population summary

Snapshot of every public repository tagged `base_model:adapter:CohereLabs/tiny-aya-global` on the Hugging Face Hub, taken 2026-11-14T17:55:36+00:00 UTC. Aggregate counts only: repository ids, author names and model cards stay in the private manifest.

- Manifest SHA-256: `887bf12c73cc17db65ed2d641f7958d12d57b724d8d17fd00f2abf3e69d92544`
- Salted manifest commitment (PREREG section 2): `9a441ebb37da5fb940babf2e4b244951cafaddb0231b6f88d2ef0b24377778fa`
- Code: weighbridge 0.1.0; hub.py sha256:120125a2bb94; population.yaml sha256:61bdb2ba6335

## Totals

- Listed repositories: 235 from 78 authors.
- Included: 187 adapters from 70 authors. Excluded: 48.
- Confirmatory set (at most 2 per author, seed 20261114): 107 adapters from 70 authors.

### Exclusions by reason

| reason | repositories |
|---|---:|
| gated | 9 |
| no_adapter_config | 4 |
| full_model_mislabelled | 2 |
| no_adapter_weights | 1 |
| unreadable | 0 |
| not_lora | 0 |
| base_not_resolvable | 32 |

## Breakdowns

Intent and method are keyword classifications of each repository's name, tags and model card; they are conservative, so ambiguous cases are `unknown`.

### Declared intent

| | all listed | included |
|---|---:|---:|
| attack_or_drift | 9 | 8 |
| safety_hardened | 16 | 15 |
| control | 7 | 7 |
| task | 158 | 124 |
| unknown | 45 | 33 |

### Training method

| | all listed | included |
|---|---:|---:|
| dpo | 30 | 29 |
| grpo | 51 | 22 |
| rft | 2 | 2 |
| sft | 56 | 53 |
| other | 3 | 3 |
| unknown | 93 | 78 |

### LoRA rank

| | all listed | included |
|---|---:|---:|
| 4 | 14 | 14 |
| 8 | 20 | 19 |
| 16 | 150 | 119 |
| 32 | 20 | 20 |
| 64 | 15 | 15 |
| n/a | 16 | 0 |

### Rank-stabilised LoRA (rsLoRA)

| | all listed | included |
|---|---:|---:|
| False | 231 | 183 |
| True | 4 | 4 |

### Target modules

| | all listed | included |
|---|---:|---:|
| all 7 | 186 | 154 |
| all 7, layer subset | 2 | 2 |
| n/a | 16 | 0 |
| o+down | 2 | 2 |
| q+k+v+o | 16 | 16 |
| q+k+v+o, layer subset | 1 | 1 |
| q+v | 9 | 9 |
| q+v+gate+up+down | 3 | 3 |

### Month created

| | all listed | included |
|---|---:|---:|
| 2026-02 | 1 | 1 |
| 2026-03 | 24 | 24 |
| 2026-05 | 23 | 22 |
| 2026-06 | 1 | 1 |
| 2026-07 | 2 | 1 |
| 2026-08 | 5 | 5 |
| 2026-10 | 1 | 1 |
| 2026-11 | 178 | 132 |

### Era (hackathon = created on or after 2026-11-01)

| | all listed | included |
|---|---:|---:|
| hackathon | 178 | 132 |
| earlier | 57 | 55 |

### Included adapters per author

| adapters per author | authors |
|---:|---:|
| 1 | 33 |
| 2 | 16 |
| 3 | 10 |
| 4 | 4 |
| 5 | 3 |
| 6 | 1 |
| 9 | 1 |
| 22 | 1 |
| 24 | 1 |
