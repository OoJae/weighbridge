Leak check: PASS (threshold: character-5-gram Jaccard > 0.5; gated pairs involve an outcome panel and two model-input panels)

| panel A | panel B | gated | exact | normalised | max Jaccard | pairs > threshold | twins |
|---|---|---|---|---|---|---|---|
| en_outcome_harmful | y2_heldout | yes | 0 | 0 | 0.486 | 0 | 0 |
| d1_probe | en_outcome_harmful | yes | 0 | 0 | 0.471 | 0 | 0 |
| en_outcome_harmful | fit_directions | yes | 0 | 0 | 0.471 | 0 | 0 |
| fit_directions | y2_heldout | yes | 0 | 0 | 0.462 | 0 | 0 |
| d1_probe | y2_heldout | yes | 0 | 0 | 0.427 | 0 | 0 |
| en_outcome_benign | en_outcome_harmful | yes | 0 | 0 | 0.252 | 0 | 50 |
| en_outcome_benign | fit_directions | yes | 0 | 0 | 0.233 | 0 | 0 |
| d1_probe | en_outcome_benign | yes | 0 | 0 | 0.214 | 0 | 0 |
| en_outcome_benign | y2_heldout | yes | 0 | 0 | 0.204 | 0 | 0 |
| en_outcome_harmful | yo_tukabench | yes | 0 | 0 | 0.087 | 0 | 200 |
| en_outcome_benign | yo_tukabench | yes | 0 | 0 | 0.074 | 0 | 100 |
| fit_directions | yo_tukabench | yes | 0 | 0 | 0.055 | 0 | 0 |
| d1_probe | yo_tukabench | yes | 0 | 0 | 0.029 | 0 | 0 |
| y2_heldout | yo_tukabench | yes | 0 | 0 | 0.025 | 0 | 0 |
| en_outcome_benign | judge_validation | no | 22 | 22 | 1.000 | 22 | 0 |
| en_outcome_harmful | judge_validation | no | 0 | 0 | 0.466 | 0 | 0 |
| d1_probe | fit_directions | no | 0 | 0 | 0.463 | 0 | 0 |
| fit_directions | judge_validation | no | 0 | 0 | 0.259 | 0 | 0 |
| d1_probe | judge_validation | no | 0 | 0 | 0.253 | 0 | 0 |
| judge_validation | y2_heldout | no | 0 | 0 | 0.233 | 0 | 0 |
| judge_validation | yo_tukabench | no | 0 | 0 | 0.037 | 0 | 0 |
