# Steering evaluation (BAAI/bge-small-en-v1.5)

Test negatives per concept: {'sentiment': 300, 'sports': 300, 'business': 300, 'scitech': 300}. STS-B test pairs: 690 (base Spearman 0.853). Runtime 92.4s.

Eval-probe accuracy (held-out readers, never used by the method): sentiment 0.870, sports 0.977, business 0.870, scitech 0.902

## 1. Sweep at alpha = 0.3 (mean over concepts)

| method | gain (target prob) | success rate | leakage (other concepts) | selectivity = leak/gain | STS-B drop | monotonicity |
|---|---|---|---|---|---|---|
| random | 0.023 | 0.112 | 0.023 | 0.997 | 0.0008 | 0.928 |
| naive | 0.555 | 0.974 | 0.095 | 0.172 | 0.0062 | 1.000 |
| lowdin | 0.539 | 0.974 | 0.059 | 0.110 | 0.0067 | 1.000 |
| nullspace | 0.460 | 0.915 | 0.016 | 0.035 | 0.0063 | 1.000 |

## 2. Per concept, nullspace vs naive, alpha = 0.3 (95% bootstrap CI)

| concept | method | gain [CI] | leakage [CI] |
|---|---|---|---|
| sentiment | naive | 0.494 [0.482, 0.506] | 0.045 [0.045, 0.046] |
| sentiment | nullspace | 0.478 [0.466, 0.489] | 0.012 [0.012, 0.012] |
| sports | naive | 0.631 [0.623, 0.638] | 0.178 [0.173, 0.183] |
| sports | nullspace | 0.481 [0.474, 0.488] | 0.022 [0.021, 0.022] |
| business | naive | 0.535 [0.523, 0.547] | 0.078 [0.074, 0.081] |
| business | nullspace | 0.417 [0.409, 0.425] | 0.016 [0.016, 0.017] |
| scitech | naive | 0.560 [0.549, 0.571] | 0.080 [0.076, 0.084] |
| scitech | nullspace | 0.462 [0.453, 0.470] | 0.015 [0.014, 0.015] |

## 3. Steering contract vs baselines (judged by held-out eval probes)

| method | requests | refused | per-input target met | within budget |
|---|---|---|---|---|
| contract | 24 | 0 | 0.623 | 0.750 |
| open_loop_calibrated | 24 | 0 | 0.727 | 0.833 |
| uncalibrated_0.3 | 24 | 0 | 0.604 | 1.000 |

Contract per-input claims: says 'verified' for 0.721 of inputs; when it says so, the independent probe agrees 0.809 of the time (baselines make no per-input claim).

## 4. Visualizer faithfulness (n=1000, k=10)

| projection | trustworthiness | continuity | kNN preservation | distance Spearman | steered-point kNN |
|---|---|---|---|---|---|
| random | 0.557 | 0.623 | 0.021 | 0.109 | - |
| pca | 0.789 | 0.876 | 0.065 | 0.676 | 0.102 |
| umap | 0.925 | 0.936 | 0.352 | 0.597 | 0.176 |
