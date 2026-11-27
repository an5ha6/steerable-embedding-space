# Steering evaluation (sentence-transformers/all-MiniLM-L6-v2)

Test negatives per concept: {'sentiment': 300, 'sports': 300, 'business': 300, 'scitech': 300}. STS-B test pairs: 690 (base Spearman 0.819). Runtime 622.9s.

Eval-probe accuracy (held-out readers, never used by the method): sentiment 0.770, sports 0.982, business 0.882, scitech 0.883

## 1. Sweep at alpha = 0.3 (mean over concepts)

| method | gain (target prob) | success rate | leakage (other concepts) | selectivity = leak/gain | STS-B drop | monotonicity |
|---|---|---|---|---|---|---|
| random | 0.019 | 0.140 | 0.023 | 1.192 | 0.0003 | 0.778 |
| naive | 0.490 | 0.941 | 0.085 | 0.174 | 0.0002 | 1.000 |
| lowdin | 0.472 | 0.915 | 0.052 | 0.111 | -0.0004 | 1.000 |
| nullspace | 0.405 | 0.792 | 0.016 | 0.038 | -0.0008 | 1.000 |

## 2. Per concept, nullspace vs naive, alpha = 0.3 (95% bootstrap CI)

| concept | method | gain [CI] | leakage [CI] |
|---|---|---|---|
| sentiment | naive | 0.428 [0.420, 0.436] | 0.049 [0.048, 0.050] |
| sentiment | nullspace | 0.422 [0.414, 0.430] | 0.023 [0.022, 0.024] |
| sports | naive | 0.552 [0.542, 0.562] | 0.143 [0.139, 0.148] |
| sports | nullspace | 0.382 [0.371, 0.392] | 0.014 [0.013, 0.014] |
| business | naive | 0.482 [0.471, 0.492] | 0.075 [0.072, 0.079] |
| business | nullspace | 0.382 [0.373, 0.391] | 0.014 [0.013, 0.014] |
| scitech | naive | 0.500 [0.488, 0.511] | 0.074 [0.070, 0.078] |
| scitech | nullspace | 0.432 [0.422, 0.442] | 0.012 [0.012, 0.013] |

## 3. Steering contract vs baselines (judged by held-out eval probes)

| method | requests | refused | per-input target met | within budget |
|---|---|---|---|---|
| contract | 24 | 0 | 0.585 | 0.708 |
| open_loop_calibrated | 24 | 0 | 0.627 | 0.958 |
| uncalibrated_0.3 | 24 | 0 | 0.538 | 1.000 |

Contract per-input claims: says 'verified' for 0.714 of inputs; when it says so, the independent probe agrees 0.723 of the time (baselines make no per-input claim).

## 4. Visualizer faithfulness (n=1000, k=10)

| projection | trustworthiness | continuity | kNN preservation | distance Spearman | steered-point kNN |
|---|---|---|---|---|---|
| random | 0.550 | 0.619 | 0.022 | 0.089 | - |
| pca | 0.754 | 0.840 | 0.065 | 0.429 | 0.114 |
| umap | 0.914 | 0.919 | 0.345 | 0.401 | 0.147 |
