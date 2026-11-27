# Robustness checks (BAAI/bge-small-en-v1.5, alpha = 0.3, mean over 4 concepts)

Reader accuracy on held-out test (trained on the probe split, never seen by the method):

| concept | linear | MLP | kNN |
|---|---|---|---|
| sentiment | 0.870 | 0.833 | 0.858 |
| sports | 0.977 | 0.982 | 0.962 |
| business | 0.870 | 0.855 | 0.875 |
| scitech | 0.902 | 0.902 | 0.915 |

## A + B. Gain and leakage under three reader families

| method | linear gain | linear leak | MLP gain | MLP leak | kNN gain | kNN leak | STS-B drop |
|---|---|---|---|---|---|---|---|
| random | 0.023 | 0.023 | 0.043 | 0.033 | 0.018 | 0.046 | 0.0008 |
| naive | 0.555 | 0.095 | 0.810 | 0.106 | 0.493 | 0.127 | 0.0062 |
| lowdin | 0.539 | 0.059 | 0.798 | 0.068 | 0.466 | 0.104 | 0.0067 |
| nullspace | 0.460 | 0.016 | 0.698 | 0.027 | 0.372 | 0.083 | 0.0063 |
| prefix | 0.139 | 0.035 | 0.234 | 0.049 | 0.120 | 0.053 | 0.0299 |

## B. Steered search on AG News test (300 queries per target, top-10)

| method | precision@10 of target | relevance to original query |
|---|---|---|
| unsteered | 0.049 | 0.717 |
| prefix | 0.068 | 0.715 |
| naive | 0.205 | 0.704 |
| nullspace | 0.150 | 0.708 |
