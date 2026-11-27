# Extras (BAAI/bge-small-en-v1.5)

## A. Steered search on AG News test (300 queries per target, top-10)

Mean over targets sports / business / sci-tech. Queries come from the other 3 classes.

| method | alpha | precision@10 of target | drift to unrelated classes | relevance to original query |
|---|---|---|---|---|
| random | 0.1 | 0.049 | 0.107 | 0.717 |
| random | 0.2 | 0.049 | 0.106 | 0.716 |
| random | 0.3 | 0.049 | 0.107 | 0.715 |
| unsteered | 0.0 | 0.049 | 0.107 | 0.717 |
| naive | 0.1 | 0.071 | 0.100 | 0.716 |
| naive | 0.2 | 0.113 | 0.091 | 0.713 |
| naive | 0.3 | 0.205 | 0.077 | 0.704 |
| lowdin | 0.1 | 0.070 | 0.100 | 0.716 |
| lowdin | 0.2 | 0.112 | 0.091 | 0.713 |
| lowdin | 0.3 | 0.194 | 0.076 | 0.705 |
| nullspace | 0.1 | 0.065 | 0.105 | 0.716 |
| nullspace | 0.2 | 0.095 | 0.101 | 0.714 |
| nullspace | 0.3 | 0.150 | 0.093 | 0.708 |

## B. Concept erasure (fresh linear reader trained after erasure; chance = 0.5)

| concept | k removed dims | target acc (ours) | target acc (random dims) | other concepts acc (ours) | STS-B drop (ours) |
|---|---|---|---|---|---|
| sentiment | 0 | 0.870 | 0.870 | 0.916 | 0.0000 |
| sentiment | 1 | 0.802 | 0.873 | 0.913 | 0.0037 |
| sentiment | 2 | 0.615 | 0.870 | 0.914 | 0.0052 |
| sentiment | 4 | 0.563 | 0.873 | 0.917 | 0.0064 |
| sentiment | 8 | 0.563 | 0.872 | 0.913 | 0.0045 |
| sentiment | 16 | 0.563 | 0.870 | 0.913 | 0.0045 |
| sports | 0 | 0.977 | 0.977 | 0.881 | 0.0000 |
| sports | 1 | 0.748 | 0.978 | 0.867 | 0.0004 |
| sports | 2 | 0.620 | 0.977 | 0.862 | 0.0006 |
| sports | 4 | 0.608 | 0.975 | 0.854 | 0.0009 |
| sports | 8 | 0.583 | 0.977 | 0.841 | 0.0024 |
| sports | 16 | 0.583 | 0.977 | 0.841 | 0.0024 |
| business | 0 | 0.870 | 0.870 | 0.916 | 0.0000 |
| business | 1 | 0.767 | 0.873 | 0.913 | -0.0001 |
| business | 2 | 0.657 | 0.870 | 0.909 | 0.0001 |
| business | 4 | 0.638 | 0.867 | 0.904 | 0.0012 |
| business | 8 | 0.635 | 0.867 | 0.889 | 0.0016 |
| business | 16 | 0.635 | 0.867 | 0.889 | 0.0016 |
| scitech | 0 | 0.902 | 0.902 | 0.906 | 0.0000 |
| scitech | 1 | 0.770 | 0.902 | 0.904 | 0.0008 |
| scitech | 2 | 0.623 | 0.903 | 0.898 | 0.0011 |
| scitech | 4 | 0.582 | 0.898 | 0.880 | 0.0016 |
| scitech | 8 | 0.583 | 0.905 | 0.867 | 0.0020 |
| scitech | 16 | 0.583 | 0.903 | 0.867 | 0.0020 |
