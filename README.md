# Steerable Embedding Space

**Resurrect 2026, Problem 5:** an embedding model whose representations can be steered in controlled ways, plus a visualizer whose fidelity is measured.
**Bonus:** the same embedding space also solves **Problem 6** (ambiguity detection and resolution) and **Problem 1** (research scope map over 1,200 papers).

Everything runs on a laptop CPU. One command reproduces every number in this file.

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt          # CPU torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/eval_steering.py          # Problem 5  -> results/steering.md / .json
python scripts/eval_ambiguity.py         # Problem 6  -> results/ambiguity.json
python scripts/eval_papers.py            # Problem 1  -> results/papers.json
streamlit run app.py --server.address localhost   # demo (first load ~3 min, then cached)
```

## What we built

A **steerable adapter over a frozen encoder** (`BAAI/bge-small-en-v1.5`, 384-D). We did not train a new foundation model on a laptop, and we don't claim to.

`e' = normalize(e + Σ_c α_c · d_c)`

1. **Null-space concept directions** (`steer/engine.py`). For each concept (sentiment, sports, business, sci/tech), the direction is the difference of class means, with every *other* concept's linear readout projected out. Steering sports should not change what a reader concludes about business.
2. **Steering contract.** The user does not move a raw slider. They state: *concept, wanted effect, maximum quality loss, maximum leakage*. The system then:
   - picks the largest α the budgets allow, using calibration curves;
   - searches per input for the smallest α that reaches the target (closed loop on a verifier probe), raised by a **conformal safety margin** fitted against 5 independent calibration readers;
   - refuses when the budget can't reach the target;
   - returns a per-input "verified" claim.
3. **Visualizer with a fidelity certificate** (`steer/fidelity.py`, `app.py`). A UMAP map with trustworthiness, continuity and kNN preservation shown next to it, measured against the true 384-D geometry. Steering paths are drawn on the map.

## Evaluation

### Definition of success
Steering is (a) **controllable**: it moves the target concept; (b) **selective**: it does not move other concepts; (c) **predictable**: the effect grows monotonically with α, and the contract's claims hold; (d) **quality-preserving**: general semantic similarity survives. The visualizer is (e) **faithful** to the high-dimensional neighbourhoods.

### Protocol (why the numbers can be trusted)
- **Data:** SST-2 (sentiment), AG News (sports / business / sci-tech), STS-B test. All public. Fixed seed 0. 1,500 sentences per side per concept.
- **Four disjoint splits per concept:** `fit` learns directions and the verifier; `cal` calibrates the contract; `probe` trains the **evaluation probes**; `test` produces the reported numbers on 300 sentences per concept, steered from the negative class.
- **Every reported effect is read by evaluation probes the method never saw.** Their held-out accuracy: sentiment 0.870, sports 0.977, business 0.870, sci-tech 0.902. All probes are trained on the same amount of data (300 per side), so they share one probability scale. An earlier version without this got contract precision of 0.05 (see failure cases).
- **STS-B is split in half:** calibration vs test. Only the query side is steered (asymmetric), which is the harder and more realistic case.
- 95% bootstrap CIs (1,000 resamples).

### Metrics and why
| metric | measures | does NOT measure / limitation |
|---|---|---|
| gain = Δ P(target) under eval probe | controllability | a linear reader's view, not human judgement |
| success rate (P > 0.5 after steering) | does it actually flip the class | threshold-dependent |
| leakage = mean \|Δ P(other concept)\| | selectivity | only covers the 4 concepts we track |
| STS-B Spearman drop | is general semantic similarity preserved | one benchmark; asymmetric steering only |
| monotonicity = Spearman(α, P) | predictability of dose → response | per-sentence averaged |
| contract claim precision | when it says "verified", is it true under an independent reader | concept-dependent (see business) |
| trustworthiness / continuity (k=10) | 2D neighbours are real / real neighbours stay close | local structure only; global UMAP distances aren't meaningful |

### Results: Problem 5 (`results/steering.md`)

**Sweep at α = 0.3, mean over 4 concepts × 300 test sentences**

| method | gain | success | leakage | leak / gain | STS-B drop | monotonicity |
|---|---|---|---|---|---|---|
| random direction (control) | 0.023 | 0.112 | 0.023 | 0.997 | 0.0008 | 0.928 |
| naive difference-of-means | 0.555 | 0.974 | 0.095 | 0.172 | 0.0062 | 1.000 |
| Löwdin-orthogonal directions | 0.539 | 0.974 | 0.059 | 0.110 | 0.0067 | 1.000 |
| **null-space (ours)** | 0.460 | 0.915 | **0.016** | **0.035** | 0.0063 | 1.000 |

- **Leakage falls 5.9× vs naive (0.095 → 0.016) and 3.7× vs orthogonal directions, keeping 83% of the gain.** Per concept, all CIs are disjoint; e.g. sports leakage 0.178 [0.173, 0.183] → 0.022 [0.021, 0.022].
- **Insight:** making the *directions* mutually orthogonal (Löwdin) is not the same as making the *effects* independent. Our null-space directions are not orthogonal to each other (sports·sci-tech cosine 0.56), yet they leak 3.7× less, because what matters is orthogonality to the other concepts' *readouts*.
- STS-B Spearman (base 0.853) drops by 0.006 at α = 0.3 and 0.02–0.04 at α = 0.6.

**Steering contract (24 requests = 4 concepts × 3 targets × 2 budgets)**

| method | per-input target met | within budget | per-input claims |
|---|---|---|---|
| contract (ours) | 0.623 | 0.750 | "verified" on 72%; **true 81% of the time** under the independent probe |
| fixed α calibrated on the mean | 0.727 | 0.833 | none |
| uncalibrated slider α = 0.3 | 0.604 | 1.000 | none |

Honest reading: the contract uses the *smallest* α that should reach the target, so its average hit rate is not higher than a fixed α. What it adds is a **per-input, checkable claim**: 81% precise overall, 0.92–0.98 for sports, 0.55 for business. The baselines can't tell you which inputs worked.

**Visualizer faithfulness (1,000 points, k = 10)**

| projection | trustworthiness | continuity | kNN kept | distance Spearman |
|---|---|---|---|---|
| random projection (control) | 0.557 | 0.623 | 0.021 | 0.109 |
| PCA | 0.789 | 0.876 | 0.065 | **0.676** |
| **UMAP (used)** | **0.925** | **0.936** | **0.352** | 0.597 |

UMAP is far better locally; PCA is better globally. The UI says so and shows these numbers next to the map. Steered points placed with `transform` keep only 0.18 of their true kNN, so the drawn steering path is a guide, not a measurement. Claims come from the 384-D numbers.

### Failure cases
- **Business** is the hardest concept: contract precision 0.55. It shares vocabulary with sci-tech and world news, so its direction loses the most signal when the other readouts are projected out (gain 0.535 → 0.417).
- **First contract version** (verifier trained on more data than the eval probe): claim precision 0.05. The verifier's probabilities were on a different scale. Fixed with equal-size probes plus a 5-reader conformal margin. The margin's nominal 90% coverage is still not reached (81%), because the calibration readers come from the same pool and understate how much an independent reader differs.
- Over-steering: at α = 0.5, the sentiment STS-B drop reaches 0.027 and keeps growing. The contract caps α so this stays inside the user's budget.
- Only linear concepts; only the 4 concepts we track are protected from leakage.

## Bonus: Problem 6, ambiguity detection and resolution (`steer/ambiguity.py`)

**Idea, in the same space:** a noun's coarse senses (WordNet supersenses, e.g. *bank* → group/organization vs. place) are prototype points. A text is **ambiguous when its context moves the sense profile no further than the bare word does**: the context carries no sense information. Then the system asks a clarifying question, and the chosen sense becomes a steering direction (Problem 5 machinery).

**Definitions:** target words are nouns with ≥ 2 supersenses each covering ≥ 20% of their SemCor occurrences (no safe default). *Clear* = a real SemCor sentence, sense-annotated by humans. *Clear-short* = a short request plus a sense hint ("Tell me about the bank (the place)"), which controls for "the detector just flags short text". *Ambiguous* = a short request with no hint. Words are split dev/test **by word**, and the threshold is fitted on dev only.

AMB_RESULTS

## Bonus: Problem 1, research scope map (`steer/papers.py`)

1,200 arXiv abstracts from 8 CS areas (150 each, labels hidden from the method) are mapped and clustered. Each cluster gets automatic names and a **"start here" paper** (closest to its centroid). Readers steer the reading list with their own concept ("more theoretical, less systems") using the same steering machinery.

| measured (5 seeds) | ours (embedding) | baseline TF-IDF + LSA |
|---|---|---|
| NMI vs true areas | **0.755 ± 0.004** | 0.580 ± 0.009 |
| ARI vs true areas | **0.746 ± 0.005** | 0.460 ± 0.010 |

Map fidelity: trustworthiness 0.953, continuity 0.953. The clusters are named automatically, e.g. *"secret, encryption, cipher, attacks"* (cs.CR) and *"manipulators, kinematic"* (cs.RO).
**Not measured:** whether people understand the field faster. That needs a user study, which we did not run.

## Layout
```
steer/encoder.py    frozen encoder + disk cache
steer/engine.py     directions (random/naive/lowdin/nullspace), steering, Contract
steer/fidelity.py   trustworthiness, continuity, kNN preservation, out-of-sample fidelity
steer/ambiguity.py  bonus #6
steer/papers.py     bonus #1
scripts/eval_*.py   reproducible evaluations -> results/
app.py              Streamlit demo (5 tabs, incl. live evidence)
```
