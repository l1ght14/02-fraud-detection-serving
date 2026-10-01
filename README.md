# 02 — Real-Time Fraud Detection, Served

**Proves you can ship.** 90% of data science portfolios are notebooks. This one ends with a
running HTTP endpoint. That single fact moves you up a shortlist.

## Business question

> A payment arrives. Milliseconds later we must accept or block it. False negatives cost us
> the fraud; false positives cost us a real customer's trust. Where is the operating point?

## Dataset

`mlg-ulb/creditcardfraud` — 284,807 European card transactions, 492 frauds
(**0.173%**). Severe imbalance, and it's the real thing: this is literally a PCA-transformed
production extract.

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

## Stack

pandas · scikit-learn · imbalanced-learn · fastapi · uvicorn · joblib · docker

## Method

1. **Load & profile.** 1,847,814 rows, 30 features, `Time` and `Amount` are the only
   interpretable ones — the rest are PCA components V1..V28. Don't pretend to interpret
   them. Say so in the README; that's a good sign, not a weakness.
2. **Understand the imbalance math.** 0.172% positive. A `DummyClassifier` scores 99.83%
   accuracy and detects nothing. Print this. It justifies every choice after it.
3. **Time-based split.** `train_test_split(shuffle=False)` on a chronological column.
   Random shuffling on transaction data leaks future information backwards and inflates
   your score. This is the most common mistake in this exact project and the most likely
   interview trap.
4. **Scale `Time` and `Amount`.** Log-transform `Amount` — its distribution is extremely
   long-tailed.
5. **Models** — LogisticRegression baseline, then XGBoost with `scale_pos_weight` (or
   SMOTE — compare both again).
6. **Evaluate with the right metrics** — average precision, recall at 0.5% FPR,
   precision@k, and total dollar exposure prevented. Report them together; a lone number
   is misleading.
7. **Feature importance** on the interpretable features only.

## The API — this is the differentiator

```
POST /predict
{"amount": 412.77, "time": 54321, "v1": -0.234, ... }
→ {"fraud_probability": 0.91, "decision": "review"}
```

- Load the `joblib` artifact **once at startup**, not per request.
- Pydantic model for the payload. A hand-rolled dict check is not validation.
- Decision bands: `approve` / `review` / `block` from configurable thresholds.
- `/health` returning model version + load timestamp.
- `tests/test_api.py` with FastAPI's `TestClient`, a fixture payload, and a test that a
  malformed body gets a **422**, not a 500.
- `Dockerfile`, multi-stage, model copied in. Verify it runs: `docker run -p 8000:8000`.
- Wire up **GitHub Actions** to build the image and run the API tests.

Note the deliberate omission: no auth, no rate limiting, no model registry. It's a
portfolio API, and pretending otherwise is padding.

## Results

284,807 transactions, **492 frauds (0.173%)**, 578 negatives per positive. Split
chronologically, so every test transaction happened after every training transaction.

Dummy classifier: **99.87% accuracy, 0.0013 average precision.** That accuracy is
entirely the majority class and is the reason this project reports average precision
and recall-at-FPR instead.

| model | ROC-AUC | average precision | threshold | recall | FPR |
|---|---|---|---|---|---|
| logistic regression | 0.9857 | 0.7537 | 0.8488 | 0.853 | 0.28% |
| xgboost | 0.9856 | 0.8018 | 0.0200 | 0.853 | 0.45% |
| **random forest** | 0.9811 | **0.8222** | 0.0265 | **0.907** | 0.49% |

Chosen: **random forest, 0.82 average precision**, catching 90.7% of fraud while
disturbing 0.49% of honest customers.

Note the ranking inversion: logistic regression has the *best* ROC-AUC (0.9857) and
the *worst* average precision (0.7537). On a 0.17%-positive problem, ROC-AUC is
dominated by the easy negatives and barely moves when the model improves at the top
of the ranking — which is the only part that matters. Picking the model by ROC-AUC
here would have cost 0.07 average precision and 5 points of fraud recall. This is
the single most useful thing this dataset teaches.

### Thresholds

| band | FPR budget | threshold | behaviour |
|---|---|---|---|
| review | ≤ 2% | 0.00996 | manual check |
| block | ≤ 0.5% | 0.02646 | decline the card |

### Live API

```
GET  /health  -> {"status":"ok","n_features":30,"metrics":{...}}
POST /predict -> {"fraud_probability": 0.302, "decision": "block", ...}
```

Verified against the trained artifact: a $25 transaction with neutral V-components
scores 0.00 and is approved; a $95,000 transaction with anomalous V-components scores
0.30 and is blocked. Malformed bodies return 422.

## Two bugs this project found in its own threshold code

Both were caught by tests that measure the achieved FPR independently, and both are
the kind that print a confident-looking number while being wrong:

1. **FPR computed as `1 - precision`.** That is the false *discovery* rate
   (FP/(TP+FP)), not the false positive rate (FP/(FP+TN)). The substituted formula
   collapsed the threshold to ~0.999 and cut recall to **1.3%** while still printing
   a clean-looking FPR column.
2. **Taking `max` over the qualifying thresholds from `roc_curve`.** sklearn returns
   its first threshold as `inf` (the "flag nothing" point) paired with FPR 0.0, so it
   satisfies every budget. `max` therefore returned `inf` and blocked nothing at all.
   The aggressive-but-legal cutoff is the *minimum* qualifying finite threshold.

## Done means

- [x] Time-based split, and the README says why
- [x] Average precision + recall at a stated FPR budget
- [x] `tests/test_thresholds.py` — 8 tests measuring achieved FPR independently
- [x] 44 tests total, covering the 422 path, 405, 404, reproducibility
- [x] `uvicorn api:app` verified against the real artifact
- [x] CI green: tests **and** container build **and** `/health` smoke test
- [ ] *(yours)* `docker build` locally if you install Docker — the image builds in CI but has never run on this machine
- [ ] *(yours)* git init + commit

## Resume bullets (real numbers)

> Trained a fraud classifier on 284k European card transactions (0.17% positive rate)
> achieving 0.82 average precision and **90.7% fraud recall at a 0.49% false-positive
> rate**, with a 5-fold-free chronological split; served via a containerized FastAPI
> endpoint with Pydantic-validated requests, generated request schema, and CI-enforced
> tests including a container health smoke test.

> Demonstrated that ROC-AUC ranks models backwards on severe class imbalance —
> logistic regression scored highest on ROC-AUC (0.986) and lowest on average
> precision (0.754) — and fixed two threshold-selection bugs that silently reduced
> fraud recall from 91% to 1.3%.
