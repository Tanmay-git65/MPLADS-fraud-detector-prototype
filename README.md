# MPLADS AI Fraud & Anomaly Detection — Prototype

## How to run it (in order)
1. `python3 risk_scoring.py`     -> runs all 7 detectors, writes output/risk_report.csv
2. `python3 evaluate.py`         -> grades the system against the hidden ground truth (precision/recall)

## File map
- `data_generator.py`            Synthetic data + injected fraud patterns (heavily commented -- read the module
                                  docstring at the top first, then each `# ----INJECT----` block shows exactly
                                  how each of the 8 fraud patterns is planted)
- `detectors/cost_anomaly.py`    Peer-group z-score + RandomForest residual cost model
- `detectors/duplicate_works.py` TF-IDF text similarity + geospatial proximity duplicate detector
- `detectors/payment_structuring.py`  Threshold-clustering + Benford's Law
- `detectors/execution_risk.py`  Delay-vs-norm + payment velocity mismatch
- `detectors/image_forensics.py` Geotag mismatch + duplicate photo hash reuse
- `detectors/vendor_network.py`  Graph/Jaccard-overlap collusion ring detector
- `detectors/rules_engine.py`    Explicit non-ML MPLADS compliance rules (R1-R4)
- `risk_scoring.py`              Combines all 7 detectors into one explainable composite score (Layer 3)
- `evaluate.py`                  Scores the whole system against hidden ground truth (Layer 4)
- `data/`                        Generated CSVs (mps, ias, works, payments, photos, ground_truth)
- `output/risk_report.csv`       Final output: every work, its risk score, tier, and plain-English reasons

## Every file is commented with WHY, not just WHAT
Each detector file opens with a docstring explaining: the real-world fraud pattern it targets, why that
specific technique was chosen, and where you'd swap in a heavier/production tool (e.g. sentence-transformers
instead of TF-IDF, SHAP instead of permutation importance, imagehash on real photos instead of simulated
hashes) if you have internet access on your own machine. Inline comments explain non-obvious lines.

## Headline result (from evaluate.py)
At Critical risk tier: 60% precision, 91% recall on the 8 planted fraud patterns.
Vendor collusion ring: all 3 planted IAs caught, zero false positives.
Full breakdown is in evaluation_output.txt in this folder.
