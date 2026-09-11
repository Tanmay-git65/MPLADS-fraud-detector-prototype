"""
Detector 1: COST ANOMALY
=========================
Catches works whose sanctioned cost is implausible for what was actually
built, given similar works elsewhere.

Two complementary techniques, deliberately kept separate so you can explain
each independently to a judge:

  A) PEER-GROUP BENCHMARKING (robust statistics, no ML)
     Group works by (category, state) and compute the *median* and *MAD*
     (median absolute deviation) of cost-per-unit-scale within that group.
     MAD is used instead of standard deviation because MAD is robust to the
     very outliers we're trying to find -- a normal std-dev z-score gets
     dragged around by the fraud itself, MAD barely moves. This is the
     "simple, auditable" half of the detector -- an official can verify it
     with a spreadsheet.

  B) ML RESIDUAL MODEL (RandomForestRegressor)
     Predict "expected sanctioned amount" from category, scale, state,
     district, and year using a RandomForest. The residual (actual - predicted)
     captures anomalies that simple peer-grouping might miss -- e.g. a work
     that's individually unusual across *combinations* of features, not just
     one category bucket. We use feature_importances_ / permutation
     importance for explainability instead of SHAP (SHAP isn't available
     offline in this sandbox; permutation importance answers the same
     question -- "which inputs drove this prediction" -- for our purposes).

Final sub-score = max of the two normalized signals, so a work only needs to
look anomalous by ONE credible method to be flagged, but we report BOTH
numbers in the explanation so a reviewer can see which one fired.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


def _mad_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        mad = series.std() + 1e-9  # fallback for degenerate groups
    # 0.6745 scales MAD to be comparable to a normal std-dev z-score
    return 0.6745 * (series - median) / mad


def run(works: pd.DataFrame) -> pd.DataFrame:
    df = works.copy()
    df["cost_per_scale"] = df["sanctioned_amount"] / df["scale"].clip(lower=0.01)
    df["sanction_year"] = pd.to_datetime(df["sanction_date"]).dt.year

    # ---- A) Peer-group robust z-score ----
    df["peer_zscore"] = df.groupby(["category", "state"])["cost_per_scale"].transform(_mad_zscore)
    # Normalize to 0-1 sub-score: |z| of 3 or more -> score 1.0
    peer_score = (df["peer_zscore"].abs() / 3.0).clip(0, 1)

    # ---- B) ML residual model ----
    features = df[["category", "state", "district", "scale", "sanction_year"]].copy()
    target = df["sanctioned_amount"].values

    cat_cols = ["category", "state", "district"]
    num_cols = ["scale", "sanction_year"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="passthrough")

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)),
    ])
    model.fit(features, target)
    predicted = model.predict(features)
    residual = target - predicted
    # Normalize residual by predicted amount so we're looking at *relative*
    # overspend, not raw rupee magnitude (a small hall and a big road need
    # different absolute thresholds but the same relative-overspend logic).
    relative_residual = residual / np.maximum(predicted, 1)
    df["ml_relative_residual"] = relative_residual
    # 0-1 score: 100%+ over predicted cost -> score 1.0. Only positive
    # residuals matter here (underspend isn't a fraud signal by itself).
    ml_score = relative_residual.clip(0, 1.0)

    df["cost_anomaly_score"] = np.maximum(peer_score, ml_score)

    def explain(row):
        reasons = []
        if abs(row["peer_zscore"]) >= 2.5:
            reasons.append(
                f"cost is {row['peer_zscore']:.1f}x the typical spread for "
                f"'{row['category']}' works in {row['state']} (peer-group check)"
            )
        if row["ml_relative_residual"] >= 0.5:
            reasons.append(
                f"ML model predicted ~₹{predicted[row.name]:,.0f} for a work "
                f"of this type/scale/location, actual sanctioned amount is "
                f"₹{row['sanctioned_amount']:,.0f} "
                f"({row['ml_relative_residual']*100:.0f}% above prediction)"
            )
        return "; ".join(reasons) if reasons else ""

    df["cost_anomaly_reason"] = df.apply(explain, axis=1)

    # Feature importance summary -- printed once for the "how does the model
    # decide" slide in your pitch deck.
    result = permutation_importance(model, features, target, n_repeats=5, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=features.columns).sort_values(ascending=False)

    return df[["work_id", "peer_zscore", "ml_relative_residual", "cost_anomaly_score", "cost_anomaly_reason"]], importances


if __name__ == "__main__":
    works = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\works.csv")
    scored, importances = run(works)
    print("Feature importances (what drives the cost prediction):")
    print(importances, "\n")
    print("Top 10 most cost-anomalous works:")
    print(scored.sort_values("cost_anomaly_score", ascending=False).head(10).to_string(index=False))
