"""
Detector 7: COMPLIANCE RULES ENGINE
=======================================
Deliberately NOT machine learning. Encodes explicit, known MPLADS
guidelines directly as auditable rules. This matters for the pitch: judges
(and a real Ministry) will want to see that you're not relying on a
statistical model to catch things that are simply auditable facts -- e.g.
"sanctioned amount exceeds the per-work ceiling" is a yes/no compliance
check, not something that needs a probability score.

Rules implemented (each individually explainable, individually toggleable):
  R1: Sanctioned amount should not deviate more than a fixed % from the MP's
      recommended amount without a documented reason (data proxy: >20% jump
      flagged for review).
  R2: A work should not remain "Sanctioned" (i.e. not even started) for more
      than ~90 days -- a sign of a stalled handoff to the Implementing Agency.
  R3: Total payments for a work should never exceed its sanctioned amount
      (a basic financial-control check that should never fail in a
      well-governed system -- if it does, it's a serious integrity flag,
      not just an anomaly).
  R4: MP's cumulative recommended amount in a year should not exceed their
      annual entitlement (a hard ceiling check).

In production this module would be the easiest to keep updated: whenever
MPLADS guidelines are revised, only this file changes -- no retraining.
"""

import pandas as pd


REC_VS_SANCTION_DEVIATION_FLAG = 0.20
STALLED_SANCTION_DAYS = 90


def run(works: pd.DataFrame, payments: pd.DataFrame, mps: pd.DataFrame) -> pd.DataFrame:
    df = works.copy()
    flags = []

    # R1: sanctioned amount deviates significantly from recommended amount
    df["rec_vs_sanction_pct"] = (df["sanctioned_amount"] - df["recommended_amount"]) / df["recommended_amount"].clip(lower=1)
    r1 = df["rec_vs_sanction_pct"].abs() > REC_VS_SANCTION_DEVIATION_FLAG

    # R2: stuck at "Sanctioned" (not started) too long
    df["sanction_date_dt"] = pd.to_datetime(df["sanction_date"])
    days_since = (pd.Timestamp("2026-09-04") - df["sanction_date_dt"]).dt.days
    r2 = (df["status"] == "Sanctioned") & (days_since > STALLED_SANCTION_DAYS)

    # R3: total payments exceed sanctioned amount (hard financial-control breach)
    total_paid = payments.groupby("work_id")["amount"].sum().rename("total_paid")
    df = df.merge(total_paid, on="work_id", how="left")
    df["total_paid"] = df["total_paid"].fillna(0)
    r3 = df["total_paid"] > df["sanctioned_amount"] * 1.001  # small float tolerance

    df["rule_score"] = 0.0
    df["rule_reason"] = ""

    # NOTE: Use list comprehensions throughout to avoid Arrow-backed dtype
    # incompatibility with Series string concatenation (newer pandas).
    df.loc[r1, "rule_score"] = 0.5
    r1_reasons = df.loc[r1].apply(
        lambda r: f"[R1] sanctioned amount deviates {r['rec_vs_sanction_pct']*100:.0f}% from MP-recommended amount without documented justification; ", axis=1)
    if r1.any():
        r1_idx = df.index[r1]
        df.loc[r1_idx, "rule_reason"] = [
            str(a) + str(b) for a, b in zip(df.loc[r1_idx, "rule_reason"].tolist(), r1_reasons.tolist())
        ]

    df.loc[r2, "rule_score"] = df.loc[r2, "rule_score"].clip(lower=0.4)
    if r2.any():
        r2_idx = df.index[r2]
        df.loc[r2_idx, "rule_reason"] = [
            str(s) + "[R2] work sanctioned but not started within 90 days; "
            for s in df.loc[r2_idx, "rule_reason"].tolist()
        ]

    df.loc[r3, "rule_score"] = 1.0  # hard breach -- always maximum severity
    r3_reasons = df.loc[r3].apply(
        lambda r: f"[R3] SEVERE: total payments (₹{r['total_paid']:,.0f}) exceed sanctioned amount (₹{r['sanctioned_amount']:,.0f}) -- financial control breach; ", axis=1)
    if r3.any():
        r3_idx = df.index[r3]
        df.loc[r3_idx, "rule_reason"] = [
            str(a) + str(b) for a, b in zip(df.loc[r3_idx, "rule_reason"].tolist(), r3_reasons.tolist())
        ]

    # R4: MP annual entitlement ceiling check (portfolio-level, reported separately)
    df["rec_year"] = pd.to_datetime(df["recommended_date"]).dt.year
    mp_year_totals = df.groupby(["mp_id", "rec_year"])["recommended_amount"].sum().reset_index()
    mp_with_entitlement = mp_year_totals.merge(mps[["mp_id", "annual_entitlement"]], on="mp_id", how="left")
    mp_with_entitlement["over_entitlement"] = mp_with_entitlement["recommended_amount"] > mp_with_entitlement["annual_entitlement"]
    r4_breaches = mp_with_entitlement[mp_with_entitlement["over_entitlement"]]

    return df[["work_id", "rule_score", "rule_reason"]], r4_breaches


if __name__ == "__main__":
    works = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\works.csv")
    payments = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\payments.csv")
    mps = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\mps.csv")
    scored, entitlement_breaches = run(works, payments, mps)
    flagged = scored[scored.rule_score > 0]
    print(f"Flagged {len(flagged)} works by rule engine (R1/R2/R3)\n")
    print(flagged.sort_values("rule_score", ascending=False).head(10).to_string(index=False))
    print(f"\nMP annual entitlement breaches (R4): {len(entitlement_breaches)}")
