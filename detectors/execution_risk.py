"""
Detector 4: EXECUTION RISK (delay/stall + payment-velocity mismatch)
=======================================================================
Two related but distinct inefficiency/fraud signals, both about whether
money and physical progress are moving in step with each other and with
MPLADS' one-year completion norm.

  A) STALLED / OVERDUE
     MPLADS guidelines expect works to complete within ~1 year of sanction.
     We compute `overdue_ratio = days_since_sanction / expected_duration_days`
     for every work still "In Progress" or "Sanctioned". A ratio > 1.3 means
     the work has already blown past its expected window by 30%+; the higher
     the ratio, the more urgent. This is a genuinely PREDICTIVE signal for
     works not yet complete (it tells you which are AT RISK of becoming a
     dead/abandoned project before they're officially declared stalled) --
     this is what turns the platform from "detect fraud after the fact" into
     "flag inefficiency before it becomes a write-off," which is what the
     problem statement means by "predictive insights."

  B) VELOCITY MISMATCH
     Compare cumulative amount paid so far against work status. If a work is
     NOT marked "Completed" but 95-100% of its sanctioned funds have already
     been disbursed, that's a red flag: either progress reporting is stale
     (an integrity/process problem), or funds were released without
     commensurate work (a fraud/diversion problem). Either way it needs
     human eyes.
"""

import numpy as np
import pandas as pd
from datetime import datetime

TODAY = datetime(2026, 9, 4)
OVERDUE_RATIO_FLAG = 1.3
VELOCITY_FLAG_THRESHOLD = 0.95


def run(works: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    df = works.copy()
    df["sanction_date_dt"] = pd.to_datetime(df["sanction_date"])
    df["days_since_sanction"] = (TODAY - df["sanction_date_dt"]).dt.days
    df["overdue_ratio"] = df["days_since_sanction"] / df["expected_duration_days"].clip(lower=1)

    not_done = df["status"] != "Completed"
    df["stall_score"] = 0.0
    df.loc[not_done, "stall_score"] = ((df.loc[not_done, "overdue_ratio"] - 1) / 2).clip(0, 1)

    df["stall_reason"] = ""
    stalled_mask = not_done & (df["overdue_ratio"] > OVERDUE_RATIO_FLAG)
    df.loc[stalled_mask, "stall_reason"] = df.loc[stalled_mask].apply(
        lambda r: (f"sanctioned {r['days_since_sanction']} days ago but expected duration was only "
                   f"{r['expected_duration_days']} days ({r['overdue_ratio']:.1f}x overdue), still "
                   f"marked '{r['status']}'"),
        axis=1,
    )

    # ---- Velocity mismatch ----
    paid = payments.groupby("work_id")["amount"].sum().rename("total_paid")
    df = df.merge(paid, on="work_id", how="left")
    df["total_paid"] = df["total_paid"].fillna(0)
    df["disbursed_fraction"] = (df["total_paid"] / df["sanctioned_amount"].clip(lower=1)).clip(0, 2)

    velocity_mask = not_done & (df["disbursed_fraction"] >= VELOCITY_FLAG_THRESHOLD)
    df["velocity_score"] = 0.0
    df.loc[velocity_mask, "velocity_score"] = 1.0
    df["velocity_reason"] = ""
    df.loc[velocity_mask, "velocity_reason"] = df.loc[velocity_mask].apply(
        lambda r: (f"{r['disbursed_fraction']*100:.0f}% of sanctioned funds already disbursed, but "
                   f"work is still marked '{r['status']}' -- payment has outpaced physical progress"),
        axis=1,
    )

    df["execution_risk_score"] = np.maximum(df["stall_score"], df["velocity_score"])
    df["execution_risk_reason"] = [
        (a + " | " + b).strip(" |")
        for a, b in zip(df["stall_reason"].tolist(), df["velocity_reason"].tolist())
    ]

    return df[["work_id", "overdue_ratio", "stall_score", "disbursed_fraction", "velocity_score",
               "execution_risk_score", "execution_risk_reason"]]


if __name__ == "__main__":
    works = pd.read_csv("/home/claude/mplads_ai/data/works.csv")
    payments = pd.read_csv("/home/claude/mplads_ai/data/payments.csv")
    scored = run(works, payments)
    flagged = scored[scored.execution_risk_score > 0.3].sort_values("execution_risk_score", ascending=False)
    print(f"Flagged {len(flagged)} works for execution risk (stall/velocity)\n")
    print(flagged.head(10)[["work_id", "overdue_ratio", "disbursed_fraction", "execution_risk_score", "execution_risk_reason"]].to_string(index=False))
