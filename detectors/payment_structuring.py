"""
Detector 3: PAYMENT STRUCTURING
=================================
"Structuring" is a well-known financial-crime pattern: instead of one large
payment (which triggers extra scrutiny/approval above a threshold), the
amount is split into several smaller payments each just under the
threshold. This detector looks for exactly that, at the WORK level (all
payments belonging to one work_id).

Two techniques:

  A) THRESHOLD-CLUSTERING (the direct check)
     For each work, look at all its payments. If 2+ payments each fall in
     the "just under a known approval threshold" band (85-99% of the
     threshold) AND they were made close together in time (within 10 days),
     that's a strong, explainable structuring signal -- this is exactly
     what a bank AML analyst looks for.

  B) BENFORD'S LAW DEVIATION (portfolio-level sanity check)
     In naturally-occurring financial datasets, the leading digit of amounts
     follows Benford's Law (1 appears ~30% of the time, 9 only ~4.6%) --
     NOT a uniform distribution. Fabricated/manipulated numbers tend to
     deviate from this. We compute this at the IMPLEMENTING AGENCY level
     (an IA's full set of payments) since Benford's Law needs a reasonably
     large sample to be meaningful -- it's not reliable per-work. An IA
     whose payment amounts deviate significantly from the expected Benford
     distribution gets an elevated "portfolio risk" flag, which feeds into
     the composite score as context alongside the work-level flags.
"""

import numpy as np
import pandas as pd

APPROVAL_THRESHOLD = 1_000_000  # INR 10 lakh -- representative approval ceiling
BAND_LOW, BAND_HIGH = 0.80, 0.99
CLUSTER_WINDOW_DAYS = 10

BENFORD_EXPECTED = {d: np.log10(1 + 1 / d) for d in range(1, 10)}


def _leading_digit(x):
    x = abs(x)
    if x < 1:
        return None
    return int(str(int(x))[0])


def benford_deviation(amounts: pd.Series) -> float:
    """Chi-square-like deviation score (0 = perfect Benford fit, higher =
    more suspicious). Requires a reasonable sample size to be meaningful."""
    digits = amounts.apply(_leading_digit).dropna()
    if len(digits) < 30:
        return np.nan
    observed = digits.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    expected = pd.Series(BENFORD_EXPECTED)
    deviation = ((observed - expected) ** 2 / expected).sum()
    return deviation


def run(payments: pd.DataFrame, works: pd.DataFrame) -> pd.DataFrame:
    results = []
    for work_id, grp in payments.groupby("work_id"):
        grp = grp.sort_values("payment_date")
        band_payments = grp[(grp.amount >= APPROVAL_THRESHOLD * BAND_LOW) &
                             (grp.amount <= APPROVAL_THRESHOLD * BAND_HIGH)]
        score = 0.0
        reason = ""
        if len(band_payments) >= 2:
            dates = pd.to_datetime(band_payments.payment_date)
            span_days = (dates.max() - dates.min()).days
            if span_days <= CLUSTER_WINDOW_DAYS:
                score = min(1.0, 0.5 + 0.15 * len(band_payments))
                reason = (
                    f"{len(band_payments)} payments totalling "
                    f"₹{band_payments.amount.sum():,.0f} were each made just under the "
                    f"₹{APPROVAL_THRESHOLD:,.0f} approval threshold, within {span_days} days of "
                    f"each other -- consistent with structuring to avoid extra review"
                )
        results.append({"work_id": work_id, "structuring_score": score, "structuring_reason": reason})

    df = pd.DataFrame(results)

    # ---- Benford check at IA level, merged back onto works ----
    payments_with_ia = payments.merge(works[["work_id", "ia_id"]], on="work_id", how="left")
    ia_benford = payments_with_ia.groupby("ia_id")["amount"].apply(benford_deviation).rename("ia_benford_deviation")
    ia_benford_flag = (ia_benford > ia_benford.quantile(0.90)) & ia_benford.notna()
    ia_flagged = ia_benford_flag[ia_benford_flag].index.tolist()

    df = df.merge(works[["work_id", "ia_id"]], on="work_id", how="left")
    df["ia_benford_flagged"] = df["ia_id"].isin(ia_flagged)

    return df[["work_id", "structuring_score", "structuring_reason", "ia_id", "ia_benford_flagged"]], ia_benford.sort_values(ascending=False)


if __name__ == "__main__":
    payments = pd.read_csv("/home/claude/mplads_ai/data/payments.csv")
    works = pd.read_csv("/home/claude/mplads_ai/data/works.csv")
    scored, ia_benford = run(payments, works)

    flagged = scored[scored.structuring_score > 0]
    print(f"Flagged {len(flagged)} works for payment structuring:\n")
    print(flagged.head(10).to_string(index=False))

    print("\nTop 5 IAs by Benford deviation (higher = payment amounts look less 'natural'):")
    print(ia_benford.head(5))
