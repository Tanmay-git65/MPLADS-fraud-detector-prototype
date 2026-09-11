"""
LAYER 4: EVALUATION AGAINST GROUND TRUTH
============================================
The ONLY place in this whole project that touches ground_truth.csv. None of
the detectors or the risk_scoring pipeline ever see it -- it exists purely
so we can grade ourselves the way a real evaluator would grade a fraud
system: precision (of what we flagged, how much was really planted fraud)
and recall (of what we planted, how much did we catch).

Why this matters for your pitch: "our system flags anomalies" is an
unfalsifiable claim. "our system caught 91% of known injected fraud
patterns at a Medium-risk threshold or above, with X% precision" is a
number a judge can interrogate, and that's what separates a hackathon toy
from something that reads as production-credible.
"""

import pandas as pd

RISK_REPORT_PATH = r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\output\risk_report.csv"
GROUND_TRUTH_PATH = r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\ground_truth.csv"


DELAY_PATTERNS = {"STALLED_OVERDUE", "VELOCITY_MISMATCH"}


def evaluate(flag_threshold_tier=("Medium", "High", "Critical"), delay_tiers=("Delayed", "Severely Stalled")):
    risk = pd.read_csv(RISK_REPORT_PATH)
    gt = pd.read_csv(GROUND_TRUTH_PATH)

    # Ground truth entities that are WORKS (most patterns) -- IA/vendor-level
    # patterns (VENDOR_COLLUSION_RING) are evaluated separately since they're
    # not per-work ground truth. STALLED_OVERDUE / VELOCITY_MISMATCH are
    # evaluated against the separate delay_tier axis, everything else
    # against the fraud risk_tier axis -- matching how the pipeline actually
    # scores them (see risk_scoring.py's note on why delay is kept separate).
    gt_work = gt[gt.entity_type == "work"]
    gt_fraud = gt_work[~gt_work.pattern.isin(DELAY_PATTERNS)]
    gt_delay = gt_work[gt_work.pattern.isin(DELAY_PATTERNS)]

    fraud_flagged = set(risk[risk.risk_tier.isin(flag_threshold_tier)].work_id)
    delay_flagged = set(risk[risk.delay_tier.isin(delay_tiers)].work_id)
    flagged_work_ids = fraud_flagged | delay_flagged
    gt_work = pd.concat([gt_fraud, gt_delay])  # recombine for the overall summary below
    planted_work_ids = set(gt_work.entity_id)
    all_work_ids = set(risk.work_id)

    true_positives = flagged_work_ids & planted_work_ids
    false_positives = flagged_work_ids - planted_work_ids
    false_negatives = planted_work_ids - flagged_work_ids

    precision = len(true_positives) / len(flagged_work_ids) if flagged_work_ids else 0
    recall = len(true_positives) / len(planted_work_ids) if planted_work_ids else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("=" * 70)
    print(f"EVALUATION -- flagging threshold: risk_tier in {flag_threshold_tier}")
    print("=" * 70)
    print(f"Total works in dataset:                 {len(all_work_ids)}")
    print(f"Works with an injected fraud pattern:   {len(planted_work_ids)}")
    print(f"Works flagged by the system:            {len(flagged_work_ids)}")
    print(f"  True positives  (correctly caught):   {len(true_positives)}")
    print(f"  False positives (flagged, not fraud): {len(false_positives)}")
    print(f"  False negatives (missed fraud):       {len(false_negatives)}")
    print(f"\nPrecision: {precision*100:5.1f}%   (of what we flagged, how much was real)")
    print(f"Recall:    {recall*100:5.1f}%   (of planted fraud, how much we caught)")
    print(f"F1 score:  {f1*100:5.1f}%")

    print("\n--- Recall by fraud pattern type ---")
    for pattern, grp in gt_work.groupby("entity_id" if False else "pattern"):
        planted_ids = set(grp.entity_id)
        caught = planted_ids & flagged_work_ids
        print(f"  {pattern:22s}: caught {len(caught):3d} / {len(planted_ids):3d}  ({len(caught)/len(planted_ids)*100:5.1f}%)")

    print("\n--- Missed cases (false negatives) -- worth showing honestly in your demo ---")
    missed = gt_work[gt_work.entity_id.isin(false_negatives)]
    if len(missed):
        print(missed[["entity_id", "pattern", "note"]].head(10).to_string(index=False))
    else:
        print("  None -- every planted work-level fraud case was flagged at this threshold.")

    # IA-level collusion ring check (separate because ground truth is at IA level)
    gt_ia = gt[(gt.entity_type == "ia") & (gt.pattern == "VENDOR_COLLUSION_RING")]
    planted_ias = set(gt_ia.entity_id)
    flagged_ia_works = risk[risk.risk_tier.isin(flag_threshold_tier)]
    # (network detector output isn't in risk.csv directly as ia list, so we
    # just check via the risk_explanation text mentioning "collusion")
    ring_mentioned = risk[risk.risk_explanation.str.contains("collusion", case=False, na=False)]
    flagged_ring_ias = set(ring_mentioned.ia_id) if "ia_id" in ring_mentioned.columns else set()
    print(f"\n--- Vendor collusion ring (IA-level) ---")
    print(f"Planted ring IAs: {sorted(planted_ias)}")
    print(f"IAs whose works triggered a 'collusion' explanation: {sorted(flagged_ring_ias) if flagged_ring_ias else 'see network detector output directly'}")

    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positives": true_positives, "false_positives": false_positives, "false_negatives": false_negatives}


def fraud_only_precision_recall():
    """The headline number for your pitch: precision/recall computed ONLY
    on genuine fraud patterns (excludes the two delay patterns, which are
    scored on their own separate axis -- see risk_scoring.py). This is the
    metric to put on a slide, at each tier, so a judge can see the
    precision/recall tradeoff explicitly rather than a single cherry-picked
    number."""
    risk = pd.read_csv(RISK_REPORT_PATH)
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    gt_work = gt[gt.entity_type == "work"]
    gt_fraud = gt_work[~gt_work.pattern.isin(DELAY_PATTERNS)]
    planted_fraud = set(gt_fraud.entity_id)

    print("\n" + "=" * 70)
    print("HEADLINE METRIC: fraud-pattern precision/recall by risk tier")
    print("(delay patterns excluded -- scored separately on the delay_tier axis)")
    print("=" * 70)
    for tiers in [("Critical",), ("High", "Critical"), ("Medium", "High", "Critical")]:
        flagged = set(risk[risk.risk_tier.isin(tiers)].work_id)
        tp = flagged & planted_fraud
        precision = len(tp) / len(flagged) if flagged else 0
        recall = len(tp) / len(planted_fraud) if planted_fraud else 0
        print(f"  Tier >= {tiers[-1] if len(tiers)==1 else '/'.join(tiers):20s}: "
              f"flagged={len(flagged):4d}  precision={precision*100:5.1f}%  recall={recall*100:5.1f}%")

    print("\nRecommended review workflow: Critical tier -> immediate investigation queue; "
          "High -> District Authority review within a week; Medium -> flagged for routine audit sampling.")


if __name__ == "__main__":
    print("### Combined (fraud + delay) view, Medium+ threshold ###")
    evaluate(flag_threshold_tier=("Medium", "High", "Critical"))

    fraud_only_precision_recall()
