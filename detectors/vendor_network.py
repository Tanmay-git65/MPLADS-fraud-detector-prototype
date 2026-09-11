"""
Detector 6: VENDOR / IA NETWORK ANALYSIS
============================================
Individual works can each look perfectly normal while still being part of a
collusion ring -- a small clique of Implementing Agencies and vendors that
capture a disproportionate share of works/payments within a district, often
cycling money between a small set of related entities. This is invisible to
row-level detectors (each individual work/payment looks fine) and only shows
up when you look at the RELATIONSHIP GRAPH.

Method: for each district, build each IA's "significant vendor set" -- the
vendors that account for a meaningful share (>=15%) of that IA's total
payment volume (i.e. its real regular partners, not one-off/incidental
vendors). Then compute the JACCARD SIMILARITY of significant-vendor-sets
between every pair of IAs in the district: |shared vendors| / |union of
vendors|. Two IAs that are legitimately independent should have low overlap
(they each pick from a large vendor pool); IAs secretly funnelling work
through the same small clique of preferred vendors will show high overlap.
We then group IAs into connected "rings" wherever pairwise overlap exceeds
a threshold (using a graph + connected components), so a ring of 3+ IAs
sharing a vendor pool gets flagged as one cluster with a shared explanation.

This mirrors how real anti-corruption/AML units use network analysis --
"who transacts disproportionately with the same small group" is a much
harder pattern to hide than any single transaction, because it only shows
up when you look at the relationship graph, not any one row of data.
"""

import pandas as pd
import numpy as np
import networkx as nx

SIGNIFICANT_VENDOR_SHARE = 0.15   # vendor must be >=15% of an IA's volume to count as a "regular partner"
JACCARD_RING_THRESHOLD = 0.4      # pairwise overlap above this links two IAs into the same ring


def _significant_vendor_set(grp: pd.DataFrame) -> set:
    vshare = grp.groupby("vendor_id")["amount"].sum()
    vshare = vshare / vshare.sum()
    return set(vshare[vshare >= SIGNIFICANT_VENDOR_SHARE].index)


def run(payments: pd.DataFrame, works: pd.DataFrame, ias: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pw = payments.merge(works[["work_id", "ia_id", "district"]], on="work_id", how="left")

    ia_flags = []
    for district, grp in pw.groupby("district"):
        ia_vendor_sets = {ia_id: _significant_vendor_set(g) for ia_id, g in grp.groupby("ia_id")}
        ia_ids = [i for i in ia_vendor_sets if len(ia_vendor_sets[i]) > 0]

        ring_graph = nx.Graph()
        ring_graph.add_nodes_from(ia_ids)
        overlap_detail = {}
        for i in range(len(ia_ids)):
            for j in range(i + 1, len(ia_ids)):
                a, b = ia_ids[i], ia_ids[j]
                set_a, set_b = ia_vendor_sets[a], ia_vendor_sets[b]
                union = set_a | set_b
                jaccard = len(set_a & set_b) / len(union) if union else 0
                if jaccard >= JACCARD_RING_THRESHOLD:
                    ring_graph.add_edge(a, b, weight=jaccard)
                    overlap_detail[(a, b)] = (jaccard, set_a & set_b)

        for component in nx.connected_components(ring_graph):
            if len(component) < 2:
                continue  # need at least 2 IAs sharing vendors to call it a "ring"
            component = list(component)
            shared_vendors = set.intersection(*[ia_vendor_sets[c] for c in component])
            if not shared_vendors:
                # fall back to union of all pairwise shared vendors within this component
                shared_vendors = set()
                for (a, b), (jac, common) in overlap_detail.items():
                    if a in component and b in component:
                        shared_vendors |= common
            avg_jaccard = np.mean([jac for (a, b), (jac, _) in overlap_detail.items()
                                    if a in component and b in component]) if overlap_detail else 0
            score = min(1.0, 0.5 + avg_jaccard)
            reason = (
                f"{len(component)} Implementing Agencies in {district} ({', '.join(sorted(component))}) "
                f"share an unusually overlapping small set of regular vendors "
                f"({', '.join(sorted(shared_vendors)) or 'overlapping pool'}) -- "
                f"avg vendor-overlap {avg_jaccard*100:.0f}%, consistent with a closed-loop vendor ring"
            )
            for ia_id in component:
                ia_flags.append({"ia_id": ia_id, "district": district, "network_score": score,
                                  "network_reason": reason, "n_vendors": len(ia_vendor_sets[ia_id])})

    ia_df = pd.DataFrame(ia_flags) if ia_flags else pd.DataFrame(
        columns=["ia_id", "district", "network_score", "network_reason", "n_vendors"])
    # Map IA-level network risk onto every work executed by that IA
    work_level = works[["work_id", "ia_id"]].merge(
        ia_df[["ia_id", "network_score", "network_reason"]], on="ia_id", how="left"
    ).fillna({"network_score": 0.0, "network_reason": ""})

    return work_level[["work_id", "network_score", "network_reason"]], ia_df.sort_values("network_score", ascending=False)


if __name__ == "__main__":
    payments = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\payments.csv")
    works = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\works.csv")
    ias = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\ias.csv")
    work_scored, ia_scored = run(payments, works, ias)

    print("Top flagged IAs (possible collusion rings):")
    print(ia_scored[ia_scored.network_score > 0].head(10).to_string(index=False))
    print(f"\nWorks touched by a flagged IA: {(work_scored.network_score > 0).sum()}")
