"""
Detector 2: DUPLICATE / PHANTOM WORKS
=======================================
Catches the same physical asset being recommended/sanctioned more than once
under different work_ids -- i.e. double-billing the public purse for one
asset.

Method: combine TWO signals, and only flag when BOTH agree, which keeps
false positives low (two unrelated works can share a location by chance;
two unrelated works can share similar wording by chance; both together is
much rarer).

  1. TEXT SIMILARITY on work_description, via TF-IDF + cosine similarity.
     (We'd use sentence-transformer embeddings in production for better
     semantic matching -- e.g. "drainage line" vs "sewage channel" -- but
     TF-IDF is a fully offline, zero-dependency substitute that still works
     well here because MPLADS descriptions are short and formulaic.)

  2. GEOSPATIAL PROXIMITY via haversine distance between recorded
     coordinates. Works within ~500m of each other are candidates.

We only compare works WITHIN THE SAME DISTRICT (a duplicate across two
different states is not a realistic pattern and comparing all 1400+ works
pairwise nationally would also be computationally wasteful) -- this is a
reasonable, explainable scoping decision.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, sqrt, atan2


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


TEXT_SIM_THRESHOLD = 0.6
DISTANCE_THRESHOLD_KM = 0.5


def run(works: pd.DataFrame) -> pd.DataFrame:
    df = works.copy().reset_index(drop=True)
    df["dup_score"] = 0.0
    df["dup_reason"] = ""

    pair_flags = {wid: [] for wid in df.work_id}

    for district, group in df.groupby("district"):
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        texts = group["work_description"].tolist()
        vec = TfidfVectorizer().fit_transform(texts)
        sim_matrix = cosine_similarity(vec)

        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                text_sim = sim_matrix[a, b]
                if text_sim < TEXT_SIM_THRESHOLD:
                    continue
                lat1, lon1 = group.iloc[a]["latitude"], group.iloc[a]["longitude"]
                lat2, lon2 = group.iloc[b]["latitude"], group.iloc[b]["longitude"]
                dist_km = haversine_km(lat1, lon1, lat2, lon2)
                if dist_km > DISTANCE_THRESHOLD_KM:
                    continue
                # Both signals agree -> likely duplicate
                wid_a, wid_b = group.iloc[a]["work_id"], group.iloc[b]["work_id"]
                score = min(1.0, text_sim * (1 - dist_km / DISTANCE_THRESHOLD_KM) + 0.3)
                pair_flags[wid_a].append((wid_b, text_sim, dist_km, score))
                pair_flags[wid_b].append((wid_a, text_sim, dist_km, score))

    for i, row in df.iterrows():
        matches = pair_flags[row.work_id]
        if not matches:
            continue
        best = max(matches, key=lambda m: m[3])
        df.loc[i, "dup_score"] = best[3]
        df.loc[i, "dup_reason"] = (
            f"description + location closely match work {best[0]} "
            f"(text similarity {best[1]:.2f}, {best[2]*1000:.0f}m apart) -- possible duplicate/phantom billing"
        )

    return df[["work_id", "dup_score", "dup_reason"]]


if __name__ == "__main__":
    works = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\works.csv")
    scored = run(works)
    flagged = scored[scored.dup_score > 0].sort_values("dup_score", ascending=False)
    print(f"Flagged {len(flagged)} works as possible duplicates out of {len(works)} total\n")
    print(flagged.head(10).to_string(index=False))
