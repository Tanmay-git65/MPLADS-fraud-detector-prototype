"""
Detector 5: IMAGE / DOCUMENT FORENSICS
=========================================
eSAKSHI requires Implementing Agencies to upload photographs at each payment
stage, and a final completion photo. This is a rich, almost entirely
underused signal -- most anomaly-detection approaches to schemes like this
stay purely tabular. Two checks:

  A) GEOTAG MISMATCH
     Every modern phone photo carries GPS EXIF metadata. Compare the photo's
     geotag against the work's own recorded sanction-location coordinates.
     A photo taken more than ~2km away from where the work is supposed to be
     is a strong, hard-to-explain-away signal -- either the wrong photo was
     uploaded, or the work doesn't exist where claimed.
     (In this sandbox we don't have real photo files, so `geotag_lat/lon`
     are simulated fields in the synthetic photos table -- in production
     these come directly from `exifread` / `Pillow.ExifTags` reading the
     actual uploaded JPEG's GPSInfo tag.)

  B) DUPLICATE PHOTO REUSE
     Compute a perceptual hash (pHash) per photo -- a fingerprint where
     visually similar/identical images hash to the same or very close value,
     even after resizing/recompression. If the SAME hash appears against
     MULTIPLE DIFFERENT work_ids, the same physical photo was reused to
     "prove" completion of more than one claimed work -- a well-documented
     fraud pattern in scheme audits (CAG reports on other local-development
     schemes have flagged exactly this).
     (Here `phash` is a simulated field for the same reason as above -- in
     production, swap in `imagehash.phash(Image.open(photo_path))`, and
     compare with Hamming distance <= 5 rather than exact match, to also
     catch lightly-edited reuploads, not just byte-identical files.)
"""

import pandas as pd
from math import radians, sin, cos, sqrt, atan2

GEOTAG_DISTANCE_FLAG_KM = 2.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def run(photos: pd.DataFrame, works: pd.DataFrame) -> pd.DataFrame:
    df = photos.merge(works[["work_id", "latitude", "longitude"]], on="work_id", how="left")

    # ---- A) Geotag mismatch ----
    df["geo_distance_km"] = df.apply(
        lambda r: haversine_km(r["geotag_lat"], r["geotag_lon"], r["latitude"], r["longitude"]), axis=1
    )
    df["geotag_score"] = (df["geo_distance_km"] / (GEOTAG_DISTANCE_FLAG_KM * 3)).clip(0, 1)
    df["geotag_reason"] = ""
    geo_mask = df["geo_distance_km"] > GEOTAG_DISTANCE_FLAG_KM
    df.loc[geo_mask, "geotag_reason"] = df.loc[geo_mask].apply(
        lambda r: f"completion photo GPS tag is {r['geo_distance_km']:.1f}km from the sanctioned work location", axis=1
    )

    # ---- B) Duplicate photo reuse ----
    hash_counts = df.groupby("phash")["work_id"].nunique()
    reused_hashes = hash_counts[hash_counts > 1].index
    df["dup_photo_score"] = 0.0
    df["dup_photo_reason"] = ""
    for h in reused_hashes:
        matches = df[df.phash == h]
        other_works = matches.work_id.unique()
        mask = df.phash == h
        df.loc[mask, "dup_photo_score"] = 1.0
        df.loc[mask, "dup_photo_reason"] = (
            f"identical photo (hash {h}) also submitted as completion evidence for "
            f"{len(other_works)-1} other unrelated work(s): {', '.join([w for w in other_works if w not in matches.work_id.values[:1]][:3])}"
        )

    df["image_forensics_score"] = df[["geotag_score", "dup_photo_score"]].max(axis=1)
    df["image_forensics_reason"] = [
        (a + " | " + b).strip(" |")
        for a, b in zip(df["geotag_reason"].tolist(), df["dup_photo_reason"].tolist())
    ]

    # Aggregate to work-level (a work can have multiple photos; take the max/worst)
    work_level = df.groupby("work_id").agg(
        image_forensics_score=("image_forensics_score", "max"),
        image_forensics_reason=("image_forensics_reason", lambda s: " || ".join([x for x in s if x])),
    ).reset_index()

    return work_level


if __name__ == "__main__":
    photos = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\photos.csv")
    works = pd.read_csv(r"C:\Users\tanma\Desktop\mywork\projs&exp\sih\mplads_ai_prototype\data\works.csv")
    scored = run(photos, works)
    # Note: score is continuous (raw GPS noise gives every photo a tiny
    # nonzero score), so "flagged" means score above a real review
    # threshold, not merely > 0.
    flagged = scored[scored.image_forensics_score > 0.3].sort_values("image_forensics_score", ascending=False)
    print(f"Flagged {len(flagged)} works via image forensics\n")
    print(flagged.head(10).to_string(index=False))
