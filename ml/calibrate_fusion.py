import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.utils.metadata_loader import load_metadata  # noqa: E402


RESULTS_PATH = ROOT / "ml" / "outputs" / "results.csv"


def main():
    meta = load_metadata()
    meta = meta[["filename", "expected_label"]].copy()
    meta["expected_label"] = meta["expected_label"].str.strip().str.lower()

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"results.csv not found at {RESULTS_PATH}")

    res = pd.read_csv(RESULTS_PATH)

    df = pd.merge(meta, res, on="filename", how="inner")
    if df.empty:
        print("No overlapping filenames between metadata and results.")
        return

    # Features: individual module risks in [0,1]
    X = df[["ela", "fft", "vit", "expiry_score"]].values
    y = (df["expected_label"] == "reject").astype(int).values

    # Fit simple logistic regression (calibrated risk in [0,1])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]

    # Grid-search thresholds to enforce: no expected Approve samples go to Reject
    best = None
    for t1 in np.linspace(0.1, 0.6, 11):  # approve/manual boundary
        for t2 in np.linspace(t1 + 0.05, 0.9, 10):  # manual/reject boundary
            decisions = np.full_like(p, "manual", dtype=object)
            decisions[p < t1] = "approve"
            decisions[p >= t2] = "reject"

            exp = df["expected_label"].values

            # False positive hard rejects: expected Approve but predicted Reject
            fp_reject = np.sum((exp == "approve") & (decisions == "reject"))
            if fp_reject > 0:
                continue

            # High-risk recall: among expected Reject, how many predicted Reject
            mask_reject = exp == "reject"
            if mask_reject.sum() == 0:
                continue
            recall = np.mean(decisions[mask_reject] == "reject")

            # Prefer higher recall; tie-breaker: fewer in manual review
            manual_rate = np.mean(decisions == "manual")
            score = (recall, -manual_rate)
            if best is None or score > best[0]:
                best = (score, (t1, t2))

    print("=== Learned logistic fusion (Reject vs not) ===")
    print("intercept:", float(clf.intercept_[0]))
    print("coef_ (ela, fft, vit, expiry):", " ".join(f"{c:.6f}" for c in clf.coef_[0]))

    if best is None:
        print("Could not find thresholds with zero false-positive rejects.")
        return

    (t1, t2) = best[1]
    print("\n=== Selected decision thresholds on fused risk ===")
    print(f"approve if risk < {t1:.3f}")
    print(f"manual  if {t1:.3f} <= risk < {t2:.3f}")
    print(f"reject  if risk >= {t2:.3f}")


if __name__ == "__main__":
    main()
