import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.utils.metadata_loader import load_metadata  # noqa: E402


RESULTS_PATH = ROOT / "ml" / "outputs" / "results.csv"


def main():
    # Ground truth from metadata
    meta = load_metadata()
    meta = meta[["filename", "expected_label", "category"]].copy()

    # Model outputs
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"results.csv not found at {RESULTS_PATH}")

    res = pd.read_csv(RESULTS_PATH)

    # Join on filename
    df = pd.merge(meta, res, on="filename", how="inner")

    if df.empty:
        print("No overlapping filenames between metadata and results.")
        return

    # Normalise labels
    df["expected_label"] = df["expected_label"].str.strip().str.lower()
    df["decision"] = df["decision"].str.strip().str.lower()

    # Basic accuracy (Approve vs Reject, counting Manual Review as its own bucket)
    df["is_correct_strict"] = df["expected_label"] == df["decision"]

    # A softer metric: treat Manual Review as not strictly wrong, but not correct either
    df["is_high_risk_expected"] = df["expected_label"] == "reject"
    df["is_high_risk_pred"] = df["decision"] == "reject"

    overall_acc = df["is_correct_strict"].mean()

    print("=== Overall Strict Accuracy (decision vs expected_label) ===")
    print(f"{overall_acc:.3f}  ({df['is_correct_strict'].sum()} / {len(df)})")
    print()

    print("=== Confusion Matrix (rows = expected, cols = decision) ===")
    cm = pd.crosstab(df["expected_label"], df["decision"])
    print(cm)
    print()

    # Per-category breakdown (original / edited / ai_generated) if category is present
    if "category" in df.columns:
        print("=== Strict Accuracy by Category ===")
        by_cat = (
            df.groupby("category")["is_correct_strict"]
            .mean()
            .sort_values(ascending=False)
        )
        print(by_cat)
        print()

    # High-risk recall: among expected Reject, how many did we mark Reject (not Manual Review)?
    high_risk = df[df["is_high_risk_expected"]]
    if not high_risk.empty:
        recall = (high_risk["decision"] == "reject").mean()
        print("=== High-Risk Recall (expected Reject, predicted Reject) ===")
        print(f"{recall:.3f}  ({(high_risk['decision'] == 'reject').sum()} / {len(high_risk)})")
    else:
        print("No expected Reject samples to evaluate high-risk recall.")


if __name__ == "__main__":
    main()
