import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on the path so the ml package can be imported.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.utils.metadata_loader import load_metadata, get_image_path  # noqa: E402
from ml.pipeline import run_pipeline  # noqa: E402

OUTPUT_PATH = ROOT / "ml" / "outputs" / "results.csv"


df = load_metadata()


def _normalize_date(val):
    """
    Normalize delivery dates to ISO (YYYY-MM-DD) where possible so that
    downstream parsing is stable across different input formats.
    """
    if pd.isna(val):
        return None
    for dayfirst in (True, False):
        try:
            ts = pd.to_datetime(val, dayfirst=dayfirst, errors="raise")
            return ts.strftime("%Y-%m-%d")
        except Exception:
            continue
    return val


df["delivery_date"] = df["delivery_date"].apply(_normalize_date)

results = []

for _, row in df.iterrows():
    image_path = get_image_path(row["filename"])
    delivery_date = row["delivery_date"]

    # Skip rows where we cannot resolve an image path.
    if image_path is None:
        continue

    output = run_pipeline(image_path, delivery_date)

    results.append(
        {
            "filename": row["filename"],
            "category": row.get("category"),
            "score": output["score"],
            "decision": output["decision"],
            "timeline": output["timeline"],
            "expiry_text": output["expiry_text"],
            "ela": output["ela"],
            "fft": output["fft"],
            "vit": output["vit"],
            "expiry_score": output["expiry_score"],
            "ocr_confidence": output.get("ocr_confidence"),
            "fusion_confidence": output.get("confidence"),
            "tags": ";".join(output.get("tags", [])),
            "highlight_path": output.get("highlight_path"),
        }
    )

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

print(f"Processing completed. Results saved to {OUTPUT_PATH}.")