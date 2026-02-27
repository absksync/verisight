import sys
import os
from pathlib import Path

# Add parent directory to path so 'ml' package can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.utils.metadata_loader import load_metadata, get_image_path
from ml.pipeline import run_pipeline
import pandas as pd

OUTPUT_PATH = r"C:/Users/Apoorva/Desktop/VERISIGHT/ml/outputs/results.csv"

df = load_metadata()

results = []

for _, row in df.iterrows():
    image_path = get_image_path(row['filename'])
    delivery_date = row['delivery_date']

    output = run_pipeline(image_path, delivery_date)

    results.append({
        "filename": row['filename'],
        "category": row['category'],
        "score": output['score'],
        "confidence": output['confidence'],
        "timeline": output['timeline']
    })

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

print("Processing completed. Results saved.")