import pandas as pd
import os

EXPIRY_DATA_PATH = "ml/data/expiry_dataset/expiry_dataset.csv"

class ExpiryValidator:
    def __init__(self):
        if os.path.exists(EXPIRY_DATA_PATH):
            self.df = pd.read_csv(EXPIRY_DATA_PATH)
        else:
            self.df = pd.DataFrame()

    def check_expiry(self, filename: str, ocr_text: str):
        """
        Compare OCR output with ground truth expiry text.
        Returns score adjustment + tags.
        """

        if self.df.empty:
            return 0, []

        row = self.df[self.df["image_name"] == filename]

        if row.empty:
            return 0, []

        expected = str(row.iloc[0]["expiry_text"]).strip()

        if expected and expected not in ocr_text:
            return 10, ["Expiry Mismatch"]

        return 0, []