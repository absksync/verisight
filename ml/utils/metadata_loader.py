import pandas as pd
import os

DATA_PATH = r"C:/Users/Apoorva/Desktop/VERISIGHT/data/Hack_data"

def load_metadata():
    meta_path = os.path.join(DATA_PATH, "metadata.csv")

    df = pd.read_csv(meta_path)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove rows where filename is missing
    df = df.dropna(subset=["filename"])

    # Convert filename to string
    df["filename"] = df["filename"].astype(str)

    return df


def get_image_path(relative_path):

    # Safety check
    if not isinstance(relative_path, str):
        return None

    full_path = os.path.join(DATA_PATH, relative_path)

    # Check file exists
    if not os.path.exists(full_path):
        print("File not found:", full_path)
        return None

    return full_path