import pandas as pd
import os

# Build DATA_PATH relative to this file so it works regardless of CWD
ROOT_ML = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(ROOT_ML, "data", "Hack_data")


def load_metadata():
    meta_path = os.path.join(DATA_PATH, "metadata.csv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"metadata.csv not found at {meta_path}. Expected it under the ml/data/Hack_data folder."
        )

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

    # First, check directly under DATA_PATH
    full_path = os.path.join(DATA_PATH, relative_path)
    if os.path.exists(full_path):
        return full_path

    # If not found, search immediate subdirectories (e.g., 'original', 'edited', 'ai')
    try:
        for entry in os.listdir(DATA_PATH):
            entry_path = os.path.join(DATA_PATH, entry)
            if os.path.isdir(entry_path):
                candidate = os.path.join(entry_path, relative_path)
                if os.path.exists(candidate):
                    return candidate
    except FileNotFoundError:
        # DATA_PATH doesn't exist
        return None

    # Not found anywhere
    print("File not found:", full_path)
    return None