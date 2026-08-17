"""
VeriSight — Tamper Detection Model Training (v2)
=================================================
Improvements over v1:
 - Uses ALL 414 authentic + ALL 1813 tampered IMD2020 samples (no cap)
 - Adds ExpDate-Real images as additional authentic samples
 - Richer 12-dim feature vector including color channel stats,
   JPEG artifact ratio, and edge continuity
 - GradientBoostingClassifier for better generalisation
 - 5-fold cross-validation reported

Saves:
  models/tamper_classifier.pkl
  models/tamper_scaler.pkl
  reports/tamper_model_report.json
"""

import io
import json
import logging
import random
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_tamper_model")

BASE_DIR      = Path(__file__).resolve().parent.parent
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature extraction ────────────────────────────────────────────────────────

def ela_score(img_bgr, quality=90):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    diff = ImageChops.difference(pil, resaved)
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema]) if extrema else 1
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff if max_diff < 50 else 15.0
    ela = np.array(ImageEnhance.Brightness(diff).enhance(scale))
    gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY) if len(ela.shape)==3 else ela
    return (float(np.percentile(gray, 98)) / 255.0,
            float(np.mean(gray)) / 255.0,
            float(np.std(gray)) / 255.0)

def noise_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap  = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    k    = 15
    mf   = cv2.blur(lap,    (k,k))
    msf  = cv2.blur(lap**2, (k,k))
    ns   = np.sqrt(np.maximum(msf - mf**2, 0))
    norm = cv2.normalize(ns, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    p98, p50 = np.percentile(norm, 98), np.percentile(norm, 50)
    return float(min(1.0, (p98 - p50) / 120.0))

def edge_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mag  = cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3),
                         cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return float(min(1.0, (np.sum(norm > 180)/norm.size) * 40.0))

def dct_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    scores = []
    for y in range(0, h-8, 8):
        for x in range(0, w-8, 8):
            d = cv2.dct(gray[y:y+8, x:x+8])
            scores.append(float(np.sum(np.abs(d[2:,2:]))))
    if not scores: return 0.0
    a = np.array(scores)
    return float(min(1.0, np.std(a)/(np.mean(a)+1e-6)))

def color_channel_stats(img_bgr):
    """Per-channel mean and std — color inconsistencies reveal copy-paste."""
    feats = []
    for ch in cv2.split(img_bgr):
        feats.append(float(np.mean(ch))/255.0)
        feats.append(float(np.std(ch))/128.0)
    return feats  # 6 values: [b_mean, b_std, g_mean, g_std, r_mean, r_std]

def jpeg_blocking(img_bgr):
    """Measure blockiness from JPEG artifacts (8x8 grid discontinuities)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    h_grid = np.mean(np.abs(np.diff(gray[:, 7::8], axis=1))) if w > 8 else 0.0
    v_grid = np.mean(np.abs(np.diff(gray[7::8, :], axis=0))) if h > 8 else 0.0
    h_rand = np.mean(np.abs(np.diff(gray[:, 3::8], axis=1))) if w > 3 else 1.0
    v_rand = np.mean(np.abs(np.diff(gray[3::8, :], axis=0))) if h > 3 else 1.0
    block_ratio_h = float(h_grid / (h_rand + 1e-6))
    block_ratio_v = float(v_grid / (v_rand + 1e-6))
    return float(min(2.0, (block_ratio_h + block_ratio_v) / 2.0)) / 2.0  # normalised 0-1

def extract_features(img_bgr):
    """12-dim feature vector."""
    img_bgr = cv2.resize(img_bgr, (256,256), interpolation=cv2.INTER_AREA)
    ela_pk, ela_mn, ela_sd = ela_score(img_bgr)
    ns    = noise_score(img_bgr)
    es    = edge_score(img_bgr)
    ds    = dct_score(img_bgr)
    cs    = color_channel_stats(img_bgr)  # 6 values
    jb    = jpeg_blocking(img_bgr)

    return np.array(
        [ela_pk, ela_mn, ela_sd, ns, es, ds] + cs + [jb, ela_pk*ns],
        dtype=np.float32
    )

# ── Data loading ──────────────────────────────────────────────────────────────

def load_entries(manifest_path, base_dir):
    with open(manifest_path) as f:
        manifest = json.load(f)

    # IMD2020: all authentic + all tampered
    imd      = [e for e in manifest if e.get("dataset") == "IMD2020"]
    auth_imd = [e for e in imd if not e["is_tampered"]]
    tamp_imd = [e for e in imd if e["is_tampered"]]

    # ExpDate-Real: genuine packaging → authentic class
    exp = [e for e in manifest if e.get("dataset") == "ExpDate-Real"]

    # OpenFoodFacts: real commercial photos → authentic class
    off = [e for e in manifest if e.get("dataset") == "OpenFoodFacts"]

    authentic = auth_imd + exp + off
    tampered  = tamp_imd

    logger.info(f"Authentic pool: {len(auth_imd)} IMD2020 + {len(exp)} ExpDate + {len(off)} OFF = {len(authentic)}")
    logger.info(f"Tampered pool : {len(tampered)} IMD2020")

    # Balance by capping majority class
    min_count = min(len(authentic), len(tampered))
    random.seed(42)
    authentic = random.sample(authentic, min_count)
    tampered  = random.sample(tampered,  min_count)
    logger.info(f"Balanced at {min_count} per class ({min_count*2} total)")

    all_entries = [(e, 0) for e in authentic] + [(e, 1) for e in tampered]
    random.shuffle(all_entries)
    return all_entries

def extract_dataset(all_entries, base_dir):
    features, labels, errors = [], [], 0
    t0 = time.time()
    for idx, (entry, label) in enumerate(all_entries):
        img = cv2.imread(str(base_dir / entry["path"]))
        if img is None:
            errors += 1
            continue
        try:
            features.append(extract_features(img))
            labels.append(label)
        except Exception:
            errors += 1
            continue
        if (idx+1) % 200 == 0:
            el  = time.time()-t0
            eta = (el/(idx+1))*(len(all_entries)-idx-1)
            logger.info(f"  {idx+1}/{len(all_entries)} | {el:.0f}s elapsed | ETA {eta:.0f}s")

    logger.info(f"Done — valid: {len(features)}, errors: {errors}")
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    logger.info("="*60)
    logger.info("VeriSight — Tamper Detection Training (v2)")
    logger.info("="*60)

    manifest = PROCESSED_DIR / "benchmark_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("Run scripts/prepare_dataset.py first.")

    all_entries = load_entries(manifest, BASE_DIR)
    X, y        = extract_dataset(all_entries, BASE_DIR)
    if len(X) < 50:
        raise RuntimeError(f"Too few samples ({len(X)})")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    # ── 5-fold CV on training set ─────────────────────────────────────────────
    logger.info("Running 5-fold cross-validation on train set...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                              learning_rate=0.1, random_state=42))
    ])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5), scoring="f1", n_jobs=-1)
    logger.info(f"CV F1: {cv_scores.round(3)} | mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Final training ─────────────────────────────────────────────────────────
    logger.info("Training final GradientBoostingClassifier (200 trees)...")
    t0 = time.time()
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                     learning_rate=0.1, random_state=42)
    clf.fit(X_train_sc, y_train)
    train_time = time.time()-t0
    logger.info(f"Done in {train_time:.1f}s")

    y_pred = clf.predict(X_val_sc)
    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec  = recall_score(y_val, y_pred, zero_division=0)
    f1   = f1_score(y_val, y_pred, zero_division=0)
    cm   = confusion_matrix(y_val, y_pred).tolist()

    logger.info("\n" + classification_report(y_val, y_pred, target_names=["Authentic","Tampered"]))
    logger.info(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    feat_names = ["ela_peak","ela_mean","ela_std","noise","edge","dct",
                  "b_mean","b_std","g_mean","g_std","r_mean","r_std",
                  "jpeg_block","ela_x_noise"]
    importances = {n: round(float(v),4) for n,v in zip(feat_names, clf.feature_importances_)}
    logger.info(f"Feature importances: {importances}")

    clf_path    = MODELS_DIR / "tamper_classifier.pkl"
    scaler_path = MODELS_DIR / "tamper_scaler.pkl"
    joblib.dump(clf,    clf_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Model  → {clf_path}")
    logger.info(f"Scaler → {scaler_path}")

    report = {
        "model": "GradientBoostingClassifier",
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "training_time_seconds": round(train_time, 2),
        "cv_f1_scores": cv_scores.round(4).tolist(),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "metrics": {"accuracy": round(acc,4), "precision": round(prec,4),
                    "recall": round(rec,4), "f1_score": round(f1,4)},
        "confusion_matrix": {"labels": ["Authentic","Tampered"], "matrix": cm},
        "feature_importances": importances,
        "model_path": str(clf_path),
        "scaler_path": str(scaler_path),
    }
    rpt = REPORTS_DIR / "tamper_model_report.json"
    with open(rpt,"w") as f: json.dump(report, f, indent=2)
    logger.info(f"Report → {rpt}")
    logger.info("✅ Training complete!")
    return report

if __name__ == "__main__":
    train()
