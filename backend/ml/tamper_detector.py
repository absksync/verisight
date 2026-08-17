import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

_BASE_DIR    = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR  = _BASE_DIR / "models"
_CLF_PATH    = _MODELS_DIR / "tamper_classifier.pkl"
_SCALER_PATH = _MODELS_DIR / "tamper_scaler.pkl"

class PackagingTamperDetector:
    """
    Forensic Packaging Integrity & Manipulation Detector.
    Analyzes images for digital tampering, re-stamped dates, spliced text,
    and localized compression/noise anomalies.
    """

    def __init__(self, ela_quality: int = 90, ela_scale: float = 15.0):
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale
        self.ml_classifier: Optional[Any] = None
        self.ml_scaler: Optional[Any] = None
        self._load_trained_model()

    def _load_trained_model(self) -> None:
        """Attempt to load the trained RandomForest tamper classifier."""
        if not _JOBLIB_AVAILABLE:
            logger.info("joblib not available — using classical CV tamper scoring.")
            return
        if _CLF_PATH.exists() and _SCALER_PATH.exists():
            try:
                self.ml_classifier = joblib.load(str(_CLF_PATH))
                self.ml_scaler     = joblib.load(str(_SCALER_PATH))
                logger.info(f"✅ Trained tamper model loaded from {_CLF_PATH}")
            except Exception as e:
                logger.warning(f"Could not load trained model: {e} — falling back to classical CV.")
        else:
            logger.info("No trained model found — using classical CV tamper scoring.")

    def _extract_ml_features(self, pil_img: Image.Image, cv_img: np.ndarray,
                              ela_score: float, noise_score: float,
                              edge_score: float) -> np.ndarray:
        """Build the 7-dim feature vector matching train_tamper_model.py."""
        # DCT high-frequency inconsistency
        gray_f = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray_f.shape
        dct_scores = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = gray_f[y:y+8, x:x+8]
                dct   = cv2.dct(block)
                dct_scores.append(float(np.sum(np.abs(dct[2:, 2:]))))
        if dct_scores:
            arr = np.array(dct_scores)
            dct_score = float(min(1.0, np.std(arr) / (np.mean(arr) + 1e-6)))
        else:
            dct_score = 0.0

        gray_u8 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        brightness  = float(np.mean(gray_u8)) / 255.0
        contrast    = float(np.std(gray_u8))  / 128.0
        interaction = ela_score * noise_score

        return np.array(
            [ela_score, noise_score, edge_score, dct_score,
             brightness, contrast, interaction],
            dtype=np.float32
        )

    def compute_ela(self, pil_image: Image.Image) -> Tuple[np.ndarray, float]:
        """
        Computes Error Level Analysis (ELA).
        Re-saves the image at a known JPEG quality and measures pixel error differential.
        Altered areas or pasted digital text show distinct compression variances.
        """
        # Save original to JPEG buffer
        buffer = io.BytesIO()
        rgb_img = pil_image.convert('RGB')
        rgb_img.save(buffer, 'JPEG', quality=self.ela_quality)
        buffer.seek(0)
        resaved_img = Image.open(buffer)

        # Calculate difference
        ela_diff = ImageChops.difference(rgb_img, resaved_img)

        # Scale difference to enhance visibility
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff if max_diff < 50 else self.ela_scale

        enhancer = ImageEnhance.Brightness(ela_diff)
        ela_enhanced = enhancer.enhance(scale)

        ela_arr = np.array(ela_enhanced)
        gray_ela = cv2.cvtColor(ela_arr, cv2.COLOR_RGB2GRAY) if len(ela_arr.shape) == 3 else ela_arr

        # High variance across patches indicates inconsistent compression levels
        ela_std = float(np.std(gray_ela))
        ela_mean = float(np.mean(gray_ela))
        ela_peak_ratio = float(np.percentile(gray_ela, 98)) / 255.0

        return gray_ela, ela_peak_ratio

    def compute_noise_inconsistency(self, cv_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates localized noise floor variance using Laplacian filtering.
        Digital insertions and pasted text lack the native sensor noise of the camera.
        """
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img

        # High-pass filter via Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = np.abs(lap)

        # Local standard deviation in sliding 16x16 window
        kernel_size = 15
        mean_filter = cv2.blur(lap_abs, (kernel_size, kernel_size))
        mean_sq_filter = cv2.blur(lap_abs**2, (kernel_size, kernel_size))
        local_var = np.maximum(mean_sq_filter - mean_filter**2, 0)
        local_std = np.sqrt(local_var)

        # Normalize to 0-255
        norm_std = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Measure anomaly: difference between top 2% loudest noise regions and median
        p98 = np.percentile(norm_std, 98)
        p50 = np.percentile(norm_std, 50)
        noise_discrepancy = float(min(1.0, (p98 - p50) / 120.0))

        return norm_std, noise_discrepancy

    def compute_edge_gradient_discontinuity(self, cv_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates edge gradient continuity around text/labels using Sobel gradients.
        Altered stamps often have artificial haloing or abrupt edge falloffs.
        """
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Check for localized extreme gradient spikes
        high_grad_ratio = float(np.sum(norm_mag > 180)) / float(norm_mag.size)
        gradient_anomaly = float(min(1.0, high_grad_ratio * 40.0))

        return norm_mag, gradient_anomaly

    def generate_heatmap_and_regions(
        self,
        orig_img_bgr: np.ndarray,
        ela_map: np.ndarray,
        noise_map: np.ndarray,
        edge_map: np.ndarray
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Combines ELA, noise inconsistency, and edge maps into a unified forensic heatmap
        and identifies bounding boxes of anomalous / suspicious regions.
        """
        h, w = orig_img_bgr.shape[:2]

        # Resize all maps to match original dimensions
        ela_resized = cv2.resize(ela_map, (w, h))
        noise_resized = cv2.resize(noise_map, (w, h))
        edge_resized = cv2.resize(edge_map, (w, h))

        # Weighted combination of forensic channels
        fused = (
            0.50 * (ela_resized.astype(np.float32) / 255.0) +
            0.30 * (noise_resized.astype(np.float32) / 255.0) +
            0.20 * (edge_resized.astype(np.float32) / 255.0)
        )
        fused_u8 = np.clip(fused * 255.0, 0, 255).astype(np.uint8)

        # Apply Gaussian blur for smooth thermal representation
        blurred = cv2.GaussianBlur(fused_u8, (21, 21), 0)

        # Apply JET / TURBO colormap (Blue -> Cyan -> Yellow -> Red)
        heatmap_color = cv2.applyColorMap(blurred, cv2.COLORMAP_TURBO)

        # Alpha blend with original packaging image (45% heatmap, 55% original)
        overlay = cv2.addWeighted(orig_img_bgr, 0.55, heatmap_color, 0.45, 0)

        # Extract anomalous bounding boxes (spots where anomaly > 75th percentile + threshold)
        thresh_val = max(110, int(np.percentile(blurred, 88)))
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        anomalous_regions = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise contours & overly huge full-image contours
            if 300 < area < (h * w * 0.4):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                roi_score = float(np.mean(blurred[by:by+bh, bx:bx+bw])) / 255.0
                anomalous_regions.append({
                    "bbox": [int(bx), int(by), int(bx + bw), int(by + bh)],
                    "score": round(roi_score, 3),
                    "type": "potential_tamper_hotspot",
                    "severity": "HIGH" if roi_score > 0.65 else "MEDIUM"
                })

        # Draw visual bounding boxes on heatmap overlay for top anomalies
        for reg in anomalous_regions:
            x1, y1, x2, y2 = reg["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                overlay,
                f"Tamper Alert ({int(reg['score']*100)}%)",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        # Convert overlay to Base64 JPEG
        _, enc_buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(enc_buf).decode("utf-8")

        return overlay_b64, anomalous_regions

    def analyze(self, image_input) -> Dict[str, Any]:
        """
        Main analysis method.
        Accepts PIL Image, numpy array (BGR), or file path.
        Returns comprehensive forensic report.
        """
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            cv_img = image_input
            rgb_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_arr)
        elif isinstance(image_input, str):
            cv_img = cv2.imread(image_input)
            if cv_img is None:
                raise ValueError(f"Could not load image from {image_input}")
            rgb_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_arr)
        else:
            raise TypeError("Unsupported image format for tamper analysis")

        # 1. Error Level Analysis
        ela_map, ela_score = self.compute_ela(pil_img)

        # 2. Noise Floor Consistency
        noise_map, noise_score = self.compute_noise_inconsistency(cv_img)

        # 3. Gradient / Edge Discontinuity
        edge_map, edge_score = self.compute_edge_gradient_discontinuity(cv_img)

        # 4. Generate Heatmap & Anomalous Hotspots
        heatmap_b64, hotspots = self.generate_heatmap_and_regions(cv_img, ela_map, noise_map, edge_map)

        # 5. Calculate Aggregate Tamper Score
        if self.ml_classifier is not None and self.ml_scaler is not None:
            # ── ML path: use trained RandomForest probability ──────────────
            try:
                feat = self._extract_ml_features(pil_img, cv_img, ela_score, noise_score, edge_score)
                feat_sc = self.ml_scaler.transform(feat.reshape(1, -1))
                # probability of class 1 (Tampered)
                ml_prob = float(self.ml_classifier.predict_proba(feat_sc)[0][1])
                # Blend: 70% ML signal + 30% classical (keeps heatmap coherent)
                classical_score = (0.50 * ela_score) + (0.30 * noise_score) + (0.20 * edge_score)
                tamper_score = 0.70 * ml_prob + 0.30 * classical_score
                logger.debug(f"ML tamper prob={ml_prob:.3f}, classical={classical_score:.3f}, blended={tamper_score:.3f}")
            except Exception as e:
                logger.warning(f"ML scoring failed ({e}), falling back to classical CV.")
                tamper_score = (0.50 * ela_score) + (0.30 * noise_score) + (0.20 * edge_score)
        else:
            # ── Classical CV path (no model loaded) ───────────────────────
            tamper_score = (0.50 * ela_score) + (0.30 * noise_score) + (0.20 * edge_score)

        tamper_score = min(1.0, max(0.0, tamper_score))

        # Adjust score if explicit high-severity hotspot contours are detected
        if len(hotspots) > 0:
            top_spot_score = max([h["score"] for h in hotspots])
            if top_spot_score > 0.6:
                tamper_score = max(tamper_score, top_spot_score * 0.85)

        authenticity_score = round(max(0.0, (1.0 - tamper_score) * 100.0), 1)

        if tamper_score >= 0.60:
            status = "TAMPERED"
            verdict_desc = "High probability of digital manipulation, stamped overlay, or spliced packaging date."
        elif tamper_score >= 0.38:
            status = "SUSPICIOUS"
            verdict_desc = "Noticeable compression or noise discrepancies detected in localized packaging regions."
        else:
            status = "AUTHENTIC"
            verdict_desc = "Packaging exhibits uniform compression substrate and continuous noise characteristics."

        return {
            "status": status,
            "tamper_score": round(tamper_score, 3),
            "authenticity_score": authenticity_score,
            "verdict_description": verdict_desc,
            "heatmap_image": heatmap_b64,
            "anomalous_regions": hotspots,
            "ml_model_active": self.ml_classifier is not None,
            "forensic_breakdown": {
                "ela_compression_error": round(ela_score, 3),
                "noise_discrepancy": round(noise_score, 3),
                "gradient_edge_anomaly": round(edge_score, 3),
                "hotspots_detected": len(hotspots)
            }
        }
