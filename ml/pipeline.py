from modules.ocr_module import extract_expiry
from modules.expiry_logic import check_expiry
from modules.ela_module import compute_ela
from modules.fft_module import compute_fft
from modules.vit_module import compute_vit
from utils.image_utils import highlight_region
from fusion import compute_score
from confidence_engine import compute_confidence
from tags_engine import generate_tags

def run_pipeline(image_path, delivery_date):

    expiry_text, bbox, ocr_conf = extract_expiry(image_path)
    highlight = highlight_region(image_path, bbox)

    timeline, expiry_score = check_expiry(expiry_text, delivery_date)

    ela_score, heatmap = compute_ela(image_path)
    fft_score = compute_fft(image_path)
    vit_score = compute_vit(image_path)

    score = compute_score(ela_score, fft_score, vit_score, expiry_score)
    confidence = compute_confidence([ela_score, fft_score, vit_score, expiry_score])
    tags = generate_tags(ela_score, fft_score, vit_score, ocr_conf)

    return {
        "score": score,
        "confidence": confidence,
        "expiry": expiry_text,
        "timeline": timeline,
        "highlight": highlight,
        "heatmap": heatmap,
        "tags": tags
    }