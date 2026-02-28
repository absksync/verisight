"""
Unit tests for the VeriSight score-adjustment and decision logic.

These tests exercise the helper functions extracted in api.py without
starting the full FastAPI server or running the ML pipeline.
"""

import sys
from pathlib import Path

# Ensure the project root is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import adjust_score, make_decision  # noqa: E402


# ── adjust_score tests ───────────────────────────────────────────────────────

class TestAdjustScore:
    """Graduated boost scenarios."""

    def test_no_boost_when_neither_condition(self):
        """Plain image: no tags, low ViT → score unchanged."""
        assert adjust_score(40, "", 0.3) == 40

    def test_synthetic_tag_only(self):
        """'Synthetic Pattern' tag present, ViT below threshold → +25."""
        assert adjust_score(30, "Synthetic Pattern", 0.5) == 55

    def test_vit_above_07_only(self):
        """No tag, but ViT > 0.7 → +20."""
        assert adjust_score(25, "", 0.75) == 45

    def test_both_synthetic_and_vit(self):
        """Both conditions → cumulative +25 + +20 = +45."""
        assert adjust_score(20, "Synthetic Pattern", 0.75) == 65

    def test_very_high_vit(self):
        """ViT > 0.85 → gets both the >0.7 boost (+20) and >0.85 (+10) = +30."""
        assert adjust_score(30, "", 0.9) == 60

    def test_all_conditions_combined(self):
        """Synthetic tag + ViT > 0.85 → +25 + +20 + +10 = +55."""
        assert adjust_score(30, "Synthetic Pattern;Compression Artifact", 0.9) == 85

    def test_score_capped_at_100(self):
        """Boost should never exceed 100."""
        assert adjust_score(90, "Synthetic Pattern", 0.9) == 100

    def test_zero_raw_score(self):
        """Starting from 0 with all boosts → 55."""
        assert adjust_score(0, "Synthetic Pattern", 0.9) == 55

    def test_tags_with_other_entries(self):
        """'Synthetic Pattern' buried among other tags still triggers boost."""
        assert adjust_score(40, "OCR Confidence Drop;Synthetic Pattern;Texture Irregularity", 0.5) == 65


# ── make_decision tests ──────────────────────────────────────────────────────

class TestMakeDecision:
    """Threshold boundary tests."""

    def test_approve_low(self):
        assert make_decision(0) == "Approve"

    def test_approve_just_under(self):
        assert make_decision(29) == "Approve"

    def test_manual_review_at_30(self):
        assert make_decision(30) == "Manual Review"

    def test_manual_review_at_59(self):
        assert make_decision(59) == "Manual Review"

    def test_reject_at_60(self):
        assert make_decision(60) == "Reject"

    def test_reject_at_100(self):
        assert make_decision(100) == "Reject"
