from datetime import datetime


def parse_expiry(expiry_text):
    """
    Try multiple common formats for printed expiry dates on packaging.
    Returns a datetime or None if parsing fails.
    """
    if expiry_text is None:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%m/%Y",
        "%m-%Y",
        "%m/%y",
        "%Y",
        "%d.%m.%y",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%m.%d.%Y",
        "%m-%d-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(expiry_text, fmt)
        except Exception:
            continue

    return None


def parse_date(date_text):
    """
    Parse the delivery date, being tolerant to different user / system formats.
    """
    if date_text is None:
        return None

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d.%m.%y",
        "%d.%m.%Y",
        "%m.%d.%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%y",
        "%m-%d-%y",
        "%Y/%m/%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_text, fmt)
        except Exception:
            continue

    return None


def check_expiry(expiry_text, delivery_date):
    """
    Context-aware expiry check.

    Returns:
        timeline_message (str)
        timeline_risk (float in [0, 1])
    """

    expiry_date = parse_expiry(expiry_text)

    # If OCR failed, we keep a moderate risk but do not over-penalise.
    if expiry_date is None:
        return "No expiry found", 0.3

    delivery = parse_date(delivery_date)
    if delivery is None:
        return "Invalid delivery date", 0.5

    if expiry_date < delivery:
        # Example: Delivered on 25 Feb but expired on 20 Feb.
        return "Expired before delivery", 1.0

    return "Valid timeline", 0.0