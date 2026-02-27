from datetime import datetime

def check_expiry(expiry_text, delivery_date):
    if expiry_text is None:
        return "No expiry found", 0.4

    try:
        expiry_date = datetime.strptime(expiry_text, "%d/%m/%y")
    except:
        return "Format error", 0.3

    delivery = datetime.strptime(delivery_date, "%Y-%m-%d")

    if expiry_date < delivery:
        return "Expired before delivery", 1.0
    else:
        return "Valid timeline", 0.0