def compute_score(ela, fft, vit, expiry):
    """
    Multi-signal fraud risk in [0, 100].

    Each input is a risk-like value in [0, 1]:
      - ela: image editing risk (ELA)
      - fft: texture / frequency anomaly
      - vit: AI / synthetic probability proxy
      - expiry: expiry-vs-delivery timeline risk

    We combine them with fixed weights to keep the behaviour
    simple and interpretable:

        risk = w_ela*ela + w_fft*fft + w_vit*vit + w_exp*expiry

    where weights sum to 1. The returned score is this risk
    scaled to [0, 100].
    """

    w_ela = 0.10
    w_fft = 0.40
    w_vit = 0.30
    w_exp = 0.20

    risk = (
        w_ela * float(ela)
        + w_fft * float(fft)
        + w_vit * float(vit)
        + w_exp * float(expiry)
    )

    # Clamp to [0, 1] just in case of numerical drift.
    risk = max(0.0, min(1.0, risk))

    return int(risk * 100)