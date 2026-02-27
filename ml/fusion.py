def compute_score(ela, fft, vit, expiry):
    risk = (
        0.35 * ela +
        0.25 * fft +
        0.20 * vit +
        0.20 * expiry
    )
    return int(risk * 100)