from __future__ import annotations

import hashlib

import numpy as np


SCORE_FINGERPRINT_SCHEMA = "sleep_scores_i16_epoch_text_v1"


def score_fingerprint(
    scores: np.ndarray,
    epoch_seconds: float,
) -> str:
    values = np.asarray(scores, dtype="<i2").reshape(-1)
    digest = hashlib.sha256()
    digest.update(values.tobytes())
    digest.update(
        f"|epoch_seconds={float(epoch_seconds):.9g}".encode("ascii")
    )
    return digest.hexdigest()
