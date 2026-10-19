"""Uniform regional analysis metrics accumulated before taking a square root."""

import numpy as np


def require_even_validation(samples, world_size, monthly_rtma):
    if monthly_rtma and samples % world_size:
        raise ValueError(
            f"validation samples ({samples}) must be divisible by world_size "
            f"({world_size}) for exact cross-rank metrics"
        )


def physical_metric_sums(prediction, analysis, background):
    """Return SSE, source-valid count, analysis-valid count, total, bad predictions.

    Inputs have channels last. The mask is for metrics only, never training loss.
    """
    prediction, analysis, background = [
        np.asarray(x) for x in (prediction, analysis, background)
    ]
    if prediction.shape != analysis.shape or background.shape != analysis.shape:
        raise ValueError("physical metric shapes do not match")
    axes = tuple(range(analysis.ndim - 1))
    valid_analysis = np.isfinite(analysis)
    valid = valid_analysis & np.isfinite(background)
    finite_prediction = np.isfinite(prediction)
    error = np.zeros(analysis.shape, dtype=np.float64)
    np.subtract(
        prediction.astype(np.float64),
        analysis.astype(np.float64),
        out=error,
        where=valid & finite_prediction,
    )
    return np.stack(
        (
            np.square(error).sum(axis=axes),
            valid.sum(axis=axes),
            valid_analysis.sum(axis=axes),
            np.full(analysis.shape[-1], np.prod(analysis.shape[:-1])),
            (valid & ~finite_prediction).sum(axis=axes),
        )
    ).astype(np.float64)


def finalize_physical_metrics(sums):
    # Fail AFTER all-reduce so a bad prediction on one rank cannot strand others.
    if np.any(sums[4] > 0):
        raise ValueError(
            "non-finite predictions on analysis/background-valid points: "
            + str(sums[4].astype(np.int64).tolist())
        )
    if np.any(sums[1] == 0):
        raise ValueError("physical RMSE has an empty channel after masking")
    return {
        "rmse": np.sqrt(sums[0] / sums[1]).tolist(),
        "valid_count": sums[1].astype(np.int64).tolist(),
        "analysis_valid_count": sums[2].astype(np.int64).tolist(),
        "coverage": (sums[1] / sums[2]).tolist(),
        "analysis_nan_fraction": (1 - sums[2] / sums[3]).tolist(),
    }
