"""Performance measures used in the manuscript (Methods: Statistical analysis)."""
import numpy as np
from scipy import stats


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y, scores):
    """Fast DeLong (Sun & Xu, 2014). scores: (k, n) for k models on the same patients."""
    y = np.asarray(y).astype(bool)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(s) for s in np.concatenate([pos, neg], axis=1)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def auc_delong(y, prob, alpha=0.05):
    """AUC with DeLong (1 - alpha) confidence interval."""
    aucs, cov = _delong_components(y, prob)
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return aucs[0], max(0.0, aucs[0] - z * se), min(1.0, aucs[0] + z * se)


def delong_paired_test(y, prob_a, prob_b):
    """Two-sided paired DeLong test of AUC(a) = AUC(b) on the same patients."""
    aucs, cov = _delong_components(y, np.vstack([prob_a, prob_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return 2 * stats.norm.sf(abs(z))


def threshold_metrics(y, prob, threshold):
    """Sensitivity, specificity, predictive values and accuracy at probability >= threshold."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "accuracy": div(tp + tn, len(y))}


def brier(y, prob):
    return float(np.mean((np.asarray(prob) - np.asarray(y)) ** 2))


def youden_threshold(y, prob):
    """Probability cut-off maximizing sensitivity + specificity - 1 (ties: lowest cut-off)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, prob)
    best = np.argmax(tpr - fpr)
    return float(thr[best])
