
# fiedler_plateau_validator.py
import numpy as np

def _best_two_level_fit(v_sorted: np.ndarray):
    """
    Best one-cut 1D k=2 fit for a sorted vector.
    Returns (R2, idx, (mean_left, mean_right), (std_left, std_right)).
    """
    n = len(v_sorted)
    if n < 3:
        return 0.0, None, (None, None), (None, None)
    ps = np.cumsum(v_sorted)
    ps2 = np.cumsum(v_sorted**2)
    total_var = np.var(v_sorted, ddof=0) + 1e-12

    best_R2, best_i = 0.0, None
    best_means = (None, None)
    best_stds = (None, None)

    for i in range(1, n):  # cut after i-1
        n1, n2 = i, n - i
        if n2 == 0:
            break
        sum1 = ps[i-1]; sum2 = ps[-1] - sum1
        mean1 = sum1 / n1; mean2 = sum2 / n2
        ss1 = ps2[i-1] - n1 * mean1**2
        ss2 = (ps2[-1] - ps2[i-1]) - n2 * mean2**2
        within_var = (ss1 + ss2) / n
        R2 = 1.0 - within_var / total_var
        if R2 > best_R2:
            best_R2 = float(R2)
            best_i = i
            std1 = float(np.sqrt(max(ss1/n1, 1e-12)))
            std2 = float(np.sqrt(max(ss2/n2, 1e-12)))
            best_means = (float(mean1), float(mean2))
            best_stds = (std1, std2)

    return best_R2, best_i, best_means, best_stds

def _conductance(W: np.ndarray, labels: np.ndarray) -> float:
    W = 0.5*(W + W.T)
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    A = labels.astype(bool)
    if A.sum() == 0 or A.sum() == len(A):
        return 1.0
    B = ~A
    cut = float(W[A][:, B].sum())
    volA = float(d[A].sum())
    volB = float(d[B].sum())
    denom = max(min(volA, volB), 1e-12)
    return cut / denom

def assess_fiedler_vector(v: np.ndarray,
                          W: np.ndarray | None = None,
                          r2_min: float = 0.80,
                          d_min: float = 1.0):
    """
    Assess if a provided Fiedler vector v exhibits two clear plateaus.

    Parameters
    ----------
    v : array-like
        The Fiedler vector (1D numpy array).
    W : array-like or None
        Optional similarity/adjacency matrix. If provided, we also compute
        conductance of the induced split.
    r2_min : float
        Minimum R^2 for the two-plateau fit to accept.
    d_min : float
        Minimum Cohen's d (effect size) between plateaus to accept.

    Returns
    -------
    dict with fields:
      - step_R2
      - threshold
      - means (left, right)
      - stds  (left, right)
      - sizes (n_left, n_right)
      - cohens_d
      - conductance (if W provided)
      - is_valid
    """
    v = np.asarray(v, float).reshape(-1)
    order = np.argsort(v)
    v_sorted = v[order]

    step_R2, idx, (m1, m2), (s1, s2) = _best_two_level_fit(v_sorted)
    if idx is None or idx <= 0 or idx >= len(v_sorted):
        thr = float(np.median(v_sorted))
        labels_sorted = v_sorted >= thr
    else:
        thr = float(0.5*(v_sorted[idx-1] + v_sorted[idx]))
        labels_sorted = np.zeros_like(v_sorted, dtype=bool)
        labels_sorted[idx:] = True

    # Map labels back to original order
    labels = np.zeros_like(labels_sorted, dtype=bool)
    labels[order] = labels_sorted

    n1 = int(labels.sum())
    n0 = int(len(labels) - n1)

    pooled_std = np.sqrt(((max(n0-1,0)*(s1 if s1 is not None else 0.0)**2) + 
                          (max(n1-1,0)*(s2 if s2 is not None else 0.0)**2)) / 
                         max(n0 + n1 - 2, 1))
    cohens_d = float(abs((m2 if m2 is not None else 0.0) - (m1 if m1 is not None else 0.0)) / (pooled_std + 1e-12))

    result = {
        "step_R2": float(step_R2),
        "threshold": thr,
        "means": (m1, m2),
        "stds": (s1, s2),
        "sizes": (n0, n1),
        "cohens_d": cohens_d,
        "is_valid": (step_R2 >= r2_min) and (cohens_d >= d_min)
    }

    if W is not None:
        result["conductance"] = float(_conductance(np.asarray(W, float), labels))
        # If conductance is available, tighten decision by also requiring it to be reasonable.
        result["is_valid"] = result["is_valid"] and (result["conductance"] <= 0.4)

    return result
