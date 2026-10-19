"""Helper functions for Bass diffusion model estimation and forecasting.

Formulas follow the DS 223 lecture (Bass, 1969):
    F(t) = (1 - e^{-(p+q)t}) / (1 + (q/p) e^{-(p+q)t})
    f(t) = ((p+q)^2 / p) e^{-(p+q)t} / (1 + (q/p) e^{-(p+q)t})^2
    a(t) = M * f(t)   (new adopters in period t)
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def bass_F(t, p, q):
    """Cumulative fraction of the market that has adopted by time t."""
    e = np.exp(-(p + q) * t)
    return (1 - e) / (1 + (q / p) * e)


def bass_f(t, p, q):
    """Fraction of the market adopting at time t (density)."""
    e = np.exp(-(p + q) * t)
    return ((p + q) ** 2 / p) * e / (1 + (q / p) * e) ** 2


def bass_sales(t, m, p, q):
    """Number of adopters at time t: a(t) = M * f(t)."""
    return m * bass_f(t, p, q)


def load_series(path, start_year=None, end_year=None):
    """Read a year/sales csv (handles the UTF-8 BOM in the course files)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if start_year is not None:
        df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]
    return df.reset_index(drop=True)


def fit_bass_nls(sales, p0=0.01, q0=0.3):
    """Method 1: non-linear least squares on a(t) = M f(t) (like R's nls())."""
    sales = np.asarray(sales, dtype=float)
    t = np.arange(1, len(sales) + 1)
    params, cov = curve_fit(
        bass_sales, t, sales,
        p0=[sales.sum() * 1.5, p0, q0],
        bounds=([sales.sum(), 1e-6, 1e-6], [np.inf, 1.0, 2.0]),
        maxfev=100000,
    )
    se = np.sqrt(np.diag(cov))
    fitted = bass_sales(t, *params)
    r2 = 1 - ((sales - fitted) ** 2).sum() / ((sales - sales.mean()) ** 2).sum()
    return {"m": params[0], "p": params[1], "q": params[2],
            "se_m": se[0], "se_p": se[1], "se_q": se[2],
            "r2": r2, "fitted": fitted}


def fit_bass_ols(sales):
    """Method 2: Bass (1969) discrete OLS  a(t) = b0 + b1 A(t-1) + b2 A(t-1)^2.

    Then  M = (-b1 - sqrt(b1^2 - 4 b0 b2)) / (2 b2),  p = b0 / M,  q = -M b2.
    """
    sales = np.asarray(sales, dtype=float)
    A_prev = np.concatenate([[0.0], np.cumsum(sales)[:-1]])
    X = np.column_stack([np.ones_like(A_prev), A_prev, A_prev ** 2])
    b0, b1, b2 = np.linalg.lstsq(X, sales, rcond=None)[0]
    disc = b1 ** 2 - 4 * b0 * b2
    m = (-b1 - np.sqrt(disc)) / (2 * b2)
    return {"m": m, "p": b0 / m, "q": -m * b2, "b0": b0, "b1": b1, "b2": b2}


def peak_time(p, q):
    """Time of peak adoptions: t* = ln(q/p) / (p+q)."""
    return np.log(q / p) / (p + q)


def adopter_categories(p, q):
    """Rogers adopter-category boundaries derived from the Bass curve.

    Uses the standard result that the inflection points of F(t) are at
    t* -/+ 1.317/(p+q) (Mahajan, Muller & Srivastava, 1990).
    """
    tp = peak_time(p, q)
    w = 1.317 / (p + q)
    # before t1: innovators + early adopters | t1-peak: early majority
    # peak-t2: late majority | after t2: laggards
    return {"t1_early_majority_starts": tp - w, "peak_late_majority_starts": tp,
            "t2_laggards_start": tp + w}
