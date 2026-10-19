"""Step 4: estimate Bass parameters (p, q, M) for the look-alike innovation.

Look-alike: digital still cameras. Data: worldwide camera shipments by
CIPA member (Japanese) manufacturers, thousand units, 1951-2020
(data/cameras_and_lenses.csv).

Main fit window = 1999-2020, the digital-camera life cycle (CIPA starts
reporting digital still cameras in 1999; the earlier years are the film
camera generation, a different innovation). The full 1951-2020 fit is
reported for comparison, and the smartphone series from the lecture is
fitted as a benchmark.
"""
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import load_series, fit_bass_nls, fit_bass_ols, peak_time

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

rows = []

# --- main: digital era -------------------------------------------------------
cam = load_series("data/cameras_and_lenses.csv", start_year=1999)
nls = fit_bass_nls(cam["sales"], p0=0.01, q0=0.3)
ols = fit_bass_ols(cam["sales"])
rows.append({"series": "Cameras 1999-2020 (digital era)", "method": "NLS",
             "M_thousand_units": nls["m"], "p": nls["p"], "q": nls["q"],
             "se_p": nls["se_p"], "se_q": nls["se_q"], "R2": nls["r2"],
             "peak_t": peak_time(nls["p"], nls["q"])})
rows.append({"series": "Cameras 1999-2020 (digital era)", "method": "OLS (Bass 1969)",
             "M_thousand_units": ols["m"], "p": ols["p"], "q": ols["q"],
             "peak_t": peak_time(ols["p"], ols["q"]) if ols["p"] > 0 else None})

# --- comparison: full history -----------------------------------------------
full = load_series("data/cameras_and_lenses.csv")
nls_full = fit_bass_nls(full["sales"], p0=0.002, q0=0.2)
rows.append({"series": "Cameras 1951-2020 (full)", "method": "NLS",
             "M_thousand_units": nls_full["m"], "p": nls_full["p"], "q": nls_full["q"],
             "R2": nls_full["r2"], "peak_t": peak_time(nls_full["p"], nls_full["q"])})

# --- benchmark: smartphones (lecture data, million units) -------------------
sp = load_series("data/smartphones.csv")
nls_sp = fit_bass_nls(sp["sales"], p0=0.02, q0=0.4)
rows.append({"series": "Smartphones 2007-2020 (benchmark, mln units)", "method": "NLS",
             "M_thousand_units": nls_sp["m"] * 1000, "p": nls_sp["p"], "q": nls_sp["q"],
             "R2": nls_sp["r2"], "peak_t": peak_time(nls_sp["p"], nls_sp["q"])})

res = pd.DataFrame(rows)
res.to_csv("data/bass_parameters.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))

# --- plots -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(full["year"], full["sales"] / 1000, color="#9bb7d4", label="Shipments (actual)")
ax.axvspan(1998.5, 2020.5, color="#f2e2c4", alpha=.45, zorder=0, label="Digital era (fit window)")
ax.set_title("Camera shipments, CIPA manufacturers, 1951–2020")
ax.set_ylabel("Million units"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("img/camera_shipments_history.png")

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(cam["year"], cam["sales"] / 1000, color="#9bb7d4", label="Actual shipments")
ax.plot(cam["year"], nls["fitted"] / 1000, color="#c0392b", marker="o", lw=2,
        label=f"Bass fit (p={nls['p']:.4f}, q={nls['q']:.3f}, R²={nls['r2']:.2f})")
ax.set_title("Digital cameras: actual vs Bass model fit")
ax.set_ylabel("Million units"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("img/camera_bass_fit.png")

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(sp["year"], sp["sales"], color="#b9d4b0", label="Actual sales")
ax.plot(sp["year"], nls_sp["fitted"], color="#c0392b", marker="o", lw=2,
        label=f"Bass fit (p={nls_sp['p']:.4f}, q={nls_sp['q']:.3f})")
ax.set_title("Benchmark: smartphones, actual vs Bass fit")
ax.set_ylabel("Million units"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("img/smartphone_bass_fit.png")
