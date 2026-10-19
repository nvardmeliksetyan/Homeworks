"""Steps 5-7: predict the diffusion of the Camp Snap screen-free digital camera
(TIME Best Inventions 2025) in the United States.

Inputs
------
q (imitation)  : from the digital-camera look-alike (script1, NLS, 1999-2020)
M (potential)  : Fermi estimate for the US, built from data/fermi_inputs.csv
p (innovation) : two scenarios
    A - look-alike p unchanged (0.0106)
    B - p calibrated so the model matches Camp Snap's real sales: launched 2023,
        "more than 1 million cameras" sold by early 2026 (Modern Retail, 2026)
        -> cumulative A(3) = 1,000,000 at end-2025.  B is the main forecast,
        run for M = 6M (low), 12M (base) and 20M (high).

New adopters per year are M * [F(t) - F(t-1)], so yearly and cumulative
numbers add up exactly.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from helper_functions import bass_F, peak_time, adopter_categories

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

# ---- look-alike parameters (from script1) ------------------------------------
params = pd.read_csv("data/bass_parameters.csv")
main = params[(params["series"].str.startswith("Cameras 1999")) & (params["method"] == "NLS")].iloc[0]
p_analogue, q = float(main["p"]), float(main["q"])

# ---- Fermi estimate of M (United States) -----------------------------------
fermi = pd.read_csv("data/fermi_inputs.csv").set_index("input")["value"]
pop = fermi["US population (2024)"]
children = pop * fermi["Share of US population under 18"]
camp_kids = children * fermi["Share of US children who attend summer camp"]
adults = pop - children
M_fermi = (camp_kids * fermi["Camp children who eventually get a screen-free camera"]
           + (children - camp_kids) * fermi["Other children who eventually get a screen-free camera"]
           + adults * fermi["Adults who eventually buy one (nostalgia/travel/gifts)"])
M_base = round(M_fermi / 1e6) * 1e6
print(f"Fermi M = {M_fermi:,.0f}  -> rounded to {M_base:,.0f}")
scenarios = {"low": fermi["Market potential M - low scenario"],
             "base": M_base,
             "high": fermi["Market potential M - high scenario"]}

# ---- calibrate p to observed sales ------------------------------------------
LAUNCH_YEAR = int(fermi["Camp Snap launch year"])
OBSERVED_CUM = fermi["Camp Snap cumulative cameras sold by early 2026"]
OBSERVED_T = 3                                   # end of 2025

def calibrate_p(M):
    return brentq(lambda p: bass_F(OBSERVED_T, p, q) - OBSERVED_CUM / M, 1e-5, 0.9)

years = np.arange(LAUNCH_YEAR, LAUNCH_YEAR + 18)       # 2023-2040
t = years - LAUNCH_YEAR + 1
out = pd.DataFrame({"year": years, "t": t})

def add_run(tag, label, M, p):
    out[f"new_adopters_{tag}"] = (M * (bass_F(t, p, q) - bass_F(t - 1, p, q))).round(0).astype(int)
    out[f"cumulative_{tag}"] = (M * bass_F(t, p, q)).round(0).astype(int)
    tp = peak_time(p, q)
    cats = adopter_categories(p, q)
    return {"scenario": label, "M": M, "p": round(p, 4), "q": round(q, 4),
            "peak_t": round(tp, 2), "peak_year": round(LAUNCH_YEAR - 1 + tp, 1),
            "early_majority_from": round(LAUNCH_YEAR - 1 + cats["t1_early_majority_starts"], 1),
            "laggards_from": round(LAUNCH_YEAR - 1 + cats["t2_laggards_start"], 1),
            "peak_year_new_adopters": int(out[f"new_adopters_{tag}"].max()),
            "model_cumulative_end_2025": int(round(M * bass_F(OBSERVED_T, p, q)))}

summary = [add_run(tag, f"B - calibrated p, {tag} M", M, calibrate_p(M)) for tag, M in scenarios.items()]
summary.append(add_run("A", "A - look-alike p, base M", M_base, p_analogue))
summary = pd.DataFrame(summary)

out.to_csv("data/camp_snap_forecast.csv", index=False)
summary.to_csv("data/camp_snap_scenarios.csv", index=False)

# ---- check against the one real data point ----------------------------------
validation = pd.DataFrame([
    {"check": "Cumulative cameras sold by end-2025", "actual": int(OBSERVED_CUM),
     "scenario_A_lookalike_p": int(summary.iloc[3]["model_cumulative_end_2025"]),
     "scenario_B_calibrated_p": int(summary.iloc[1]["model_cumulative_end_2025"])}])
validation.to_csv("data/camp_snap_validation.csv", index=False)

print(f"q (analogue) = {q:.4f}; analogue p = {p_analogue:.4f}")
print(summary.to_string(index=False))
print(validation.to_string(index=False))
print(out[["year", "new_adopters_base", "cumulative_base"]].to_string(index=False))

# ---- plots -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(years, out["new_adopters_base"] / 1e6, color="#e6a157", label="Base (M = 12M)")
ax.plot(years, out["new_adopters_low"] / 1e6, color="#7a7a7a", ls="--", marker=".", label="Low (M = 6M)")
ax.plot(years, out["new_adopters_high"] / 1e6, color="#c0392b", ls="--", marker=".", label="High (M = 20M)")
ax.set_title("Camp Snap: predicted new adopters per year (United States)")
ax.set_ylabel("Million units"); ax.set_xticks(years[::2]); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("img/camp_snap_new_adopters.png")

fig, ax = plt.subplots(figsize=(9, 4.4))
for tag, c in [("low", "#7a7a7a"), ("base", "#e6a157"), ("high", "#c0392b")]:
    ax.plot(years, out[f"cumulative_{tag}"] / 1e6, color=c, marker="o", ms=3, label=tag)
ax.scatter([2025], [1.0], color="black", zorder=5, label="Observed: >1M sold by early 2026")
ax.set_title("Camp Snap: predicted cumulative adopters (United States)")
ax.set_ylabel("Million units"); ax.set_xticks(years[::2]); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("img/camp_snap_cumulative.png")

fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(years, out["cumulative_A"] / 1e6, color="#4a78a8", marker="o", ms=3,
        label=f"A: look-alike p = {p_analogue:.4f}")
ax.plot(years, out["cumulative_base"] / 1e6, color="#e6a157", marker="o", ms=3,
        label=f"B: calibrated p = {summary.iloc[1]['p']:.4f}")
ax.scatter([2025], [1.0], color="black", zorder=5, label="Observed: >1M sold by early 2026")
ax.set_title("Camp Snap, base M = 12M: look-alike p vs calibrated p")
ax.set_ylabel("Cumulative adopters, million"); ax.set_xticks(years[::2]); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("img/camp_snap_scenario_A_vs_B.png")
