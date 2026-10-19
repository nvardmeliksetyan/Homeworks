# Homework on Bass Model: Camp Snap

DS 223 Marketing Analytics (AUA) · Nvard Meliksetyan

For this homework I picked **Camp Snap**, a screen-free digital camera from TIME's Best Inventions of 2025, and used the Bass diffusion model to predict how fast people in the US will adopt it.
TIME page: <https://time.com/collections/best-inventions-2025/7318270/camp-snap/>

## The idea in short

The Bass model needs three numbers:

- **p**: how many people buy on their own, because of the product itself or advertising.
- **q**: how many buy because people around them already have it (word of mouth).
- **M**: the total number of people who will eventually buy it.

Camp Snap is only a few years old, so its own sales are too short to estimate these. Instead I used the **digital camera** as a look-alike. Camp Snap *is* a digital camera with the screen taken out: same technology, a similar kind of buyer, and it also spreads by people seeing the photos and buying one. I fitted the Bass model to worldwide camera shipments for 1999–2020 (the digital-camera years) and reused the result for Camp Snap:

- **q** is taken from the digital camera, because word of mouth should work the same way.
- **p** is chosen so the model gives exactly the **1 million cameras** Camp Snap really sold by early 2026. With the digital camera's p unchanged, the model would predict only about 0.61 million, which is too slow.
- **M** comes from a Fermi estimate for the US: children who go to summer camp, other children and adults, which comes to about **12 million**. The adoption shares are my own assumptions, so I also tested M = 6 million and 20 million.

## Main results

| | p | q | M |
|---|---|---|---|
| Digital cameras 1999–2020 (fitted, R² = 0.91) | 0.0106 | 0.310 | 1,375 million |
| Camp Snap (main scenario, United States) | 0.0178 | 0.310 | 12 million |

The model says Camp Snap sales keep growing until a **peak of about 1.0 million cameras in 2031**, and half of the US market (about 6 million) is reached around 2031–2032. With a smaller market (6 million) the peak comes earlier, in 2028. With a larger one (20 million) it comes later, in 2033.

## How to run it

You need Python 3 and a few libraries. Install them once:

```bash
pip install numpy pandas scipy matplotlib
```

Then, **from this folder** (the one that contains `data` and the scripts), run the two scripts in this order:

```bash
python script1_estimate_parameters.py
python script2_predict_diffusion.py
```

The first one fits the model to the digital-camera data. The second one uses that result to forecast Camp Snap. Both print their results in the terminal and save tables in `data/` and charts in `img/`.

If you prefer a notebook, open `analysis.ipynb` and run all cells. It does the same thing step by step, and the outputs are already saved in it (you will also need `pip install jupyter` if you don't have it).

If you get a "No such file or directory: data/..." error, your terminal is in the wrong folder. Use `cd` to go into the folder that holds `data`.

## What is in the project

**The code**

- `helper_functions.py` holds the Bass model formulas and the functions that estimate p, q and M in two ways: non-linear least squares, and Bass's original regression. The other files use it.
- `script1_estimate_parameters.py` fits the model to the camera data, using the digital era, the full history for comparison, and the smartphone data from the lecture as a benchmark. It also makes the first three charts.
- `script2_predict_diffusion.py` builds the Camp Snap forecast. It calculates M from the Fermi inputs, compares the look-alike p with the calibrated p, runs the low, base and high scenarios, and makes the other three charts.
- `analysis.ipynb` is the notebook version of the whole analysis, with the outputs already shown.

**The data (`data/`)**

Three files are inputs:

- `cameras_and_lenses.csv` is worldwide camera shipments in thousand units per year, 1951–2020, by CIPA member companies (CIPA data via Statista). This is the main dataset and was provided with the course.
- `smartphones.csv` is smartphone sales in million units, 2007–2020, from the lecture. It is used only as a benchmark.
- `fermi_inputs.csv` has every number behind the market potential M, each marked as data (with its source) or as my own assumption.

The other four files are results that the scripts create, so you don't need to edit them:

- `bass_parameters.csv` holds the estimated p, q and M for every fit.
- `camp_snap_scenarios.csv` lists p, q, M, peak year and adopter-category years for each scenario: A uses the look-alike p, and B uses the calibrated p with low, base and high M.
- `camp_snap_forecast.csv` has the predicted new and cumulative adopters for each year 2023–2040, for every scenario.
- `camp_snap_validation.csv` compares the model with Camp Snap's real sales (1 million by end of 2025).

**Charts (`img/`)**

- `camera_shipments_history.png` shows camera shipments 1951–2020, with the digital-era fit window shaded.
- `camera_bass_fit.png` shows digital-camera shipments with the fitted Bass curve.
- `smartphone_bass_fit.png` shows the benchmark fit on the lecture's smartphone data.
- `camp_snap_new_adopters.png` shows the predicted Camp Snap buyers per year in the three scenarios.
- `camp_snap_cumulative.png` shows the predicted total buyers over time, with the real 1 million point.
- `camp_snap_scenario_A_vs_B.png` compares the look-alike p with the calibrated p against the real sales.

**Report and references**

- `report/` has the full written report as `report.pdf`, with its source `report_source.md`. It goes through all seven steps of the assignment.
- `references/` has all the sources with links, as `references.md` and `references.pdf`.

## Things to keep in mind

- The numbers are cameras sold, not people, since some people own more than one.
- The adoption shares behind M (50%, 7.5% and 1%) are my own assumptions, which is why the report also shows M = 6 million and 20 million.
- p is fitted to a single reported number ("more than 1 million" sold by early 2026). If the real number is higher, the market is bigger or the product is spreading faster.
- The Bass model does not include price, new models or competitors such as the Kodak Charmera.
