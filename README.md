# NoisyLabels

Code, results and manuscript for

> V. Lakmini, D. Vimalajeewa and B. Vidakovic. *Learning Wavelet Shrinkage from Noisy Labels: Stein Calibration with Interval Teachers.* Draft 06, 2026.

Several wavelet shrinkage rules are blended by a gate that chooses, coefficient by coefficient, between a strong and a weak rule. The gate is fitted to Stein's unbiased risk estimate (SURE), and noisy labels of signal membership act as a teacher. Label noise with bounded flip rates identifies the clean label probability only up to an interval. The paper turns that interval into a penalty that never increases the SURE-calibrated risk when the interval contains the target (Theorem 1). The simulations measure how often this guarantee holds when the interval is estimated, and compare the interval teacher with point teachers.

## Contents

```
paper/              manuscript, supplement, bibliography and all figures (Figs/)
code/python/        known-prior simulation of Section 7 and the figure scripts
code/matlab/        BAMS two-rule gates of Section 8 (base MATLAB only)
results/            saved simulation output and summary tables
```

| File | Reproduces |
|---|---|
| `code/python/sim7.py` | Section 7.2: the main design and the reliability-bound sensitivity (Tables 2 and 3), plus the secondary checks on the number of cells, pilot size and an affine gate |
| `code/python/sim7_pointwise.py` | Section 7.2: risks of the label-based pointwise rules |
| `code/python/stress.py` | Section 7.3: the boundary stress test (Table 4) |
| `code/python/sim7_report.py` | Section 7 tables and Figure `sim7_excess.pdf`, from the saved results |
| `code/python/fig_theorem1.py` | Figure `theorem1_schematic.pdf` |
| `code/python/quadrature_tables03.py` | The ζ(π) and population-risk tables of Draft 03 (kept for reference; Draft 06 gives ζ in closed form) |
| `code/matlab/BAMS2gate1.m` | Section 8, Gate 1: two BAMS rules with different slab scales, winner labels and a trained gate |
| `code/matlab/BAMS2gate2.m` | Section 8, Gate 2: spike-probability rules and a noisy membership teacher |

## Running the Python code

Requirements: Python 3.10 or later with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`). Figures use TeX Gyre Termes or Times New Roman when available and fall back to a serif font otherwise.

```bash
cd code/python
python3 sim7.py            # R = 500 replicates by default; python3 sim7.py 50 for a quick run
python3 sim7_pointwise.py
python3 stress.py          # R = 500 by default
python3 sim7_report.py     # tables to results/sim7_summary.txt, figure to paper/Figs/
python3 fig_theorem1.py
```

Results are written to `results/` and figures to `paper/Figs/`, whatever the working directory. The saved `results/*.npy` files let `sim7_report.py` run without repeating the simulation. The seed is fixed (`SEED = 20261125` in `sim7.py`), and all risks are computed exactly by quadrature, so a rerun reproduces the tables.

## Running the MATLAB code

Base MATLAB only, no toolboxes. From `code/matlab`:

```matlab
R1 = BAMS2gate1();                                  % Doppler, both tau pairs
R1 = BAMS2gate1(struct('signal','bumps'));          % same design on Bumps
R2 = BAMS2gate2();                                  % 200 repeats
R2 = BAMS2gate2(struct('nRep',10,'doPlot',false));  % quick check
```

Each script writes CSV tables and PDF/PNG figures with the prefix `cfg.outPrefix`. To write straight into the paper's figure folder, pass for example `struct('outPrefix','../../paper/Figs/BAMS2gate1')`.

## Building the manuscript

```bash
cd paper
pdflatex MLShrinkNoisyLabels06 && bibtex MLShrinkNoisyLabels06 && pdflatex MLShrinkNoisyLabels06 && pdflatex MLShrinkNoisyLabels06
```

and likewise for `MLShrinkNoisyLabels06_supplement`, which holds the recorruption analysis. Compiled PDFs of both are included.

## Citation

See `CITATION.cff`. Until the paper appears, please cite the draft as above.

## License

Code is released under the MIT License (see `LICENSE`). The manuscript and figures are © the authors; all rights reserved.
