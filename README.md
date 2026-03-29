# MadDM automation scripts

Python scripts for automating **MadDM** analyses of a **t-channel dark matter model** with dark matter coupled to **right-handed bottom quarks**.
**Please read the manual before using the scripts ! Documentation is also referenced.** This repository is used for the **DMSimpt_LOMassive** dark matter model by **B. Fuks** with the restrict_card provided.

## Overview

This repository contains two main scripts:

- `scan_relic_paral_py3.py`  
  Runs automated MadDM scans, generates projects, evaluates relic density, and extracts direct/indirect detection observables.

- `scan4_heatmap.py`  
  Post-processes Scan 4 CSV outputs to produce heatmaps, exclusion regions, coupling rescalings, and iso-\(\Omega h^2\) curves.

## What these scripts do

The workflow is:

```text
scan_relic_paral_py3.py  --->  CSV outputs
                                   |
                                   v
                           scan4_heatmap.py
                                   |
                                   v
                           plots / heatmaps / contours
```

Main features:

* automated relic density scans,
* extraction of direct detection observables,
* extraction of indirect detection observables,
* CSV export,
* heatmap and contour plotting for the ((M_x, M_y)) plane,
* coupling rescaling for (\Omega h^2).

## Requirements

These scripts are designed to run with:

* Python 3
* `MadDM`
* `MG5_aMC`
* standard scientific Python packages (`numpy`, `matplotlib`, `scipy`, `pandas` if needed)

You must have a valid `MadDM` installation inside your `MG5_aMC` directory.

## Basic usage

### Run scans

```bash
python scan_relic_paral_py3.py --mg5 /path/to/MG5_aMC --outdir scan_outputs
```

### Run only Scan 4

```bash
python scan_relic_paral_py3.py \
  --mg5 /path/to/MG5_aMC \
  --scans 4 \
  --scan4-lambda 1 \
  --scan4-step 100 \
  --scan4-mx 100to2000 \
  --scan4-my 200to4000
```

### Produce a heatmap from Scan 4

```bash
python scan4_heatmap.py scan4_omega_map_lambda_1.csv --all
```

## Outputs

Depending on the selected scan, the scripts generate:

* CSV files with relic density and detection observables,
* PNG plots for 1D scans,
* 2D heatmaps and iso-Omegah² contour plots for Scan 4.

## Important note

**Please read the manual before using the scripts ! Documentation is also referenced.**

