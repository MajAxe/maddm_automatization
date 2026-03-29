#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import shutil
from multiprocessing import Pool, cpu_count, current_process

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"


# Omega extraction

OMEGA_REGEXES = [
    re.compile(r"^\s*Omegah2\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", re.MULTILINE),
]

def extract_omega(output_text):
    for rgx in OMEGA_REGEXES:
        m = rgx.search(output_text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return float("nan")


# Direct detection extraction (SigmaN) from stdout/files text

_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

SIGMAN_KV_REGEX = re.compile(
    r"^\s*(SigmaN_(?:SI|SD)_[pn])\s*=\s*\[\s*(" + _NUM_RE + r")\s*(?:,|\s+)\s*(" + _NUM_RE + r")\s*\]\s*(?:#.*)?$",
    re.MULTILINE
)

SIGMAN_KEYS = ["SigmaN_SI_p", "SigmaN_SI_n", "SigmaN_SD_p", "SigmaN_SD_n"]

def extract_sigman_pairs(output_text):
    """
    Extract 4 pairs from lines like:
      SigmaN_SI_p                   = [2.88e-50,1.00e-46]           # LZ2024
      SigmaN_SI_n                   = [2.88e-50,1.00e-46]           # LZ2024
      SigmaN_SD_p                   = [0.00e+00,-1.00e+00]          # Pico60 (2019)
      SigmaN_SD_n                   = [0.00e+00,-1.00e+00]          # LZ2024
    Returns dict {key: (pred, UL)}; missing -> (-1,-1).
    """
    out = {k: (-1.0, -1.0) for k in SIGMAN_KEYS}
    try:
        for m in SIGMAN_KV_REGEX.finditer(output_text):
            name = m.group(1).strip()
            if name not in out:
                continue
            try:
                a = float(m.group(2))
                b = float(m.group(3))
                out[name] = (a, b)
            except Exception:
                pass
    except Exception:
        pass
    return out


# Indirect detection extraction (xsxs_bbx / Total_xsec)

INDIRECT_KEYS = ["xsxs_bbx", "Total_xsec"]

INDIRECT_KV_REGEX = re.compile(
    r"^\s*(xsxs_bbx|Total_xsec)\s*=\s*\[\s*(" + _NUM_RE + r")\s*(?:,|\s+)\s*(" + _NUM_RE + r")\s*\]\s*(?:#.*)?$",
    re.MULTILINE
)

def extract_indirect_pairs(output_text):
    """
    Extract 2 pairs from lines like:
      xsxs_bbx                      = [1.01e-53,5.98e-27]
      Total_xsec                    = [1.01e-53,-1.00e+00]
    Returns dict {key: (pred, UL)}. Missing -> (-1,-1).
    """
    out = {k: (-1.0, -1.0) for k in INDIRECT_KEYS}
    try:
        for m in INDIRECT_KV_REGEX.finditer(output_text):
            name = m.group(1).strip()
            if name not in out:
                continue
            try:
                a = float(m.group(2))
                b = float(m.group(3))
                out[name] = (a, b)
            except Exception:
                pass
    except Exception:
        pass
    return out

def status_from_pair(pair):
    """
    For terminal printing only.
      - "No Limit" if UL == -1
      - Allowed  <=> pred < UL
      - Excluded <=> pred >= UL
    """
    try:
        a, b = pair
        if b == -1.0:
            return "No Limit", False
        if a < b:
            return "Allowed", False
        return "Excluded", True
    except Exception:
        return "No Limit", False

def fmt_pair(pair):
    try:
        a, b = pair
        return "(%.3e vs %.3e)" % (float(a), float(b))
    except Exception:
        return "(? vs ?)"


# MadDM runner (safe for parallel)

def run_maddm(maddm_exe, script_text, workdir):
    """
    Run ./bin/maddm.py with a temporary .maddm script (UNIQUE per call).
    Returns combined stdout.
    """
    tmp_path = None
    try:
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".maddm", prefix="_tmp_run_", dir=workdir, delete=False)
        tmp_path = tf.name
        tf.write(script_text)
        tf.flush()
        tf.close()

        p = subprocess.Popen(
            [maddm_exe, tmp_path],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        out = p.communicate()[0]
        try:
            out = out.decode("utf-8", "ignore")
        except Exception:
            pass
        return out
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# Extract from project files fallback

def extract_omega_from_project_files(project_dir):
    best_mtime = -1
    best_val = float("nan")

    for root, dirs, files in os.walk(project_dir):
        for fn in files:
            path = os.path.join(root, fn)

            try:
                if os.path.getsize(path) > 20 * 1024 * 1024:
                    continue
            except Exception:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue

            if "Omegah2" not in txt:
                continue

            val = extract_omega(txt)
            if val == val:
                try:
                    mt = os.path.getmtime(path)
                except Exception:
                    mt = 0
                if mt > best_mtime:
                    best_mtime = mt
                    best_val = val

    return best_val

def extract_sigman_from_project_files(project_dir):
    best = {k: (-1.0, -1.0) for k in SIGMAN_KEYS}
    best_mtime = {k: -1 for k in SIGMAN_KEYS}

    for root, dirs, files in os.walk(project_dir):
        for fn in files:
            path = os.path.join(root, fn)

            try:
                if os.path.getsize(path) > 20 * 1024 * 1024:
                    continue
            except Exception:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue

            if "SigmaN_" not in txt:
                continue

            pairs = extract_sigman_pairs(txt)
            has_any = any((pairs[k] != (-1.0, -1.0)) for k in SIGMAN_KEYS)
            if not has_any:
                continue

            try:
                mt = os.path.getmtime(path)
            except Exception:
                mt = 0

            for k in SIGMAN_KEYS:
                if pairs[k] != (-1.0, -1.0) and mt > best_mtime[k]:
                    best_mtime[k] = mt
                    best[k] = pairs[k]

    return best

def extract_indirect_from_project_files(project_dir):
    best = {k: (-1.0, -1.0) for k in INDIRECT_KEYS}
    best_mtime = {k: -1 for k in INDIRECT_KEYS}

    for root, dirs, files in os.walk(project_dir):
        for fn in files:
            path = os.path.join(root, fn)

            try:
                if os.path.getsize(path) > 20 * 1024 * 1024:
                    continue
            except Exception:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue

            if ("xsxs_bbx" not in txt) and ("Total_xsec" not in txt):
                continue

            pairs = extract_indirect_pairs(txt)
            has_any = any((pairs[k] != (-1.0, -1.0)) for k in INDIRECT_KEYS)
            if not has_any:
                continue

            try:
                mt = os.path.getmtime(path)
            except Exception:
                mt = 0

            for k in INDIRECT_KEYS:
                if pairs[k] != (-1.0, -1.0) and mt > best_mtime[k]:
                    best_mtime[k] = mt
                    best[k] = pairs[k]

    return best

def get_omega_from_run_or_files(maddm_stdout, project_dir):
    om = extract_omega(maddm_stdout)
    if (om == om) and (not math.isinf(om)):
        return om
    return extract_omega_from_project_files(project_dir)

def get_sigman_from_run_or_files(maddm_stdout, project_dir):
    pairs = extract_sigman_pairs(maddm_stdout)
    if any((pairs[k] != (-1.0, -1.0)) for k in SIGMAN_KEYS):
        return pairs
    return extract_sigman_from_project_files(project_dir)

def get_indirect_from_run_or_files(maddm_stdout, project_dir):
    pairs = extract_indirect_pairs(maddm_stdout)
    if any((pairs[k] != (-1.0, -1.0)) for k in INDIRECT_KEYS):
        return pairs
    return extract_indirect_from_project_files(project_dir)

def is_finite_positive(x):
    try:
        return (x == x) and (not math.isinf(x)) and (x > 0.0)
    except Exception:
        return False

def omega_default_if_bad(om, default_val):
    if is_finite_positive(om):
        return om
    return float(default_val)


# Utils

def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

def ensure_project(maddm_exe, mg5_dir, project, model, dm_name):
    """
    Ensure the project directory exists.
    Returns True if created during this call, False if it already existed.
    """
    proj_dir = os.path.join(mg5_dir, project)
    if os.path.isdir(proj_dir):
        return False

    init_script = "\n".join([
        "import model {0}".format(model),
        "define darkmatter {0}".format(dm_name),
        "generate relic_density",
        "add direct_detection", # Comment if you don't use scan 4 !! IT WILL SAVE TIME
        "add indirect_detection", # Comment if you don't use scan 4 !! IT WILL SAVE TIME
        "output {0}".format(project),
        "quit",
        ""
    ])
    out = run_maddm(maddm_exe, init_script, mg5_dir)

    if not os.path.isdir(proj_dir):
        raise RuntimeError(
            "MadDM project '{0}' was not created.\n--- MadDM output ---\n{1}".format(project, out)
        )

    return True

def maddm_point_script(project, m_dm, m_med, coupling,
                       dm_pdg, med_pdg,
                       coup_block, coup_i, coup_j):
    lines = []
    lines.append("launch {0}".format(project))
    lines.append("set param_card mass {0} {1}".format(dm_pdg, m_dm))
    lines.append("set param_card mass {0} {1}".format(med_pdg, m_med))
    lines.append("set param_card {0} {1} {2} {3}".format(coup_block, coup_i, coup_j, coupling))

    # INTERACTIVES CHOICES !!
    lines.append("4") # sigmav -> flux_source
    lines.append("4") # flux_source -> flux_earth
    lines.append("done")
    lines.append("quit")
    lines.append("")
    return "\n".join(lines)

def frange(start, stop, step):
    x = start
    while x <= stop + 1e-9:
        yield x
        x += step

def grid_offset_first_point(offset, stop, step):
    """
    Default grid (as before):
      Example step=100 -> [5, 100, 200, ..., 4000]
    """
    vals = [float(offset)]
    for x in frange(0.0, float(stop), float(step)):
        if abs(x) < 1e-12:
            continue
        vals.append(float(x))

    out = []
    seen = set()
    for v in sorted(vals):
        key = "%.12g" % v
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out

def parse_range_to(s):
    """
    Parse strings like:
      "5to4000"
      "5 to 4000"
      "5,4000"
      "5:4000"
      "5..4000"
    Returns (a,b) floats or None if s is None/empty.
    """
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None

    # normalize
    ss2 = ss.lower().replace(" ", "")
    # supported separators
    for sep in ["to", ",", ":", ".."]:
        if sep in ss2:
            parts = ss2.split(sep)
            if len(parts) >= 2:
                try:
                    a = float(parts[0])
                    b = float(parts[1])
                    return (a, b)
                except Exception:
                    return None

    # last resort regex "a...b"
    m = re.match(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*.*\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", ss)
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except Exception:
            return None
    return None


def build_linear_scan_values(range_str, step, default_start, default_stop):
    rng = parse_range_to(range_str)
    if rng is None:
        a, b = float(default_start), float(default_stop)
    else:
        a, b = rng
    if b < a:
        a, b = b, a
    if step <= 0:
        raise ValueError("Step must be > 0")
    return list(frange(float(a), float(b), float(step)))

def build_log_scan_values(range_str, n, default_start, default_stop):
    rng = parse_range_to(range_str)
    if rng is None:
        a, b = float(default_start), float(default_stop)
    else:
        a, b = rng
    if b < a:
        a, b = b, a
    if a <= 0 or b <= 0:
        raise ValueError("Log scan bounds must be > 0")
    return logspace(float(a), float(b), int(n))

def logspace(a, b, n):
    la = math.log10(a)
    lb = math.log10(b)
    if n <= 1:
        return [10 ** la]
    vals = []
    for k in range(n):
        t = float(k) / float(n - 1)
        vals.append(10 ** (la + t * (lb - la)))
    return vals

def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def format_duration(seconds):
    try:
        seconds = float(seconds)
    except Exception:
        return "{0}s".format(seconds)
    if seconds < 60.0:
        return "{0:.2f}s".format(seconds)
    m = int(seconds // 60)
    s = seconds - 60.0 * m
    if m < 60:
        return "{0:d}m {1:.1f}s".format(m, s)
    h = int(m // 60)
    m2 = m - 60 * h
    return "{0:d}h {1:d}m {2:.1f}s".format(h, m2, s)


# Fit helpers (scipy)

def power_law_model(x, a, p):
    return a * (x ** p)

def power_law_fit(x, y):
    valid = []
    for xi, yi in zip(x, y):
        try:
            xi = float(xi)
            yi = float(yi)
        except Exception:
            continue
        if xi > 0.0 and yi > 0.0 and (not math.isinf(xi)) and (not math.isinf(yi)):
            valid.append((xi, yi))

    if len(valid) < 2:
        return None, None

    xv = [v[0] for v in valid]
    yv = [v[1] for v in valid]

    try:
        lx = [math.log(v) for v in xv]
        ly = [math.log(v) for v in yv]
        n = float(len(lx))
        sx = sum(lx)
        sy = sum(ly)
        sxx = sum(v * v for v in lx)
        sxy = sum(a * b for a, b in zip(lx, ly))
        den = n * sxx - sx * sx
        if abs(den) < 1e-15:
            p0 = -1.0
            a0 = yv[0]
        else:
            p0 = (n * sxy - sx * sy) / den
            ln_a0 = (sy - p0 * sx) / n
            a0 = math.exp(ln_a0)
        popt, _ = curve_fit(power_law_model, xv, yv, p0=(a0, p0), maxfev=100000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return None, None

def make_power_curve(x, a, p):
    try:
        return [power_law_model(float(xi), a, p) for xi in x]
    except Exception:
        return None

def float_to_latex(x, precision=3):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x == 0.0:
        return '0'
    s = ("%." + str(int(precision)) + "g") % x
    if 'e' in s or 'E' in s:
        mant, exp = re.split('[eE]', s)
        return r'%s\times 10^{%d}' % (mant, int(exp))
    return s

def format_plain_number_no_exp(x, precision=15):
    try:
        x = float(x)
    except Exception:
        return str(x)
    s = ("%." + str(int(precision)) + "f") % x
    s = s.rstrip('0').rstrip('.')
    if s == '-0':
        s = '0'
    return s

def power_law_fit_label(a, p, variable_tex):
    return r'Fit: $\Omega h^2 = %s\,%s^{%.3f}$' % (float_to_latex(a), variable_tex, float(p))

def omega_ylabel():
    return r'$\Omega h^2$'

def mx_xlabel():
    return r'$M_x\,[\mathrm{GeV}]$'

def my_xlabel():
    return r'$M_y\,[\mathrm{GeV}]$'

def lambda_xlabel():
    return r'$\lambda$'

def scan1_title(mx, lam, loglog=False):
    return r'$\Omega h^2$ vs $M_y$ ($M_x=%s\,\mathrm{GeV}$, $\lambda=%.3g$)' % (format_plain_number_no_exp(mx), float(lam))

def scan2_title(my, lam, loglog=False):
    return r'$\Omega h^2$ vs $M_x$ ($M_y=%s\,\mathrm{GeV}$, $\lambda=%.3g$)' % (format_plain_number_no_exp(my), float(lam))

def scan3_title(mx, my):
    return r'$\Omega h^2$ vs $\lambda$ ($M_x=%s\,\mathrm{GeV}$, $M_y=%s\,\mathrm{GeV}$)' % (format_plain_number_no_exp(mx), format_plain_number_no_exp(my))


# Plots

def plot_scan_with_fit(x, y, fit_y, xlabel, ylabel, title, out_png, fit_label=None):
    plt.figure()
    plt.scatter(x, y, label="MadDM")
    if fit_y is not None:
        lbl = fit_label if fit_label else "Power-law fit"
        plt.plot(x, fit_y, linestyle="--", label=lbl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_scan_loglog_with_fit(x, y, fit_y, xlabel, ylabel, title, out_png,
                              fit_label=None,
                              extra_curve=None, extra_label=None):
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(x, y, label="MadDM")
    if fit_y is not None:
        lbl = fit_label if fit_label else "Power-law fit"
        plt.plot(x, fit_y, linestyle="--", label=lbl)
    if extra_curve is not None:
        plt.plot(x, extra_curve, linestyle=":", label=extra_label if extra_label else "Reference")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# Scan selection parsing

def parse_scans(scans_str):
    if scans_str is None:
        return set([1, 2, 3, 4])

    s = scans_str.strip().lower()
    if s == "" or s == "all":
        return set([1, 2, 3, 4])

    parts = s.split(",")
    out = set()
    for p in parts:
        p = p.strip()
        if p == "":
            continue
        try:
            out.add(int(p))
        except Exception:
            pass
    out = set([x for x in out if x in (1, 2, 3, 4)])
    return out

# Scan4 parallel worker machinery
#   - one MadDM project per process: <base>_1, <base>_2, ...

_WORK = {}

def _scan4_init_worker(mg5_dir, maddm_exe, project_names, proj_dirs,
                       dm_pdg, med_pdg, coup_block, coup_i, coup_j, lambda_val):
    idx = 0
    try:
        ident = current_process()._identity
        if ident:
            idx = int(ident[0]) - 1
    except Exception:
        idx = 0
    idx = idx % max(1, len(project_names))

    _WORK["mg5_dir"] = mg5_dir
    _WORK["maddm_exe"] = maddm_exe
    _WORK["project"] = project_names[idx]
    _WORK["proj_dir"] = proj_dirs[idx]
    _WORK["dm_pdg"] = dm_pdg
    _WORK["med_pdg"] = med_pdg
    _WORK["coup_block"] = coup_block
    _WORK["coup_i"] = coup_i
    _WORK["coup_j"] = coup_j
    _WORK["lambda_val"] = float(lambda_val)

def _scan4_worker(task):
    # task: (m_dm, m_med)
    m_dm, m_med = task

    mg5_dir = _WORK["mg5_dir"]
    maddm_exe = _WORK["maddm_exe"]
    project = _WORK["project"]
    proj_dir = _WORK["proj_dir"]

    script = maddm_point_script(
        project=project,
        m_dm=m_dm,
        m_med=m_med,
        coupling=_WORK["lambda_val"],
        dm_pdg=_WORK["dm_pdg"],
        med_pdg=_WORK["med_pdg"],
        coup_block=_WORK["coup_block"],
        coup_i=_WORK["coup_i"],
        coup_j=_WORK["coup_j"],
    )

    out = run_maddm(maddm_exe, script, mg5_dir)

    om = get_omega_from_run_or_files(out, proj_dir)
    dd_pairs = get_sigman_from_run_or_files(out, proj_dir)
    id_pairs = get_indirect_from_run_or_files(out, proj_dir)

    # Return only numeric values (pred, UL) for CSV requirement
    return (m_dm, m_med, om, dd_pairs, id_pairs)


# Main

def main():
    t0 = time.time()

    ap = argparse.ArgumentParser()

    ap.add_argument("--mg5", required=True,
                    help="Path to MG5_aMC directory (contains ./bin/maddm.py)")
    ap.add_argument("--model", default="DMSimpt_v2_0_LOmassive-F3S_br", help="UFO model name")
    ap.add_argument("--project", default="scan_bottom_RD", help="MadDM base project name")
    ap.add_argument("--dmname", default="xs", help="DM particle name (often 'xs')")
    ap.add_argument("--outdir", default="scan_outputs", help="Output directory (CSV + PNG)")

    ap.add_argument("--scans", default="all",
                    help="Which scans to run: 'all' or comma list like '1,3,4'")

    ap.add_argument("--dm-pdg", type=int, default=51)
    ap.add_argument("--med-pdg", type=int, default=5920005)
    ap.add_argument("--coup-block", default="dmf3d")
    ap.add_argument("--coup-i", type=int, default=3)
    ap.add_argument("--coup-j", type=int, default=3)

    ap.add_argument("--scan3-n", type=int, default=30, help="Nb points for scan3 (logspace)")

    ap.add_argument("--scan1-mx", type=float, default=1000.0,
                    help="Fixed Mx (GeV) for scan1")
    ap.add_argument("--scan1-my", default="1100to4100",
                    help="My range for scan1, e.g. '1100to4100'")
    ap.add_argument("--scan1-step", type=float, default=250.0,
                    help="Step (GeV) for scan1 My sweep")
    ap.add_argument("--scan1-lambda", type=float, default=1.0,
                    help="Fixed lambda for scan1")

    ap.add_argument("--scan2-my", type=float, default=2000.0,
                    help="Fixed My (GeV) for scan2")
    ap.add_argument("--scan2-mx", default="100to1900",
                    help="Mx range for scan2, e.g. '100to1900'")
    ap.add_argument("--scan2-step", type=float, default=100.0,
                    help="Step (GeV) for scan2 Mx sweep")
    ap.add_argument("--scan2-lambda", type=float, default=1.0,
                    help="Fixed lambda for scan2")

    ap.add_argument("--scan3-mx", type=float, default=1000.0,
                    help="Fixed Mx (GeV) for scan3")
    ap.add_argument("--scan3-my", type=float, default=2000.0,
                    help="Fixed My (GeV) for scan3")
    ap.add_argument("--scan3-lambda", default="1e-3to10",
                    help="Lambda range for scan3, e.g. '1e-3to10'")

    ap.add_argument("--scan4-step", type=float, default=50.0,
                    help="Step (GeV) for scan4 grid (default grid anchored at 0, first point is 5 GeV).")

    ap.add_argument("--scan4-lambda", type=float, default=1.0,
                    help="Coupling lambda for scan4 (fixed value)")

    ap.add_argument("--scan4-mx", default=None,
                    help="Scan4 Mx range like '5to4000' (or '5 to 4000', '5:4000', '5,4000'). If omitted, uses default offset grid.")
    ap.add_argument("--scan4-my", default=None,
                    help="Scan4 My range like '5to4000' (or '5 to 4000', '5:4000', '5,4000'). If omitted, uses default offset grid.")

    args = ap.parse_args()

    scans_to_run = parse_scans(args.scans)
    if len(scans_to_run) == 0:
        raise RuntimeError("No valid scan selected. Use --scans all or --scans 1,2,3,4")

    mg5_dir = os.path.abspath(os.path.expanduser(args.mg5))
    maddm_exe = os.path.join(mg5_dir, "bin", "maddm")
    if not os.path.isfile(maddm_exe):
        raise IOError("Cannot find {0}. Check --mg5".format(maddm_exe))

    outdir = os.path.abspath(os.path.expanduser(args.outdir))
    ensure_dir(outdir)

    done_outputs = []
    created_projects = []  # projects created during this run and safe to delete

    # NOTE: projects created by THIS AND ONLY THIS are deleted at "finally". COMMENT IF YOU WANT SAVE THE FILES

    try:
        base_proj_dir = os.path.join(mg5_dir, args.project)


        # Scan 1

        if 1 in scans_to_run:
            if ensure_project(maddm_exe, mg5_dir, args.project, args.model, args.dmname):
                created_projects.append(args.project)

            print("\n====================")
            print("SCAN 1 : Omega vs My")
            print("====================")
            m_dm_1 = float(args.scan1_mx)
            g_1 = float(args.scan1_lambda)
            med_vals_1 = build_linear_scan_values(args.scan1_my, args.scan1_step, 1100.0, 4100.0)
            rows1, omega1 = [], []

            for m_med in med_vals_1:
                script = maddm_point_script(args.project, m_dm_1, m_med, g_1,
                                            args.dm_pdg, args.med_pdg,
                                            args.coup_block, args.coup_i, args.coup_j)
                out = run_maddm(maddm_exe, script, mg5_dir)
                om = get_omega_from_run_or_files(out, base_proj_dir)

                om = omega_default_if_bad(om, 1.0)

                rows1.append([m_med, om])
                omega1.append(om)
                print("[Scan1] My={0:.1f}  Omega={1}".format(m_med, om))

            # fits
            A1, p1 = power_law_fit(med_vals_1, omega1)
            fit1 = None
            fit1_label = None
            if A1 is not None:
                fit1 = make_power_curve(med_vals_1, A1, p1)
                fit1_label = power_law_fit_label(A1, p1, r'M_y')
                print("  -> Fit Scan1: A={0:.6g}, p={1:.4f} (expected ~ -2)".format(A1, p1))
            else:
                print("  -> Fit Scan1 failed (not enough valid points)")

            csv1 = os.path.join(outdir, "scan1_omega_vs_mMed.csv")
            png1 = os.path.join(outdir, "scan1_omega_vs_mMed.png")
            png1log = os.path.join(outdir, "scan1_omega_vs_mMed_loglog.png")
            save_csv(csv1, ["mMed_GeV", "Omega_h2"], rows1)

            plot_scan_with_fit(
                med_vals_1, omega1, fit1,
                my_xlabel(), omega_ylabel(),
                scan1_title(m_dm_1, g_1),
                png1,
                fit_label=fit1_label
            )
            plot_scan_loglog_with_fit(
                med_vals_1, omega1, fit1,
                my_xlabel(), omega_ylabel(),
                scan1_title(m_dm_1, g_1, loglog=True),
                png1log,
                fit_label=fit1_label
            )

            done_outputs.append((csv1, png1))
            done_outputs.append((None, png1log))


        # Scan 2

        if 2 in scans_to_run:
            if ensure_project(maddm_exe, mg5_dir, args.project, args.model, args.dmname):
                created_projects.append(args.project)

            print("\n====================")
            print("SCAN 2 : Omega vs Mx")
            print("====================")
            m_med_2 = float(args.scan2_my)
            g_2 = float(args.scan2_lambda)
            dm_vals_2 = build_linear_scan_values(args.scan2_mx, args.scan2_step, 100.0, 1900.0)
            rows2, omega2 = [], []

            for m_dm in dm_vals_2:
                script = maddm_point_script(args.project, m_dm, m_med_2, g_2,
                                            args.dm_pdg, args.med_pdg,
                                            args.coup_block, args.coup_i, args.coup_j)
                out = run_maddm(maddm_exe, script, mg5_dir)
                om = get_omega_from_run_or_files(out, base_proj_dir)

                om = omega_default_if_bad(om, 1.0)

                rows2.append([m_dm, om])
                omega2.append(om)
                print("[Scan2] Mx={0:.1f}  Omega={1}".format(m_dm, om))

            A2, p2 = power_law_fit(dm_vals_2, omega2)
            fit2 = None
            fit2_label = None
            if A2 is not None:
                fit2 = make_power_curve(dm_vals_2, A2, p2)
                fit2_label = power_law_fit_label(A2, p2, r'M_x')
                print("  -> Fit Scan2: A={0:.6g}, p={1:.4f} (expected ~ +2)".format(A2, p2))
            else:
                print("  -> Fit Scan2 failed (not enough valid points)")

            csv2 = os.path.join(outdir, "scan2_omega_vs_mDM.csv")
            png2 = os.path.join(outdir, "scan2_omega_vs_mDM.png")
            png2log = os.path.join(outdir, "scan2_omega_vs_mDM_loglog.png")
            save_csv(csv2, ["mDM_GeV", "Omega_h2"], rows2)

            plot_scan_with_fit(
                dm_vals_2, omega2, fit2,
                mx_xlabel(), omega_ylabel(),
                scan2_title(m_med_2, g_2),
                png2,
                fit_label=fit2_label
            )
            plot_scan_loglog_with_fit(
                dm_vals_2, omega2, fit2,
                mx_xlabel(), omega_ylabel(),
                scan2_title(m_med_2, g_2, loglog=True),
                png2log,
                fit_label=fit2_label
            )

            done_outputs.append((csv2, png2))
            done_outputs.append((None, png2log))


        # Scan 3

        if 3 in scans_to_run:
            if ensure_project(maddm_exe, mg5_dir, args.project, args.model, args.dmname):
                created_projects.append(args.project)

            print("\n=========================")
            print("SCAN 3 : Omega vs Couplage")
            print("=========================")
            m_dm_3 = float(args.scan3_mx)
            m_med_3 = float(args.scan3_my)
            g_vals_3 = build_log_scan_values(args.scan3_lambda, args.scan3_n, 1e-3, 10.0)
            rows3, omega3 = [], []

            for l in g_vals_3:
                script = maddm_point_script(args.project, m_dm_3, m_med_3, l,
                                            args.dm_pdg, args.med_pdg,
                                            args.coup_block, args.coup_i, args.coup_j)
                out = run_maddm(maddm_exe, script, mg5_dir)
                om = get_omega_from_run_or_files(out, base_proj_dir)

                rows3.append([l, om])
                omega3.append(om)
                print("[Scan3] l={0:.6g}  Omega={1}".format(l, om))

            A3, p3 = power_law_fit(g_vals_3, omega3)
            fit3 = None
            fit3_label = None
            if A3 is not None:
                fit3 = make_power_curve(g_vals_3, A3, p3)
                fit3_label = power_law_fit_label(A3, p3, r'\lambda')
                print("  -> Fit Scan3: A={0:.6g}, p={1:.4f} (expected ~ -4)".format(A3, p3))
            else:
                print("  -> Fit Scan3 failed (not enough valid points)")

            csv3 = os.path.join(outdir, "scan3_omega_vs_couplage.csv")
            png3 = os.path.join(outdir, "scan3_omega_vs_couplage_loglog.png")
            save_csv(csv3, ["couplage", "Omega_h2"], rows3)

            plot_scan_loglog_with_fit(
                g_vals_3, omega3, fit3,
                lambda_xlabel(), omega_ylabel(),
                scan3_title(m_dm_3, m_med_3),
                png3,
                fit_label=fit3_label
            )

            done_outputs.append((csv3, png3))


        # Scan 4 (PARALLEL)

        if 4 in scans_to_run:
            print("\n============================")
            print("SCAN 4 : Omega(Mx,My) -> CSV + external --all heatmap (PARALLEL)")
            print("============================")

            g_4 = float(args.scan4_lambda)
            step = float(args.scan4_step)

            mx_rng = parse_range_to(args.scan4_mx)
            my_rng = parse_range_to(args.scan4_my)

            if mx_rng is None:
                dm_vals_4 = grid_offset_first_point(5.0, 4000.0, step)
            else:
                mx_a, mx_b = mx_rng
                if mx_b < mx_a:
                    mx_a, mx_b = mx_b, mx_a
                dm_vals_4 = list(frange(mx_a, mx_b, step))

            if my_rng is None:
                med_vals_4 = grid_offset_first_point(5.0, 4000.0, step)
            else:
                my_a, my_b = my_rng
                if my_b < my_a:
                    my_a, my_b = my_b, my_a
                med_vals_4 = list(frange(my_a, my_b, step))

            print("[Scan4] Mx grid: {0} points (min={1}, max={2})".format(len(dm_vals_4), dm_vals_4[0], dm_vals_4[-1]))
            print("[Scan4] My grid: {0} points (min={1}, max={2})".format(len(med_vals_4), med_vals_4[0], med_vals_4[-1]))

            # Build tasks (only My > Mx)
            tasks = []
            for m_dm in dm_vals_4:
                for m_med in med_vals_4:
                    if m_med > m_dm:
                        tasks.append((m_dm, m_med))
            print("[Scan4] Total points (My>Mx):", len(tasks))

            # Create one MadDM project per CPU core
            n_workers = max(1, cpu_count())
            project_names = ["{0}_{1}".format(args.project, i + 1) for i in range(n_workers)]
            proj_dirs = [os.path.join(mg5_dir, p) for p in project_names]

            print("[Scan4] Using {0} workers / projects: {1}_1 .. {1}_{0}".format(n_workers, args.project))

            # Ensure all projects exist (sequential init)
            for p in project_names:
                if ensure_project(maddm_exe, mg5_dir, p, args.model, args.dmname):
                    created_projects.append(p)

            # CSV header with values + UL immediately to the right
            csv_header = ["mDM_GeV", "mMed_GeV", "Omega_h2"]
            for k in SIGMAN_KEYS:
                csv_header.append(k)
                csv_header.append("UL_" + k)
            for k in INDIRECT_KEYS:
                csv_header.append(k)
                csv_header.append("UL_" + k)

            rows4 = []

            pool = Pool(
                processes=n_workers,
                initializer=_scan4_init_worker,
                initargs=(mg5_dir, maddm_exe, project_names, proj_dirs,
                          args.dm_pdg, args.med_pdg, args.coup_block, args.coup_i, args.coup_j, g_4)
            )

            try:
                for (m_dm, m_med, om, dd_pairs, id_pairs) in pool.imap_unordered(_scan4_worker, tasks, chunksize=1):
                    # Terminal printing
                    parts = []
                    for k in SIGMAN_KEYS:
                        st, _ = status_from_pair(dd_pairs.get(k, (-1.0, -1.0)))
                        if st in ("Allowed", "Excluded"):
                            parts.append("{0}: {1} {2}".format(k, st, fmt_pair(dd_pairs.get(k, (-1.0, -1.0)))))
                        else:
                            parts.append("{0}: {1}".format(k, st))
                    for k in INDIRECT_KEYS:
                        st, _ = status_from_pair(id_pairs.get(k, (-1.0, -1.0)))
                        if st in ("Allowed", "Excluded"):
                            parts.append("{0}: {1} {2}".format(k, st, fmt_pair(id_pairs.get(k, (-1.0, -1.0)))))
                        else:
                            parts.append("{0}: {1}".format(k, st))

                    msg = "[Scan4] Mx={0:.1f}  My={1:.1f}  Omega={2}".format(m_dm, m_med, om)
                    msg += " | " + " | ".join(parts)
                    print(msg)

                    row = [m_dm, m_med, om]
                    for k in SIGMAN_KEYS:
                        pred, ul = dd_pairs.get(k, (-1.0, -1.0))
                        row.append(pred)
                        row.append(ul)
                    for k in INDIRECT_KEYS:
                        pred, ul = id_pairs.get(k, (-1.0, -1.0))
                        row.append(pred)
                        row.append(ul)

                    rows4.append(row)
            finally:
                try:
                    pool.close()
                except Exception:
                    pass
                try:
                    pool.join()
                except Exception:
                    pass

            lam_tag = ("%.6g" % g_4).replace(".", "p")
            csv4 = os.path.join(outdir, "scan4_omega_map_lambda_{0}.csv".format(lam_tag))
            save_csv(csv4, csv_header, rows4)
            done_outputs.append((csv4, None))

            # Keep your external heatmap call as-is (NOTE: your heatmap script must be adapted to read numeric cols now)
            heatmap_script = os.path.join(os.getcwd(), "scan4_heatmap.py")
            if not os.path.isfile(heatmap_script):
                print("[Scan4] WARNING: cannot find scan4_heatmap.py in current directory:")
                print("        {0}".format(heatmap_script))
                print("        -> CSV saved only: {0}".format(csv4))
            else:
                cmd = [sys.executable, heatmap_script, csv4, "--all"]
                print("[Scan4] Running:", " ".join(cmd))
                try:
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    out_heat = p.communicate()[0]
                    try:
                        out_heat = out_heat.decode("utf-8", "ignore")
                    except Exception:
                        pass
                    print(out_heat)
                except Exception as e:
                    print("[Scan4] ERROR while running scan4_heatmap.py:", e)


        # Summary

        print("\nDone.")
        print("Scans executed: {0}".format(sorted(list(scans_to_run))))
        print("Outputs in: {0}".format(outdir))
        for a, b in done_outputs:
            if a is not None:
                print("- {0}".format(os.path.basename(a)))
            if b is not None:
                print("- {0}".format(os.path.basename(b)))

    finally:
        # Remove only projects that were created by THIS run
        # Equivalent to: rm -rf <mg5_dir>/<project_name>
        for p in list(dict.fromkeys(created_projects)):  # unique, preserve order
            proj_path = os.path.join(mg5_dir, p)
            try:
                if os.path.isdir(proj_path):
                    shutil.rmtree(proj_path)
            except Exception:
                pass

        dt = time.time() - t0
        print("Temps mis: {0}".format(format_duration(dt)))

if __name__ == "__main__":
    main()
