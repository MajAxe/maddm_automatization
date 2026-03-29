#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import csv
import math
import os
import re
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.tri as mtri
from matplotlib.patches import Patch

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"

DD_KEYS = ["SigmaN_SI_p", "SigmaN_SI_n", "SigmaN_SD_p", "SigmaN_SD_n"]
ID_KEYS = ["xsxs_bbx", "Total_xsec"]


# ---------------------------------------------
# Piecewise normalization:
#   - outside [vlow, vhigh] -> constant white (0 or 1)
#   - inside                -> map to a sub-interval (lo_color..hi_color)
#     with center at 0.12 darkest.
# ---------------------------------------------
class PiecewiseRelicNormalize(colors.Normalize):
    def __init__(self, vmin, vlow, vcenter, vhigh, vmax,
                 lo_color=0.20, mid_color=0.50, hi_color=0.80, clip=False):
        super(PiecewiseRelicNormalize, self).__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.vlow = float(vlow)
        self.vcenter = float(vcenter)
        self.vhigh = float(vhigh)

        self.lo_color = float(lo_color)
        self.mid_color = float(mid_color)
        self.hi_color = float(hi_color)

    @staticmethod
    def _safe_lin(x, x0, x1, y0, y1):
        if x1 == x0:
            return y1
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def __call__(self, value, clip=None):
        x = np.array(value, dtype=float)
        out = np.empty_like(x, dtype=float)
        out[:] = np.nan

        finite = np.isfinite(x)
        lo = finite & (x < self.vlow)
        hi = finite & (x > self.vhigh)
        mid1 = finite & (x >= self.vlow) & (x <= self.vcenter)
        mid2 = finite & (x > self.vcenter) & (x <= self.vhigh)

        out[lo] = 0.0
        out[hi] = 1.0

        if self.vcenter != self.vlow:
            out[mid1] = self.lo_color + (x[mid1] - self.vlow) * (self.mid_color - self.lo_color) / (self.vcenter - self.vlow)
        else:
            out[mid1] = self.mid_color

        if self.vhigh != self.vcenter:
            out[mid2] = self.mid_color + (x[mid2] - self.vcenter) * (self.hi_color - self.mid_color) / (self.vhigh - self.vcenter)
        else:
            out[mid2] = self.hi_color

        out = np.clip(out, 0.0, 1.0)
        return np.ma.array(out, mask=~finite)

    def inverse(self, value):
        y = np.array(value, dtype=float)
        out = np.empty_like(y, dtype=float)

        m0 = (y <= self.lo_color)
        m1 = (y > self.lo_color) & (y <= self.mid_color)
        m2 = (y > self.mid_color) & (y < self.hi_color)
        m3 = (y >= self.hi_color)

        denom0 = (self.lo_color - 0.0) if self.lo_color != 0.0 else 1.0
        denom1 = (self.mid_color - self.lo_color) if self.mid_color != self.lo_color else 1.0
        denom2 = (self.hi_color - self.mid_color) if self.hi_color != self.mid_color else 1.0
        denom3 = (1.0 - self.hi_color) if self.hi_color != 1.0 else 1.0

        out[m0] = float(self.vmin) + (y[m0] - 0.0) * (self.vlow - float(self.vmin)) / denom0
        out[m1] = self.vlow + (y[m1] - self.lo_color) * (self.vcenter - self.vlow) / denom1
        out[m2] = self.vcenter + (y[m2] - self.mid_color) * (self.vhigh - self.vcenter) / denom2
        out[m3] = self.vhigh + (y[m3] - self.hi_color) * (float(self.vmax) - self.vhigh) / denom3

        return out


def parse_lambda_from_filename(path):
    base = os.path.basename(path)
    m = re.search(r"lambda_([A-Za-z0-9eE\+\-p]+)(?:\.|$)", base)
    if not m:
        return None
    tag = m.group(1)
    try:
        return float(tag.replace("p", "."))
    except Exception:
        return None


def lambda_to_tag(lam):
    return ("%.6g" % float(lam)).replace(".", "p")


def _safe_float(s, default=float("nan")):
    try:
        return float(s)
    except Exception:
        return default


def float_to_latex(x, precision=3):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x == 0.0:
        return "0"
    s = ("%." + str(int(precision)) + "g") % x
    if "e" in s or "E" in s:
        mant, exp = re.split("[eE]", s)
        return r"%s\times 10^{%d}" % (mant, int(exp))
    return s


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

    ss2 = ss.lower().replace(" ", "")
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

    m = re.match(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*.*\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", ss)
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except Exception:
            return None
    return None


def load_scan4_csv(csv_path):
    points = []
    dm_vals = set()
    med_vals = set()

    has_dd = False
    has_id = False

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames if r.fieldnames else []

        for k in DD_KEYS:
            if (k in fields) and (("UL_" + k) in fields):
                has_dd = True
        for k in ID_KEYS:
            if (k in fields) and (("UL_" + k) in fields):
                has_id = True

        for rr in r:
            try:
                m_dm = float(rr["mDM_GeV"])
                m_med = float(rr["mMed_GeV"])
                omega = float(rr["Omega_h2"])
            except Exception:
                continue

            dm_vals.add(m_dm)
            med_vals.add(m_med)

            dd_val, dd_ul = {}, {}
            id_val, id_ul = {}, {}

            if has_dd:
                for k in DD_KEYS:
                    dd_val[k] = _safe_float(rr.get(k, -1.0), -1.0)
                    dd_ul[k] = _safe_float(rr.get("UL_" + k, -1.0), -1.0)

            if has_id:
                for k in ID_KEYS:
                    id_val[k] = _safe_float(rr.get(k, -1.0), -1.0)
                    id_ul[k] = _safe_float(rr.get("UL_" + k, -1.0), -1.0)

            points.append((m_dm, m_med, omega, dd_val, dd_ul, id_val, id_ul))

    return points, sorted(dm_vals), sorted(med_vals), has_dd, has_id


def build_matrices(points, dm_vals, med_vals, scale_factor=1.0):
    def fkey(x):
        return ("%.12g" % float(x))

    dm_to_i = {fkey(v): i for i, v in enumerate(dm_vals)}
    med_to_j = {fkey(v): j for j, v in enumerate(med_vals)}

    omega_mat = [[float("nan") for _ in med_vals] for __ in dm_vals]
    computed_mat = [[False for _ in med_vals] for __ in dm_vals]

    dd_excl_mat = {k: [[False for _ in med_vals] for __ in dm_vals] for k in DD_KEYS}
    id_excl_mat = {k: [[False for _ in med_vals] for __ in dm_vals] for k in ID_KEYS}

    for (m_dm, m_med, omega, dd_val, dd_ul, id_val, id_ul) in points:
        i = dm_to_i.get(fkey(m_dm), None)
        j = med_to_j.get(fkey(m_med), None)
        if i is None or j is None:
            continue

        computed_mat[i][j] = True

        try:
            omega_scaled = float(omega) * float(scale_factor)
        except Exception:
            omega_scaled = omega

        omega_mat[i][j] = omega_scaled

        # Excluded if UL != -1 AND UL < value (prediction exceeds UL)
        if isinstance(dd_val, dict) and isinstance(dd_ul, dict):
            for k in DD_KEYS:
                v = float(dd_val.get(k, -1.0))
                ul = float(dd_ul.get(k, -1.0))
                dd_excl_mat[k][i][j] = bool(
                    (ul != -1.0) and (v != -1.0) and np.isfinite(v) and np.isfinite(ul) and (ul < v)
                )

        if isinstance(id_val, dict) and isinstance(id_ul, dict):
            for k in ID_KEYS:
                v = float(id_val.get(k, -1.0))
                ul = float(id_ul.get(k, -1.0))
                id_excl_mat[k][i][j] = bool(
                    (ul != -1.0) and (v != -1.0) and np.isfinite(v) and np.isfinite(ul) and (ul < v)
                )

    return omega_mat, computed_mat, dd_excl_mat, id_excl_mat


def contourf_hatch_colored_with_outline(X, Y, mask, hatch, color, outline_lw=0.9):
    prev = matplotlib.rcParams.get("hatch.color", "k")
    try:
        matplotlib.rcParams["hatch.color"] = color
    except Exception:
        pass

    cf = plt.contourf(
        X, Y, mask,
        levels=[0.5, 1.5],
        colors="none",
        hatches=[hatch]
    )

    try:
        for coll in cf.collections:
            try:
                coll.set_edgecolor(color)
            except Exception:
                pass
            try:
                coll.set_linewidth(outline_lw)
            except Exception:
                pass
    except Exception:
        pass

    try:
        plt.contour(X, Y, mask, levels=[0.5], colors=[color], linewidths=outline_lw)
    except Exception:
        pass

    try:
        matplotlib.rcParams["hatch.color"] = prev
    except Exception:
        pass

    return cf


def make_omega_interpolator(points, scale_factor=1.0):
    xs, ys, zs = [], [], []
    pts_used = []

    for (mx, my, omega, dd_val, dd_ul, id_val, id_ul) in points:
        if not (my > mx):
            continue
        try:
            z = float(omega) * float(scale_factor)
        except Exception:
            z = omega
        if z == -1.0:
            continue
        if (z == z) and (not math.isinf(z)):
            xs.append(float(my))  # x=My
            ys.append(float(mx))  # y=Mx
            zs.append(float(z))
            pts_used.append((float(mx), float(my), float(z)))

    if len(xs) < 3:
        return None, None, pts_used

    tri = mtri.Triangulation(np.array(xs), np.array(ys))
    interp = mtri.LinearTriInterpolator(tri, np.array(zs))
    return tri, interp, pts_used


def build_interpolated_grid(interp, med_min, med_max, dm_min, dm_max, nx=350, ny=350):
    xgrid = np.linspace(med_min, med_max, int(nx))  # My
    ygrid = np.linspace(dm_min, dm_max, int(ny))    # Mx
    Xg, Yg = np.meshgrid(xgrid, ygrid)  # Xg=My, Yg=Mx

    Zm = interp(Xg, Yg)  # masked array
    mask_my_le_mx = (Xg <= Yg)
    if hasattr(Zm, "mask"):
        Zm = np.ma.array(Zm, mask=(Zm.mask | mask_my_le_mx))
    else:
        Zm = np.ma.array(Zm, mask=mask_my_le_mx)

    Z = Zm.filled(np.nan)
    return xgrid, ygrid, Z


def build_raw_grid_from_csv(points, dm_vals, med_vals, scale_factor=1.0):
    """
    Returns:
      xgrid_my = sorted(med_vals)
      ygrid_mx = sorted(dm_vals)
      z_grid[y_i, x_j] = Omega (scaled) if available and My>Mx, else nan
      computed_grid[y_i, x_j] = True if CSV point exists for that (Mx,My)
    """
    xgrid_my = np.array(list(med_vals), dtype=float)
    ygrid_mx = np.array(list(dm_vals), dtype=float)
    xgrid_my.sort()
    ygrid_mx.sort()

    dm_to_i = {(" %.12g" % float(v)).strip(): i for i, v in enumerate(ygrid_mx)}
    med_to_j = {(" %.12g" % float(v)).strip(): j for j, v in enumerate(xgrid_my)}

    z = np.full((len(ygrid_mx), len(xgrid_my)), np.nan, dtype=float)
    computed = np.zeros((len(ygrid_mx), len(xgrid_my)), dtype=bool)

    for (mx, my, omega, dd_val, dd_ul, id_val, id_ul) in points:
        key_i = (" %.12g" % float(mx)).strip()
        key_j = (" %.12g" % float(my)).strip()
        i = dm_to_i.get(key_i, None)
        j = med_to_j.get(key_j, None)
        if i is None or j is None:
            continue

        computed[i, j] = True

        if not (float(my) > float(mx)):
            continue

        try:
            zz = float(omega) * float(scale_factor)
        except Exception:
            zz = float(omega)

        if zz == -1.0 or (not np.isfinite(zz)):
            continue

        z[i, j] = zz

    # ensure My<=Mx masked to nan
    Xg, Yg = np.meshgrid(xgrid_my, ygrid_mx)  # Xg=My, Yg=Mx
    z[(Xg <= Yg)] = np.nan

    return xgrid_my, ygrid_mx, z, computed


def _draw_mx_eq_my_line_for_all(med_min, med_max, dm_min, dm_max, **plot_kwargs):
    # In --all (My vs Mx), the line Mx=My is y=x within overlap of ranges.
    lo = max(float(med_min), float(dm_min))
    hi = min(float(med_max), float(dm_max))
    if hi >= lo:
        plt.plot([lo, hi], [lo, hi], **plot_kwargs)


def _make_piecewise_gray_cmap():
    cmap_gray = colors.LinearSegmentedColormap.from_list(
        "relic_gray_piecewise",
        [
            (0.00, "white"),
            (0.20, "#BFBFBF"),
            (0.50, "#000000"),
            (0.80, "#BFBFBF"),
            (1.00, "white"),
        ]
    )
    cmap_gray.set_bad("white")
    return cmap_gray


def _make_simple_gray_cmap():
    # Light gray -> dark gray (avoid pure white/black so it stays readable under hatches)
    cmap = colors.LinearSegmentedColormap.from_list(
        "relic_gray_simple",
        [
            (0.00, "#D9D9D9"),
            (1.00, "#1A1A1A"),
        ]
    )
    cmap.set_bad("white")
    return cmap


def plot_heatmap(
    z_grid, xgrid_my, ygrid_mx,
    omega_mat,
    computed_mat,
    dd_excl_mat,
    id_excl_mat,
    dm_vals,
    med_vals,
    out_png,
    lambda_val,
    include_all_constraints=False,
    graph2=False,
    zoom_mx=None,
    zoom_my=None,
):
    dm_min, dm_max = float(ygrid_mx[0]), float(ygrid_mx[-1])
    med_min, med_max = float(xgrid_my[0]), float(xgrid_my[-1])

    omega_target = 0.12
    tol = 0.20
    omega_min_ok = omega_target * (1.0 - tol)
    omega_max_ok = omega_target * (1.0 + tol)

    finite = np.isfinite(z_grid)
    if np.any(finite):
        data_vmin = float(np.nanmin(z_grid))
        data_vmax = float(np.nanmax(z_grid))
    else:
        data_vmin, data_vmax = omega_min_ok, omega_max_ok

    # ---- Colormap + norm selection
    if graph2:
        # simple grayscale min->max
        vmin = data_vmin
        vmax = data_vmax if data_vmax > data_vmin else (data_vmin + 1.0)
        cmap_gray = _make_simple_gray_cmap()
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    else:
        vmin = min(data_vmin, omega_min_ok)
        vmax = max(data_vmax, omega_max_ok)
        cmap_gray = _make_piecewise_gray_cmap()
        norm = PiecewiseRelicNormalize(
            vmin=vmin,
            vlow=omega_min_ok,
            vcenter=omega_target,
            vhigh=omega_max_ok,
            vmax=vmax,
            lo_color=0.20,
            mid_color=0.50,
            hi_color=0.80
        )

    plt.figure(figsize=(10.0, 7.0))

    im = plt.imshow(
        z_grid,
        origin="lower",
        extent=[med_min, med_max, dm_min, dm_max],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_gray,
        norm=norm
    )

    cb = plt.colorbar(im)
    cb.set_label(r"$\Omega h^2$")

    if graph2:
        cb.set_ticks([vmin, 0.5 * (vmin + vmax), vmax])
        cb.set_ticklabels([
            "{:.3g}".format(vmin),
            "{:.3g}".format(0.5 * (vmin + vmax)),
            "{:.3g}".format(vmax),
        ])
    else:
        cb.set_ticks([vmin, omega_min_ok, omega_target, omega_max_ok, vmax])
        cb.set_ticklabels([
            "{:.3g}".format(vmin),
            "{:.3g}".format(omega_min_ok),
            "{:.3g}".format(omega_target),
            "{:.3g}".format(omega_max_ok),
            "{:.3g}".format(vmax),
        ])

    plt.xlabel(r"$M_y\,[\mathrm{GeV}]$")
    plt.ylabel(r"$M_x\,[\mathrm{GeV}]$")
    plt.title(r"Scan4: $\Omega h^2$ map ($\lambda=%s$)" % float_to_latex(lambda_val))

    _draw_mx_eq_my_line_for_all(med_min, med_max, dm_min, dm_max, color="k")

    if zoom_my is not None:
        plt.xlim(zoom_my[0], zoom_my[1])
    if zoom_mx is not None:
        plt.ylim(zoom_mx[0], zoom_mx[1])

    # Hatching grid from CSV (no interpolation for constraints)
    X = [list(med_vals) for _ in dm_vals]
    Y = [[m_dm for _ in med_vals] for m_dm in dm_vals]

    if include_all_constraints:
        dd_si_color = "violet"
        dd_sd_color = "blue"
        id_bbx_color = "green"
        id_tot_color = "red"

        dd_si_hatch = "\\\\"
        dd_sd_hatch = "||||"
        id_bbx_hatch = "////"
        id_tot_hatch = "----"

        mask_dd_si = []
        mask_dd_sd = []
        mask_id_bbx = []
        mask_id_tot = []

        for i in range(len(dm_vals)):
            r_si, r_sd, r_bbx, r_tot = [], [], [], []
            for j in range(len(med_vals)):
                if not computed_mat[i][j]:
                    r_si.append(0); r_sd.append(0); r_bbx.append(0); r_tot.append(0)
                    continue

                si_excl = bool(dd_excl_mat["SigmaN_SI_p"][i][j] or dd_excl_mat["SigmaN_SI_n"][i][j])
                sd_excl = bool(dd_excl_mat["SigmaN_SD_p"][i][j] or dd_excl_mat["SigmaN_SD_n"][i][j])
                bbx_excl = bool(id_excl_mat["xsxs_bbx"][i][j])
                tot_excl = bool(id_excl_mat["Total_xsec"][i][j])

                r_si.append(1 if si_excl else 0)
                r_sd.append(1 if sd_excl else 0)
                r_bbx.append(1 if bbx_excl else 0)
                r_tot.append(1 if tot_excl else 0)

            mask_dd_si.append(r_si)
            mask_dd_sd.append(r_sd)
            mask_id_bbx.append(r_bbx)
            mask_id_tot.append(r_tot)

        contourf_hatch_colored_with_outline(X, Y, mask_dd_si, dd_si_hatch, dd_si_color, outline_lw=0.9)
        contourf_hatch_colored_with_outline(X, Y, mask_dd_sd, dd_sd_hatch, dd_sd_color, outline_lw=0.9)
        contourf_hatch_colored_with_outline(X, Y, mask_id_bbx, id_bbx_hatch, id_bbx_color, outline_lw=0.9)
        contourf_hatch_colored_with_outline(X, Y, mask_id_tot, id_tot_hatch, id_tot_color, outline_lw=0.9)

        legend_handles = [
            Patch(facecolor="none", edgecolor=dd_si_color, hatch=dd_si_hatch, label="DD SI"),
            Patch(facecolor="none", edgecolor=dd_sd_color, hatch=dd_sd_hatch, label="DD SD"),
            Patch(facecolor="none", edgecolor=id_bbx_color, hatch=id_bbx_hatch, label="ID xsxs_bbx"),
            Patch(facecolor="none", edgecolor=id_tot_color, hatch=id_tot_hatch, label="ID Total_xsec"),
        ]

        plt.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            framealpha=0.9,
            borderaxespad=0.0
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# -----------------------------
# --all2 (log-log) plotting
#   x = Mx
#   y = (My/Mx - 1)
# -----------------------------
def build_interpolated_grid_all2(interp, med_min, med_max, dm_min, dm_max, nx=350, ny=350):
    """
    Build a regular grid in:
      x = Mx (log-spaced)
      y = r = (My/Mx - 1) (log-spaced, r>0)
    Then sample omega via My = Mx*(1+r), using interp(My, Mx).
    """
    xmin = max(float(dm_min), 1e-12)
    xmax = max(float(dm_max), xmin * 1.0001)

    rmax = (float(med_max) / float(xmin)) - 1.0
    rmax = max(rmax, 1e-6)

    rmin = max(1e-6, min(1e-3, 0.5 * rmax))

    xgrid = np.logspace(np.log10(xmin), np.log10(xmax), int(nx))
    ygrid = np.logspace(np.log10(rmin), np.log10(rmax), int(ny))  # r

    Xg, Rg = np.meshgrid(xgrid, ygrid)  # Xg=Mx, Rg=r
    Myg = Xg * (1.0 + Rg)

    valid = (Myg > Xg) & (Myg >= float(med_min)) & (Myg <= float(med_max)) & (Xg >= float(dm_min)) & (Xg <= float(dm_max))

    Zm = interp(Myg, Xg)
    if hasattr(Zm, "mask"):
        Zm = np.ma.array(Zm, mask=(Zm.mask | (~valid)))
    else:
        Zm = np.ma.array(Zm, mask=(~valid))

    Z = Zm.filled(np.nan)
    return xgrid, ygrid, Z


def build_raw_grid_all2_from_csv(points, dm_vals, med_vals, scale_factor=1.0, nx=350, ny=350):
    """
    Raw/no-interp version for --all2:
    Build a regular log-log grid in (Mx, r) and fill with nearest-neighbor from CSV points
    after transforming each CSV point to (Mx, r=My/Mx-1). Only uses points with My>Mx.
    """
    dm_min, dm_max = float(min(dm_vals)), float(max(dm_vals))
    med_min, med_max = float(min(med_vals)), float(max(med_vals))

    xmin = max(dm_min, 1e-12)
    xmax = max(dm_max, xmin * 1.0001)

    rmax = (med_max / xmin) - 1.0
    rmax = max(rmax, 1e-6)
    rmin = max(1e-6, min(1e-3, 0.5 * rmax))

    xgrid = np.logspace(np.log10(xmin), np.log10(xmax), int(nx))
    ygrid = np.logspace(np.log10(rmin), np.log10(rmax), int(ny))  # r

    # collect transformed CSV points
    xs = []
    ys = []
    zs = []
    for (mx, my, omega, dd_val, dd_ul, id_val, id_ul) in points:
        mx = float(mx); my = float(my)
        if not (my > mx and mx > 0.0):
            continue
        try:
            zz = float(omega) * float(scale_factor)
        except Exception:
            zz = float(omega)
        if zz == -1.0 or (not np.isfinite(zz)):
            continue
        r = my / mx - 1.0
        if r <= 0.0:
            continue
        xs.append(mx)
        ys.append(r)
        zs.append(zz)

    if len(xs) == 0:
        return xgrid, ygrid, np.full((len(ygrid), len(xgrid)), np.nan, dtype=float)

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    zs = np.array(zs, dtype=float)

    # nearest neighbor on log-log (distance in log space)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    lx = np.log(xs)
    ly = np.log(ys)

    Zout = np.full_like(Xg, np.nan, dtype=float)

    # vectorized nearest neighbor: compute distances to all points (can be heavy but typically small CSV)
    # If huge scans, consider KDTree; but keep dependencies minimal.
    lX = np.log(Xg.ravel())
    lY = np.log(Yg.ravel())

    # chunk to avoid huge memory
    chunk = 50000
    for a in range(0, lX.size, chunk):
        b = min(lX.size, a + chunk)
        dx = lX[a:b][:, None] - lx[None, :]
        dy = lY[a:b][:, None] - ly[None, :]
        d2 = dx * dx + dy * dy
        nn = np.argmin(d2, axis=1)
        Zout.ravel()[a:b] = zs[nn]

    # mask invalid region by geometry bounds on My
    Myg = Xg * (1.0 + Yg)
    valid = (Myg > Xg) & (Myg >= med_min) & (Myg <= med_max) & (Xg >= dm_min) & (Xg <= dm_max)
    Zout[~valid] = np.nan

    return xgrid, ygrid, Zout


def plot_heatmap_all2(
    z_grid, xgrid_mx, ygrid_r,
    omega_mat,
    computed_mat,
    dd_excl_mat,
    id_excl_mat,
    dm_vals,
    med_vals,
    out_png,
    lambda_val,
    include_all_constraints=False,
    graph2=False,
    zoom_mx=None,
    zoom_my=None,
):
    finite = np.isfinite(z_grid)
    if np.any(finite):
        data_vmin = float(np.nanmin(z_grid))
        data_vmax = float(np.nanmax(z_grid))
    else:
        data_vmin, data_vmax = 0.0, 1.0

    omega_target = 0.12
    tol = 0.20
    omega_min_ok = omega_target * (1.0 - tol)
    omega_max_ok = omega_target * (1.0 + tol)

    if graph2:
        vmin = data_vmin
        vmax = data_vmax if data_vmax > data_vmin else (data_vmin + 1.0)
        cmap_gray = _make_simple_gray_cmap()
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    else:
        vmin = min(data_vmin, omega_min_ok)
        vmax = max(data_vmax, omega_max_ok)
        cmap_gray = _make_piecewise_gray_cmap()
        norm = PiecewiseRelicNormalize(
            vmin=vmin,
            vlow=omega_min_ok,
            vcenter=omega_target,
            vhigh=omega_max_ok,
            vmax=vmax,
            lo_color=0.20,
            mid_color=0.50,
            hi_color=0.80
        )

    plt.figure(figsize=(10.0, 7.0))

    X, Y = np.meshgrid(xgrid_mx, ygrid_r)

    pcm = plt.pcolormesh(
        X, Y, z_grid,
        cmap=cmap_gray,
        norm=norm,
        shading="auto"
    )

    cb = plt.colorbar(pcm)
    cb.set_label(r"$\Omega h^2$")

    if graph2:
        cb.set_ticks([vmin, 0.5 * (vmin + vmax), vmax])
        cb.set_ticklabels([
            "{:.3g}".format(vmin),
            "{:.3g}".format(0.5 * (vmin + vmax)),
            "{:.3g}".format(vmax),
        ])
    else:
        cb.set_ticks([vmin, omega_min_ok, omega_target, omega_max_ok, vmax])
        cb.set_ticklabels([
            "{:.3g}".format(vmin),
            "{:.3g}".format(omega_min_ok),
            "{:.3g}".format(omega_target),
            "{:.3g}".format(omega_max_ok),
            "{:.3g}".format(vmax),
        ])

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$M_x\,[\mathrm{GeV}]$")
    plt.ylabel(r"$\left(\frac{M_y}{M_x}-1\right)$")
    plt.title(r"Scan4: $\Omega h^2$ map ($\lambda=%s$)" % float_to_latex(lambda_val))

    # "Mx=My" corresponds to r=0; in log scale we draw it at the lower y-bound (plot boundary)
    ymin = float(np.nanmin(ygrid_r))
    xmin = float(np.nanmin(xgrid_mx))
    xmax = float(np.nanmax(xgrid_mx))
    plt.plot([xmin, xmax], [ymin, ymin], color="k")

    if zoom_mx is not None:
        plt.xlim(zoom_mx[0], zoom_mx[1])
    if zoom_my is not None:
        plt.ylim(zoom_my[0], zoom_my[1])

    # Hatching overlay: transform CSV grid points (My, Mx) -> (x=Mx, y=r)
    Xtrans = []
    Ytrans = []
    for i, mx in enumerate(dm_vals):
        rowx = []
        rowy = []
        for j, my in enumerate(med_vals):
            rowx.append(float(mx))
            if my > mx and mx > 0.0:
                rowy.append(float(my) / float(mx) - 1.0)
            else:
                rowy.append(float("nan"))
        Xtrans.append(rowx)
        Ytrans.append(rowy)

    if include_all_constraints:
        dd_si_color = "violet"
        dd_sd_color = "blue"
        id_bbx_color = "green"
        id_tot_color = "red"

        dd_si_hatch = "\\\\"
        dd_sd_hatch = "||||"
        id_bbx_hatch = "////"
        id_tot_hatch = "----"

        mask_dd_si = []
        mask_dd_sd = []
        mask_id_bbx = []
        mask_id_tot = []

        for i in range(len(dm_vals)):
            r_si, r_sd, r_bbx, r_tot = [], [], [], []
            for j in range(len(med_vals)):
                if not computed_mat[i][j]:
                    r_si.append(0); r_sd.append(0); r_bbx.append(0); r_tot.append(0)
                    continue

                si_excl = bool(dd_excl_mat["SigmaN_SI_p"][i][j] or dd_excl_mat["SigmaN_SI_n"][i][j])
                sd_excl = bool(dd_excl_mat["SigmaN_SD_p"][i][j] or dd_excl_mat["SigmaN_SD_n"][i][j])
                bbx_excl = bool(id_excl_mat["xsxs_bbx"][i][j])
                tot_excl = bool(id_excl_mat["Total_xsec"][i][j])

                r_si.append(1 if si_excl else 0)
                r_sd.append(1 if sd_excl else 0)
                r_bbx.append(1 if bbx_excl else 0)
                r_tot.append(1 if tot_excl else 0)

            mask_dd_si.append(r_si)
            mask_dd_sd.append(r_sd)
            mask_id_bbx.append(r_bbx)
            mask_id_tot.append(r_tot)

        contourf_hatch_colored_with_outline(Xtrans, Ytrans, mask_dd_si, dd_si_hatch, dd_si_color, outline_lw=0.9)
        contourf_hatch_colored_with_outline(Xtrans, Ytrans, mask_dd_sd, dd_sd_hatch, dd_sd_color, outline_lw=0.9)
        contourf_hatch_colored_with_outline(Xtrans, Ytrans, mask_id_bbx, id_bbx_hatch, id_bbx_color, outline_lw=0.9)
        contourf_hatch_colored_with_outline(Xtrans, Ytrans, mask_id_tot, id_tot_hatch, id_tot_color, outline_lw=0.9)

        legend_handles = [
            Patch(facecolor="none", edgecolor=dd_si_color, hatch=dd_si_hatch, label="DD SI"),
            Patch(facecolor="none", edgecolor=dd_sd_color, hatch=dd_sd_hatch, label="DD SD"),
            Patch(facecolor="none", edgecolor=id_bbx_color, hatch=id_bbx_hatch, label="ID xsxs_bbx"),
            Patch(facecolor="none", edgecolor=id_tot_color, hatch=id_tot_hatch, label="ID Total_xsec"),
        ]

        plt.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            framealpha=0.9,
            borderaxespad=0.0
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# -------------------------------
# Iso-omega (unchanged)
# -------------------------------
def _extract_contour_vertices(contour_set):
    level_to_paths = {}
    levels = list(getattr(contour_set, "levels", []))
    colls = list(getattr(contour_set, "collections", []))
    for lvl, coll in zip(levels, colls):
        paths = []
        for p in coll.get_paths():
            v = p.vertices
            if v is not None and len(v) >= 2:
                paths.append(v.copy())
        level_to_paths[float(lvl)] = paths
    return level_to_paths


def _point_to_polyline_min_dist(p, poly):
    if poly is None or len(poly) == 0:
        return float("inf")
    d2 = (poly[:, 0] - p[0])**2 + (poly[:, 1] - p[1])**2
    return float(np.sqrt(np.min(d2)))


def _choose_label_point_for_level(level, paths_for_level, all_other_paths, xlim, ylim):
    if not paths_for_level:
        return None

    xmin, xmax = xlim
    ymin, ymax = ylim
    mx = 0.03 * (xmax - xmin) if xmax > xmin else 0.0
    my = 0.03 * (ymax - ymin) if ymax > ymin else 0.0

    best_p = None
    best_score = -1.0

    for poly in paths_for_level:
        n = len(poly)
        if n < 5:
            cand_idx = range(n)
        else:
            step = max(1, int(n / 30))
            cand_idx = range(0, n, step)

        for k in cand_idx:
            p = poly[k, :]
            if not (xmin + mx <= p[0] <= xmax - mx and ymin + my <= p[1] <= ymax - my):
                continue

            mind = float("inf")
            for op in all_other_paths:
                d = _point_to_polyline_min_dist(p, op)
                if d < mind:
                    mind = d
                    if mind < best_score:
                        break

            if mind > best_score:
                best_score = mind
                best_p = (float(p[0]), float(p[1]))

    return best_p


def plot_iso_omega(
    z_grid, xgrid_my, ygrid_mx,
    out_png,
    lambda_val,
    iso_levels,
    zoom_mx=None,
    zoom_my=None,
):
    dm_min, dm_max = float(ygrid_mx[0]), float(ygrid_mx[-1])
    med_min, med_max = float(xgrid_my[0]), float(xgrid_my[-1])

    Xg, Yg = np.meshgrid(xgrid_my, ygrid_mx)

    plt.figure(figsize=(9.5, 7.0))

    Zm = np.ma.array(z_grid, mask=~np.isfinite(z_grid))

    cs = plt.contour(
        Xg, Yg, Zm,
        levels=iso_levels
    )

    _draw_mx_eq_my_line_for_all(med_min, med_max, dm_min, dm_max, color="k")

    plt.xlabel(r"$M_y\,[\mathrm{GeV}]$")
    plt.ylabel(r"$M_x\,[\mathrm{GeV}]$")
    plt.title(r"Scan4: Iso-$\Omega h^2$ ($\lambda=%s$)" % float_to_latex(lambda_val))

    if zoom_my is not None:
        plt.xlim(zoom_my[0], zoom_my[1])
    if zoom_mx is not None:
        plt.ylim(zoom_mx[0], zoom_mx[1])

    level_paths = _extract_contour_vertices(cs)

    manual_positions = []
    ordered_levels = list(cs.levels)
    for lvl in ordered_levels:
        plist = level_paths.get(float(lvl), [])
        others = []
        for olvl, oplist in level_paths.items():
            if float(olvl) == float(lvl):
                continue
            for poly in oplist:
                others.append(poly)

        pos = _choose_label_point_for_level(
            level=float(lvl),
            paths_for_level=plist,
            all_other_paths=others,
            xlim=(med_min, med_max),
            ylim=(dm_min, dm_max),
        )
        if pos is not None:
            manual_positions.append(pos)

    try:
        labels = plt.clabel(
            cs,
            inline=True,
            inline_spacing=8,
            fontsize=14,
            fmt="%.3g",
            colors="k",
            manual=manual_positions if len(manual_positions) > 0 else None
        )
        for t in labels:
            try:
                t.set_fontweight("bold")
            except Exception:
                pass
            try:
                t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.25))
            except Exception:
                pass
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def parse_list_of_floats(s):
    if s is None:
        return []
    s = s.strip()
    if s == "":
        return []
    parts = re.split(r"[,\s]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if p == "":
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvfile", help="Scan4 CSV file (e.g. scan4_omega_map_lambda_6p5.csv)")

    ap.add_argument("--all", action="store_true",
                    help="Plot DD+ID hatches (if present). RD is shown by background.")
    ap.add_argument("--all2", action="store_true",
                    help="Same as --all but log-log with x=Mx and y=(My/Mx - 1).")

    ap.add_argument("--graph2", action="store_true",
                    help="Use simple grayscale (min->light gray, max->dark gray) instead of 0.12-centered mapping.")

    ap.add_argument("--nointerpolation", action="store_true",
                    help="Disable 2D interpolation: use only raw CSV values for the background.")

    ap.add_argument("--lambdas", default=None,
                    help="Comma-separated lambdas to rescale Omega (RD-only). Example: --lambdas=0.5,1,2")

    ap.add_argument("--iso_omega", default=None,
                    help="Comma-separated Omega levels to plot iso-curves ONLY (independent mode). Example: --iso_omega=0.08,0.12,0.2")

    ap.add_argument("--interp-n", type=int, default=350,
                    help="Interpolation grid resolution (nx=ny). Default 350.")

    ap.add_argument("--zoom-mx", default=None,
                    help="Mx range like '100to1000' (or '100,1000', '100:1000', '100..1000').")
    ap.add_argument("--zoom-my", default=None,
                    help="My range like '100to1000' (or '100,1000', '100:1000', '100..1000').")

    args = ap.parse_args()

    zoom_mx = parse_range_to(args.zoom_mx)
    zoom_my = parse_range_to(args.zoom_my)

    csv_path = os.path.abspath(os.path.expanduser(args.csvfile))
    if not os.path.isfile(csv_path):
        raise IOError("CSV not found: {0}".format(csv_path))

    points, dm_vals, med_vals, has_dd, has_id = load_scan4_csv(csv_path)

    lam0 = parse_lambda_from_filename(csv_path)
    if lam0 is None:
        lam0 = 1.0

    outdir = os.path.dirname(csv_path)

    iso_levels = parse_list_of_floats(args.iso_omega) if args.iso_omega is not None else []
    if args.iso_omega is not None:
        if len(iso_levels) == 0:
            raise RuntimeError("No valid values provided in --iso_omega")

        lambdas_new = parse_list_of_floats(args.lambdas) if args.lambdas is not None else [lam0]

        for lam in lambdas_new:
            if lam <= 0.0:
                continue
            scale = (lam0 / lam) ** 4

            tri, interp, pts_used = make_omega_interpolator(points, scale_factor=scale)
            if interp is None:
                raise RuntimeError("Not enough points to build 2D interpolator (need at least 3).")

            dm_min, dm_max = dm_vals[0], dm_vals[-1]
            med_min, med_max = med_vals[0], med_vals[-1]

            xgrid, ygrid, z_grid = build_interpolated_grid(
                interp, med_min, med_max, dm_min, dm_max,
                nx=args.interp_n, ny=args.interp_n
            )

            out_png = os.path.join(outdir, "scan4_iso_omega_lambda_{0}.png".format(lambda_to_tag(lam)))
            plot_iso_omega(
                z_grid=z_grid,
                xgrid_my=xgrid,
                ygrid_mx=ygrid,
                out_png=out_png,
                lambda_val=lam,
                iso_levels=iso_levels,
                zoom_mx=zoom_mx,
                zoom_my=zoom_my
            )
            print("Saved:", out_png)

        return

    # Heatmap / rescale modes
    if args.lambdas is None:
        omega_mat, computed_mat, dd_excl_mat, id_excl_mat = build_matrices(
            points, dm_vals, med_vals, scale_factor=1.0
        )

        include_all = (bool(args.all) or bool(args.all2)) and (bool(has_dd) or bool(has_id))

        dm_min, dm_max = dm_vals[0], dm_vals[-1]
        med_min, med_max = med_vals[0], med_vals[-1]

        # Background field (interp or raw)
        did_interpolate = False

        if args.nointerpolation:
            xgrid_raw, ygrid_raw, z_raw, computed_raw = build_raw_grid_from_csv(
                points, dm_vals, med_vals, scale_factor=1.0
            )
            if args.all2:
                # for all2, build a regular (Mx,r) grid filled from CSV (nearest in log space)
                xgrid_mx, ygrid_r, z_grid2 = build_raw_grid_all2_from_csv(
                    points, dm_vals, med_vals, scale_factor=1.0,
                    nx=args.interp_n, ny=args.interp_n
                )
                out_png = os.path.join(outdir, "scan4_omega_map_lambda_{0}_all2.png".format(lambda_to_tag(lam0)))

                plot_heatmap_all2(
                    z_grid=z_grid2,
                    xgrid_mx=xgrid_mx,
                    ygrid_r=ygrid_r,
                    omega_mat=omega_mat,
                    computed_mat=computed_mat,
                    dd_excl_mat=dd_excl_mat,
                    id_excl_mat=id_excl_mat,
                    dm_vals=dm_vals,
                    med_vals=med_vals,
                    out_png=out_png,
                    lambda_val=lam0,
                    include_all_constraints=include_all,
                    graph2=bool(args.graph2),
                    zoom_mx=zoom_mx,
                    zoom_my=zoom_my,
                )
                print("Saved:", out_png)
            else:
                out_png = os.path.join(outdir, "scan4_omega_map_lambda_{0}.png".format(lambda_to_tag(lam0)))

                plot_heatmap(
                    z_grid=z_raw,
                    xgrid_my=xgrid_raw,
                    ygrid_mx=ygrid_raw,
                    omega_mat=omega_mat,
                    computed_mat=computed_mat,
                    dd_excl_mat=dd_excl_mat,
                    id_excl_mat=id_excl_mat,
                    dm_vals=dm_vals,
                    med_vals=med_vals,
                    out_png=out_png,
                    lambda_val=lam0,
                    include_all_constraints=include_all,
                    graph2=bool(args.graph2),
                    zoom_mx=zoom_mx,
                    zoom_my=zoom_my,
                )
                print("Saved:", out_png)
        else:
            tri, interp, pts_used = make_omega_interpolator(points, scale_factor=1.0)
            if interp is None:
                raise RuntimeError("Not enough points to build 2D interpolator (need at least 3).")
            did_interpolate = True

            if args.all2:
                xgrid_mx, ygrid_r, z_grid2 = build_interpolated_grid_all2(
                    interp, med_min, med_max, dm_min, dm_max,
                    nx=args.interp_n, ny=args.interp_n
                )
                out_png = os.path.join(outdir, "scan4_omega_map_lambda_{0}_all2.png".format(lambda_to_tag(lam0)))

                plot_heatmap_all2(
                    z_grid=z_grid2,
                    xgrid_mx=xgrid_mx,
                    ygrid_r=ygrid_r,
                    omega_mat=omega_mat,
                    computed_mat=computed_mat,
                    dd_excl_mat=dd_excl_mat,
                    id_excl_mat=id_excl_mat,
                    dm_vals=dm_vals,
                    med_vals=med_vals,
                    out_png=out_png,
                    lambda_val=lam0,
                    include_all_constraints=include_all,
                    graph2=bool(args.graph2),
                    zoom_mx=zoom_mx,
                    zoom_my=zoom_my,
                )
                print("Saved:", out_png)
            else:
                xgrid, ygrid, z_grid = build_interpolated_grid(
                    interp, med_min, med_max, dm_min, dm_max,
                    nx=args.interp_n, ny=args.interp_n
                )
                out_png = os.path.join(outdir, "scan4_omega_map_lambda_{0}.png".format(lambda_to_tag(lam0)))

                plot_heatmap(
                    z_grid=z_grid,
                    xgrid_my=xgrid,
                    ygrid_mx=ygrid,
                    omega_mat=omega_mat,
                    computed_mat=computed_mat,
                    dd_excl_mat=dd_excl_mat,
                    id_excl_mat=id_excl_mat,
                    dm_vals=dm_vals,
                    med_vals=med_vals,
                    out_png=out_png,
                    lambda_val=lam0,
                    include_all_constraints=include_all,
                    graph2=bool(args.graph2),
                    zoom_mx=zoom_mx,
                    zoom_my=zoom_my,
                )
                print("Saved:", out_png)

        # Print interpolation check only if we actually interpolated
        if did_interpolate:
            finite_pts = []
            for (mx, my, omega, dd_val, dd_ul, id_val, id_ul) in points:
                if not (my > mx):
                    continue
                if omega == -1.0:
                    continue
                if (omega == omega) and (not math.isinf(omega)):
                    finite_pts.append((mx, my, omega))

            sample = random.sample(finite_pts, min(3, len(finite_pts))) if len(finite_pts) > 0 else []
            print("\n[Interpolation check] 3 points aléatoires (CSV vs interpolé):")
            for (mx, my, omega_csv) in sample:
                z_int = interp(np.array([my], dtype=float), np.array([mx], dtype=float))
                try:
                    z_int_val = float(z_int.filled(np.nan)[0])
                except Exception:
                    try:
                        z_int_val = float(z_int[0])
                    except Exception:
                        z_int_val = float("nan")

                if (omega_csv != 0.0) and np.isfinite(omega_csv) and np.isfinite(z_int_val):
                    err_pct = abs(z_int_val - omega_csv) / abs(omega_csv) * 100.0
                else:
                    err_pct = float("nan")

                print("  (Mx={:.6g}, My={:.6g})  CSV={:.6g}  interp={:.6g}  err={:.3g}%".format(
                    mx, my, omega_csv, z_int_val, err_pct
                ))

        return

    # Rescale for lambdas (RD-only)
    lambdas_new = parse_list_of_floats(args.lambdas)
    if len(lambdas_new) == 0:
        raise RuntimeError("No valid lambdas provided in --lambdas")

    for lam in lambdas_new:
        if lam <= 0.0:
            continue

        scale = (lam0 / lam) ** 4

        omega_mat, computed_mat, dd_excl_mat, id_excl_mat = build_matrices(
            points, dm_vals, med_vals, scale_factor=scale
        )

        dm_min, dm_max = dm_vals[0], dm_vals[-1]
        med_min, med_max = med_vals[0], med_vals[-1]

        if args.nointerpolation:
            xgrid_raw, ygrid_raw, z_raw, computed_raw = build_raw_grid_from_csv(
                points, dm_vals, med_vals, scale_factor=scale
            )
            out_png = os.path.join(outdir, "scan4_omega_map_lambda_{0}.png".format(lambda_to_tag(lam)))
            plot_heatmap(
                z_grid=z_raw,
                xgrid_my=xgrid_raw,
                ygrid_mx=ygrid_raw,
                omega_mat=omega_mat,
                computed_mat=computed_mat,
                dd_excl_mat=dd_excl_mat,
                id_excl_mat=id_excl_mat,
                dm_vals=dm_vals,
                med_vals=med_vals,
                out_png=out_png,
                lambda_val=lam,
                include_all_constraints=False,
                graph2=bool(args.graph2),
                zoom_mx=zoom_mx,
                zoom_my=zoom_my,
            )
            print("Saved:", out_png)
        else:
            tri, interp, pts_used = make_omega_interpolator(points, scale_factor=scale)
            if interp is None:
                raise RuntimeError("Not enough points to build 2D interpolator (need at least 3).")

            xgrid, ygrid, z_grid = build_interpolated_grid(
                interp, med_min, med_max, dm_min, dm_max,
                nx=args.interp_n, ny=args.interp_n
            )

            out_png = os.path.join(outdir, "scan4_omega_map_lambda_{0}.png".format(lambda_to_tag(lam)))

            plot_heatmap(
                z_grid=z_grid,
                xgrid_my=xgrid,
                ygrid_mx=ygrid,
                omega_mat=omega_mat,
                computed_mat=computed_mat,
                dd_excl_mat=dd_excl_mat,
                id_excl_mat=id_excl_mat,
                dm_vals=dm_vals,
                med_vals=med_vals,
                out_png=out_png,
                lambda_val=lam,
                include_all_constraints=False,
                graph2=bool(args.graph2),
                zoom_mx=zoom_mx,
                zoom_my=zoom_my,
            )
            print("Saved:", out_png)

    return


if __name__ == "__main__":
    main()
