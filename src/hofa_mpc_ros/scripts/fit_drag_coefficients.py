#!/usr/bin/env python3
"""Fit an effective surge drag model from constant-force experiments.

The fitted model is the same scalar form used by hofa_mpc_ros::ThreeDOFModel:

    F_drag_x = d_l_eff * u + d_q_eff * abs(u) * u

For each trial the residual force is estimated as

    F_drag_x = F_x_actual - m_eff * du/dt

where ``F_x_actual`` is a body-frame generalized force.  In a steady-state
window the acceleration term is small, but retaining it makes the script useful
for short transients as well.  At least three distinct force levels are
recommended; one force level cannot identify both coefficients reliably.

Examples:

  # Current ROS text export format, using commanded constant forces.
  python3 fit_drag_coefficients.py \
      --odom force_4N_odom.txt force_6N_odom.txt force_8N_odom.txt force_10N_odom.txt \
      --force-x 4 6 8 10 --output drag_fit.json

  # CSV exports from the experiment tools (columns: stamp,u; force can be
  # supplied per file as above).
  python3 fit_drag_coefficients.py \
      --odom force_4N.csv force_6N.csv force_8N.csv \
      --force-x 4 6 8 --mass-effective 7.94
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np


def _number(value):
    return float(value)


def _field(row, names, default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return _number(row[name])
    return default


def load_csv(filename):
    """Load a CSV containing stamp/time and body surge velocity ``u``."""
    with open(filename, "r") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CSV is empty: %s" % filename)
    output = []
    for row in rows:
        stamp = _field(row, ("stamp", "time", "t", "t_rel"))
        u = _field(row, ("u", "linear_x", "twist_linear_x"))
        if stamp is None or u is None:
            continue
        output.append((stamp, u))
    if len(output) < 3:
        raise ValueError("CSV needs stamp/time and u columns: %s" % filename)
    return np.asarray(output, dtype=float)


def load_ros_echo_text(filename):
    """Load the ``rostopic echo -b`` YAML-like text used in this project."""
    text = open(filename, "r").read()
    blocks = re.split(r"\n---\s*\n", text)
    output = []
    for block in blocks:
        stamp = re.search(r"secs:\s*(\d+)\s+nsecs:\s*(\d+)", block)
        linear = re.search(
            r"twist:\s*\n\s*twist:\s*\n\s*linear:\s*\n"
            r"\s*x:\s*([-+0-9.eE]+)", block)
        if stamp is None or linear is None:
            continue
        t = int(stamp.group(1)) + int(stamp.group(2)) * 1e-9
        output.append((t, float(linear.group(1))))
    if len(output) < 3:
        raise ValueError("Could not parse ROS odometry text: %s" % filename)
    return np.asarray(output, dtype=float)


def load_odom(filename):
    if filename.lower().endswith(".csv"):
        return load_csv(filename)
    # Text exports have no reliable extension requirement, so try CSV first
    # and fall back to the ROS echo parser.
    try:
        return load_csv(filename)
    except Exception:
        return load_ros_echo_text(filename)


def moving_average(values, width):
    width = int(max(1, width))
    if width <= 1 or len(values) < width:
        return values.copy()
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=float) / float(width)
    padded = np.pad(values, (width // 2,), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def select_window(data, steady_window, steady_start, steady_end):
    t = data[:, 0] - data[0, 0]
    if steady_start is None:
        start = max(0.0, t[-1] - steady_window)
    else:
        start = max(0.0, float(steady_start))
    end = t[-1] if steady_end is None else min(t[-1], float(steady_end))
    mask = (t >= start) & (t <= end)
    if int(mask.sum()) < 3:
        raise ValueError("steady window has too few samples")
    return t[mask], data[mask, 1]


def fit_drag(trials, nonnegative=True):
    """Fit coefficients from one representative point per trial."""
    u = np.asarray([trial["u_mean"] for trial in trials], dtype=float)
    force = np.asarray([trial["drag_force_mean"] for trial in trials], dtype=float)
    if len(trials) < 2 or np.ptp(u) < 1e-5:
        raise ValueError(
            "At least two distinct steady speeds are required to fit d_l and d_q")
    design = np.column_stack((u, np.abs(u) * u))
    if nonnegative:
        try:
            from scipy.optimize import nnls
            coeff, residual_norm = nnls(design, force)
        except ImportError:
            coeff = np.linalg.lstsq(design, force, rcond=None)[0]
            residual_norm = float(np.linalg.norm(design @ coeff - force))
    else:
        coeff = np.linalg.lstsq(design, force, rcond=None)[0]
        residual_norm = float(np.linalg.norm(design @ coeff - force))

    prediction = design @ coeff
    residual = force - prediction
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((force - force.mean()) ** 2))
    r2 = None if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "drag_linear_eff": float(coeff[0]),
        "drag_quadratic_eff": float(coeff[1]),
        "fit_rmse_N": float(np.sqrt(np.mean(residual * residual))),
        "fit_r2": r2,
        "nnls_residual_norm": float(residual_norm),
        "trial_predictions_N": [float(x) for x in prediction],
        "trial_residuals_N": [float(x) for x in residual],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odom", nargs="+", required=True,
                        help="one odometry text/CSV per force level")
    parser.add_argument("--force-x", nargs="+", type=float, required=True,
                        help="actual/commanded body Fx for each odometry file")
    parser.add_argument("--mass-effective", type=float, default=7.94,
                        help="effective surge mass kg (rigid + added mass estimate)")
    parser.add_argument("--steady-window", type=float, default=5.0,
                        help="seconds at the end of each trial used for fitting")
    parser.add_argument("--steady-start", type=float, default=None,
                        help="optional relative start time instead of final window")
    parser.add_argument("--steady-end", type=float, default=None,
                        help="optional relative end time")
    parser.add_argument("--smooth-window", type=int, default=11,
                        help="odd sample count for u smoothing before du/dt")
    parser.add_argument("--allow-negative", action="store_true",
                        help="allow unconstrained negative drag coefficients")
    parser.add_argument("--output", default=None,
                        help="JSON output path; default prints only")
    args = parser.parse_args(argv)

    if len(args.odom) != len(args.force_x):
        parser.error("--odom and --force-x must have the same number of values")
    if args.mass_effective <= 0.0:
        parser.error("--mass-effective must be positive")

    trials = []
    for filename, force_x in zip(args.odom, args.force_x):
        data = load_odom(filename)
        t, u = select_window(data, args.steady_window,
                             args.steady_start, args.steady_end)
        u_smooth = moving_average(u, args.smooth_window)
        du_dt = np.gradient(u_smooth, t)
        drag = float(force_x) - args.mass_effective * du_dt
        trial = {
            "file": os.path.abspath(filename),
            "force_x_N": float(force_x),
            "window_start_s": float(t[0]),
            "window_end_s": float(t[-1]),
            "samples": int(len(t)),
            "u_mean_mps": float(np.mean(u_smooth)),
            "u_std_mps": float(np.std(u_smooth)),
            "du_dt_mean_mps2": float(np.mean(du_dt)),
            "du_dt_std_mps2": float(np.std(du_dt)),
            "drag_force_mean_N": float(np.mean(drag)),
            "drag_force_std_N": float(np.std(drag)),
        }
        trials.append(trial)

    result = {
        "model": "F_drag_x = d_l_eff*u + d_q_eff*abs(u)*u",
        "formula": "F_drag_x = F_x_actual - mass_effective*du_dt",
        "mass_effective_kg": float(args.mass_effective),
        "steady_window_s": float(args.steady_window),
        "trials": trials,
    }
    result.update(fit_drag(trials, nonnegative=not args.allow_negative))
    result["yaml_snippet"] = {
        "drag_linear_x": result["drag_linear_eff"],
        "drag_quadratic_x": result["drag_quadratic_eff"],
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        with open(args.output, "w") as stream:
            stream.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print("fit_drag_coefficients.py: %s" % exc, file=sys.stderr)
        sys.exit(2)
