#!/usr/bin/env python3
"""Identify hydrodynamic drag and added mass per axis from experiments.

The model is the scalar form used by hofa_mpc_ros::ThreeDOFModel, one axis at
a time (surge/sway/yaw)::

    F = m_eff * dx/dt + d_l * x + d_q * abs(x) * x

Identify in two stages, in this order:

1. ``--mode drag`` -- constant force, held until the vehicle reaches terminal
   velocity.  At steady state dx/dt = 0, the inertia term drops out entirely
   and the drag coefficients come out *independent of whatever mass you
   believe in*.  Needs at least three distinct force levels; one level cannot
   separate d_l from d_q.

2. ``--mode added-mass`` -- step response, with the stage-1 drag held fixed.
   The only remaining unknown is the effective inertia, and
   ``added_mass = m_eff - dry_mass``.

Fitting inertia and drag together on a single transient is ill-conditioned
when the speed range is narrow: during simulation work that produced a
*negative* sway drag coefficient, which is why the stages are separate.

Examples:

  # Stage 1: three constant-force surge trials.
  python3 fit_drag_coefficients.py --mode drag --axis surge \
      --odom f4.csv f6.csv f8.csv --force 4 6 8 --output drag_surge.json

  # Stage 1 for sway, using the recorded body-y velocity and Fy.
  python3 fit_drag_coefficients.py --mode drag --axis sway \
      --odom s2.csv s4.csv s6.csv --force 2 4 6

  # Stage 2: inertia from a step, drag pinned from stage 1.
  python3 fit_drag_coefficients.py --mode added-mass --axis surge \
      --odom step20N.csv --force 20 \
      --drag-linear 54.0 --drag-quadratic 2.0 --dry-mass 7.94
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


# Per-axis column aliases and the wrench component that drives that axis.
AXES = {
    "surge": {"velocity": ("u", "linear_x", "twist_linear_x", "vx_body"),
              "force": "fx", "force_unit": "N",
              "inertia_unit": "kg", "velocity_unit": "m/s"},
    "sway":  {"velocity": ("v", "linear_y", "twist_linear_y", "vy_body"),
              "force": "fy", "force_unit": "N",
              "inertia_unit": "kg", "velocity_unit": "m/s"},
    "yaw":   {"velocity": ("r", "angular_z", "twist_angular_z", "yaw_rate"),
              "force": "nz", "force_unit": "N.m",
              "inertia_unit": "kg.m^2", "velocity_unit": "rad/s"},
}


def _number(value):
    return float(value)


def _field(row, names, default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return _number(row[name])
    return default


def load_csv(filename, velocity_names):
    """Load a CSV containing stamp/time and the axis velocity column."""
    with open(filename, "r") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CSV is empty: %s" % filename)
    output = []
    for row in rows:
        stamp = _field(row, ("stamp", "time", "t", "t_rel"))
        x = _field(row, velocity_names)
        if stamp is None or x is None:
            continue
        output.append((stamp, x))
    if len(output) < 3:
        raise ValueError("CSV needs stamp/time and one of %s: %s"
                         % ("/".join(velocity_names), filename))
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


def load_odom(filename, velocity_names):
    if filename.lower().endswith(".csv"):
        return load_csv(filename, velocity_names)
    # Text exports have no reliable extension requirement, so try CSV first
    # and fall back to the ROS echo parser (surge only).
    try:
        return load_csv(filename, velocity_names)
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
    u = np.asarray([trial["u_mean_mps"] for trial in trials], dtype=float)
    force = np.asarray([trial["drag_force_mean_N"] for trial in trials], dtype=float)
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


def fit_added_mass(t, x, force, drag_linear, drag_quadratic, dry_mass,
                   smooth_window):
    """Fit effective inertia from a transient, with drag held fixed.

    ``F - D(x) = m_eff * dx/dt`` has a single unknown once the drag is known,
    so this is a one-parameter least squares through the origin.
    """
    x_smooth = moving_average(x, smooth_window)
    dx_dt = np.gradient(x_smooth, t)
    residual_force = force - (drag_linear * x_smooth
                              + drag_quadratic * np.abs(x_smooth) * x_smooth)
    denominator = float(np.sum(dx_dt * dx_dt))
    if denominator < 1e-9:
        raise ValueError(
            "no usable acceleration in this trial: dx/dt is ~0 everywhere. "
            "Use a step large enough to accelerate the vehicle, and a window "
            "that covers the transient rather than the steady tail.")
    m_eff = float(np.sum(dx_dt * residual_force) / denominator)
    prediction = m_eff * dx_dt
    error = residual_force - prediction
    ss_tot = float(np.sum((residual_force - residual_force.mean()) ** 2))
    # How much of the force is actually explained by acceleration rather than
    # drag.  When this is small the fit is extrapolating and m_eff is fragile.
    accel_share = float(np.mean(np.abs(prediction))
                        / max(np.mean(np.abs(force)), 1e-9))
    return {
        "mass_effective_kg": m_eff,
        "dry_mass_kg": float(dry_mass),
        "added_mass_kg": m_eff - float(dry_mass),
        "fit_rmse_N": float(np.sqrt(np.mean(error * error))),
        "fit_r2": None if ss_tot < 1e-12 else 1.0 - float(np.sum(error * error)) / ss_tot,
        "acceleration_share_of_force": accel_share,
        "dx_dt_max": float(np.max(np.abs(dx_dt))),
        "samples": int(len(t)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("drag", "added-mass"),
                        default="drag",
                        help="drag: steady-state sweep (mass-independent). "
                             "added-mass: transient, needs --drag-* from the "
                             "drag stage")
    parser.add_argument("--axis", choices=sorted(AXES), default="surge",
                        help="which body axis is being identified")
    parser.add_argument("--odom", nargs="+", required=True,
                        help="one odometry text/CSV per force level")
    parser.add_argument("--force", "--force-x", nargs="+", type=float,
                        required=True, dest="force",
                        help="actual/commanded body force (or torque for yaw) "
                             "for each odometry file")
    parser.add_argument("--mass-effective", type=float, default=7.94,
                        help="drag mode only: effective inertia used to remove "
                             "the residual acceleration term inside the steady "
                             "window; has little influence there by design")
    parser.add_argument("--drag-linear", type=float, default=None,
                        help="added-mass mode: d_l from the drag stage")
    parser.add_argument("--drag-quadratic", type=float, default=None,
                        help="added-mass mode: d_q from the drag stage")
    parser.add_argument("--dry-mass", type=float, default=None,
                        help="added-mass mode: rigid-body inertia for this "
                             "axis, so added_mass = m_eff - dry_mass")
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

    if len(args.odom) != len(args.force):
        parser.error("--odom and --force must have the same number of values")
    if args.mass_effective <= 0.0:
        parser.error("--mass-effective must be positive")
    axis = AXES[args.axis]

    if args.mode == "added-mass":
        missing = [name for name, value in
                   (("--drag-linear", args.drag_linear),
                    ("--drag-quadratic", args.drag_quadratic),
                    ("--dry-mass", args.dry_mass)) if value is None]
        if missing:
            parser.error("added-mass mode requires %s (run --mode drag first)"
                         % ", ".join(missing))
        if len(args.odom) != 1:
            parser.error("added-mass mode takes exactly one transient trial")
        data = load_odom(args.odom[0], axis["velocity"])
        t, x = select_window(data, args.steady_window,
                             args.steady_start, args.steady_end)
        result = {
            "mode": "added-mass",
            "axis": args.axis,
            "model": "F = m_eff*dx/dt + d_l*x + d_q*abs(x)*x",
            "file": os.path.abspath(args.odom[0]),
            "force": float(args.force[0]),
            "force_unit": axis["force_unit"],
            "inertia_unit": axis["inertia_unit"],
            "drag_linear": float(args.drag_linear),
            "drag_quadratic": float(args.drag_quadratic),
            "window_start_s": float(t[0]),
            "window_end_s": float(t[-1]),
        }
        result.update(fit_added_mass(
            t, x, float(args.force[0]), float(args.drag_linear),
            float(args.drag_quadratic), float(args.dry_mass),
            args.smooth_window))
        if result["acceleration_share_of_force"] < 0.1:
            result["warning"] = (
                "acceleration explains under 10%% of the applied force; this "
                "window is nearly steady state and the inertia estimate is "
                "poorly conditioned. Use a window covering the transient.")
        index = {"surge": 0, "sway": 1, "yaw": 2}[args.axis]
        result["yaml_snippet"] = {
            "vehicle": {"added_mass[%d]" % index: result["added_mass_kg"]}}
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            parent = os.path.dirname(os.path.abspath(args.output))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            with open(args.output, "w") as stream:
                stream.write(text + "\n")
        print(text)
        return 0

    if len(args.odom) < 3:
        print("warning: %d force level(s) given; three or more distinct "
              "levels are recommended to separate d_l from d_q"
              % len(args.odom), file=sys.stderr)

    trials = []
    for filename, force_x in zip(args.odom, args.force):
        data = load_odom(filename, axis["velocity"])
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
        "mode": "drag",
        "axis": args.axis,
        "model": "F_drag = d_l_eff*x + d_q_eff*abs(x)*x",
        "formula": "F_drag = F_applied - mass_effective*dx_dt",
        "note": ("at true steady state dx_dt ~ 0, so these coefficients do "
                 "not depend on mass_effective"),
        "force_unit": axis["force_unit"],
        "velocity_unit": axis["velocity_unit"],
        "mass_effective_kg": float(args.mass_effective),
        "steady_window_s": float(args.steady_window),
        "trials": trials,
    }
    result.update(fit_drag(trials, nonnegative=not args.allow_negative))
    residual_accel = max(abs(trial["du_dt_mean_mps2"]) for trial in trials)
    if residual_accel > 0.01:
        result["warning"] = (
            "mean |dx/dt| reaches %.3f in the steady window, so these trials "
            "had not settled; the fit is leaning on mass_effective and is no "
            "longer mass-independent. Hold each force level longer."
            % residual_accel)
    index = {"surge": 0, "sway": 1, "yaw": 2}[args.axis]
    result["yaml_snippet"] = {
        "vehicle": {
            "drag_linear[%d]" % index: result["drag_linear_eff"],
            "drag_quadratic[%d]" % index: result["drag_quadratic_eff"],
        }}

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
