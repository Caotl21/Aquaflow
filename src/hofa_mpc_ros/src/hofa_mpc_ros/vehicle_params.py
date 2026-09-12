"""Resolution of vehicle inertia from configuration.

The number the controllers need is the *effective* inertia -- rigid-body mass
plus hydrodynamic added mass -- because that is what resists acceleration in
water.  Storing only that sum hides where it came from, and the two halves
behave very differently across platforms:

* ``dry_mass`` is a property of the vehicle.  It is the same in simulation and
  on the real robot, and you get it from a scale.
* ``added_mass`` is a property of the vehicle *and the fluid model*.  In
  Stonefish it is derived from a minimum-volume ellipsoid enclosing the
  collision mesh, which for an open-frame hull encloses far more water than
  the hull actually displaces; the resulting values are roughly an order of
  magnitude above what the physical vehicle shows.

Keeping them as separate fields makes that difference visible in the config
instead of buried in one number, and means a platform switch only has to
replace the half that actually changes.
"""
import numpy as np

# Fallbacks used only when a config supplies neither form.
DEFAULT_DRY_MASS = (7.94, 7.94, 0.15)
DEFAULT_ADDED_MASS = (0.0, 0.0, 0.0)


def _triplet(values, name):
    values = [float(v) for v in values]
    if len(values) != 3:
        raise ValueError("%s must have 3 entries [x, y, yaw], got %d"
                         % (name, len(values)))
    return values


def resolve_inertia(vehicle):
    """Return (mass_matrix 3x3, info dict) for a ``vehicle`` parameter dict.

    Preferred form::

        vehicle:
          dry_mass:   [7.94, 7.94, 0.15]
          added_mass: [233.0, 286.0, 1.88]

    The legacy ``mass_matrix`` (a flat 9-element row-major matrix) is still
    accepted so existing configs keep working; it is treated as already
    effective, with added mass folded in and therefore unattributable.
    """
    if "dry_mass" in vehicle:
        dry = _triplet(vehicle["dry_mass"], "dry_mass")
        added = _triplet(vehicle.get("added_mass", DEFAULT_ADDED_MASS),
                         "added_mass")
        effective = [d + a for d, a in zip(dry, added)]
        source = "dry_mass+added_mass"
    elif "mass_matrix" in vehicle:
        flat = [float(v) for v in vehicle["mass_matrix"]]
        if len(flat) != 9:
            raise ValueError("mass_matrix must have 9 entries, got %d"
                             % len(flat))
        matrix = np.asarray(flat, dtype=float).reshape(3, 3)
        effective = [matrix[0, 0], matrix[1, 1], matrix[2, 2]]
        dry = added = None
        source = "mass_matrix (legacy)"
        return matrix, {"source": source, "effective": effective,
                        "dry": dry, "added": added}
    else:
        dry = list(DEFAULT_DRY_MASS)
        added = list(DEFAULT_ADDED_MASS)
        effective = list(dry)
        source = "defaults"

    return np.diag(effective), {"source": source, "effective": effective,
                                "dry": dry, "added": added}


def describe_inertia(info):
    """One-line human summary for a startup log."""
    effective = info["effective"]
    if info["dry"] is None:
        return ("inertia %s: effective m=[%.2f, %.2f] kg Iz=%.3f kg.m^2 "
                "(added mass not separable)"
                % (info["source"], effective[0], effective[1], effective[2]))
    return ("inertia %s: dry=[%.2f, %.2f, %.3f] + added=[%.2f, %.2f, %.3f] "
            "-> effective=[%.2f, %.2f, %.3f]"
            % (info["source"], info["dry"][0], info["dry"][1], info["dry"][2],
               info["added"][0], info["added"][1], info["added"][2],
               effective[0], effective[1], effective[2]))
