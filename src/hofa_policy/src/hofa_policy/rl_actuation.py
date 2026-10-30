"""Reuse the actual ROS allocator's calibration without importing rospy."""
import ast
import bisect
import math
from pathlib import Path
import numpy as np


class Actuation:
    def __init__(self, workspace):
        path = Path(workspace)/"src/aquaflow_stonefish/scripts/brics6_thruster_allocator.py"
        tree = ast.parse(path.read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                       and node.name in ("invert4", "Brics6Allocator")]
        namespace = dict(bisect=bisect, math=math)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
        cls = namespace["Brics6Allocator"]
        self.allocator = cls.__new__(cls)
        self.pinv = np.asarray(self.allocator._build_pseudoinverse())
        self.matrix = np.linalg.pinv(self.pinv)
        self.minimum = -6.0*9.80665
        self.maximum = 6.9*9.80665

    def allocate(self, wrench, fz=0.0):
        proposed = self.pinv@np.r_[wrench, float(fz)]
        forces = np.clip(proposed, self.minimum, self.maximum)
        pwm = np.asarray([self.allocator.force_to_pwm(float(f), i)
                          for i, f in enumerate(forces)])
        # Include startup deadzone in reconstructed force, matching the actual
        # PWM-to-thrust model. Never report a clipped command as executed.
        actual = []
        for command in pwm:
            curve = (self.allocator.positive_curve if command>=0 else
                     self.allocator.negative_curve)
            value = np.interp(abs(command), [c[0] for c in curve], [c[1] for c in curve])
            actual.append(value if command>=0 else -value)
        realized = self.matrix@np.asarray(actual)
        saturated = bool(np.any(np.abs(forces-proposed)>1e-8))
        return pwm, realized[:3], forces, saturated


class DepthPID:
    def __init__(self, target=1.0):
        self.target = float(target)
        self.integral = 0.0

    def command(self, state, dt):
        error = self.target-state["z"]
        self.integral = float(np.clip(self.integral+error*dt, -5, 5))
        return float(np.clip(8*error-4*state["w"]+0.5*self.integral, -15, 15))
