"""Fixed-step subprocess bridge to the actual Stonefish 1.6 physics engine."""
import copy
import hashlib
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import xml.etree.ElementTree as ET

import numpy as np


def workspace_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "src/aquaflow_stonefish/scenarios").is_dir():
            return parent
    raise RuntimeError("locate workspace with aquaflow_stonefish under src/")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten_scenario(source, output, workspace):
    """Resolve ROS package/include arguments without changing physical XML.

    Only ROS transport tags are removed.  Geometry, environment, actuators,
    hydrodynamics, sensor parameters and physical asset paths are preserved.
    """
    sources = {}
    workspace = Path(workspace)

    def substitute(value, args):
        value = re.sub(r"\$\(find ([^)]+)\)",
                       lambda m: str(workspace / "src" / m.group(1)), value)
        return re.sub(r"\$\(arg ([^)]+)\)", lambda m: args[m.group(1)], value)

    def expand(filename, args):
        filename = Path(filename).resolve()
        sources[str(filename)] = sha256(filename)
        root = ET.parse(filename).getroot()

        def visit(node):
            for key, value in list(node.attrib.items()):
                node.set(key, substitute(value, args))
            children = list(node)
            for child in children:
                if child.tag.startswith("ros_"):
                    node.remove(child)
                elif child.tag == "include":
                    include_path = Path(substitute(child.attrib["file"], args))
                    if not include_path.is_absolute():
                        include_path = filename.parent / include_path
                    include_args = {a.attrib["name"]: substitute(a.attrib["value"], args)
                                    for a in child.findall("arg")}
                    included = expand(include_path, include_args)
                    index = list(node).index(child)
                    node.remove(child)
                    for offset, item in enumerate(list(included)):
                        node.insert(index + offset, copy.deepcopy(item))
                else:
                    visit(child)
        visit(root)
        return root

    root = expand(source, {})
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return sources


class StonefishBackend:
    """The world advances only in reset/step, never during an optimizer update."""
    def __init__(self, executable, data_dir, scenario, physics_hz=100,
                 log_dir=None, timeout_s=60.0):
        self.physics_hz = int(physics_hz)
        self.timeout_s = float(timeout_s)
        self.queue = queue.Queue()
        self.closed = False
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (log_dir / "stonefish.log").open("w")
        command = [str(Path(executable).resolve()), str(Path(data_dir).resolve()),
                   str(Path(scenario).resolve()), str(self.physics_hz)]
        if not Path(command[0]).is_file():
            raise FileNotFoundError("build the Stonefish adapter first: %s" % command[0])
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=os.environ.copy())
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self._receive("HOFA_READY ")
        except Exception:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.log.write(line)
                self.log.flush()
                if line.startswith("HOFA_"):
                    self.queue.put(line.strip())
        finally:
            self.queue.put("HOFA_ERROR adapter exited")

    def _receive(self, expected):
        try:
            line = self.queue.get(timeout=self.timeout_s)
        except queue.Empty as error:
            raise TimeoutError("Stonefish response timed out; inspect stonefish.log") from error
        if not line.startswith(expected):
            raise RuntimeError(line)
        return line

    def _command(self, text):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("Stonefish adapter is not running")
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        line = self._receive("HOFA_STATE ")
        values = np.asarray([float(v) for v in line.split()[1:]])
        if values.shape != (14,) or not np.isfinite(values).all():
            raise RuntimeError("invalid Stonefish state: %s" % line)
        t, x, y, z, u, v, w, qx, qy, qz, qw, p, q, r = values
        yaw = np.arctan2(2 * (qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        return dict(stamp=t, x=x, y=y, z=z, u=u, v=v, w=w, yaw=yaw, r=r,
                    qx=qx, qy=qy, qz=qz, qw=qw, p=p, q=q,
                    speed_world=float(np.hypot(u, v)))

    def reset(self, pose):
        return self._command("RESET " + " ".join("%.17g" % float(v) for v in pose))

    def step(self, pwm, dt):
        ticks = round(float(dt) * self.physics_hz)
        if ticks < 1 or abs(ticks/self.physics_hz-dt) > 1e-9:
            raise ValueError("control interval must be an integer number of physics ticks")
        pwm = np.asarray(pwm, dtype=float)
        if pwm.shape != (6,) or not np.isfinite(pwm).all() or np.any(np.abs(pwm)>1):
            raise ValueError("invalid PWM action")
        return self._command("STEP %d " % ticks + " ".join("%.17g" % v for v in pwm))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()
