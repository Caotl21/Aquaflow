#!/usr/bin/env python3
"""Plot artifacts from ``run_tracking_experiment.py`` without ROS."""
from __future__ import print_function

import importlib.util
import os
import sys


try:
    from hofa_mpc_ros.tracking_plots import main
except ImportError:
    source = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../src/hofa_mpc_ros/tracking_plots.py"))
    package_root = os.path.dirname(os.path.dirname(source))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    spec = importlib.util.spec_from_file_location(
        "hofa_mpc_ros.tracking_plots", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = module.main


if __name__ == "__main__":
    main()
