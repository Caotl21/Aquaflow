"""Checks for action bounds, crossings and physical-scenario preservation."""
import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from hofa_policy.rl_backend import flatten_scenario, workspace_root
from hofa_policy.rl_reference import Reference, make_route
from hofa_policy.rl_actuation import Actuation
from hofa_policy.residual_core import compose, OBSERVATION_SIZE
from hofa_policy.residual_env import read_config


def test_zero_residual_is_exact_baseline_and_limits_are_physical():
    base=np.array([8.0,-0.5,0.4])
    limits=np.array([1.0,0.4,0.08])
    slew=np.array([0.3,0.15,0.025])
    command,residual=compose(base,np.zeros(3),np.zeros(3),limits,slew)
    np.testing.assert_array_equal(command,base)
    for _ in range(10):
        command,new=compose(base,np.array([9,-9,9]),residual,limits,slew)
        assert np.all(np.abs(new-residual)<=slew+1e-12)
        assert np.all(np.abs(new)<=limits+1e-12)
        residual=new
    with pytest.raises(ValueError):
        compose(base,[0,np.nan,0],residual,limits,slew)


def test_crossing_hairpin_cannot_jump_progress_to_later_branch():
    ref=Reference(make_route("hairpin"),read_config()["profile"])
    arc,_,_=ref.project(-3,0,update=True)
    assert arc<=0.6
    # Walk the first straight to its crossing with a later branch.
    for x in np.linspace(-5,-3,30):
        arc,_,_=ref.project(x,0,update=True)
    assert arc==pytest.approx(2.0)


def test_scenario_flattening_preserves_thruster_and_mass_data(tmp_path):
    root=workspace_root()
    out=tmp_path/"scenario.scn"
    sources=flatten_scenario(root/"src/aquaflow_stonefish/scenarios/aquaflow_empty_pool_control.scn",out,root)
    flattened=ET.parse(out).getroot()
    original=ET.parse(root/"src/aquaflow_stonefish/scenarios/bricsbot_control_reference.scn").getroot()
    assert len(sources)==2
    assert len(flattened.findall(".//actuator"))==6
    for original_actuator,new_actuator in zip(original.findall(".//actuator"),flattened.findall(".//actuator")):
        assert ET.tostring(original_actuator)==ET.tostring(new_actuator)
    assert [x.attrib for x in original.findall(".//mass")]==[x.attrib for x in flattened.findall(".//mass")]
    assert not any(node.tag.startswith("ros_") for node in flattened.iter())
    assert "$(" not in out.read_text()


def test_allocator_reports_deadzone_and_realized_wrench():
    allocator=Actuation(workspace_root())
    pwm,realized,forces,saturated=allocator.allocate(np.array([8,1,0.4]))
    assert np.all(np.abs(pwm)<=1)
    assert not saturated
    np.testing.assert_allclose(realized,[8,1,0.4],atol=0.08)
    pwm,realized,_,_=allocator.allocate(np.zeros(3))
    np.testing.assert_array_equal(pwm,np.zeros(6))
    np.testing.assert_array_equal(realized,np.zeros(3))
    _,_,_,saturated=allocator.allocate(np.array([1000,0,0]))
    assert saturated


def test_reference_profile_and_window_obey_contract():
    ref=Reference(make_route("right_angle"),read_config()["profile"])
    window=ref.window(0)
    assert len(window)==20
    assert window[0]["dx_ned"]==0
    assert max(abs(p["dpsi_ned"]) for p in window)<=0.5+1e-8
    assert OBSERVATION_SIZE==212
