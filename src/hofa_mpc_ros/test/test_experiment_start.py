"""Run in the sourced ROS environment; no master or simulator required."""
import importlib.util
import pathlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rospy
import hofa_mpc_ros
from pkgutil import extend_path
# Other pure-Python tests prepend the source tree; include generated ROS msgs.
hofa_mpc_ros.__path__ = extend_path(hofa_mpc_ros.__path__, hofa_mpc_ros.__name__)
from std_msgs.msg import Time
from hofa_mpc_ros.experiment_gate import ExperimentGate


def load_script(name):
    path = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script('run_mpc_experiment')
reference = load_script('reference_processor')


class StartTests(unittest.TestCase):
    def test_missing_ack_cancels_start(self):
        barrier = runner.ExperimentBarrier.__new__(runner.ExperimentBarrier)
        barrier.recorder = SimpleNamespace(start_stamp=None, done=True)
        barrier.acks = {}
        barrier.pub, barrier.stop = Mock(), Mock()
        with patch.object(rospy.Time, 'now', side_effect=[rospy.Time(10), rospy.Time(11)]), patch.object(rospy, 'is_shutdown', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'acknowledgement missing'):
                barrier.start('/unused')
        barrier.stop.assert_called_once()

    def test_late_start_rejected_and_future_start_cancelled(self):
        gate = ExperimentGate.__new__(ExperimentGate)
        gate.topic, gate.start_at, gate.ack = '/unique/start', None, Mock()
        with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(10)), patch.object(rospy, 'logerr'):
            gate._start(Time(data=rospy.Time.from_sec(9)))
            self.assertFalse(gate.allows(10))
            gate.ack.publish.assert_not_called()
            gate._start(Time(data=rospy.Time.from_sec(11)))
            self.assertFalse(gate.allows(10.99))
            self.assertTrue(gate.allows(11))
            gate._start(Time(data=rospy.Time()))
            self.assertFalse(gate.allows(12))

    def test_reference_does_not_advance_or_rebase_clock(self):
        obj = reference.ReferenceProcessor.__new__(reference.ReferenceProcessor)
        gate = ExperimentGate.__new__(ExperimentGate)
        gate.topic, gate.start_at = '/start', 100.
        obj.experiment_gate = gate
        obj._publish_schedule_lag = Mock()
        obj.progress_mode = 'schedule'
        obj.arc_path = {'s': [0., 10.], 'time': [0., 10.]}
        obj.max_schedule_lead = .1
        obj.schedule_t0 = None
        self.assertEqual(obj._schedule_anchor(rospy.Time.from_sec(99), 0), 0)
        self.assertEqual(obj._schedule_anchor(rospy.Time.from_sec(105), 0), 5)
        self.assertEqual(obj.schedule_t0.to_sec(), 100)

    def test_delayed_first_sample_does_not_reset_scoring_origin(self):
        path = {'points': [(100., 0., 0., 1., 0.), (100., 5., 0., 1., 0.)]}
        obj = runner.ExperimentRecorder.__new__(runner.ExperimentRecorder)
        obj.start_stamp, obj.profile_params = 100., {}
        row = dict(stamp=100.1, x=0., y=0., yaw=0., speed_world=0., r=0.)
        model = obj.evaluation_model(path, [row])
        self.assertEqual(float(model['times'][0]), 100.)
        metrics = runner.compute_metrics([row], model, [], [], [5., 0.], 'timeout')
        self.assertAlmostEqual(metrics['duration_s'], .1)
        self.assertGreater(metrics['schedule_lag_s']['rmse'], .09)

    def test_snapshot_filters_prestart_without_deleting_recording(self):
        obj = runner.ExperimentRecorder.__new__(runner.ExperimentRecorder)
        obj.lock = threading.RLock()
        obj.start_stamp = 100.
        obj.odom = [{'stamp': 99.9}, {'stamp': 100.02}]
        for key in ('wrench', 'status', 'window', 'pwm', 'virtual_accel'):
            setattr(obj, key, [{'stamp': 99.9}, {'stamp': 100.02}])
        obj.initial_path = obj.global_path = None
        obj.finish_reason = 'timeout'
        data = obj.snapshot()
        self.assertEqual(len(data['odom']), 1)
        self.assertAlmostEqual(data['odom'][0]['t_rel'], .02)
        self.assertEqual(len(obj.odom), 2)


if __name__ == '__main__':
    unittest.main()
