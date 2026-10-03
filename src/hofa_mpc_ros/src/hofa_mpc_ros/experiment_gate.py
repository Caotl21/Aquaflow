"""Opt-in timestamped start barrier and near-level experimental depth hold."""
import rospy
from std_msgs.msg import Time
from nav_msgs.msg import Odometry

class ExperimentGate:
    def __init__(self, role, depth_hold=False):
        self.topic = rospy.get_param('~experiment_start_topic', '')
        self.start_at = None
        self.odom = None
        self.integral = 0.0
        self.last_depth_stamp = None
        self.target_depth = float(rospy.get_param('~experiment_depth', 1.0))
        self.depth_kp = float(rospy.get_param('~depth_kp', 8.0))
        self.depth_kd = float(rospy.get_param('~depth_kd', 4.0))
        self.depth_ki = float(rospy.get_param('~depth_ki', 0.5))
        self.depth_integral_limit = float(
            rospy.get_param('~depth_integral_limit', 5.0))
        self.depth_max_force = float(
            rospy.get_param('~depth_max_force_n', 15.0))
        if self.topic:
            self.ack = rospy.Publisher(self.topic + '/' + role + '_ack', Time, queue_size=1, latch=True)
            self.sub = rospy.Subscriber(self.topic, Time, self._start, queue_size=1)
            if depth_hold:
                vehicle = rospy.get_param('~vehicle_name', 'bricsbot')
                self.odom_sub = rospy.Subscriber('/' + vehicle + '/odometry', Odometry, self._odom, queue_size=1)

    def _start(self, msg):
        value = msg.data.to_sec()
        if value == 0:
            self.start_at = None
        elif self.start_at == value:
            pass
        elif self.start_at is None and value > rospy.Time.now().to_sec():
            self.start_at = value
        else:
            rospy.logerr('Rejected late or conflicting experiment start')
            return
        self.ack.publish(msg)

    def allows(self, now):
        return not self.topic or (self.start_at is not None and now >= self.start_at)

    def _odom(self, msg):
        self.odom = msg

    def depth_force(self):
        if not self.topic or self.odom is None:
            return 0.0
        msg = self.odom
        stamp = msg.header.stamp.to_sec()
        if not 0 <= rospy.Time.now().to_sec() - stamp < .25:
            return 0.0
        dt = 0 if self.last_depth_stamp is None else max(0., min(.2, stamp-self.last_depth_stamp))
        self.last_depth_stamp = stamp
        error = self.target_depth - msg.pose.pose.position.z
        self.integral = max(
            -self.depth_integral_limit,
            min(self.depth_integral_limit, self.integral + error * dt))
        command = (self.depth_kp * error
                   - self.depth_kd * msg.twist.twist.linear.z
                   + self.depth_ki * self.integral)
        return max(-self.depth_max_force,
                   min(self.depth_max_force, command))
