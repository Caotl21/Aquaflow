"""Time-indexed local reference with consistent pose and derivative sampling."""
from dataclasses import dataclass
import numpy as np
from .geometry import rotation


@dataclass
class Trajectory:
    # Each row: [time_s, x_m, y_m, yaw_rad], fixed frame at submission pose.
    points: np.ndarray

    def __post_init__(self):
        self.points = np.array(self.points,dtype=float,copy=True)
        p = self.points
        if p.ndim != 2 or p.shape[1] != 4 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError("trajectory must contain finite N x 4 time/x/y/yaw points")
        if p[0,0] != 0 or np.any(np.diff(p[:,0]) <= 0):
            raise ValueError("trajectory starts at t=0 with strictly increasing times")
        if np.linalg.norm(p[0,1:]) > 1e-8:
            raise ValueError("local trajectory must start at the submission pose")
        p[:,3] = np.unwrap(p[:,3])

    @classmethod
    def constant_velocity(cls, velocity, dt=0.1, count=20):
        velocity = np.asarray(velocity,dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("velocity must be finite [vx,vy,yaw_rate]")
        t = np.arange(count)*dt
        return cls(np.column_stack([t,t[:,None]*velocity]))

    def sample(self, age, anchor):
        p = self.points
        index = min(np.searchsorted(p[:,0],max(age,0),side="right")-1,len(p)-2)
        interval = p[index+1,0]-p[index,0]
        fraction = np.clip((age-p[index,0])/interval,0,1)
        local = p[index,1:]*(1-fraction)+p[index+1,1:]*fraction
        derivative = (p[index+1,1:]-p[index,1:])/interval
        if age >= p[-1,0]:
            derivative = np.zeros(3)
        rot = rotation(anchor[2])
        pose = np.r_[anchor[:2]+rot@local[:2],anchor[2]+local[2]]
        velocity = np.r_[rot@derivative[:2],derivative[2]]
        return pose, velocity
