"""Planar trajectory tracking surrogate, not an identified BricsBot model."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from .geometry import camera_tokens, clearance, first_contact, rotation, swept_clearance, wrap
from .trajectory import Trajectory


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    u: float = 0.0
    v: float = 0.0
    r: float = 0.0

    def vector(self):
        return np.array([self.x,self.y,self.yaw,self.u,self.v,self.r],dtype=float)


@dataclass
class Config:
    dt: float = 0.1
    substeps: int = 5
    sensor_dt: float = 0.1
    sensor_delay: float = 0.1
    command_delay: float = 0.1
    max_time: float = 120.0
    max_speed: float = 0.20
    max_yaw_rate: float = 0.50
    max_accel: float = 0.12
    max_yaw_accel: float = 0.25
    tracking_tau: tuple = (0.25,0.35,0.20)
    position_gain: float = 1.0
    yaw_gain: float = 1.0
    ideal_tracking: bool = False
    current_world: tuple = (0.0,0.0)
    robot_radius: float = 0.365
    safety_margin: float = 0.08
    goal_tolerance: float = 0.20
    bounds: tuple = (-2.0,10.0,-4.0,4.0)
    camera_offset: tuple = (0.20,0.0)
    fov: float = math.radians(80)
    sensor_range: float = 8.0
    rays: int = 641
    bearing_noise: float = math.radians(0.2)
    dropout: float = 0.05
    history: int = 12
    max_tracks: int = 8

    def __post_init__(self):
        positive = [self.dt,self.sensor_dt,self.max_time,self.max_speed,self.max_yaw_rate,
                    self.max_accel,self.max_yaw_accel,self.robot_radius,self.goal_tolerance,
                    self.sensor_range,self.position_gain,self.yaw_gain,*self.tracking_tau]
        nonnegative = [self.sensor_delay,self.command_delay,self.safety_margin,self.bearing_noise]
        if not np.isfinite(positive+nonnegative).all() or min(positive)<=0 or min(nonnegative)<0:
            raise ValueError("invalid time, dynamics or sensor parameters")
        for n in (self.substeps,self.history,self.max_tracks,self.rays):
            if not isinstance(n,int) or n<1:
                raise ValueError("counts must be positive integers")
        if self.rays<3 or not 0<self.fov<math.pi or not 0<=self.dropout<=1:
            raise ValueError("invalid camera parameters")
        if len(self.tracking_tau)!=3 or len(self.camera_offset)!=2 or len(self.current_world)!=2:
            raise ValueError("invalid vector dimensions")
        if len(self.bounds)!=4 or not np.isfinite([*self.bounds,*self.camera_offset,*self.current_world]).all():
            raise ValueError("invalid geometry")
        xmin,xmax,ymin,ymax=self.bounds
        if xmax-xmin<=2*self.robot_radius or ymax-ymin<=2*self.robot_radius:
            raise ValueError("bounds smaller than robot")
        if np.linalg.norm(self.camera_offset)>=self.robot_radius:
            raise ValueError("camera must be inside the circular footprint")
        step=self.dt/self.substeps
        for period in (self.sensor_dt,self.sensor_delay,self.command_delay,self.max_time):
            if abs(period/step-round(period/step))>1e-7:
                raise ValueError("timing values must be integer multiples of integration step")


class BearingEnv:
    """step(Trajectory) executes one policy period; step(None) keeps old plan.

    Observation is exclusively angles/masks, ideal association IDs, relative
    odometry, velocity and goal. Ground truth only appears in the info dict.
    Delays preserve original capture and submission times.
    """
    def __init__(self, obstacles=(), goal=(5.,0.), config=None, state=None, seed=0):
        self.obstacles=tuple(obstacles)
        self.goal=np.asarray(goal,dtype=float).copy()
        self.config=config or Config()
        self.initial=(state or State()).vector()
        self.seed=seed
        if self.goal.shape!=(2,) or not np.isfinite(self.goal).all():
            raise ValueError("goal must be finite xy")
        self.reset()

    @property
    def time(self):
        return self.tick*self.config.dt/self.config.substeps

    def reset(self, seed=None):
        if seed is not None:
            self.seed=seed
        self.rng=np.random.default_rng(self.seed)
        self.state=self.initial.copy()
        c=self.config
        if not np.isfinite(self.state).all() or clearance(self.state[:2],self.obstacles,c.robot_radius,c.bounds)<=0:
            raise ValueError("initial state must be finite and collision-free")
        if np.linalg.norm(self.state[3:5])>c.max_speed or abs(self.state[5])>c.max_yaw_rate:
            raise ValueError("initial velocity exceeds configured limits")
        if clearance(self.goal,self.obstacles,c.robot_radius,c.bounds)<=0:
            raise ValueError("goal is inside a physical obstacle or outside bounds")
        self.tick=0
        self.done=False
        self.collision=False
        self.reached=bool(np.linalg.norm(self.goal-self.state[:2])<=c.goal_tolerance)
        self.done=self.reached
        self.frames=deque(maxlen=c.history)
        self.pending_frames=deque()
        self.commands=deque()
        self.active=None
        self.path_length=0.0
        self.min_clearance=clearance(self.state[:2],self.obstacles,c.robot_radius,c.bounds)
        self._capture()
        self._deliver()
        return self.observation(),self._info()

    def _capture(self):
        tok,mask,ids=camera_tokens(self.state[:3],self.obstacles,self.config.bounds,self.config,self.rng)
        self.pending_frames.append((self.time,tok,mask,ids,self.state.copy()))

    def _deliver(self):
        while self.pending_frames and self.pending_frames[0][0]+self.config.sensor_delay<=self.time+1e-9:
            self.frames.append(self.pending_frames.popleft())

    def observation(self):
        c=self.config
        tokens=np.zeros((c.history,c.max_tracks,7),dtype=np.float32)
        masks=np.zeros((c.history,c.max_tracks),dtype=bool)
        ids=np.full((c.history,c.max_tracks),-1,dtype=np.int64)
        frame_mask=np.zeros(c.history,dtype=bool)
        age=np.zeros(c.history,dtype=np.float32)
        ego=np.zeros((c.history,7),dtype=np.float32)
        rot=rotation(self.state[2]).T
        for index,frame in enumerate(self.frames,start=c.history-len(self.frames)):
            stamp,tok,mask,identity,state=frame
            tokens[index],masks[index],ids[index]=tok,mask,identity
            frame_mask[index]=True
            age[index]=self.time-stamp
            delta=rot@(state[:2]-self.state[:2])
            dyaw=wrap(state[2]-self.state[2])
            ego[index]=[*delta,math.sin(dyaw),math.cos(dyaw),*state[3:]]
        goal=rot@(self.goal-self.state[:2])
        return dict(tokens=tokens,track_mask=masks,track_ids=ids,frame_mask=frame_mask,
                    age=age,ego_history=ego,velocity=self.state[3:].astype(np.float32).copy(),
                    goal=goal.astype(np.float32))

    def _info(self):
        c=self.config
        return dict(time=float(self.time),state=self.state.tolist(),collision=bool(self.collision),
                    reached=bool(self.reached),min_clearance=float(self.min_clearance),
                    margin_violation=bool(self.min_clearance<c.safety_margin),path_length=float(self.path_length),
                    goal_distance=float(np.linalg.norm(self.goal-self.state[:2])))

    def step(self, trajectory=None):
        if self.done:
            raise RuntimeError("reset required after episode end")
        c=self.config
        if trajectory is not None:
            if not isinstance(trajectory,Trajectory):
                raise TypeError("action must be a local Trajectory or None")
            plan=Trajectory(trajectory.points)
            speed=np.diff(plan.points[:,1:],axis=0)/np.diff(plan.points[:,0,None],axis=0)
            if np.any(np.linalg.norm(speed[:,:2],axis=1)>c.max_speed+1e-8) or np.any(abs(speed[:,2])>c.max_yaw_rate+1e-8):
                raise ValueError("reference exceeds configured speed or yaw rate")
            self.commands.append((self.time+c.command_delay,self.time,plan,self.state[:3].copy()))
        previous_distance=float(np.linalg.norm(self.goal-self.state[:2]))
        h=c.dt/c.substeps
        interval_min=float("inf")
        while_count=0
        while while_count<c.substeps and not self.done:
            while self.commands and self.commands[0][0]<=self.time+1e-9:
                _,stamp,plan,anchor=self.commands.popleft()
                self.active=(stamp,plan,anchor)
            target=np.zeros(3)
            if self.active is not None:
                stamp,plan,anchor=self.active
                reference,feedforward=plan.sample(self.time-stamp,anchor)
                target[:2]=rotation(self.state[2]).T@(feedforward[:2]+c.position_gain*(reference[:2]-self.state[:2]))
                target[2]=feedforward[2]+c.yaw_gain*wrap(reference[2]-self.state[2])
            norm=np.linalg.norm(target[:2])
            if norm>c.max_speed:
                target[:2]*=c.max_speed/norm
            target[2]=np.clip(target[2],-c.max_yaw_rate,c.max_yaw_rate)
            if c.ideal_tracking:
                self.state[3:]=target
            else:
                delta=(target-self.state[3:])*(1-np.exp(-h/np.asarray(c.tracking_tau)))
                norm=np.linalg.norm(delta[:2])
                if norm>c.max_accel*h:
                    delta[:2]*=c.max_accel*h/norm
                delta[2]=np.clip(delta[2],-c.max_yaw_accel*h,c.max_yaw_accel*h)
                self.state[3:]+=delta
                actual_speed=np.linalg.norm(self.state[3:5])
                if actual_speed>c.max_speed:
                    self.state[3:5]*=c.max_speed/actual_speed
            start=self.state[:2].copy()
            ground_velocity=rotation(self.state[2])@self.state[3:5]+np.asarray(c.current_world)
            end=start+h*ground_velocity
            physical=swept_clearance(start,end,self.obstacles,c.robot_radius,c.bounds)
            fraction=first_contact(start,end,self.obstacles,c.robot_radius,c.bounds) if physical<=0 else 1.0
            self.state[:2]=start+fraction*(end-start)
            self.state[2]=wrap(self.state[2]+fraction*h*self.state[5])
            executed_clearance=swept_clearance(start,self.state[:2],self.obstacles,c.robot_radius,c.bounds)
            self.min_clearance=min(self.min_clearance,executed_clearance)
            interval_min=min(interval_min,executed_clearance)
            self.path_length+=float(np.linalg.norm(self.state[:2]-start))
            self.tick+=1
            while_count+=1
            self.collision=physical<=0
            self.reached=np.linalg.norm(self.goal-self.state[:2])<=c.goal_tolerance and not self.collision
            self.done=self.collision or self.reached or self.time>=c.max_time-1e-9
            if self.tick % round(c.sensor_dt/h)==0:
                self._capture()
            self._deliver()
        info=self._info()
        info["interval_clearance"]=interval_min
        reward=previous_distance-info["goal_distance"]-0.01*h*while_count
        reward-=10.0*int(self.collision)
        reward+=1.0*int(self.reached)
        terminated=self.collision or self.reached
        truncated=self.done and not terminated
        return self.observation(),float(reward),bool(terminated),bool(truncated),info
