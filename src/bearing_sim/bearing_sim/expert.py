"""Privileged single-circle route teacher used only for labels."""
import numpy as np
from .geometry import rotation,swept_clearance
from .trajectory import Trajectory

class RouteTeacher:
    def __init__(self,env):
        self.config=env.config
        start,goal=env.state[:2].copy(),env.goal.copy()
        self.mode,self.index=0,0
        if len(env.obstacles)>1: raise ValueError('v1 teacher supports <=1 circle')
        safe=self.config.robot_radius+self.config.safety_margin+.22
        if swept_clearance(start,goal,env.obstacles,safe,self.config.bounds)>0:
            self.route=[goal]
            return
        obstacle=env.obstacles[0]
        forward=(goal-start)/np.linalg.norm(goal-start)
        left=np.array([-forward[1],forward[0]])
        center=np.array([obstacle.x,obstacle.y]); options=[]
        for sign in (1,-1):
            extent=obstacle.radius+safe+.35
            points=[start,center-forward*extent+left*sign*extent,center+forward*extent+left*sign*extent,goal]
            if all(swept_clearance(a,b,env.obstacles,safe,self.config.bounds)>0 for a,b in zip(points[:-1],points[1:])):
                length=sum(np.linalg.norm(b-a) for a,b in zip(points[:-1],points[1:]))
                options.append((length,sign,points[1:]))
        if not options: raise ValueError('no safe teacher route')
        _,sign,self.route=min(options,key=lambda item:item[0])
        self.mode=1 if sign>0 else 2

    def action(self,state,horizon=19,dt=.1):
        while self.index<len(self.route)-1 and np.linalg.norm(self.route[self.index]-state[:2])<.15: self.index+=1
        virtual,index=state[:2].copy(),self.index
        rot=rotation(state[2]).T; actions=[]
        for _ in range(horizon):
            while index<len(self.route)-1 and np.linalg.norm(self.route[index]-virtual)<.15: index+=1
            offset=self.route[index]-virtual
            velocity=offset/max(np.linalg.norm(offset),1e-12)*min(self.config.max_speed,np.linalg.norm(offset))
            actions.append([*(rot@velocity),0.]); virtual+=velocity*dt
        actions=np.asarray(actions,dtype=np.float32)
        points=np.zeros((horizon+1,4)); points[:,0]=np.arange(horizon+1)*dt
        points[1:,1:]=np.cumsum(actions.astype(float)*dt,axis=0)
        return Trajectory(points),actions,self.mode
