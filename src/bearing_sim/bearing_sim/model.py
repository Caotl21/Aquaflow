"""Observation-only masked recurrent multi-trajectory policy."""
import numpy as np
import torch
from torch import nn
from .trajectory import Trajectory
from .records import FIELDS
DEFAULTS=dict(hidden=64,candidates=3,horizon=19,dt=.1,max_speed=.2,max_yaw_rate=.5,max_accel=.12,max_yaw_accel=.25)

class BearingPolicy(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.config=dict(DEFAULTS,**kwargs)
        c,h=self.config,self.config['hidden']
        if c['candidates']!=3: raise ValueError('v1 heads: direct/left/right')
        if c['horizon'] < 2 or c['hidden'] < 1:
            raise ValueError('invalid model dimensions')
        if any(not np.isfinite(c[k]) or c[k] <= 0 for k in
               ('dt','max_speed','max_yaw_rate','max_accel','max_yaw_accel')):
            raise ValueError('invalid reference limits')
        self.token=nn.Sequential(nn.Linear(7,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.temporal=nn.GRUCell(2*h+9,h)
        self.fuse=nn.Sequential(nn.Linear(h+5,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.logits=nn.Linear(h,3)
        self.actions=nn.Linear(h,3*c['horizon']*3)
        self.register_buffer('ego_scale',torch.tensor([1.,1.,1.,1.,.2,.2,.5]))

    def forward(self,tokens,track_mask,frame_mask,age,ego_history,velocity,goal):
        valid=track_mask.bool() & frame_mask.bool()[...,None]
        safe=torch.where(valid[...,None],tokens,torch.zeros_like(tokens)).float()
        z=self.token(safe)
        mean=(z*valid[...,None]).sum(2)/valid.sum(2,keepdim=True).clamp_min(1)
        maximum=z.masked_fill(~valid[...,None],-1e9).max(2).values
        maximum=torch.where(valid.any(2)[...,None],maximum,torch.zeros_like(maximum))
        ego=torch.where(frame_mask[...,None],ego_history,torch.zeros_like(ego_history)).float()/self.ego_scale
        ages=torch.where(frame_mask,age,torch.zeros_like(age)).float()
        count=valid.sum(2).float()/tokens.shape[2]
        seq=torch.cat([mean,maximum,ego,ages[...,None],count[...,None]],-1)
        hidden=torch.zeros(len(tokens),self.config['hidden'],device=tokens.device)
        for i in range(tokens.shape[1]):
            update=self.temporal(seq[:,i],hidden)
            hidden=torch.where(frame_mask[:,i,None],update,hidden)
        state=torch.cat([velocity.float()/self.ego_scale[-3:],goal.float()/5],-1)
        fused=self.fuse(torch.cat([hidden,state],-1))
        raw=torch.tanh(self.actions(fused)).view(len(tokens),3,self.config['horizon'],3)
        xy=raw[...,:2]/torch.linalg.vector_norm(raw[...,:2],dim=-1,keepdim=True).clamp_min(1)*self.config['max_speed']
        return dict(logits=self.logits(fused),actions=torch.cat([xy,raw[...,2:]*self.config['max_yaw_rate']],-1))

def policy_loss(output,target,mode):
    if target.shape != (len(mode),output['actions'].shape[2],3):
        raise ValueError('teacher and model trajectory horizons differ')
    selected=output['actions'][torch.arange(len(mode),device=mode.device),mode]
    scale=selected.new_tensor([.2,.2,.5])
    fit=((selected-target)/scale).square().mean()
    classification=nn.functional.cross_entropy(output['logits'],mode)
    smooth=((selected[:,1:]-selected[:,:-1])/scale).square().mean()
    return fit+.15*classification+.01*smooth

def decode_actions(actions,velocity,config):
    actions=np.asarray(actions,dtype=float)
    previous=np.asarray(velocity,dtype=float).copy()
    if (actions.shape!=(config['horizon'],3) or not np.isfinite(actions).all()
            or previous.shape != (3,) or not np.isfinite(previous).all()):
        raise ValueError('invalid predicted actions or initial velocity')
    if np.linalg.norm(previous[:2]) > config['max_speed']+1e-8 or abs(previous[2]) > config['max_yaw_rate']+1e-8:
        raise ValueError('initial velocity exceeds reference limits')
    dt=config['dt']; points=np.zeros((len(actions)+1,4))
    for i,desired in enumerate(actions):
        desired=desired.copy()
        desired[:2]*=min(1,config['max_speed']/max(np.linalg.norm(desired[:2]),1e-12))
        desired[2]=np.clip(desired[2],-config['max_yaw_rate'],config['max_yaw_rate'])
        delta=desired-previous
        delta[:2]*=min(1,config['max_accel']*dt/max(np.linalg.norm(delta[:2]),1e-12))
        delta[2]=np.clip(delta[2],-config['max_yaw_accel']*dt,config['max_yaw_accel']*dt)
        previous+=delta
        points[i+1,0]=(i+1)*dt
        points[i+1,1:]=points[i,1:]+previous*dt
    return Trajectory(points)

@torch.no_grad()
def predict(model,observation,device=None):
    device=device or next(model.parameters()).device
    tensors={k:torch.as_tensor(np.asarray(observation[k]),device=device).unsqueeze(0) for k in FIELDS}
    output=model(**tensors)
    probabilities=output['logits'].softmax(-1)[0].cpu().numpy()
    actions=output['actions'][0].cpu().numpy()
    candidates=[decode_actions(a,observation['velocity'],model.config) for a in actions]
    chosen=int(probabilities.argmax())
    return candidates[chosen],dict(mode=chosen,probabilities=probabilities.tolist(),candidates=[p.points.tolist() for p in candidates])
