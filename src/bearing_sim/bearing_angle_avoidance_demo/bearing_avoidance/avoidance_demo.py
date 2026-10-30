#!/usr/bin/env python3
"""Bearing-angle PF + 前向局部轨迹采样：无 RL、无主动信息奖励。
运行 python avoidance_demo.py --seed 7 --output results --gif
模型、真值评估器与规划器接口分离；规划器不读取真实障碍物。
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from bearing_angle_pf import BearingAnglePF, observe, wrap


@dataclass
class Config:
    seed: int = 7
    particles: int = 1000                 # 每目标，而不是整个场景。
    map_samples: int = 180
    dt: float = 0.35
    horizon: float = 2.8
    max_steps: int = 220
    robot_radius: float = 0.18
    safety_margin: float = 0.10
    risk_threshold: float = 0.08          # 已发现目标的经验碰撞比例，不是安全保证。
    fov_deg: float = 140.
    sensor_range: float = 12.
    noise_beta_deg: float = 0.6
    noise_theta_deg: float = 0.35
    goal_tolerance: float = 0.30
    occlusion: bool = True
    corridor_half_width: float = 2.5
    distance_prior: tuple = (.55, 13.)
    diameter_prior: tuple = (.20, 1.25)


@dataclass
class AngularObservation:
    detections: dict                       # ID -> [bearing, 完整角宽]，rad。
    partial_sectors: list                  # 可见但不完整轮廓的角区间，rad。


def make_scene(seed):
    """可复现的受约束随机圆柱场；故意让直线参考穿过障碍物。
    场景生成使用真实几何，不把它传给规划器；没有预计算避障路径。
    """
    rng=np.random.default_rng(seed)
    circles=[]
    for x in [2.5,4.9,7.2]:
        circles.append([x+rng.uniform(-.15,.15),rng.uniform(-.16,.16),rng.uniform(.48,.65)])
    attempts=0
    while len(circles)<12 and attempts<10000:
        attempts+=1
        c=np.array([rng.uniform(1.8,8.4),rng.uniform(-1.9,1.9),rng.uniform(.35,.65)])
        if all(np.linalg.norm(c[:2]-np.array(o[:2])) > (c[2]+o[2])/2+.65 for o in circles):
            circles.append(c.tolist())
    if len(circles)!=12:raise RuntimeError('Random scene generation failed')
    return np.array(circles),np.array([0.,0.]),np.array([9.5,0.])


def ray_distances(circles,pose,angles):
    q=circles[:,:2]-pose[:2]
    r=circles[:,2]/2
    u=np.column_stack((np.cos(angles+pose[2]),np.sin(angles+pose[2])))
    b=q@u.T
    disc=b*b-(np.sum(q*q,axis=1)-r*r)[:,None]
    near=b-np.sqrt(np.maximum(disc,0))
    hit=(disc>=0)&(near>=0)
    result=np.where(hit,near,np.inf)
    result[np.sum(q*q,axis=1)<=r*r]=0
    return result


def sensor(circles,pose,cfg,rng):
    """环境端生成角度观测。内部 ray casting 只用于可见性，绝不输出真值距离。
    完整可见目标输出 bearing-angle；部分轮廓只输出相机内实际可见角区间。
    """
    half=np.deg2rad(cfg.fov_deg/2)
    grid=np.linspace(-half,half,1001)
    ranges=ray_distances(circles,pose,grid)
    nearest=np.argmin(ranges,axis=0)
    nohit=np.min(ranges,axis=0)>cfg.sensor_range
    nearest[nohit]=-1
    expected,valid=observe(circles,pose)
    detections={};partial=[]
    for i,(beta,theta) in enumerate(expected):
        in_fov=abs(beta)+theta/2 < half
        in_range=np.linalg.norm(circles[i,:2]-pose[:2]) < cfg.sensor_range
        footprint=(grid>=beta-theta/2)&(grid<=beta+theta/2)
        complete=(valid[i] and in_fov and in_range and footprint.any() and
                  (not cfg.occlusion or np.all(nearest[footprint]==i)))
        if complete:
            z=expected[i]+rng.normal(0,np.deg2rad([cfg.noise_beta_deg,cfg.noise_theta_deg]))
            z[0]=wrap(z[0]);detections[str(i)]=z
        else:
            visible=(nearest==i) if cfg.occlusion else (np.isfinite(ranges[i])&(ranges[i]<cfg.sensor_range))
            # 相机能看到的残缺轮廓作为禁行角扇区，避免忽略未初始化的部分目标。
            edges=np.diff(np.r_[False,visible,False].astype(int))
            for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                partial.append((grid[lo]-.5*(grid[1]-grid[0]),grid[hi-1]+.5*(grid[1]-grid[0])))
    return AngularObservation(detections,partial)


class Tracker:
    def __init__(self,cfg):
        self.cfg=cfg;self.filters={};self.first_seen={}
        self.rng=np.random.default_rng(cfg.seed+1000)

    def update(self,pose,obs,t):
        for key,z in obs.detections.items():
            if key not in self.filters:
                self.filters[key]=BearingAnglePF(n=self.cfg.particles,
                    seed=int(self.rng.integers(2**31)),
                    sigma_deg=(self.cfg.noise_beta_deg,self.cfg.noise_theta_deg),
                    distance_prior=self.cfg.distance_prior,
                    diameter_prior=self.cfg.diameter_prior,mh_steps=3)
                self.first_seen[key]=t
            self.filters[key].update(pose,z)

    def sample_maps(self):
        """shape=(地图样本数,已发现目标数,3)，相同地图用于所有候选。"""
        if not self.filters:return np.empty((self.cfg.map_samples,0,3))
        result=[]
        for pf in self.filters.values():
            idx=self.rng.choice(pf.n,self.cfg.map_samples,p=pf.weights)
            result.append(pf.particles[idx])
        return np.stack(result,axis=1)

    def means(self):
        return {k:np.average(pf.particles,axis=0,weights=pf.weights) for k,pf in self.filters.items()}


def segment_point_distances(a,b,points):
    """支持 points=(...,2)，精确线段距离，不只采样路径端点。"""
    ab=b-a
    fraction=np.clip(np.sum((points-a)*ab,axis=-1)/max(float(ab@ab),1e-15),0,1)
    nearest=a+fraction[...,None]*ab
    return np.linalg.norm(points-nearest,axis=-1)


def candidate_risks(position,endpoints,maps,cfg):
    """返回每条线段的地图碰撞比例与净空5%分位数。无真值参数。"""
    if maps.shape[1]==0:
        return np.zeros(len(endpoints)),np.full(len(endpoints),np.inf)
    # 批处理 K条候选 × S张地图 × N个已知障碍物。
    ab=endpoints-position
    q=maps[None,:,:,:2]-position
    fraction=np.clip(np.sum(q*ab[:,None,None,:],axis=-1)/np.maximum(np.sum(ab*ab,axis=1)[:,None,None],1e-15),0,1)
    nearest=position+fraction[:,:,:,None]*ab[:,None,None,:]
    clearance=np.linalg.norm(maps[None,:,:,:2]-nearest,axis=-1)-maps[None,:,:,2]/2-cfg.robot_radius-cfg.safety_margin
    per_map=np.min(clearance,axis=2)
    return np.mean(per_map<=0,axis=1),np.quantile(per_map,.05,axis=1)


def plan(pose,goal,previous_velocity,maps,obs,cfg):
    """局部规划器只看目标、位姿、估计地图与角度观测，绝不接收真值场景。
    原语为恒速直线轨迹；限制世界参考+x方向前进。相机朝向跟随所选速度。
    """
    speeds=np.array([.12,.24,.36])
    headings=np.deg2rad(np.linspace(-75,75,31))
    velocity=np.array([[s*np.cos(h),s*np.sin(h)] for s in speeds for h in headings])
    # 加入直接朝终点的原语（仅向前），便于最后收敛。
    to_goal=goal-pose[:2]
    if to_goal[0]>=0 and np.linalg.norm(to_goal)>1e-8:
        direct=to_goal/np.linalg.norm(to_goal)
        velocity=np.vstack((velocity,[min(.36,np.linalg.norm(to_goal)/cfg.dt)*direct]))
    endpoints=pose[:2]+cfg.horizon*velocity
    bearing=wrap(np.arctan2(velocity[:,1],velocity[:,0])-pose[2])
    permitted=np.abs(bearing)<=np.deg2rad(cfg.fov_deg/2-12)
    permitted &= np.abs(endpoints[:,1])<cfg.corridor_half_width-cfg.robot_radius-cfg.safety_margin
    permitted &= endpoints[:,0]<=goal[0]+.25
    # 已知部分轮廓的方向不进入；额外角余量为启发式，不是完整自由空间证明。
    for lo,hi in obs.partial_sectors:
        permitted &= ~((bearing>=lo-np.deg2rad(8))&(bearing<=hi+np.deg2rad(8)))
    risk,clearance=candidate_risks(pose[:2],endpoints,maps,cfg)
    permitted &= risk<=cfg.risk_threshold
    lookahead=np.array([min(goal[0],pose[0]+1.6),0.])
    # 不含信息熵/估计精度奖励；只优化进度、参考偏离、平滑与碰撞风险。
    cost=(np.linalg.norm(endpoints-lookahead,axis=1)+.10*endpoints[:,1]**2
          +.30*np.sum((velocity-previous_velocity)**2,axis=1)+2.0*risk
          +.025/np.maximum(clearance+.15,.05))
    cost[~permitted]=np.inf
    # 停止的代价；运动候选必须优于停止，或没有候选时保持。
    stop_cost=np.linalg.norm(pose[:2]-lookahead)+.10*pose[1]**2+.03
    best=int(np.argmin(cost))
    if not np.isfinite(cost[best]) or cost[best]>=stop_cost:
        hold_risk,hold_clearance=candidate_risks(pose[:2],pose[:2][None,:],maps,cfg)
        return np.zeros(2),dict(risk=float(hold_risk[0]),
            clearance_5p_m=float(hold_clearance[0]),
            feasible=int(permitted.sum()),planned_endpoint=pose[:2].copy(),
            stopped=True,reason='no_feasible' if not permitted.any() else 'cost_prefers_hold')
    return velocity[best],dict(risk=float(risk[best]),
        clearance_5p_m=float(clearance[best]),feasible=int(permitted.sum()),
        planned_endpoint=endpoints[best],stopped=False,reason='moving')


def true_clearance(circles,a,b,robot_radius):
    """仅用于环境终止条件/评估，不能用来选择动作。"""
    return float(np.min(segment_point_distances(a,b,circles[:,:2])-circles[:,2]/2-robot_radius))


def run(cfg,output,make_gif=False,plots=True):
    start_time=time.perf_counter()
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    circles,start,goal=make_scene(cfg.seed)
    pose=np.r_[start,0.];tracker=Tracker(cfg);rng=np.random.default_rng(cfg.seed+2000)
    previous_velocity=np.zeros(2);trajectory=[pose.copy()];records=[];snapshots=[];waits=0
    status='timeout';minimum_clearance=np.inf;stop_reasons={'no_feasible':0,'cost_prefers_hold':0}
    for step in range(cfg.max_steps):
        obs=sensor(circles,pose,cfg,rng)
        tracker.update(pose,obs,step)
        maps=tracker.sample_maps()
        velocity,info=plan(pose,goal,previous_velocity,maps,obs,cfg)
        next_position=pose[:2]+cfg.dt*velocity
        clearance=true_clearance(circles,pose[:2],next_position,cfg.robot_radius)
        minimum_clearance=min(minimum_clearance,clearance)
        next_yaw=np.arctan2(velocity[1],velocity[0]) if np.linalg.norm(velocity)>1e-8 else pose[2]
        snapshots.append(dict(pose=pose.copy(),means=tracker.means(),visible=list(obs.detections),endpoint=info['planned_endpoint'].copy()))
        records.append([step,step*cfg.dt,*pose,*velocity,info['risk'],info['feasible'],len(tracker.filters),len(obs.detections),clearance,int(info['stopped'])])
        pose=np.r_[next_position,next_yaw];trajectory.append(pose.copy());previous_velocity=velocity
        waits=waits+1 if info['stopped'] else 0
        if info['stopped']:stop_reasons[info['reason']]+=1
        if clearance<=0:status='collision';break
        if np.linalg.norm(pose[:2]-goal)<=cfg.goal_tolerance:status='reached';break
        if waits>=18:status='stalled';break
    trajectory=np.array(trajectory);records=np.array(records)
    target_errors=[]
    for key,estimate in tracker.means().items():
        gt=circles[int(key)]
        target_errors.append(dict(id=key,first_seen_step=tracker.first_seen[key],
          center_error_m=float(np.linalg.norm(estimate[:2]-gt[:2])),diameter_error_m=float(abs(estimate[2]-gt[2]))))
    straight_dist=segment_point_distances(start,goal,circles[:,:2])-circles[:,2]/2-cfg.robot_radius
    summary=dict(status=status,seed=cfg.seed,steps=len(records),simulated_time_s=len(records)*cfg.dt,
                 wall_time_s=time.perf_counter()-start_time,goal_distance_m=float(np.linalg.norm(pose[:2]-goal)),
                 path_length_m=float(np.sum(np.linalg.norm(np.diff(trajectory[:,:2],axis=0),axis=1))),
                 minimum_true_surface_clearance_m=float(minimum_clearance),
                 obstacle_count=len(circles),discovered_count=len(tracker.filters),
                 straight_reference_collision_count=int(np.sum(straight_dist<=0)),
                 stops=int(records[:,-1].sum()),max_selected_known_map_risk=float(records[:,7].max()),
                 config=asdict(cfg),target_errors=target_errors,stop_reasons=stop_reasons)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savetxt(out/'history.csv',records,delimiter=',',comments='',header='step,time,px,py,yaw,vx,vy,known_map_risk,feasible_candidates,discovered,visible,true_surface_clearance,stopped')
    np.savez_compressed(out/'episode.npz',trajectory=trajectory,truth=circles,start=start,goal=goal)
    if plots:plot_result(out,cfg,circles,goal,trajectory,records,snapshots,summary)
    if make_gif:animate(out,cfg,circles,goal,trajectory,snapshots,summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','target_errors')},indent=2),flush=True)
    return summary


def plot_result(out,cfg,circles,goal,traj,records,snapshots,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    fig=plt.figure(figsize=(13,9),constrained_layout=True)
    gs=fig.add_gridspec(2,2,height_ratios=[1.35,1])
    ax=fig.add_subplot(gs[0,:])
    draw_map(ax,cfg,circles,goal)
    ax.plot(traj[:,0],traj[:,1],color='#1565C0',lw=2.4,label='Executed trajectory')
    for k in [0,len(snapshots)//2,len(snapshots)-1]:
        p=snapshots[k]['pose'];yaw=np.rad2deg(p[2])
        ax.add_patch(Wedge(p[:2],.85,yaw-cfg.fov_deg/2,yaw+cfg.fov_deg/2,color='#1565C0',alpha=.08))
        ax.scatter(*p[:2],color='#1565C0',s=25)
    for key,state in snapshots[-1]['means'].items():
        ax.add_patch(Circle(state[:2],state[2]/2,fill=False,edgecolor='#D97706',ls='--',lw=1.3))
    ax.plot([],[],color='#D97706',ls='--',label='Final PF mean circles')
    final_states=list(snapshots[-1]['means'].values())
    if final_states:
        low=min(-2.7,min(c[1]-c[2]/2-.2 for c in final_states))
        high=max(2.7,max(c[1]+c[2]/2+.2 for c in final_states))
        ax.set_ylim(low,high)
    ax.legend(loc='upper right',ncol=2,fontsize=8)
    ax.set_title(f"Bearing-angle PF avoidance | {summary['status']} | seed {cfg.seed}")
    ax2=fig.add_subplot(gs[1,0])
    ax2.plot(records[:,1],records[:,11],label='True surface clearance (evaluation only)',color='#16856B')
    ax2.axhline(0,color='#CC3344',ls='--');ax2.axhline(cfg.safety_margin,color='gray',ls=':',label='Nominal safety margin')
    ax2.set(xlabel='Simulation time (s)',ylabel='Clearance (m)',title='Continuous executed-segment clearance');ax2.legend(fontsize=8);ax2.grid(alpha=.2)
    ax3=fig.add_subplot(gs[1,1])
    ax3.plot(records[:,1],records[:,9],label='Discovered IDs',color='#1565C0')
    ax3.plot(records[:,1],records[:,10],label='Current complete observations',color='#D97706')
    ax3.set(xlabel='Simulation time (s)',ylabel='Target count',title='Finite-FOV bearing-angle observations');ax3.legend(fontsize=8);ax3.grid(alpha=.2)
    fig.savefig(out/'avoidance_result.png',dpi=155);plt.close(fig)


def draw_map(ax,cfg,circles,goal):
    from matplotlib.patches import Circle
    for i,c in enumerate(circles):
        ax.add_patch(Circle(c[:2],c[2]/2+cfg.robot_radius,facecolor='#B0B0B0',alpha=.18,edgecolor='none'))
        ax.add_patch(Circle(c[:2],c[2]/2,facecolor='#555555',edgecolor='none',alpha=.85))
        ax.text(c[0],c[1],str(i),ha='center',va='center',color='white',fontsize=8)
    ax.plot([0,goal[0]],[0,0],'--',color='#888888',label='Straight reference')
    ax.scatter(0,0,c='#16856B',s=75,marker='o',label='Start')
    ax.scatter(*goal,c='#C93A48',s=110,marker='*',label='Goal')
    ax.set(xlim=(-.5,10),ylim=(-2.7,2.7),xlabel='World x (m)',ylabel='World y (m)')
    ax.set_aspect('equal');ax.grid(alpha=.12)


def animate(out,cfg,circles,goal,traj,snapshots,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    def update(index):
        ax.clear();draw_map(ax,cfg,circles,goal)
        snap=snapshots[min(index,len(snapshots)-1)];pose=traj[index];yaw=np.rad2deg(pose[2])
        ax.add_patch(Wedge(pose[:2],1.3,yaw-cfg.fov_deg/2,yaw+cfg.fov_deg/2,color='#1565C0',alpha=.10))
        ax.plot(traj[:index+1,0],traj[:index+1,1],color='#1565C0',lw=2)
        ax.add_patch(Circle(pose[:2],cfg.robot_radius,color='#1565C0'))
        endpoint=snap['endpoint'] if index<len(snapshots) else pose[:2]
        ax.plot([pose[0],endpoint[0]],[pose[1],endpoint[1]],color='#16856B',lw=2,ls='--')
        for key,c in snap['means'].items():
            ax.add_patch(Circle(c[:2],c[2]/2,fill=False,edgecolor='#E18A15',ls='--'))
        label=summary['status'] if index==len(snapshots) else f"visible {len(snap['visible'])}"
        ax.set_title(f"Step {index} | known {len(snap['means'])} | {label} | orange: PF estimates")
    ids=list(range(0,len(snapshots),2))
    ids.append(len(snapshots))
    animation=FuncAnimation(fig,update,frames=ids,interval=100)
    animation.save(out/'avoidance.gif',writer=PillowWriter(fps=10),dpi=85)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--particles',type=int,default=1000)
    parser.add_argument('--map-samples',type=int,default=180)
    parser.add_argument('--max-steps',type=int,default=220)
    parser.add_argument('--output',default='results')
    parser.add_argument('--gif',action='store_true')
    parser.add_argument('--no-occlusion',action='store_true')
    args=parser.parse_args()
    if min(args.particles,args.map_samples)<50:parser.error('particles and map-samples must be >=50')
    cfg=Config(seed=args.seed,particles=args.particles,map_samples=args.map_samples,max_steps=args.max_steps,occlusion=not args.no_occlusion)
    run(cfg,args.output,args.gif)
