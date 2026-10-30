"""Run: python test_core.py"""
import numpy as np
from avoidance_demo import (Config,Tracker,AngularObservation,plan,candidate_risks,
                            segment_point_distances,sensor,true_clearance)

cfg=Config(particles=500,map_samples=100)
# 线段中间相交：端点本身无碰撞，仍必须检测整段穿越。
assert np.isclose(segment_point_distances(np.array([0.,0.]),np.array([4.,0.]),np.array([[2.,.5]]))[0],.5)
maps=np.repeat(np.array([[[2.,0.,1.]]]),100,axis=0)
risk,cl=candidate_risks(np.array([0.,0.]),np.array([[4.,0.],[0.,2.]]),maps,cfg)
assert risk[0]==1 and risk[1]==0
# 空后验没有“已知目标碰撞”，但不表示证明整个未知空间安全。
risk,_=candidate_risks(np.zeros(2),np.array([[1.,0.]]),np.empty((100,0,3)),cfg)
assert risk[0]==0
# 遮挡时不能泄漏被遮挡目标的完整观测；只有角度，没有真值距离。
obs=sensor(np.array([[2.,0.,1.],[4.,0.,1.]]),np.zeros(3),cfg,np.random.default_rng(1))
assert '0' in obs.detections and '1' not in obs.detections
assert all(z.shape==(2,) for z in obs.detections.values())
# 不完整的轮廓提供角禁区，不提供错误完整角宽。
obs=sensor(np.array([[1.,2.5,1.2]]),np.zeros(3),cfg,np.random.default_rng(1))
assert len(obs.partial_sectors)>0 and len(obs.detections)==0
# 无观测保持；同ID复用。
t=Tracker(cfg);z=AngularObservation({'target':np.array([.2,.3])},[])
t.update(np.zeros(3),z,0);pf=t.filters['target'];old=pf.particles.copy();w=pf.weights.copy()
t.update(np.array([.1,0,0]),AngularObservation({},[]),1)
assert np.array_equal(old,pf.particles) and np.array_equal(w,pf.weights)
t.update(np.array([.1,0,0]),z,2);assert t.filters['target'] is pf
# 规划器接口不接收真值，只接收采样后验；选择的方向位于当前前视范围。
vel,info=plan(np.zeros(3),np.array([9.5,0.]),np.zeros(2),maps,AngularObservation({},[]),cfg)
assert vel[0]>=0
assert abs(np.arctan2(vel[1],vel[0]))<=np.deg2rad(cfg.fov_deg/2-12)+1e-9
assert info['stopped'] or info['risk']<=cfg.risk_threshold
# 已执行线段的真值碰撞仅由评估器检查。
assert true_clearance(np.array([[2.,0.,1.]]),np.array([0.,0.]),np.array([4.,0.]),cfg.robot_radius)<0
print('PASS: geometry, continuous collision, occlusion, truncation, ID retention, planner gating')
