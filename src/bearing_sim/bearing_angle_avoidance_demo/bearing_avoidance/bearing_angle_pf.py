#!/usr/bin/env python3
"""Bearing-angle 静态圆障碍物粒子滤波示例，仅依赖 numpy / matplotlib。

状态 = [圆心世界坐标 cx, cy, 直径 L]，观测 = [bearing beta, 完整角宽 theta]。
内部角度全部是弧度；机器人/相机位姿已知。没有给滤波器传入真值。
用 resample-move SMC 处理静态参数，MH 移动使用完整观测历史的后验，
而不是给静态障碍物注入任意过程噪声。这里只演示单个已关联目标。
运行: python bearing_angle_pf.py --output results
"""
from pathlib import Path
import argparse
import json
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def observe(states, pose):
    """几何正向模型，可接收一个或多个状态；有效性要求相机位于圆外。"""
    states = np.atleast_2d(states)
    delta = states[:, :2] - np.asarray(pose[:2])
    d = np.linalg.norm(delta, axis=1)
    valid = (states[:, 2] > 0) & (d > states[:, 2] / 2)
    beta = wrap(np.arctan2(delta[:, 1], delta[:, 0]) - pose[2])
    theta = 2 * np.arcsin(np.clip(states[:, 2] / (2 * np.maximum(d, 1e-12)), 0, 1))
    return np.column_stack((beta, theta)), valid


def visible_sector(z):
    """``[beta, theta]`` of the visible part of a complete or truncated observation."""
    if len(z) == 2:
        return float(z[0]), float(z[1])
    edge, bound = float(z[0]), float(z[1])
    return 0.5 * (edge + bound), abs(bound - edge)


def log_likelihood(states, pose, z, sigma):
    """Per-state log-likelihood; ``-inf`` where the camera is inside the circle.

    ``z=[beta, theta]`` is a complete silhouette.  ``z=[edge, bound, side]``
    is a silhouette cut by the image side: ``edge`` is the true tangent
    bearing inside the image, the pillar continues past the image edge
    ``bound`` toward ``side`` (+1: larger bearings, -1: smaller).  Only the
    tangent is measured; the far tangent must lie beyond ``bound``.  Tangent
    rays from several poses still determine centre and diameter.
    """
    predicted, valid = observe(states, pose)
    if len(z) == 2:
        residual = predicted - z
        residual[:, 0] = wrap(residual[:, 0])
        ll = -0.5 * np.sum((residual / sigma)**2, axis=1)
    else:
        edge, bound, side = z
        near = predicted[:, 0] - side * predicted[:, 1] / 2
        far = predicted[:, 0] + side * predicted[:, 1] / 2
        short = np.minimum(side * wrap(far - bound), 0.)
        ll = -0.5 * ((wrap(near - edge) / sigma[0])**2 + (short / sigma[0])**2)
    ll[~valid] = -np.inf
    return ll


def normalize_logweights(logw):
    if not np.any(np.isfinite(logw)):
        raise RuntimeError('所有粒子失效：检查先验覆盖、角度单位和观测噪声。')
    w = np.exp(logw - np.max(logw))
    return w / w.sum()


def systematic_resample(weights, rng):
    n = len(weights)
    points = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, points)


def weighted_quantiles(values, weights, qs=(0.025, 0.975)):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return np.interp(qs, cumulative, values[order])


class BearingAnglePF:
    def __init__(self, n=5000, seed=42, sigma_deg=(0.6, 0.35),
                 distance_prior=(1., 6.), diameter_prior=(0.2, 1.6), mh_steps=4):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.sigma = np.deg2rad(sigma_deg)
        self.distance_prior = distance_prior
        self.diameter_prior = diameter_prior
        self.mh_steps = mh_steps
        self.history = []
        self.particles = None
        self.weights = None
        self.anchor = None
        self.last_ess = np.nan
        self.acceptance = np.nan
        self.resamples = 0

    def initialize(self, pose, z):
        """首帧引导提议分布，不将第一帧似然重复乘两次。

        原始先验：首帧中心距离 d 均匀、世界方位 gamma 均匀、直径 L 均匀。
        提议：d 均匀，beta/theta 围绕观测按测量噪声采样。
        c = anchor + d*g; L = 2*d*sin(theta/2)。
        由变量变换 dL/dtheta = d*cos(theta/2)，重要性权重正比
        d*cos(theta/2)，并施加先验边界。不是随意给所有粒子相同权重。
        """
        self.anchor = np.asarray(pose[:2], dtype=float).copy()
        if len(z) == 3:
            self._initialize_truncated(pose, z)
            return
        d = self.rng.uniform(*self.distance_prior, self.n)
        beta = self.rng.normal(z[0], self.sigma[0], self.n)
        theta = self.rng.normal(z[1], self.sigma[1], self.n)
        gamma = beta + pose[2]
        diameter = 2 * d * np.sin(theta / 2)
        self.particles = np.column_stack((self.anchor[0] + d*np.cos(gamma),
                                         self.anchor[1] + d*np.sin(gamma), diameter))
        valid = ((theta > 0) & (theta < np.pi) &
                 (diameter >= self.diameter_prior[0]) &
                 (diameter <= self.diameter_prior[1]))
        w = np.where(valid, d * np.cos(theta / 2), 0.)
        if w.sum() <= 0:
            raise RuntimeError('先验与首帧观测不相容。')
        self.weights = w / w.sum()
        self.history.append((np.asarray(pose).copy(), np.asarray(z).copy()))
        self._maybe_resample()

    def _initialize_truncated(self, pose, z):
        """First observation cut by the image side (see :func:`log_likelihood`).

        Draw d and L from the prior, restricted to angular widths that reach
        the image edge (L >= 2 d sin(w/2), w = visible width), and place the
        centre so that the near tangent is the measured edge plus bearing
        noise.  The proposal then cancels prior x tangent likelihood, so the
        weights are the truncation masses of the restricted prior.
        """
        edge, bound, side = z
        width = abs(wrap(bound - edge))
        lo, hi = self.diameter_prior
        d_max = min(self.distance_prior[1], hi / (2 * max(np.sin(width / 2), 1e-9)))
        if d_max <= self.distance_prior[0]:
            raise RuntimeError('先验与首帧截断观测不相容。')
        d = self.rng.uniform(self.distance_prior[0], d_max, self.n)
        l_min = np.clip(2 * d * np.sin(width / 2), lo, hi)
        diameter = self.rng.uniform(l_min, hi)
        theta = 2 * np.arcsin(np.clip(diameter / (2 * d), 0, 1))
        gamma = pose[2] + self.rng.normal(edge, self.sigma[0], self.n) + side * theta / 2
        self.particles = np.column_stack((self.anchor[0] + d*np.cos(gamma),
                                         self.anchor[1] + d*np.sin(gamma), diameter))
        near = gamma - pose[2] - side * theta / 2
        short = np.minimum(side * wrap(near + side * theta - bound), 0.)
        w = (hi - l_min) * (diameter < 2 * d) * np.exp(-0.5 * (short / self.sigma[0])**2)
        if w.sum() <= 0:
            raise RuntimeError('先验与首帧截断观测不相容。')
        self.weights = w / w.sum()
        self.history.append((np.asarray(pose).copy(), np.asarray(z, dtype=float).copy()))
        self._maybe_resample()

    def update(self, pose, z):
        """z=None 表示无有效观测；静态状态与权重保持不变。"""
        self.acceptance = np.nan
        if z is None:
            return
        if self.particles is None:
            self.initialize(pose, z)
            return
        ll = log_likelihood(self.particles, pose, z, self.sigma)
        with np.errstate(divide='ignore'):
            self.weights = normalize_logweights(np.log(self.weights) + ll)
        self.history.append((np.asarray(pose).copy(), np.asarray(z).copy()))
        self._maybe_resample()

    def _logposterior(self, states):
        """MH 的完整目标分布：原始先验 × 截至目前每一帧有效似然。
        极坐标均匀距离先验在 Cartesian 坐标下的密度正比 1/d。
        """
        d0 = np.linalg.norm(states[:, :2] - self.anchor, axis=1)
        good = ((d0 >= self.distance_prior[0]) & (d0 <= self.distance_prior[1]) &
                (states[:, 2] >= self.diameter_prior[0]) &
                (states[:, 2] <= self.diameter_prior[1]))
        value = -np.log(np.maximum(d0, 1e-12))
        for pose, z in self.history:
            ll = log_likelihood(states, pose, z, self.sigma)
            good &= np.isfinite(ll)
            value += np.where(np.isfinite(ll), ll, 0.)
        return np.where(good, value, -np.inf)

    def _maybe_resample(self):
        ess = 1. / np.sum(self.weights**2)
        self.last_ess = ess  # 保存重采样前的 ESS，便于诊断。
        if ess >= 0.55 * self.n:
            return
        mean = np.average(self.particles, axis=0, weights=self.weights)
        delta = self.particles - mean
        cov = (delta * self.weights[:, None]).T @ delta
        # 在本次 MH 核内固定对称提议协方差；保留位置-尺寸相关性。
        chol = np.linalg.cholesky(0.65**2 * cov + np.eye(3)*1e-10)
        self.particles = self.particles[systematic_resample(self.weights, self.rng)].copy()
        self.weights = np.full(self.n, 1. / self.n)
        self.resamples += 1
        current = self._logposterior(self.particles)
        accepted = 0
        for _ in range(self.mh_steps):
            proposal = self.particles + self.rng.normal(size=(self.n, 3)) @ chol.T
            candidate = self._logposterior(proposal)
            accept = np.log(self.rng.random(self.n)) < candidate - current
            self.particles[accept] = proposal[accept]
            current[accept] = candidate[accept]
            accepted += accept.sum()
        self.acceptance = accepted / (self.n * self.mh_steps)

    def estimate(self):
        mean = np.average(self.particles, axis=0, weights=self.weights)
        intervals = np.array([weighted_quantiles(self.particles[:, j], self.weights)
                              for j in range(3)])
        return mean, intervals


def simulate(seed=7, frames=65, fov_deg=90):
    """真值只属于环境与评估器；滤波器接口只接收 pose 和 z。"""
    rng = np.random.default_rng(seed)
    truth = np.array([3.2, 1.1, 0.7])
    t = np.linspace(0, 1, frames)
    xy = np.column_stack((1.8*t, 0.35*np.sin(np.pi*t)))
    yaw = np.arctan2(1.-xy[:, 1], 4.-xy[:, 0])  # 面向一个预先给定的航向参考点。
    yaw[(t > 0.40) & (t < 0.56)] += np.deg2rad(110)  # 主动转头，制造暂时失视。
    poses = np.column_stack((xy, yaw))
    observations = []
    for pose in poses:
        z, valid = observe(truth, pose)
        beta, theta = z[0]
        visible = valid[0] and abs(beta) + theta/2 <= np.deg2rad(fov_deg)/2
        if visible:
            noisy = z[0] + rng.normal(0, np.deg2rad([0.6, 0.35]))
            noisy[0] = wrap(noisy[0])
            observations.append(noisy)
        else:
            observations.append(None)
    return truth, poses, observations


def run(n=5000, seed=7, output='results'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    truth, poses, observations = simulate(seed=seed)
    pf = BearingAnglePF(n=n, seed=seed+100)
    estimates, bounds, ess, visible, snapshots = [], [], [], [], {}
    for k, (pose, z) in enumerate(zip(poses, observations)):
        previous = None if pf.particles is None else pf.particles.copy()
        previous_w = None if pf.weights is None else pf.weights.copy()
        pf.update(pose, z)
        if z is None and previous is not None:
            assert np.array_equal(previous, pf.particles)
            assert np.array_equal(previous_w, pf.weights)
        mean, interval = pf.estimate()
        assert np.isfinite(pf.particles).all()
        assert np.isclose(pf.weights.sum(), 1.) and np.all(pf.weights >= 0)
        estimates.append(mean); bounds.append(interval)
        ess.append(pf.last_ess); visible.append(z is not None)
        if k in (0, 10, len(poses)-1):
            snapshots[k] = (pf.particles.copy(), pf.weights.copy())
    estimates, bounds = np.array(estimates), np.array(bounds)
    center_error = np.linalg.norm(estimates[:, :2]-truth[:2], axis=1)
    summary = {
        'seed': seed, 'particles': n, 'frames': len(poses),
        'valid_observations': int(sum(visible)), 'resampling_events': pf.resamples,
        'true_state_cx_cy_diameter_m': truth.tolist(),
        'final_estimate_m': estimates[-1].tolist(),
        'final_center_error_m': float(center_error[-1]),
        'final_diameter_error_m': float(abs(estimates[-1, 2]-truth[2])),
        'final_marginal_95_percent_intervals_m': bounds[-1].tolist(),
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    table = np.column_stack((np.arange(len(poses)), poses, visible, estimates,
                             bounds[:, :, 0], bounds[:, :, 1], center_error, ess))
    np.savetxt(output/'history.csv', table, delimiter=',', comments='',
               header='frame,px,py,yaw,visible,cx,cy,L,cx_lo,cy_lo,L_lo,cx_hi,cy_hi,L_hi,center_error,ess_before_resample')

    # Keep the estimator importable in ROS nodes that do not install plotting
    # dependencies; plotting is only needed for this standalone demo.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    colors = ['#4C78A8', '#F58518', '#54A24B']
    for ax, (k, (particles, weights)), color in zip(axes[0], snapshots.items(), colors):
        # 绘制按权重重采样的显示点，避免低权重粒子在图中被误读为同等可信。
        ids = np.random.default_rng(123).choice(n, min(n, 1000), p=weights)
        sc = ax.scatter(particles[ids, 0], particles[ids, 1], c=particles[ids, 2],
                        s=8, alpha=0.30, cmap='viridis', vmin=0.2, vmax=1.6)
        ax.add_patch(Circle(truth[:2], truth[2]/2, fill=False, color='black', lw=1.8))
        ax.scatter(*truth[:2], marker='+', s=90, color='black', label='True center')
        ax.plot(poses[:k+1, 0], poses[:k+1, 1], '-', color='#D1495B', lw=2, label='Camera path')
        ax.scatter(*poses[k, :2], marker='^', color='#D1495B', s=50)
        ax.set(xlim=(-0.3, 6.1), ylim=(-0.4, 3.1), xlabel='World x (m)', ylabel='World y (m)',
               title=f'Frame {k}: posterior center samples')
        ax.set_aspect('equal'); ax.grid(alpha=.15)
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.colorbar(sc, ax=axes[0, :], shrink=.65, label='Particle diameter L (m)')
    for j, (ax, label, color) in enumerate(zip(axes[1], ['Center x', 'Center y', 'Diameter L'], colors)):
        ax.fill_between(np.arange(len(poses)), bounds[:, j, 0], bounds[:, j, 1],
                        color=color, alpha=.2, label='95% marginal credible interval')
        ax.plot(estimates[:, j], color=color, lw=2, label='Posterior mean')
        ax.axhline(truth[j], color='black', linestyle='--', label='Ground truth')
        missing = np.flatnonzero(~np.asarray(visible))
        if len(missing):
            ax.axvspan(missing[0]-.5, missing[-1]+.5, color='gray', alpha=.15, label='Out of view')
        ax.set(xlabel='Frame', ylabel='m', title=label); ax.grid(alpha=.15)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Bearing-angle particle filter | static circle, unknown center and diameter', fontsize=15)
    fig.savefig(output/'bearing_angle_pf_result.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    if args.particles < 100:
        parser.error('--particles must be >= 100')
    run(args.particles, args.seed, args.output)
