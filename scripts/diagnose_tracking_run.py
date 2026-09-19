"""Read-only comparison of logged MPC tracking, timing and depth.

Run with python3 scripts/diagnose_tracking_run.py RUN_DIR [RUN_DIR ...].
Uses the logged online index-zero reference, not a re-zeroed schedule.
"""
import bisect
import csv
import json
import math
import pathlib
import statistics
import sys


def read(path):
    with path.open() as stream:
        return [{k: float(v) for k, v in row.items()
                 if k not in ('reference_source', 'state', 'solver_success')}
                for row in csv.DictReader(stream)]


def stats(a):
    if not a:
        return None
    s = sorted(abs(x) for x in a)
    return dict(n=len(a), mean=statistics.mean(a),
                rmse=math.sqrt(statistics.mean(x*x for x in a)),
                p95_abs=s[int(.95*(len(s)-1))], max_abs=max(s))


def interp(t, ts, vs):
    j = bisect.bisect_right(ts, t)
    if j == 0:
        return vs[0]
    if j == len(ts):
        return vs[-1]
    f = (t-ts[j-1])/(ts[j]-ts[j-1])
    return vs[j-1]*(1-f)+vs[j]*f


def project(x, y, path, arcs):
    best = (float('inf'), 0.)
    for i, (a, b) in enumerate(zip(path, path[1:])):
        dx, dy = b['x_ned']-a['x_ned'], b['y_ned']-a['y_ned']
        length = math.hypot(dx, dy)
        q = max(0., min(1., ((x-a['x_ned'])*dx+(y-a['y_ned'])*dy)/length**2))
        distance = math.hypot(x-a['x_ned']-q*dx, y-a['y_ned']-q*dy)
        if distance < best[0]:
            best = distance, arcs[i]+q*length
    return best


def analyze(directory):
    p = pathlib.Path(directory)
    odom = read(p/'odometry.csv')
    refs = [r for r in read(p/'mpc_reference_window.csv') if r['index'] == 0]
    ev = read(p/'evaluation_reference.csv')
    path = read(p/'global_reference_path.csv')
    status = read(p/'controller_status.csv')
    forces = read(p/'generalized_force.csv')
    arcs = [0.]
    for a, b in zip(path, path[1:]):
        arcs.append(arcs[-1]+math.hypot(b['x_ned']-a['x_ned'], b['y_ned']-a['y_ned']))
    t0 = odom[0]['stamp']
    ts = [r['stamp'] for r in odom]
    os = [r['actual_arc_m'] for r in ev]
    splan = [r['scheduled_arc_m'] for r in ev]
    tplan = [r['t_rel'] for r in ev]
    errors, offset, yaw = [], [], []
    for r in refs:
        t = r['stamp']
        if not ts[0] <= t <= ts[-1]:
            continue
        x = interp(t, ts, [a['x'] for a in odom])
        y = interp(t, ts, [a['y'] for a in odom])
        errors.append(math.hypot(x-r['x_ned'], y-r['y_ned']))
        # Use wrapped interpolation for yaw (these runs do not cross +/- pi).
        a = interp(t, ts, [a['yaw'] for a in odom])-r['yaw_ned']
        yaw.append(math.degrees(math.atan2(math.sin(a), math.cos(a))))
        s = project(r['x_ned'], r['y_ned'], path, arcs)[1]
        if 1. < s < min(splan[-1], arcs[-1])-1.:
            offset.append(interp(s, splan, tplan)-(t-t0))
    groups = {}
    for lo, hi in [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60)]:
        ids = [i for i, r in enumerate(odom) if lo <= r['stamp']-t0 < hi]
        if not ids:
            continue
        groups[f'{lo}-{hi}s'] = dict(
            depth_min=min(odom[i]['z'] for i in ids), depth_max=max(odom[i]['z'] for i in ids),
            cross_track=stats([project(odom[i]['x'],odom[i]['y'],path,arcs)[0] for i in ids]),
            status_position=stats([r['position_error_m'] for r in status if lo<=r['stamp']-t0<hi]))
    return dict(run=p.name, first_odom=odom[0], last_odom=odom[-1],
                online_position=stats(errors), online_yaw_deg=stats(yaw),
                schedule_offset_from_online_reference=stats(offset),
                status_position=stats([r['position_error_m'] for r in status if ts[0]<=r['stamp']<=ts[-1]]),
                first_time_depth_below={str(z):next((r['stamp']-t0 for r in odom if r['z']<z),None)
                                        for z in [.8,.5,.3,.2,.15,.1]},
                groups=groups, force_columns=list(forces[0]),
                force_stats={k:stats([r[k] for r in forces]) for k in forces[0] if k!='stamp'})


if __name__ == '__main__':
    print(json.dumps([analyze(d) for d in sys.argv[1:]], indent=2))
