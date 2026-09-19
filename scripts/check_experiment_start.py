"""Verify saved start handshake and scoring epoch: python3 SCRIPT RUN_DIR."""
import csv
import json
import math
import pathlib
import sys


def check(directory):
    p = pathlib.Path(directory)
    meta = json.loads((p / 'experiment_start.json').read_text())
    t0 = meta['start_stamp']

    def rows(name):
        with (p / name).open() as stream:
            return [{k: float(v) for k, v in row.items() if k != 'reference_source'}
                    for row in csv.DictReader(stream)]

    odom = rows('odometry.csv')
    pre = rows('prestart_wrench.csv')
    evaluation = rows('evaluation_reference.csv')
    assert all(meta['acks'][role] == t0 for role in ('reference', 'controller'))
    assert pre and all(r['stamp'] < t0 for r in pre)
    assert all(r[k] == 0 for r in pre for k in ('fx', 'fy', 'nz'))
    assert all(r['stamp'] >= t0 for r in odom)
    assert 0 <= odom[0]['stamp'] - t0 < .1
    assert odom[0]['speed_world'] < .01
    assert all(abs(r['t_rel'] - (r['stamp'] - t0)) < 1e-8 for r in evaluation)
    assert all(abs(r['t_rel'] - (r['stamp'] - t0)) < 1e-8 for r in odom)
    assert len(odom) == len(evaluation)
    assert math.hypot(odom[0]['x'] - evaluation[0]['x_ref'],
                      odom[0]['y'] - evaluation[0]['y_ref']) < .02
    result = dict(start_stamp=t0, first_sample_delay_s=odom[0]['stamp']-t0,
                  first_speed_mps=odom[0]['speed_world'],
                  prestart_wrench_samples=len(pre), prestart_horizontal_force_max=0.,
                  depth_min_m=min(r['z'] for r in odom),
                  depth_max_m=max(r['z'] for r in odom), checks_passed=True)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    check(sys.argv[1])
