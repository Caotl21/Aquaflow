'''Generate train/val/test coverage reports for HOFA policy teacher data.'''
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .actuation import ThrusterGeometry
from .contract import SCHEMA


AXES = ('fx', 'fy', 'nz')


def describe(values, signed=False):
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not data.size:
        return None
    result = {
        'count': int(data.size),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'median': float(np.median(data)),
        'p95': float(np.percentile(data, 95)),
        'p99': float(np.percentile(data, 99)),
        'max': float(np.max(data)),
    }
    if signed:
        result.update({
            'p95_abs': float(np.percentile(np.abs(data), 95)),
            'max_abs': float(np.max(np.abs(data))),
            'positive_ratio': float(np.mean(data > 1.0e-6)),
            'negative_ratio': float(np.mean(data < -1.0e-6)),
            'near_zero_ratio': float(np.mean(np.abs(data) <= 1.0e-6)),
        })
    return result


class AllocationChecker:
    '''Check exact force-box feasibility for the 3x4 BricsBot allocation.'''

    def __init__(self, geometry):
        self.geometry = geometry
        self.matrix = geometry.allocation_matrix
        self.pinv = np.linalg.pinv(self.matrix)
        self.null = np.linalg.svd(self.matrix)[2][-1]

    def solve(self, wrench):
        wrench = np.asarray(wrench, dtype=np.float64)
        force_zero = self.pinv @ wrench
        residual = float(np.linalg.norm(self.matrix @ force_zero - wrench))
        tolerance = 1.0e-7 * max(1.0, float(np.linalg.norm(wrench)))
        if residual > tolerance:
            return False, float('inf'), residual
        lower, upper = -float('inf'), float('inf')
        for i, direction in enumerate(self.null):
            if abs(direction) < 1.0e-12:
                if not (self.geometry.force_min[i] - tolerance <= force_zero[i]
                        <= self.geometry.force_max[i] + tolerance):
                    return False, float('inf'), residual
                continue
            a = (self.geometry.force_min[i] - force_zero[i]) / direction
            b = (self.geometry.force_max[i] - force_zero[i]) / direction
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
        if lower > upper + tolerance:
            return False, float('inf'), residual
        base = (force_zero - self.geometry.force_mid) / self.geometry.force_half
        direction = self.null / self.geometry.force_half
        denominator = float(direction @ direction)
        lam = 0.0 if denominator < 1.0e-15 else -float(base @ direction) / denominator
        lam = min(max(lam, lower), upper)
        force = force_zero + self.null * lam
        normalized = (force - self.geometry.force_mid) / self.geometry.force_half
        feasible = bool(np.all(force >= self.geometry.force_min - tolerance)
                        and np.all(force <= self.geometry.force_max + tolerance))
        return feasible, float(np.max(np.abs(normalized))), residual


def read_status(path):
    if not path.exists():
        return {}
    with path.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get('state') == 'ACTIVE']
    if not rows:
        return {}
    def column(name):
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    callback = column('callback_time_ms')
    return {
        'success': np.asarray([row['solver_success'].lower() in ('true', '1')
                               for row in rows], dtype=np.float64),
        'iterations': column('solver_iterations'),
        'layer1_ms': column('layer1_time_ms'),
        'layer2_ms': column('layer2_time_ms'),
        'callback_ms': callback,
        'deadline_miss': callback > 80.0,
    }


def analyse_track(directory, geometry, checker):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata['contract']['schema'] != SCHEMA:
        raise ValueError('contract mismatch: %s' % directory)
    data = np.load(directory / 'samples.npz', allow_pickle=False)
    state = np.asarray(data['state'], dtype=np.float64)
    reference = np.asarray(data['reference'], dtype=np.float64)
    wrench = np.asarray(data['target_wrench'], dtype=np.float64)
    if len(wrench) != metadata['samples']:
        raise ValueError('sample count mismatch: %s' % directory)
    first = reference[:, 0]
    arrays = {
        'wrench': wrench,
        'position_error': np.linalg.norm(first[:, :2], axis=1),
        'yaw_error_deg': np.degrees(np.abs(np.arctan2(first[:, 2], first[:, 3]))),
        'velocity_error': np.linalg.norm(state[:, :2] - first[:, 4:6], axis=1),
        'yaw_rate_error': np.abs(state[:, 2] - first[:, 6]),
        'reference_speed': np.linalg.norm(reference[:, :, 4:6], axis=2).reshape(-1),
        'reference_accel': np.linalg.norm(reference[:, :, 7:9], axis=2).reshape(-1),
        'reference_yaw_accel': np.abs(reference[:, :, 9]).reshape(-1),
    }
    yaw_rate = np.abs(reference[:, :, 6]).reshape(-1)
    arrays['curvature'] = yaw_rate / np.maximum(arrays['reference_speed'], 0.03)
    feasible, utilization, residual = [], [], []
    for command in wrench:
        ok, used, error = checker.solve(command)
        feasible.append(ok)
        utilization.append(used)
        residual.append(error)
    arrays['feasible'] = np.asarray(feasible, dtype=bool)
    arrays['thruster_utilization'] = np.asarray(utilization)
    arrays['allocation_residual'] = np.asarray(residual)
    arrays.update(read_status(Path(metadata['source_run']) / 'controller_status.csv'))
    rejected = sum(int(value) for value in metadata.get('rejected', {}).values())
    summary = {
        'id': metadata['trajectory_id'],
        'split': metadata['trajectory_split'],
        'family': metadata['trajectory_family'],
        'samples': int(len(wrench)),
        'rejected': rejected,
        'unreachable_targets': int(np.sum(~arrays['feasible'])),
        'source_run': metadata['source_run'],
        'wrench': {axis: describe(wrench[:, index], signed=True)
                   for index, axis in enumerate(AXES)},
        'position_error_m': describe(arrays['position_error']),
        'yaw_error_deg': describe(arrays['yaw_error_deg']),
        'thruster_utilization': describe(arrays['thruster_utilization']),
    }
    if 'success' in arrays:
        summary.update({
            'solver_success_rate': float(np.mean(arrays['success'])),
            'callback_deadline_miss_ratio': float(np.mean(arrays['deadline_miss'])),
            'callback_time_ms': describe(arrays['callback_ms']),
        })
    else:
        summary.update({'solver_success_rate': None,
                        'callback_deadline_miss_ratio': None,
                        'callback_time_ms': None})
    return {'summary': summary, 'arrays': arrays}


def joined(records, key, empty_shape=(0,)):
    parts = [record['arrays'][key] for record in records
             if key in record['arrays'] and record['arrays'][key].size]
    return np.concatenate(parts, axis=0) if parts else np.empty(empty_shape)


def aggregate(records, geometry):
    wrench = joined(records, 'wrench', (0, 3))
    projection = np.abs(wrench) / geometry.wrench_scale if len(wrench) else wrench
    success = joined(records, 'success')
    callback = joined(records, 'callback_ms')
    return {
        'tracks': len(records),
        'samples': int(sum(record['summary']['samples'] for record in records)),
        'rejected': int(sum(record['summary']['rejected'] for record in records)),
        'unreachable_targets': int(sum(record['summary']['unreachable_targets']
                                       for record in records)),
        'wrench': {axis: describe(wrench[:, index], signed=True)
                   for index, axis in enumerate(AXES)} if len(wrench) else {},
        'wrench_projection_utilization': {
            axis: describe(projection[:, index]) for index, axis in enumerate(AXES)
        } if len(wrench) else {},
        'thruster_utilization': describe(joined(records, 'thruster_utilization')),
        'tracking': {
            'position_error_m': describe(joined(records, 'position_error')),
            'yaw_error_deg': describe(joined(records, 'yaw_error_deg')),
            'velocity_error_mps': describe(joined(records, 'velocity_error')),
            'yaw_rate_error_radps': describe(joined(records, 'yaw_rate_error')),
        },
        'reference': {
            'speed_mps': describe(joined(records, 'reference_speed')),
            'curvature_inv_m': describe(joined(records, 'curvature')),
            'accel_mps2': describe(joined(records, 'reference_accel')),
            'yaw_accel_radps2': describe(joined(records, 'reference_yaw_accel')),
        },
        'solver': {
            'success_rate': float(np.mean(success)) if success.size else None,
            'iterations': describe(joined(records, 'iterations')),
            'layer1_time_ms': describe(joined(records, 'layer1_ms')),
            'layer2_time_ms': describe(joined(records, 'layer2_ms')),
            'callback_time_ms': describe(callback),
            'deadline_miss_ratio': float(np.mean(callback > 80.0))
                                   if callback.size else None,
        },
    }


def warnings_for(report, expected):
    warnings = []
    for split, wanted in expected.items():
        actual = report['by_split'].get(split, {}).get('tracks', 0)
        if actual != wanted:
            warnings.append('%s has %d tracks; expected %d' % (split, actual, wanted))
    for track in report['tracks']:
        if track['rejected']:
            warnings.append('%s has %d rejected samples' %
                            (track['id'], track['rejected']))
        if track['unreachable_targets']:
            warnings.append('%s has %d unreachable targets' %
                            (track['id'], track['unreachable_targets']))
        if track['solver_success_rate'] is not None and track['solver_success_rate'] < 0.99:
            warnings.append('%s solver success is below 99%%' % track['id'])
    train = report['by_split'].get('train')
    if train:
        for axis in AXES:
            signed = train['wrench'][axis]
            if min(signed['positive_ratio'], signed['negative_ratio']) < 0.01:
                warnings.append('train %s has less than 1%% coverage in one sign' % axis)
            if train['wrench_projection_utilization'][axis]['max'] < 0.05:
                warnings.append('train %s uses less than 5%% of projection range' % axis)
    return warnings


def render_markdown(report):
    lines = ['# HOFA Policy Teacher Coverage', '', '## Split summary', '',
             '| Split | Tracks | Samples | Rejected | Unreachable | Solver success | Callback P95 ms | Deadline miss |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('train', 'val', 'test'):
        item = report['by_split'].get(split)
        if not item:
            continue
        solver = item['solver']
        callback = solver['callback_time_ms'] or {'p95': float('nan')}
        lines.append('| %s | %d | %d | %d | %d | %.4f | %.2f | %.4f |' %
                     (split, item['tracks'], item['samples'], item['rejected'],
                      item['unreachable_targets'], solver['success_rate'],
                      callback['p95'], solver['deadline_miss_ratio']))
    lines += ['', '## Wrench coverage', '',
              '| Split | Axis | Min | Max | P95 abs | Positive | Negative | Max projection use |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for split in ('train', 'val', 'test'):
        item = report['by_split'].get(split)
        if not item:
            continue
        for axis in AXES:
            command = item['wrench'][axis]
            used = item['wrench_projection_utilization'][axis]
            lines.append('| %s | %s | %.4f | %.4f | %.4f | %.3f | %.3f | %.3f |' %
                         (split, axis, command['min'], command['max'],
                          command['p95_abs'], command['positive_ratio'],
                          command['negative_ratio'], used['max']))
    lines += ['', '## Family coverage', '',
              '| Family | Tracks | Samples | Position P95 m | Yaw P95 deg | Thruster P95 |',
              '|---|---:|---:|---:|---:|---:|']
    for family, item in sorted(report['by_family'].items()):
        lines.append('| %s | %d | %d | %.4f | %.3f | %.3f |' %
                     (family, item['tracks'], item['samples'],
                      item['tracking']['position_error_m']['p95'],
                      item['tracking']['yaw_error_deg']['p95'],
                      item['thruster_utilization']['p95']))
    lines += ['', '## Warnings', '']
    lines += ['- ' + warning for warning in report['warnings']] or ['- None']
    lines.append('')
    return chr(10).join(lines)


def write_tracks(path, tracks):
    fields = ['id', 'split', 'family', 'samples', 'rejected',
              'unreachable_targets', 'solver_success_rate',
              'callback_deadline_miss_ratio', 'callback_p95_ms',
              'position_error_p95_m', 'yaw_error_p95_deg',
              'fx_min', 'fx_max', 'fy_min', 'fy_max', 'nz_min', 'nz_max',
              'thruster_utilization_p95', 'source_run']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for track in tracks:
            callback = track['callback_time_ms'] or {}
            writer.writerow({
                'id': track['id'], 'split': track['split'],
                'family': track['family'], 'samples': track['samples'],
                'rejected': track['rejected'],
                'unreachable_targets': track['unreachable_targets'],
                'solver_success_rate': track['solver_success_rate'],
                'callback_deadline_miss_ratio': track['callback_deadline_miss_ratio'],
                'callback_p95_ms': callback.get('p95'),
                'position_error_p95_m': track['position_error_m']['p95'],
                'yaw_error_p95_deg': track['yaw_error_deg']['p95'],
                'fx_min': track['wrench']['fx']['min'],
                'fx_max': track['wrench']['fx']['max'],
                'fy_min': track['wrench']['fy']['min'],
                'fy_max': track['wrench']['fy']['max'],
                'nz_min': track['wrench']['nz']['min'],
                'nz_max': track['wrench']['nz']['max'],
                'thruster_utilization_p95': track['thruster_utilization']['p95'],
                'source_run': track['source_run'],
            })


def generate_report(dataset_root, output, expected=None):
    expected = expected or {'train': 20, 'val': 5, 'test': 5}
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('coverage output already exists: %s' % output)
    directories = sorted(path.parent for path in dataset_root.glob('*/metadata.json'))
    if not directories:
        raise ValueError('no exported datasets found in %s' % dataset_root)
    geometry = ThrusterGeometry.bricsbot()
    checker = AllocationChecker(geometry)
    records = [analyse_track(directory, geometry, checker) for directory in directories]
    splits, families = defaultdict(list), defaultdict(list)
    for record in records:
        splits[record['summary']['split']].append(record)
        families[record['summary']['family']].append(record)
    report = {
        'schema': 'hofa_policy_coverage_v1',
        'dataset_root': str(dataset_root),
        'tracks': [record['summary'] for record in records],
        'overall': aggregate(records, geometry),
        'by_split': {key: aggregate(value, geometry) for key, value in sorted(splits.items())},
        'by_family': {key: aggregate(value, geometry) for key, value in sorted(families.items())},
    }
    report['warnings'] = warnings_for(report, expected)
    output.mkdir(parents=True)
    (output / 'coverage.json').write_text(json.dumps(report, indent=2, sort_keys=True) + chr(10))
    (output / 'coverage.md').write_text(render_markdown(report))
    write_tracks(output / 'tracks.csv', report['tracks'])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-train', type=int, default=20)
    parser.add_argument('--expected-val', type=int, default=5)
    parser.add_argument('--expected-test', type=int, default=5)
    args = parser.parse_args()
    report = generate_report(args.datasets, args.output, {
        'train': args.expected_train, 'val': args.expected_val,
        'test': args.expected_test})
    print(json.dumps({'output': str(Path(args.output).resolve()),
                      'tracks': report['overall']['tracks'],
                      'samples': report['overall']['samples'],
                      'warnings': report['warnings']}, indent=2))


if __name__ == '__main__':
    main()
