"""Conservative pose suppression within one object, scale and assigned taxonomy."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


PHASES = ('pregrasp_qpos', 'grasp_qpos', 'squeeze_qpos')
GLOBAL_ARRAYS = {'taxonomy_names', 'taxonomy_contact_body_names'}


@dataclass(frozen=True)
class PoseThresholds:
    position_m: float = .002
    rotation_deg: float = 5.
    joint_rms_deg: float = 3.
    joint_max_deg: float = 8.

    def __post_init__(self):
        values = np.array(list(asdict(self).values()))
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('All pose thresholds must be finite and positive')
        if self.rotation_deg > 180:
            raise ValueError('Rotation threshold must not exceed 180 degrees')


def pose_features(data):
    """Use the Shadow palm, 34 mm along the free root's local z axis."""
    arrays = [np.asarray(data[phase], dtype=np.float64) for phase in PHASES]
    if any(a.ndim != 2 or a.shape[1] != 29 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('All three pose arrays must have the same (N, 29) shape')
    poses = np.stack(arrays, axis=1)
    if not np.isfinite(poses).all():
        raise ValueError('Nonfinite pose; refusing to discard it as a duplicate')
    quats = poses[:, :, 3:7].copy()
    norms = np.linalg.norm(quats, axis=2, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError('Invalid hand quaternion')
    quats /= norms
    w, x, y, z = np.moveaxis(quats, -1, 0)
    palm_offset = .034 * np.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), axis=-1)
    return poses[:, :, :3] + palm_offset, quats, poses[:, :, 7:]


def pair_distances(features, left, right):
    """Max over the three phases: metres, rotation degrees, joint RMS/max degrees."""
    palms, quats, joints = features
    position = np.linalg.norm(palms[left]-palms[right], axis=-1).max(axis=-1)
    dots = np.abs(np.sum(quats[left]*quats[right], axis=-1))
    rotation = np.rad2deg(2*np.arccos(np.clip(dots, 0., 1.))).max(axis=-1)
    delta = joints[left]-joints[right]
    rms = np.rad2deg(np.sqrt(np.mean(delta*delta, axis=-1))).max(axis=-1)
    maximum = np.rad2deg(np.abs(delta).max(axis=(-1, -2)))
    return np.stack((position, rotation, rms, maximum), axis=-1)


def partition_keys(data, count):
    """Keep distinct actual contact-body patterns and scales separately."""
    columns = []
    for name in ('contact_body_masks', 'scene_scale', 'effective_object_scale'):
        if name not in data:
            continue
        array = np.asarray(data[name])
        if array.ndim == 0 or len(array) != count:
            raise ValueError(f'{name} must be row-aligned')
        flat = np.ascontiguousarray(array.reshape(count, -1))
        # Void records retain exact uint64 masks, including bits above 2**53.
        _, labels = np.unique(flat.view(np.dtype((np.void, flat.dtype.itemsize*flat.shape[1]))),
                              return_inverse=True)
        columns.append(labels)
    if not columns:
        return np.zeros(count, dtype=np.int32)
    _, labels = np.unique(np.stack(columns, axis=1), axis=0, return_inverse=True)
    return labels


def deduplicate(data, thresholds=PoseThresholds()):
    """Return a kept row for EVERY row. Representatives themselves are never removed.

    First-row greedy suppression avoids transitive chains: every removed row is
    directly within all thresholds of its retained representative, in all phases.
    Quaternions q and -q represent the same rotation. Joints use ordinary radian
    differences because these are bounded Shadow joints, not wrapping angles.
    """
    features = pose_features(data)
    palms, quats, joints = features
    count = len(palms)
    representative = np.full(count, -1, dtype=np.int32)
    if not count:
        return representative, {'input_rows': 0, 'kept_rows': 0, 'removed_rows': 0,
                                'exact_pose_duplicates': 0, 'near_pose_duplicates': 0,
                                'max_removed_distances': [0.]*4}
    partitions = partition_keys(data, count)
    tree = cKDTree(palms[:, 1])
    position2 = thresholds.position_m**2
    cosine = np.cos(np.deg2rad(thresholds.rotation_deg)/2)
    rms2 = np.deg2rad(thresholds.joint_rms_deg)**2
    max_joint = np.deg2rad(thresholds.joint_max_deg)
    for index in range(count):
        if representative[index] >= 0:
            continue
        representative[index] = index
        nearby = np.asarray(tree.query_ball_point(palms[index, 1], thresholds.position_m), dtype=np.int32)
        nearby = nearby[(nearby > index) & (representative[nearby] < 0)
                        & (partitions[nearby] == partitions[index])]
        if not len(nearby):
            continue
        delta = palms[nearby]-palms[index]
        nearby = nearby[(np.sum(delta*delta, axis=-1) <= position2).all(axis=1)]
        if not len(nearby):
            continue
        dots = np.abs(np.sum(quats[nearby]*quats[index], axis=-1))
        nearby = nearby[(dots >= cosine).all(axis=1)]
        if not len(nearby):
            continue
        delta = joints[nearby]-joints[index]
        valid = ((np.mean(delta*delta, axis=-1) <= rms2).all(axis=1)
                 & (np.abs(delta).max(axis=(1, 2)) <= max_joint))
        representative[nearby[valid]] = index
    rows = np.arange(count)
    removed = np.flatnonzero(representative != rows)
    kept = np.flatnonzero(representative == rows)
    if not np.array_equal(representative[representative], representative):
        raise AssertionError('A representative was removed')
    if not np.array_equal(partitions, partitions[representative]):
        raise AssertionError('Contact/scale partition was crossed')
    distances = pair_distances(features, removed, representative[removed])
    limits = np.array([thresholds.position_m, thresholds.rotation_deg,
                       thresholds.joint_rms_deg, thresholds.joint_max_deg])
    if len(removed) and (distances > limits+1e-10).any():
        raise AssertionError('A removed pose is too far from its retained representative')
    exact = np.ones(len(removed), dtype=bool)
    for name in PHASES:
        poses = np.asarray(data[name])
        exact &= (poses[removed] == poses[representative[removed]]).all(axis=1)
    maxima = distances.max(axis=0).tolist() if len(removed) else [0.]*4
    return representative, {
        'input_rows': count, 'kept_rows': len(kept), 'removed_rows': len(removed),
        'exact_pose_duplicates': int(exact.sum()), 'near_pose_duplicates': int((~exact).sum()),
        'max_removed_distances': maxima,
    }


def subset_rows(data, kept, source_path):
    """Preserve original provenance and contact labels; add immediate dedup origin."""
    count = len(data['grasp_qpos'])
    output = {}
    for key, value in data.items():
        if (key not in GLOBAL_ARRAYS and isinstance(value, np.ndarray)
                and value.ndim and len(value) == count):
            output[key] = value[kept]
        else:
            output[key] = value
    output['dedup_source_path'] = str(source_path)
    output['dedup_source_indices'] = np.asarray(kept, dtype=np.int32)
    return output
