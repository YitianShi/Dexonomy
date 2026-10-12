"""Object-scale grasp diversity using interpretable pose and joint tolerances."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Tolerance:
    position_fraction: float
    rotation_deg: float
    joint_rms_deg: float
    joint_max_deg: float


TOLERANCES = {
    "strict": Tolerance(.01, 5., 2.5, 7.5),
    "default": Tolerance(.02, 10., 5., 15.),
    "relaxed": Tolerance(.05, 20., 10., 30.),
}


def pose_pair_distances(qpos, first, second, object_diagonal):
    """Position / object size, SO(3) angle, and bounded finger joint differences."""
    left, right = qpos[first], qpos[second]
    position = np.linalg.norm(left[:, :3] - right[:, :3], axis=1) / object_diagonal
    # q and -q describe the same orientation; callers normalize the quaternions.
    dot = np.abs(np.einsum("ij,ij->i", left[:, 3:7], right[:, 3:7]))
    rotation = np.rad2deg(2 * np.arccos(np.clip(dot, 0., 1.)))
    delta = np.rad2deg(left[:, 7:] - right[:, 7:])
    rms = np.sqrt(np.mean(delta * delta, axis=1))
    maximum = np.max(np.abs(delta), axis=1)
    return position, rotation, rms, maximum


def _quantiles(values):
    if not len(values):
        return None
    return dict(zip(("p05", "p25", "median", "p75", "p95"),
                    map(float, np.quantile(values, [.05, .25, .5, .75, .95]))))


def _distribution(counts):
    counts = np.asarray(counts)
    probabilities = counts[counts > 0] / counts.sum() if counts.sum() else np.empty(0)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {"occupied": len(probabilities), "entropy": entropy,
            "effective_count": float(np.exp(entropy)) if len(probabilities) else 0.}


def greedy_representatives(count, edges, labels):
    """A deterministic maximal separated set, not transitive graph components."""
    if not count:
        return np.empty(0, dtype=np.int32)
    if not len(edges):
        return np.arange(count, dtype=np.int32)
    first, second = edges.T
    graph = coo_matrix((np.ones(2 * len(edges), dtype=np.uint8),
                        (np.concatenate((first, second)), np.concatenate((second, first)))),
                       shape=(count, count)).tocsr()
    degrees = np.diff(graph.indptr)
    _, inverse, frequencies = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.lexsort((np.arange(count), frequencies[inverse], degrees))
    available = np.ones(count, dtype=bool)
    selected = []
    for index in order:
        if available[index]:
            selected.append(index)
            available[index] = False
            available[graph.indices[graph.indptr[index]:graph.indptr[index + 1]]] = False
    return np.sort(np.asarray(selected, dtype=np.int32))


def analyze_grasps(qpos, labels, object_diagonal, object_center=None,
                   pair_samples=20000, seed=12, direction_bins=48):
    """Exact local duplicate tests, plus explicitly sampled global pair quantiles.

    A near duplicate must meet all four tolerances simultaneously. Representative
    selection covers every finite input with a directly similar retained pose;
    retained poses are mutually separated at the default tolerance.
    """
    qpos = np.array(qpos, dtype=np.float64, copy=True)
    labels = np.asarray(labels)
    if qpos.ndim != 2 or qpos.shape[1] != 29 or labels.shape != (len(qpos),):
        raise ValueError("Expected Nx29 free-root Shadow hand qpos and N labels")
    if not np.isfinite(object_diagonal) or object_diagonal <= 0:
        raise ValueError("Object diagonal must be finite and positive")
    if pair_samples < 1 or direction_bins < 1:
        raise ValueError("Pair sample count and direction bins must be positive")
    input_count = len(qpos)
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    valid = np.isfinite(qpos).all(axis=1) & (norms > 1e-12)
    original_indices = np.flatnonzero(valid)
    qpos, labels, norms = qpos[valid], labels[valid], norms[valid]
    qpos[:, 3:7] /= norms[:, None]
    count = len(qpos)
    result = {"input_grasps": input_count, "finite_grasps": count,
              "invalid_grasps": input_count-count, "object_diagonal_m": float(object_diagonal),
              "tolerances": {name: asdict(value) for name, value in TOLERANCES.items()}}

    degrees = {name: np.zeros(count, dtype=np.int32) for name in TOLERANCES}
    pair_counts = dict.fromkeys(TOLERANCES, 0)
    default_edges = []
    numerical_pairs = 0
    if count > 1:
        tree = cKDTree(qpos[:, :3])
        radius = TOLERANCES["relaxed"].position_fraction * object_diagonal
        # Bound temporary memory even if many grasps share one root position.
        for start in range(0, count, 128):
            neighbors = tree.query_ball_point(qpos[start:start+128, :3], radius * (1+1e-12))
            pair_blocks = []
            for local, candidates in enumerate(neighbors):
                first_index = start+local
                candidates = np.asarray(candidates, dtype=np.int32)
                candidates = candidates[candidates > first_index]
                if len(candidates):
                    pair_blocks.append(np.column_stack((np.full(len(candidates), first_index, np.int32), candidates)))
            pairs = np.concatenate(pair_blocks) if pair_blocks else np.empty((0, 2), np.int32)
            for offset in range(0, len(pairs), 32768):
                first, second = pairs[offset:offset+32768].T
                position, rotation, rms, maximum = pose_pair_distances(qpos, first, second, object_diagonal)
                numerical_pairs += int(np.sum((position*object_diagonal <= 1e-7) &
                                              (rotation <= .005) & (maximum <= .005)))
                for name, tolerance in TOLERANCES.items():
                    close = ((position <= tolerance.position_fraction+1e-12) &
                             (rotation <= tolerance.rotation_deg+1e-9) &
                             (rms <= tolerance.joint_rms_deg+1e-9) &
                             (maximum <= tolerance.joint_max_deg+1e-9))
                    pair_counts[name] += int(np.sum(close))
                    np.add.at(degrees[name], first[close], 1)
                    np.add.at(degrees[name], second[close], 1)
                    if name == "default" and np.any(close):
                        default_edges.append(np.column_stack((first[close], second[close])))
        nearest = tree.query(qpos[:, :3], k=2)[0][:, 1] / object_diagonal
    else:
        nearest = np.empty(0)
    edges = np.concatenate(default_edges) if default_edges else np.empty((0, 2), np.int32)
    selected = greedy_representatives(count, edges, labels)
    result["numerically_identical_pair_count"] = numerical_pairs
    result["near_duplicates"] = {
        name: {"pair_count": int(pair_counts[name]),
               "grasps_with_neighbor": int(np.count_nonzero(values)),
               "fraction_with_neighbor": float(np.count_nonzero(values)/count) if count else 0.}
        for name, values in degrees.items()
    }
    result["representative_count"] = len(selected)
    result["representative_fraction"] = float(len(selected)/count) if count else 0.
    result["greedy_redundant_fraction"] = float(1-len(selected)/count) if count else 0.
    result["nearest_position_fraction_quantiles"] = _quantiles(nearest)

    possible_pairs = count*(count-1)//2
    sampled = possible_pairs > pair_samples
    if not sampled:
        first, second = np.triu_indices(count, 1)
    else:
        rng = np.random.default_rng(seed)
        first = rng.integers(count, size=pair_samples)
        second = rng.integers(count-1, size=pair_samples)
        second += second >= first
    distance = pose_pair_distances(qpos, first, second, object_diagonal)
    result["global_pairwise"] = {"pair_count": len(first), "sampled": sampled,
                                "position_fraction": _quantiles(distance[0]),
                                "rotation_deg": _quantiles(distance[1]),
                                "joint_rms_deg": _quantiles(distance[2]),
                                "joint_max_deg": _quantiles(distance[3])}

    names, counts = np.unique(labels, return_counts=True)
    representative_names, representative_counts = np.unique(labels[selected], return_counts=True)
    result["taxonomy_counts"] = dict(zip(map(str, names), map(int, counts)))
    result["representative_taxonomy_counts"] = dict(zip(map(str, representative_names), map(int, representative_counts)))
    result["taxonomy_diversity"] = _distribution(counts)

    center = np.zeros(3) if object_center is None else np.asarray(object_center)
    directions = qpos[:, :3]-center
    lengths = np.linalg.norm(directions, axis=1)
    directions = directions[lengths > 1e-12]/lengths[lengths > 1e-12, None]
    index = np.arange(direction_bins)
    z = 1-2*(index+.5)/direction_bins
    angle = index*np.pi*(3-np.sqrt(5))
    grid = np.column_stack((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z))
    assigned = np.argmax(directions @ grid.T, axis=1) if len(directions) else np.empty(0, int)
    spatial_counts = np.bincount(assigned, minlength=direction_bins)
    result["spatial_direction_diversity"] = {**_distribution(spatial_counts),
                                             "bins": direction_bins, "counts": spatial_counts.tolist(),
                                             "undefined_directions": count-len(directions)}
    flags = []
    if not count:
        flags.append("no_valid_grasps")
    else:
        if len(selected) < 64:
            flags.append("fewer_than_64_separated_representatives")
        if result["representative_fraction"] < .7:
            flags.append("more_than_30_percent_greedy_redundancy")
        if result["spatial_direction_diversity"]["occupied"] < 8:
            flags.append("fewer_than_8_of_48_root_position_directions")
    if result["invalid_grasps"]:
        flags.append("invalid_qpos")
    result["review_flags"] = flags
    result["assessment"] = "needs_review" if flags else "passes_heuristic"
    return result, original_indices[selected], edges
