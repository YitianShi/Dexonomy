from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from dexonomy.util.grasp_diversity import analyze_grasps
from scripts.analyze_grasp_diversity import load_group, load_representative_grasps


def poses(count):
    result = np.zeros((count, 29))
    result[:, 3] = 1
    return result


class GraspDiversityTests(unittest.TestCase):
    def test_quaternion_sign_does_not_create_diversity(self):
        qpos = poses(2)
        qpos[1, 3] = -3
        original = qpos.copy()
        metrics, selected, _ = analyze_grasps(qpos, ["a", "b"], .1)
        self.assertEqual(metrics["near_duplicates"]["default"]["pair_count"], 1)
        self.assertEqual(metrics["representative_count"], 1)
        self.assertEqual(metrics["taxonomy_diversity"]["occupied"], 2)
        np.testing.assert_array_equal(qpos, original)

    def test_rotation_and_finger_changes_separate_grasps_at_same_position(self):
        qpos = poses(3)
        qpos[1, 3:7] = [0, 0, 0, 1]
        qpos[2, 7] = np.deg2rad(20)
        metrics, selected, _ = analyze_grasps(qpos, ["a"]*3, .1)
        self.assertEqual(metrics["near_duplicates"]["default"]["pair_count"], 0)
        self.assertEqual(len(selected), 3)

    def test_similarity_is_scale_normalized(self):
        qpos = poses(2)
        qpos[1, 0] = .0015
        first, _, _ = analyze_grasps(qpos, ["a"]*2, .1)
        qpos[:, :3] *= 10
        second, _, _ = analyze_grasps(qpos, ["a"]*2, 1.)
        self.assertEqual(first["near_duplicates"], second["near_duplicates"])

    def test_chain_neighbors_do_not_collapse_distant_endpoints(self):
        qpos = poses(3)
        qpos[:, 0] = [0, .0015, .003]
        metrics, selected, edges = analyze_grasps(qpos, ["a"]*3, .1)
        self.assertEqual(metrics["near_duplicates"]["default"]["fraction_with_neighbor"], 1.)
        np.testing.assert_array_equal(selected, [0, 2])
        self.assertEqual(metrics["representative_count"], 2)
        for first, second in edges:
            self.assertFalse(first in selected and second in selected)

    def test_empty_and_invalid_inputs_are_not_reported_as_sufficient(self):
        empty, indices, _ = analyze_grasps(poses(0), [], .1)
        self.assertEqual(empty["review_flags"], ["no_valid_grasps"])
        self.assertEqual(len(indices), 0)
        qpos = poses(3)
        qpos[0, 0] = np.nan
        qpos[1, 3:7] = 0
        metrics, indices, _ = analyze_grasps(qpos, ["a"]*3, .1)
        self.assertEqual(metrics["invalid_grasps"], 2)
        np.testing.assert_array_equal(indices, [2])
        self.assertIn("invalid_qpos", metrics["review_flags"])

    def test_radius_duplicate_counts_match_brute_force(self):
        rng = np.random.default_rng(3)
        qpos = poses(100)
        qpos[:, :3] = rng.normal(0, .001, (100, 3))
        qpos[:, 7:] = rng.normal(0, .03, (100, 22))
        metrics, selected, _ = analyze_grasps(qpos, ["a"]*100, .1)
        count = 0
        for first in range(100):
            for second in range(first+1, 100):
                delta = np.rad2deg(qpos[first, 7:]-qpos[second, 7:])
                if (np.linalg.norm(qpos[first, :3]-qpos[second, :3]) <= .002 and
                        np.sqrt(np.mean(delta*delta)) <= 5 and np.max(abs(delta)) <= 15):
                    count += 1
        self.assertEqual(metrics["near_duplicates"]["default"]["pair_count"], count)

    def test_group_loader_preserves_files_and_rejects_mixed_scales(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "object/floating/scale_example"
            scene_path = root / "scene_cfg" / f"{relative}.npy"
            scene_path.parent.mkdir(parents=True)
            info = root / "info.json"
            info.write_text(json.dumps({"obb": [1, 2, 3], "gravity_center": [0, 0, 0]}))
            np.save(scene_path, {"scene": {"object": {"scale": [.1]*3, "pose": [0, 0, 0, 1, 0, 0, 0],
                                                       "info_path": str(info)}}, "task": {"obj_name": "object"}})
            source = root / "succ_collect/a" / f"{relative}.npy"
            source.parent.mkdir(parents=True)
            sample = {"grasp_qpos": poses(2).astype(np.float32), "scene_path": str(scene_path),
                      "target_name": "object", "effective_object_scale": np.ones((2, 3))*.1}
            np.save(source, sample)
            original = source.read_bytes()
            loaded = load_group(root, relative, ["a"], "grasp")
            self.assertAlmostEqual(loaded["diagonal"], np.sqrt(14)*.1)
            np.testing.assert_array_equal(loaded["locations"], [[0, 0], [0, 1]])
            self.assertEqual(source.read_bytes(), original)
            report = root / "analysis"
            report.mkdir()
            (report / "summary.json").write_text(json.dumps({"source_library": str(root), "phase": "grasp"}))
            (report / "object_scale_metrics.csv").write_text(
                f"relative,group_key,qpos_sha256\n{relative},g_000000,{loaded['qpos_sha256']}\n")
            np.savez(report / "representatives.npz", taxonomy_labels=np.asarray(["a"]),
                     g_000000=np.asarray([[0, 0]], np.int32))
            view = load_representative_grasps(report, relative)
            self.assertEqual(len(view["qpos"]), 1)
            self.assertEqual(source.read_bytes(), original)
            sample["grasp_qpos"][1, 0] += .1
            np.save(source, sample)
            with self.assertRaisesRegex(ValueError, "Source snapshot changed"):
                load_representative_grasps(report, relative)
            sample["effective_object_scale"][1] = .2
            np.save(source, sample)
            with self.assertRaisesRegex(ValueError, "Mixed physical scales"):
                load_group(root, relative, ["a"], "grasp")


if __name__ == "__main__":
    unittest.main()
