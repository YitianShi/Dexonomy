import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dexonomy.op.native_clutter_batch import merge_validated_target, process_native_batch, retain_valid_attempts, store_native_sampling


class NativeClutterBatchTests(unittest.TestCase):
    def test_surface_checkpoint_has_scene_coordinates_and_replays_original_seeds(self):
        with TemporaryDirectory() as temporary:
            scene_path = Path(temporary) / "scene_000023/object.npy"
            rotations = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))
            translations = np.tile(np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32), (2, 1))
            scale = np.full(3, 0.5)
            pose = np.asarray([2, 3, 4, np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
            args = (temporary, scene_path, 0, "original", 2, 2)
            _, _, path = store_native_sampling(*args, rotations, translations, scale, pose)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_allclose(archive["points_world"], [[2, 3.5, 4], [1.5, 3, 4]])
                np.testing.assert_allclose(archive["inward_normals_world"], [[0, 1, 0], [0, 1, 0]], atol=1e-15)
                self.assertEqual(json.loads(str(archive["metadata"]))["point_count"], 2)
            restored_rotations, restored_translations, _ = store_native_sampling(
                *args, rotations, np.zeros_like(translations), scale, pose)
            np.testing.assert_array_equal(restored_rotations, rotations)
            np.testing.assert_array_equal(restored_translations, translations)
            with self.assertRaisesRegex(ValueError, "Different generation settings"):
                store_native_sampling(temporary, scene_path, 0, "changed", 2, 2,
                                      rotations, translations, scale, pose)

    def test_all_1024_candidates_and_failures_survive_packing_and_resume(self):
        self._check_all_candidates(keep_failed=True)

    def test_success_only_archives_still_attempt_all_1024_and_resume_without_rerunning(self):
        self._check_all_candidates(keep_failed=False)

    def _check_all_candidates(self, keep_failed):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_path = root / "scene_000023/object.npy"
            scene_path.parent.mkdir()
            np.save(scene_path, {"scene": {"object": {"scale": np.ones(3)}},
                                 "task": {"obj_name": "object"}})
            count = 128 * 8
            qposes = np.zeros((count, 29), dtype=np.float32)
            qposes[:, 3] = 1.0
            job = {
                "output": str(root / "output"), "scene_path": str(scene_path),
                "taxonomy": "24_Tip_Pinch", "epoch": 0, "points": 128, "inplane": 8,
                "run_fingerprint": "test", "hand_cbody": ["thumb", "finger"],
                "keep_failed": keep_failed,
                "required_cbody": [["thumb"], ["finger"]],
                "grasp_config": {"hand": {"xml_path": "unused"}, "op": {"pregrasp": {"cdist": 0.045}}},
                "eval_config": {}, "filter_config": {},
                "inputs": {
                    "grasp_qpos": qposes, "obj_cpn_w": np.zeros((count, 2, 6), dtype=np.float32),
                    "hand_cpn_w": np.zeros((count, 2, 6), dtype=np.float32),
                    "ext_center": np.zeros((count, 3)), "ext_wrench": np.zeros((count, 6)),
                    "n_evo": np.zeros((count, 1)), "obj_pose": np.zeros((count, 7)),
                    "obj_scale": np.ones((count, 3)),
                },
            }
            attempts = []

            def optimize(data, cfg, env):
                index = len(attempts)
                attempts.append(index)
                data["_case"] = index % 4
                if index % 4 == 1:
                    return data, False, "grasp_scene_collision"
                data["squeeze_qpos"] = data["grasp_qpos"].copy()
                data["pregrasp_qpos"] = np.repeat(data["grasp_qpos"], 2, axis=0)
                if index % 4 == 3:
                    return data, False, "pregrasp_required_contact"
                return data, True, "optimized"

            with (
                patch("dexonomy.sim.MuJoCo_OptEnv"),
                patch("dexonomy.op.gen_grasp.optimize_grasp_data", side_effect=optimize) as optimizer,
                patch("dexonomy.op.eval_data.evaluate_grasp_data",
                      side_effect=lambda data, *args: (data["_case"] != 2, object())),
                patch("dexonomy.op.native_clutter_batch.check_clutter_grasp", return_value=None),
            ):
                metadata = process_native_batch(job)
                self.assertEqual(optimizer.call_count, count)
                self.assertEqual(metadata["attempted"], count)
                self.assertEqual(metadata["status_counts"], {
                    "valid": 256, "grasp_scene_collision": 256,
                    "eval_failed": 256, "pregrasp_required_contact": 256,
                })
                optimizer.reset_mock()
                self.assertEqual(process_native_batch(job), metadata)
                optimizer.assert_not_called()

            with np.load(metadata["archive_path"], allow_pickle=False) as archive:
                kept = np.arange(count) if keep_failed else np.arange(0, count, 4)
                self.assertEqual(len(archive["status"]), len(kept))
                np.testing.assert_array_equal(archive["candidate_indices"], kept)
                np.testing.assert_array_equal(archive["sample_point_indices"], kept % 128)
                np.testing.assert_array_equal(archive["inplane_indices"], kept // 128)
                if keep_failed:
                    self.assertTrue(np.isnan(archive["squeeze_qpos"][1]).all())
                else:
                    self.assertTrue((archive["status"] == "valid").all())
                    self.assertEqual(metadata["stored_candidates"], 256)
                    self.assertEqual(metadata["discarded_failed_candidates"], 768)
                self.assertEqual(len(archive["pregrasp_trajectory_offsets"]), len(kept) + 1)
                self.assertEqual(json.loads(str(archive["metadata"]))["attempted"], count)

            base_path = root / "base/24_Tip_Pinch/scene_000023/object.npy"
            base_path.parent.mkdir(parents=True)
            np.save(base_path, {phase: qposes[:1] for phase in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")})
            original = base_path.read_bytes()
            for _ in range(2):
                merge_validated_target(metadata, job["output"], root / "base")
                packed_path = Path(job["output"]) / "clutter/filtered_grasps/24_Tip_Pinch/scene_000023/object.npy"
                packed = np.load(packed_path, allow_pickle=True).item()
                # All 256 valid poses are identical: neither dedup nor FPS may discard them.
                self.assertEqual(len(packed["grasp_qpos"]), 257)
                self.assertEqual((packed["verified_by"] == "dexonomy_eval+clutter_filter").sum(), 256)
            self.assertEqual(base_path.read_bytes(), original)

    def test_pruning_keeps_original_candidate_ids_and_complete_ragged_trajectories(self):
        trajectory = np.arange(8 * 29, dtype=np.float32).reshape(8, 29)
        arrays = {"status": np.asarray(["valid", "eval_failed", "valid", "grasp_required_contact"]),
                  "candidate_indices": np.asarray([1024, 1025, 1026, 1027]),
                  "grasp_qpos": np.arange(4 * 29).reshape(4, 29),
                  "pregrasp_trajectory_qpos": trajectory,
                  "pregrasp_trajectory_offsets": np.asarray([0, 2, 5, 8, 8])}
        metadata = {"attempted": 4, "status_counts": {"valid": 2, "eval_failed": 1,
                    "grasp_required_contact": 1}, "run_fingerprint": "unchanged"}
        result, compact = retain_valid_attempts(arrays, metadata)
        np.testing.assert_array_equal(result["candidate_indices"], [1024, 1026])
        np.testing.assert_array_equal(result["pregrasp_trajectory_offsets"], [0, 2, 5])
        np.testing.assert_array_equal(result["pregrasp_trajectory_qpos"], trajectory[[0, 1, 5, 6, 7]])
        self.assertEqual(compact["attempted"], metadata["attempted"])
        self.assertEqual(compact["status_counts"], metadata["status_counts"])
        self.assertEqual(compact["run_fingerprint"], metadata["run_fingerprint"])
        again, again_metadata = retain_valid_attempts(result, compact)
        for name in result:
            np.testing.assert_array_equal(again[name], result[name])
        self.assertEqual(again_metadata, compact)


if __name__ == "__main__":
    unittest.main()
