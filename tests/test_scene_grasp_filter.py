import copy
from concurrent.futures import Future
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

from dexonomy.op.gen_clutter import _filter_scene, _transform_qposes
from dexonomy.op.native_clutter_batch import atomic_npy
from scripts import filter_graspnet_scene_grasps as runner


class SceneGraspFilterTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.taxonomy = "24_Tip_Pinch"
        self.scene = {"scene": {"014": {"type": "rigid_object", "scale": np.ones(3),
                                       "pose": np.asarray([0, 0, 0, 1, 0, 0, 0])},
                                "table": {"type": "plane"}}, "task": {"obj_name": "014"}}
        self.scene_path = self.root / "scene_cfg/scene_0000/014.npy"
        self.source_scene = self.root / "source/014.npy"
        self.library = self.root / "library"
        self.source = self.library / self.taxonomy / "014/floating/scale_one.npy"
        for path in (self.scene_path, self.source_scene, self.source):
            path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.scene_path, self.scene)
        np.save(self.source_scene, self.scene)
        qpos = np.zeros((2, 29), np.float32)
        qpos[:, 3] = 1
        qpos[:, 0] = [.1, .2]
        qpos[:, 7:] = np.arange(22) / 100
        self.data = {phase: qpos.copy() for phase in
                     ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")}
        self.data["pregrasp_qpos"][0, 0] = -.1
        self.data["scene_path"] = str(self.source_scene)
        np.save(self.source, self.data)
        self.args = runner.parse_args(["--scene-root", str(self.root / "scene_cfg"),
                                       "--library", str(self.library), "--output", str(self.root / "output"),
                                       "--workers", "1", "--approach-steps", "1",
                                       "--taxonomies", self.taxonomy])
        self.cfg = runner.filter_config(self.args, self.library, [self.taxonomy])
        self.cfg["output_dir"] = str(self.root / "filtered")
        self.environment = Mock()
        self.environment.reset_qpos.side_effect = self.remember_pose
        self.environment.get_scene_contacts.side_effect = self.contacts

    def remember_pose(self, qpos):
        self.pose = qpos

    def contacts(self, margin):
        # A table/neighbor blocks only the exact pregrasp of candidate zero.
        return {"dist": [-.0001]}, {"dist": [-.01] if self.pose[0] < 0 else []}, {"dist": []}

    def filter(self):
        with patch("dexonomy.sim.MuJoCo_OptEnv", return_value=self.environment):
            return _filter_scene(str(self.scene_path), OmegaConf.create(self.cfg))

    def test_strict_pregrasp_rejects_blocked_start_and_keeps_source_and_scene_immutable(self):
        source_before, scene_before = self.source.read_bytes(), self.scene_path.read_bytes()
        records = self.filter()
        record = records[0]
        self.assertEqual(record["evaluated_count"], 2)
        self.assertEqual(record["valid_count"], 1)
        self.assertEqual(record["rejected_collision"], 1)
        saved = np.load(record["output_path"], allow_pickle=True).item()
        np.testing.assert_array_equal(saved["source_indices"], [1])
        np.testing.assert_array_equal(saved["grasp_qpos"], self.data["grasp_qpos"][1:])
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(self.scene_path.read_bytes(), scene_before)
        self.assertFalse(Path(record["output_path"] + ".tmp").exists())
        self.cfg["check_pregrasp"] = False
        self.assertEqual(self.filter()[0]["valid_count"], 2)

    def test_rigid_transfer_rotates_wrist_and_preserves_finger_angles_and_inputs(self):
        qpos = self.data["grasp_qpos"].copy()
        before = qpos.copy()
        quaternion = np.sqrt(.5)
        transformed = _transform_qposes(qpos, np.array([0, 0, 0, 1, 0, 0, 0]),
                                        np.array([1, 2, 3, quaternion, 0, 0, quaternion]))
        np.testing.assert_allclose(transformed[:, :3], [[1, 2.1, 3], [1, 2.2, 3]], atol=1e-6)
        np.testing.assert_allclose(transformed[:, 3:7], [[quaternion, 0, 0, quaternion]] * 2)
        np.testing.assert_array_equal(transformed[:, 7:], qpos[:, 7:])
        np.testing.assert_array_equal(qpos, before)

    def test_zero_result_keeps_scene_and_mismatched_phases_fail_instead_of_truncating(self):
        before = self.scene_path.read_bytes()
        self.data["scene_scale"] = np.full((2, 3), 2.)
        np.save(self.source, self.data)
        record = self.filter()[0]
        self.assertEqual(record["rejected_scale"], 2)
        self.assertEqual(record["valid_count"], 0)
        self.assertEqual(record["evaluated_count"], 0)
        self.assertEqual(self.scene_path.read_bytes(), before)
        self.data["grasp_qpos"] = self.data["grasp_qpos"][:1]
        np.save(self.source, self.data)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.filter()

    def test_neighbor_contact_cannot_substitute_target_contact_and_squeeze_penetration_is_allowed(self):
        self.data["pregrasp_qpos"] = self.data["grasp_qpos"].copy()
        np.save(self.source, self.data)
        self.environment.get_scene_contacts.side_effect = lambda margin: (
            {"dist": []}, {"dist": [.0001]}, {"dist": []})
        record = self.filter()[0]
        self.assertEqual(record["rejected_target_contact"], 2)
        self.environment.get_scene_contacts.side_effect = lambda margin: (
            {"dist": [-.02] if self.pose[7] > .5 else [-.0001]}, {"dist": []}, {"dist": []})
        self.data["squeeze_qpos"][:, 7] = 1
        np.save(self.source, self.data)
        self.assertEqual(self.filter()[0]["valid_count"], 2)

    def test_resume_fingerprint_detects_replacement_addition_removal_and_source_pose_changes(self):
        original = runner.input_signature(self.scene_path, self.cfg)
        old_stat = self.source.stat()
        atomic_npy(self.source, self.data)
        os.utime(self.source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        replaced = runner.input_signature(self.scene_path, self.cfg)
        self.assertNotEqual(original, replaced)
        extra = self.source.with_name("scale_two.npy")
        np.save(extra, self.data)
        self.assertNotEqual(replaced, runner.input_signature(self.scene_path, self.cfg))
        extra.unlink()
        self.assertEqual(replaced, runner.input_signature(self.scene_path, self.cfg))
        moved = copy.deepcopy(self.scene)
        moved["scene"]["014"]["pose"][0] = 4
        np.save(self.source_scene, moved)
        self.assertNotEqual(replaced, runner.input_signature(self.scene_path, self.cfg))

    def test_target_path_penetration_short_circuits_each_candidate(self):
        self.cfg["approach_steps"] = 8
        self.environment.get_scene_contacts.side_effect = lambda margin: (
            {"dist": [-.002]}, {"dist": []}, {"dist": []})
        record = self.filter()[0]
        self.assertEqual(record["rejected_target_penetration"], 2)
        self.assertEqual(record["valid_count"], 0)
        self.assertEqual(self.environment.reset_qpos.call_count, 2)

    def test_missing_source_is_distinct_from_zero_valid_and_new_file_invalidates_resume(self):
        self.source.unlink()
        signature = runner.input_signature(self.scene_path, self.cfg)
        records = self.filter()
        self.assertEqual(records[0]["status"], "missing_source_grasp")
        row = (signature, json.dumps(records))
        self.assertEqual(runner.checkpoint_valid(row, signature), records)
        np.save(self.source, self.data)
        current = runner.input_signature(self.scene_path, self.cfg)
        self.assertNotEqual(current, signature)
        self.assertIsNone(runner.checkpoint_valid(row, current))

    def test_discard_stale_outputs_does_not_touch_source_library_or_other_targets(self):
        filtered = self.root / "results"
        stale = filtered / self.taxonomy / "scene_0000/014.npy"
        other = stale.with_name("046.npy")
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")
        other.write_bytes(b"keep")
        source_before = self.source.read_bytes()
        runner.discard_target_outputs(filtered, self.scene_path, [self.taxonomy])
        self.assertFalse(stale.exists())
        self.assertEqual(other.read_bytes(), b"keep")
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_changed_library_results_are_not_published_or_checkpoints_marked_complete(self):
        staging = self.root / "staging"
        staging.mkdir()
        staged = staging / "grasps.npy"
        np.save(staged, {"grasp_qpos": self.data["grasp_qpos"]})
        filtered = self.root / "results"
        final = filtered / self.taxonomy / "scene_0000/014.npy"
        final.parent.mkdir(parents=True)
        final.write_bytes(b"previous result")
        result = {"signature": runner.input_signature(self.scene_path, self.cfg), "stable": True,
                  "staging": str(staging), "records": [{"taxonomy": self.taxonomy,
                  "status": "saved", "valid_count": 2, "output_path": str(staged)}]}
        atomic_npy(self.source, self.data)
        self.assertIsNone(runner.publish_result(self.scene_path, result, filtered, self.cfg))
        self.assertEqual(final.read_bytes(), b"previous result")
        self.assertFalse(staging.exists())

    def test_resume_requires_positive_outputs_but_allows_checkpointed_zero_results(self):
        output = self.root / "result.npy"
        np.save(output, self.data)
        records = [{"valid_count": 2, "output_path": str(output),
                    "output_token": runner.file_token(output)}]
        row = ("signature", json.dumps(records))
        self.assertEqual(runner.checkpoint_valid(row, "signature"), records)
        output.unlink()
        self.assertIsNone(runner.checkpoint_valid(row, "signature"))
        self.assertIsNone(runner.checkpoint_valid(row, "changed"))
        zero = [{"valid_count": 0, "status": "empty", "output_path": None}]
        self.assertEqual(runner.checkpoint_valid(("signature", json.dumps(zero)), "signature"), zero)

    def test_orchestration_resumes_counts_and_recomputes_only_changed_inputs(self):
        # Run the real worker/publisher/checkpoint path synchronously in this fixture.
        executor = Mock()
        submitted = []

        def submit(function, *args):
            submitted.append(args[0])
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:
                future.set_exception(exc)
            return future

        executor.submit.side_effect = submit
        scene_root, library, output, paths, taxonomies = runner.select_inputs(self.args)
        cfg = runner.filter_config(self.args, library, taxonomies)
        with (patch.object(runner, "implementation_signature", return_value="test-version"),
              patch.object(runner, "ProcessPoolExecutor", return_value=executor),
              patch("dexonomy.sim.MuJoCo_OptEnv", return_value=self.environment)):
            self.assertEqual(runner.run_filter(self.args, scene_root, library, output, paths, taxonomies, cfg), 0)
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(progress["counts"]["valid_count"], 1)
            self.assertEqual(progress["finished_targets"], 1)
            self.assertEqual(runner.run_filter(self.args, scene_root, library, output, paths, taxonomies, cfg), 0)
            self.assertEqual(len(submitted), 1)
            self.assertEqual(json.loads((output / "progress.json").read_text())["resumed_targets"], 1)
            atomic_npy(self.source, self.data)
            self.assertEqual(runner.run_filter(self.args, scene_root, library, output, paths, taxonomies, cfg), 0)
            self.assertEqual(len(submitted), 2)
        self.assertEqual((output / "run.lock").read_text(), "")


if __name__ == "__main__":
    unittest.main()
