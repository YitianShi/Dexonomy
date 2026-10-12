from argparse import Namespace
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transforms3d.quaternions import quat2mat

from dexonomy.data.hand_loader import HandTemplateLoader, contact_template_frame
from dexonomy.op.native_clutter_batch import PHASES, merge_validated_target, process_native_batch
from dexonomy.util.np_util import np_inv_transform_points, np_transform_points
from scripts.generate_native_clutter_grasps import run_all_candidates, template_contact_counts
from scripts.generate_single_object_grasps import generation_directory


class AllHandContactsTests(unittest.TestCase):
    def test_every_anchor_preserves_template_geometry_and_batched_shapes(self):
        qpos = np.zeros(29, dtype=np.float32)
        qpos[:3] = [.2, -.1, .3]
        qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        contacts_hand = np.asarray([[.1, .2, .3, 1, 0, 0],
                                    [.2, .4, .1, 0, 1, 0],
                                    [-.1, .2, .1, 0, 0, -1]], dtype=np.float32)
        data = {"grasp_qpos": qpos,
                "hand_cpn_w": np_transform_points(contacts_hand, quat2mat(qpos[3:7]), qpos[:3]),
                "hand_skt_h": np.zeros((4, 6), np.float32), "n_evo": np.zeros(1, np.int32),
                "hand_path": "fixed.npy"}
        original = data["hand_cpn_w"].copy()
        loader = HandTemplateLoader.__new__(HandTemplateLoader)
        loader.initial_template = data
        loader.extra_info = {"hand_cbody": ["a", "b", "c"], "required_cbody": [["a"]]}
        loader.data_buffer = dict.fromkeys(["grasp_qpos", "rot_c2h", "trans_c2h",
                                           "hand_cpn_c", "hand_skt_h", "n_evo", "hand_path"])
        for index in range(3):
            frame = contact_template_frame(data, index)
            np.testing.assert_allclose(frame["hand_cpn_c"][index], [0, 0, 0, 1, 0, 0], atol=2e-7)
            reconstructed = np_inv_transform_points(frame["hand_cpn_c"], frame["rot_c2h"], frame["trans_c2h"])
            np.testing.assert_allclose(reconstructed, contacts_hand, atol=2e-7)
            batch = loader.get_contact_batched_data(index, torch.Size([2, 6]), "cpu")
            self.assertEqual(batch["hand_cpn_c"].shape, (2, 6, 3, 6))
            self.assertEqual(batch["grasp_qpos"][..., 7:].view(12, -1).shape, (12, 22))
            self.assertEqual(batch["hand_skt_h"].view(12, -1, 3).shape, (12, 8, 3))
            np.testing.assert_allclose(batch["hand_cpn_c"][1, 5], frame["hand_cpn_c"])
        np.testing.assert_array_equal(data["hand_cpn_w"], original)

    def test_new_checkpoint_directory_is_separate_even_with_identical_point_settings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / "run_config.json"
            settings.write_text(json.dumps({"points": 128, "inplane": 8, "epochs": 1}))
            original = settings.read_bytes()
            args = Namespace(points=128, inplane=8, epochs=1, all_hand_contacts=True)
            self.assertEqual(generation_directory(root, args), root / "runs/all_contacts_points_128_inplane_8_epochs_1")
            self.assertEqual(settings.read_bytes(), original)

    def test_all_anchor_point_roll_combinations_pack_resume_and_preserve_old_successes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs/all_contacts"
            relative = Path("object/floating/scale_example.npy")
            scene_path = root / "scene_cfg" / relative
            scene_path.parent.mkdir(parents=True)
            np.save(scene_path, {"scene": {"object": {"type": "rigid_object", "scale": np.ones(3)}},
                                 "task": {"obj_name": "object"}})
            taxonomy = "24_Tip_Pinch"
            old = root / "validated" / taxonomy / relative.with_suffix("") / "epoch_000.npy"
            old_archive = root / "attempts" / taxonomy / relative.with_suffix("") / "epoch_000.npz"
            old.parent.mkdir(parents=True)
            old_archive.parent.mkdir(parents=True)
            old_pose = np.full((1, 29), -2, dtype=np.float32)
            np.save(old, {**{phase: old_pose for phase in PHASES},
                          "source_grasp_paths": np.asarray([str(old_archive)]), "source_indices": np.asarray([7])})
            np.savez(old_archive, metadata=np.asarray('{}'))
            original = old.read_bytes(), old_archive.read_bytes()
            seen = set()
            records = []

            def optimize(data, cfg, env):
                row = data["grasp_qpos"][0]
                combination = tuple(int(v) for v in row[:3])
                self.assertNotIn(combination, seen)
                seen.add(combination)
                data["pregrasp_qpos"] = data["grasp_qpos"].copy()
                data["squeeze_qpos"] = data["grasp_qpos"].copy()
                return data, True, "optimized"

            with patch("dexonomy.sim.MuJoCo_OptEnv"), \
                    patch("dexonomy.op.gen_grasp.optimize_grasp_data", side_effect=optimize) as optimizer, \
                    patch("dexonomy.op.eval_data.evaluate_grasp_data", return_value=(True, object())), \
                    patch("dexonomy.op.native_clutter_batch.check_clutter_grasp", side_effect=AssertionError("Isolated only")):
                for epoch in range(2):
                    for contact in range(2):
                        count = 6
                        qpos = np.zeros((count, 29), dtype=np.float32)
                        qpos[:, 0] = epoch * 2 + contact
                        qpos[:, 1] = np.arange(count) % 3
                        qpos[:, 2] = np.arange(count) // 3
                        qpos[:, 3] = 1
                        job = {"output": str(run), "library_root": str(root), "scene_path": str(scene_path),
                               "taxonomy": taxonomy, "epoch": epoch, "points": 3, "inplane": 2,
                               "single_object": True, "source_relative": str(relative),
                               "job_key": f"{taxonomy}/{relative.with_suffix('')}/contact_{contact:03d}/{epoch}",
                               "hand_contact_index": contact, "hand_contact_count": 2,
                               "hand_cbody": ["finger", "thumb"], "required_cbody": [["finger"], ["thumb"]],
                               "run_fingerprint": "all-contacts", "keep_failed": False,
                               "grasp_config": {"hand": {"xml_path": "unused"}, "op": {"pregrasp": {"cdist": .045}}},
                               "eval_config": {}, "filter_config": {},
                               "inputs": {"grasp_qpos": qpos, "obj_cpn_w": np.zeros((count, 2, 6)),
                                          "hand_cpn_w": np.zeros((count, 2, 6)), "ext_center": np.zeros((count, 3)),
                                          "ext_wrench": np.zeros((count, 6)), "n_evo": np.zeros((count, 1)),
                                          "obj_pose": np.tile([0, 0, 0, 1, 0, 0, 0], (count, 1)), "obj_scale": np.ones((count, 3))}}
                        metadata = process_native_batch(job)
                        records.append(metadata)
                        before = optimizer.call_count
                        self.assertEqual(process_native_batch(job), metadata)
                        self.assertEqual(optimizer.call_count, before)
                        with np.load(metadata["archive_path"], allow_pickle=False) as archive:
                            np.testing.assert_array_equal(archive["candidate_indices"], np.arange(count) + (epoch * 2 + contact) * count)
                            np.testing.assert_array_equal(archive["hand_contact_indices"], [contact] * count)
                            np.testing.assert_array_equal(archive["sample_point_indices"], np.arange(count) % 3)
                            np.testing.assert_array_equal(archive["inplane_indices"], np.arange(count) // 3)
                        merge_validated_target(metadata, run, root / "unused")
            self.assertEqual(len(seen), 24)
            published = np.load(root / "succ_collect" / taxonomy / relative, allow_pickle=True).item()
            self.assertEqual(len(published["grasp_qpos"]), 25)
            np.testing.assert_array_equal(published["grasp_qpos"][:1], old_pose)
            self.assertEqual(set(published["source_indices"][1:]), set(range(24)))
            self.assertEqual(published["hand_contact_indices"][0], -1)
            self.assertEqual((old.read_bytes(), old_archive.read_bytes()), original)

            # Persist complete jobs, then prove resume counts every anchor without initializing GPU.
            db = sqlite3.connect(run / "progress.sqlite")
            db.execute("CREATE TABLE completed (job_key TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
            db.execute("CREATE TABLE status_totals (status TEXT PRIMARY KEY, count INTEGER NOT NULL)")
            db.executemany("INSERT INTO completed VALUES (?, ?)", [(m["job_key"], json.dumps(m)) for m in records])
            db.execute("INSERT INTO status_totals VALUES ('valid', 24)")
            db.commit()
            db.close()
            args = Namespace(scene_root=str(root / "scene_cfg"), scene="*/floating", target="*", single_object=True,
                             library_root=str(root), all_hand_contacts=True, epochs=2, points=3, inplane=2,
                             gpu=2, workers=2, keep_failed=False, dry_run=False, collect_only=False)
            with patch("torch.cuda.set_device", side_effect=AssertionError("Complete run must resume without GPU")):
                run_all_candidates(args, [scene_path], root / "unused", run, [taxonomy], str(root / "scene_cfg/*/floating/*.npy"))
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress["planned_groups"], 4)
            self.assertEqual(progress["completed_groups"], 4)
            self.assertEqual(progress["planned_attempts"], 24)
            self.assertEqual(progress["attempted"], 24)

    def test_official_33_templates_give_expected_exhaustive_candidate_total(self):
        root = Path(__file__).resolve().parents[1]
        names = [p.stem for p in (root / "assets/hand/shadow/init_tmpl").glob('[0-9]*.npy')]
        counts = template_contact_counts(names)
        self.assertEqual(len(counts), 33)
        self.assertEqual(sum(counts.values()), 275)
        self.assertEqual(88 * sum(counts.values()) * 128 * 8, 24780800)


if __name__ == "__main__":
    unittest.main()
