import copy
from argparse import Namespace
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from dexonomy.op.gen_clutter import _filter_scene
from dexonomy.op.native_clutter_batch import merge_validated_target, process_native_batch
from scripts import generate_single_object_grasps as single_generator
from scripts.generate_single_object_grasps import canonical_object_scene, collect_objects, generation_directory
from scripts.generate_native_clutter_grasps import run_all_candidates


class SingleObjectGraspTests(unittest.TestCase):
    def test_sampling_change_keeps_old_checkpoints_and_combines_committed_successes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_settings = {"points": 1024, "inplane": 8, "epochs": 1}
            (root / "run_config.json").write_text(json.dumps(old_settings))
            original_settings = (root / "run_config.json").read_bytes()
            args = Namespace(points=128, inplane=8, epochs=1)
            current = generation_directory(root, args)
            self.assertEqual(current, root / "runs/points_128_inplane_8_epochs_1")
            relative = Path("object/floating/scale_one.npy")
            taxonomy = "24_Tip_Pinch"
            metadata = {"single_object": True, "library_root": str(root), "source_relative": str(relative),
                        "taxonomy": taxonomy, "target": "object", "scene_path": str(root / "scene_cfg" / relative),
                        "obj_scale": [.08] * 3}
            checkpoints = []
            for run, value, epoch in ((root, 1, 0), (current, 2, 0), (current, 3, 1)):
                validated = run / "validated" / taxonomy / relative.with_suffix("") / f"epoch_{epoch:03d}.npy"
                archive = run / "attempts" / taxonomy / relative.with_suffix("") / f"epoch_{epoch:03d}.npz"
                validated.parent.mkdir(parents=True, exist_ok=True)
                np.save(validated, {phase: np.full((1, 29), value, np.float32) for phase in
                                   ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")} |
                        {"source_grasp_paths": np.asarray([str(archive)]), "source_indices": np.asarray([value])})
                if epoch == 0:
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(archive, metadata=np.asarray(json.dumps(metadata)))
                    checkpoints.append((validated, validated.read_bytes(), archive, archive.read_bytes()))
            for _ in range(2):
                merge_validated_target(metadata, current, root / "unused")
                published = np.load(root / "succ_collect" / taxonomy / relative, allow_pickle=True).item()
                np.testing.assert_array_equal(published["grasp_qpos"][:, 0], [1, 2])
                np.testing.assert_array_equal(published["source_indices"], [1, 2])
                self.assertEqual(published["source_grasp_paths"].tolist(), [str(item[2]) for item in checkpoints])
            for validated, old_data, archive, old_archive in checkpoints:
                self.assertEqual(validated.read_bytes(), old_data)
                self.assertEqual(archive.read_bytes(), old_archive)
            self.assertEqual((root / "run_config.json").read_bytes(), original_settings)
            self.assertEqual(generation_directory(root, args), current)
            self.assertEqual(generation_directory(root, Namespace(**old_settings)), root)

    def test_sampling_run_progress_includes_history_and_resumes_without_gpu(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "runs/points_128_inplane_8_epochs_1"
            current.mkdir(parents=True)
            scene_root = root / "scene_cfg"
            selected = scene_root / "selected/floating/scale_one.npy"
            selected.parent.mkdir(parents=True)
            np.save(selected, {"scene": {"selected": {"type": "rigid_object", "scale": np.ones(3) * .08}},
                               "task": {"obj_name": "selected"}})
            taxonomy = "24_Tip_Pinch"
            key = f"{taxonomy}/selected/floating/scale_one/0"
            for run, attempted, valid in ((root, 8192, 7), (current, 1024, 3)):
                database = sqlite3.connect(run / "progress.sqlite")
                database.execute("CREATE TABLE completed (job_key TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
                database.execute("CREATE TABLE status_totals (status TEXT PRIMARY KEY, count INTEGER NOT NULL)")
                database.execute("INSERT INTO completed VALUES (?, ?)",
                                 (key, json.dumps({"status_counts": {"valid": valid, "eval_failed": attempted-valid}})))
                database.executemany("INSERT INTO status_totals VALUES (?, ?)",
                                     [("valid", valid), ("eval_failed", attempted-valid)])
                database.commit()
                database.close()
            old_database = (root / "progress.sqlite").read_bytes()
            args = Namespace(scene_root=str(scene_root), scene="*/floating", target="*", single_object=True,
                             library_root=str(root), epochs=1, points=128, inplane=8, gpu=2, workers=84,
                             keep_failed=False, dry_run=False, collect_only=False)
            with patch("torch.cuda.set_device", side_effect=AssertionError("Completed selection must not initialize GPU")):
                run_all_candidates(args, [selected], root / "unused", current, [taxonomy],
                                   str(scene_root / "*/floating/*.npy"))
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress, json.loads((current / "progress.json").read_text()))
            self.assertEqual(progress["points"], 128)
            self.assertEqual(progress["attempted"], 1024)
            self.assertEqual(progress["new_valid_grasps"], 3)
            self.assertEqual(progress["library_completed_groups"], 2)
            self.assertEqual(progress["library_valid_grasps"], 10)
            self.assertEqual(progress["successful_grasp_root"], str(root / "succ_collect"))
            self.assertEqual((root / "progress.sqlite").read_bytes(), old_database)

    def test_scene_inventory_selects_deduplicated_ids_and_keeps_asset_scales(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assets, inventory = root / "assets", root / "scenes"
            for name, scales in (("target", [.05, .08]), ("distractor", [.11]), ("unused", [.14])):
                for scale in scales:
                    path = assets / name / "floating" / f"scale_{scale}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, {"scene": {name: {"type": "rigid_object", "scale": np.ones(3) * scale,
                                                   "pose": np.asarray([0, 0, 0, 1, 0, 0, 0])}},
                                   "task": {"obj_name": name}})
            scene = {"scene": {
                "target": {"type": "rigid_object", "scale": np.ones(3) * .73,
                           "pose": np.asarray([1, 2, 3, 1, 0, 0, 0])},
                "distractor": {"type": "rigid_object", "scale": np.ones(3) * .52,
                               "pose": np.asarray([4, 5, 6, 1, 0, 0, 0])},
                "table": {"type": "plane"}}, "task": {"obj_name": "target"}}
            paths = [inventory / f"scene_{i:06d}" / "target.npy" for i in range(2)]
            for path in paths:
                path.parent.mkdir(parents=True)
                np.save(path, scene)
            before = [path.read_bytes() for path in paths]
            args = Namespace(object_inventory=str(inventory), scale_inventory=None,
                             object_root=str(assets), object="*", scene="scene_*")
            catalog = collect_objects(args)
            self.assertEqual({path.parts[0] for path in catalog}, {"target", "distractor"})
            self.assertEqual(len(catalog), 3)
            self.assertEqual(sorted(float(s["scene"][s["task"]["obj_name"]]["scale"][0])
                                    for s in catalog.values()), [.05, .08, .11])
            for isolated in catalog.values():
                name = isolated["task"]["obj_name"]
                self.assertEqual(set(isolated["scene"]), {name})
                np.testing.assert_array_equal(isolated["scene"][name]["pose"], [0, 0, 0, 1, 0, 0, 0])
            self.assertEqual([path.read_bytes() for path in paths], before)
            args.object = "target"
            self.assertEqual(len(collect_objects(args)), 2)
            args.object = "unused"
            with self.assertRaisesRegex(ValueError, "No selected rigid objects"):
                collect_objects(args)

    def test_current_selection_does_not_submit_or_modify_other_cached_objects(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "assets/selected/floating/scale.npy"
            asset.parent.mkdir(parents=True)
            scene = {"scene": {"selected": {"type": "rigid_object", "scale": np.ones(3) * .08,
                                             "pose": np.asarray([0, 0, 0, 1, 0, 0, 0])}},
                     "task": {"obj_name": "selected"}}
            np.save(asset, scene)
            output = root / "library"
            cached = output / "scene_cfg/other/floating/scale_old.npy"
            cached.parent.mkdir(parents=True)
            np.save(cached, {"scene": {"other": {"type": "rigid_object", "scale": np.ones(3) * .2}},
                            "task": {"obj_name": "other"}})
            published = output / "succ_collect/24_Tip_Pinch/other/floating/scale_old.npy"
            published.parent.mkdir(parents=True)
            published.write_bytes(b"previously published grasp data")
            before = (cached.read_bytes(), published.read_bytes())
            args = Namespace(object_root=str(root / "assets"), object_inventory=None, scale_inventory=None,
                             object="selected", scene="scene_*", output=str(output),
                             taxonomies=["24_Tip_Pinch"], epochs=1, points=1024, inplane=8,
                             gpu=2, workers=84, keep_failed=False, dry_run=False)

            def run_selected(arguments, paths, baseline, destination, taxonomies, glob):
                self.assertEqual(len(paths), 1)
                self.assertEqual(paths[0].parts[-3], "selected")
                (destination / "progress.json").write_text(json.dumps({"completed_groups": 0, "planned_groups": 1}))

            with patch.object(single_generator, "parse_args", return_value=args), \
                    patch.object(single_generator, "run_all_candidates", side_effect=run_selected) as runner:
                single_generator.main()
                runner.assert_called_once()
            self.assertEqual((cached.read_bytes(), published.read_bytes()), before)

    def test_resume_counts_only_selected_jobs_and_preserves_library_history(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            scene_root = output / "scene_cfg"
            selected = scene_root / "selected/floating/scale_one.npy"
            selected.parent.mkdir(parents=True)
            np.save(selected, {"scene": {"selected": {"type": "rigid_object", "scale": np.ones(3) * .08}},
                               "task": {"obj_name": "selected"}})
            taxonomy = "24_Tip_Pinch"
            database = sqlite3.connect(output / "progress.sqlite")
            database.execute("CREATE TABLE completed (job_key TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
            database.execute("CREATE TABLE status_totals (status TEXT PRIMARY KEY, count INTEGER NOT NULL)")
            records = [
                (f"{taxonomy}/selected/floating/scale_one/0", {"status_counts": {"valid": 3, "grasp_required_contact": 8189}}),
                (f"{taxonomy}/other/floating/scale_two/0", {"status_counts": {"valid": 5, "grasp_required_contact": 8187}}),
            ]
            database.executemany("INSERT INTO completed VALUES (?, ?)", [(key, json.dumps(data)) for key, data in records])
            database.executemany("INSERT INTO status_totals VALUES (?, ?)", [("valid", 8), ("grasp_required_contact", 16376)])
            database.commit()
            database.close()
            args = Namespace(scene_root=str(scene_root), scene="*/floating", target="*", single_object=True,
                             epochs=1, points=1024, inplane=8, gpu=2, workers=84,
                             keep_failed=False, dry_run=False, collect_only=False)
            with patch("torch.cuda.set_device", side_effect=AssertionError("Completed selection must not initialize GPU")):
                run_all_candidates(args, [selected], output / "unused", output, [taxonomy],
                                   str(scene_root / "*/floating/*.npy"))
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(progress["planned_groups"], 1)
            self.assertEqual(progress["completed_groups"], 1)
            self.assertEqual(progress["attempted"], 8192)
            self.assertEqual(progress["new_valid_grasps"], 3)
            self.assertEqual(progress["library_completed_groups"], 2)
            self.assertEqual(progress["library_valid_grasps"], 8)
            database = sqlite3.connect(output / "progress.sqlite")
            self.assertEqual(database.execute("SELECT COUNT(*) FROM completed").fetchone()[0], 2)
            self.assertEqual(database.execute("SELECT count FROM status_totals WHERE status='valid'").fetchone()[0], 8)
            database.close()

    def test_empty_library_is_a_valid_filter_result_without_source_scene_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scene = {"scene": {"object": {"type": "rigid_object", "scale": np.ones(3),
                                         "pose": np.asarray([0, 0, 0, 1, 0, 0, 0])}},
                     "task": {"obj_name": "object"}}
            scene_path = root / "scene_000001/object.npy"
            scene_path.parent.mkdir()
            np.save(scene_path, scene)
            before = scene_path.read_bytes()
            source_scene = root / "object.npy"
            np.save(source_scene, scene)
            source = root / "library/24_Tip_Pinch/object/floating/scale_test.npy"
            source.parent.mkdir(parents=True)
            np.save(source, {phase: np.empty((0, 29)) for phase in
                             ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")} |
                    {"scene_path": str(source_scene)})
            cfg = OmegaConf.create({"source_grasp_dir": str(root / "library"),
                                    "output_dir": str(root / "filtered"),
                                    "single_object_library": True, "pool_scale_variants": False,
                                    "taxonomies": ["24_Tip_Pinch"], "hand_xml": "unused",
                                    "lift": {"enabled": False}})
            with patch("dexonomy.sim.MuJoCo_OptEnv"):
                records = _filter_scene(str(scene_path), cfg)
            self.assertEqual(records[0]["status"], "empty")
            self.assertEqual(records[0]["valid_count"], 0)
            self.assertEqual(scene_path.read_bytes(), before)

    def test_scene_pose_and_obstacles_do_not_change_object_scale_identity(self):
        scene = {"scene": {
            "object": {"type": "rigid_object", "file_path": "/asset/mesh.obj",
                       "xml_path": "/asset/coacd.xml", "scale": np.ones(3) * .07,
                       "pose": np.asarray([1, 2, 3, 1, 0, 0, 0])},
            "table": {"type": "plane"}, "obstacle": {"type": "rigid_object"}},
            "task": {"obj_name": "object"}}
        relative, isolated = canonical_object_scene(scene)
        moved = copy.deepcopy(scene)
        moved["scene"]["object"]["pose"] = np.asarray([8, 9, 10, 0, 1, 0, 0])
        moved["scene"].pop("obstacle")
        self.assertEqual(canonical_object_scene(moved)[0], relative)
        self.assertEqual(set(isolated["scene"]), {"object"})
        np.testing.assert_array_equal(isolated["scene"]["object"]["pose"], [0, 0, 0, 1, 0, 0, 0])
        moved["scene"]["object"]["scale"] *= 2
        self.assertNotEqual(canonical_object_scene(moved)[0], relative)
        np.testing.assert_array_equal(scene["scene"]["object"]["pose"], [1, 2, 3, 1, 0, 0, 0])

    def test_all_8192_candidates_use_single_object_validation_and_publish_reusable_library(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path("object/floating/scale_example.npy")
            path = root / "scene_cfg" / relative
            path.parent.mkdir(parents=True)
            np.save(path, {"scene": {"object": {"type": "rigid_object", "scale": np.ones(3)}},
                           "task": {"obj_name": "object"}})
            count = 1024 * 8
            qpos = np.zeros((count, 29), np.float32)
            qpos[:, 3] = 1
            qpos[:, 0] = np.arange(count) / 10000
            job = {"output": str(root), "scene_path": str(path), "taxonomy": "24_Tip_Pinch",
                   "epoch": 0, "points": 1024, "inplane": 8, "single_object": True,
                   "source_relative": str(relative), "job_key": "24_Tip_Pinch/object/floating/scale_example/0",
                   "run_fingerprint": "single-object", "keep_failed": False,
                   "hand_cbody": ["finger"], "required_cbody": [["finger"]],
                   "grasp_config": {"hand": {"xml_path": "unused"}, "op": {"pregrasp": {"cdist": .045}}},
                   "eval_config": {}, "filter_config": {},
                   "inputs": {"grasp_qpos": qpos, "obj_cpn_w": np.zeros((count, 1, 6)),
                              "hand_cpn_w": np.zeros((count, 1, 6)), "ext_center": np.zeros((count, 3)),
                              "ext_wrench": np.zeros((count, 6)), "n_evo": np.zeros((count, 1)),
                              "obj_pose": np.tile([0, 0, 0, 1, 0, 0, 0], (count, 1)),
                              "obj_scale": np.ones((count, 3))}}
            calls = []

            def optimize(data, config, environment):
                index = len(calls)
                calls.append(index)
                data["pregrasp_qpos"] = data["grasp_qpos"].copy()
                data["squeeze_qpos"] = data["grasp_qpos"].copy()
                return data, index % 4 == 0, "grasp_required_contact"

            with (
                patch("dexonomy.sim.MuJoCo_OptEnv"),
                patch("dexonomy.op.gen_grasp.optimize_grasp_data", side_effect=optimize),
                patch("dexonomy.op.eval_data.evaluate_grasp_data", return_value=(True, object())),
                patch("dexonomy.op.native_clutter_batch.check_clutter_grasp",
                      side_effect=AssertionError("Stage 1 must not apply clutter filtering")),
            ):
                metadata = process_native_batch(job)
            self.assertEqual(len(calls), count)
            self.assertEqual(metadata["status_counts"]["valid"], 2048)
            merge_validated_target(metadata, root, root / "unused")
            library = np.load(root / "succ_collect/24_Tip_Pinch" / relative, allow_pickle=True).item()
            self.assertEqual(library["scene_path"], str(path))
            np.testing.assert_array_equal(library["source_indices"], np.arange(0, count, 4))
            np.testing.assert_array_equal(library["grasp_qpos"], qpos[::4])
            with np.load(metadata["archive_path"], allow_pickle=False) as archive:
                self.assertEqual(json.loads(str(archive["metadata"]))["archive_scope"], "valid_only")
                np.testing.assert_array_equal(archive["sample_point_indices"], np.arange(0, count, 4) % 1024)
                np.testing.assert_array_equal(archive["inplane_indices"], np.arange(0, count, 4) // 1024)


if __name__ == "__main__":
    unittest.main()
