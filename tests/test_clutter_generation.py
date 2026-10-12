import json
from dataclasses import asdict
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

from dexonomy.op import gen_clutter
from dexonomy.sim.clutter_env import ClutterSimCfg, MuJoCoClutterEnv


class ClutterGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.cfg = SimpleNamespace(
            source_cfg_glob="unused",
            output_dir=str(self.root / "scene_cfg"),
            start_index=25,
            num_scenes=2,
            objects_min=1,
            objects_max=1,
            targets_per_scene=0,
            sample_dataset_scales=False,
            skip_existing=False,
            max_scene_attempts=30,
            max_spawn_attempts=10,
        )
        self.asset = {
            "name": "object",
            "scene_path": str(self.root / "source.npy"),
            "extent": np.full(3, 0.03),
            "cfg": {"type": "rigid_object", "scale": np.ones(3)},
        }
        self.poses = np.array([[0, 0, 0.015, 1, 0, 0, 0]], dtype=np.float32)
        self.penetration = {
            "reason": "table_penetration",
            "object_name": "object",
            "penetration_depth": 0.002,
            "fully_below_table": False,
        }

    def _stable_env(self):
        env = Mock(spec=MuJoCoClutterEnv)
        env.set_spawn_poses.return_value = True
        env.settle.return_value = (True, self.poses, {"reason": "stable"})
        return env

    def _generate(self, first_env, regenerate=False):
        next_env = self._stable_env()
        environments = [first_env]
        if regenerate:
            environments.append(self._stable_env())
        environments.append(next_env)
        with (
            patch.object(gen_clutter, "_scene_sim_cfg", return_value=ClutterSimCfg()),
            patch.object(gen_clutter, "_build_catalog", return_value={"object": [self.asset]}),
            patch.object(gen_clutter, "_runtime_objects", side_effect=lambda assets, cfg: assets),
            patch.object(gen_clutter, "MuJoCoClutterEnv", side_effect=environments) as factory,
        ):
            with self.assertLogs(level="INFO") as logs:
                paths, records = gen_clutter._generate_scenes(
                    self.cfg, np.random.default_rng(36)
                )
        self.assertEqual(factory.call_count, len(environments))
        for env in environments:
            env.close.assert_called_once()
        return paths, records, logs.output

    def _assert_regenerated_same_id(self, first_env):
        paths, records, logs = self._generate(first_env, regenerate=True)
        self.assertEqual(len(paths), 2)
        self.assertEqual([record["scene_index"] for record in records], [25, 26])
        self.assertEqual(records[0]["regeneration_count"], 1)
        self.assertEqual(records[0]["regeneration_failures"], {"table_penetration": 1})
        self.assertEqual(records[0]["settle"], {"reason": "stable"})
        expected = [Path(self.cfg.output_dir) / f"scene_{i:06d}" / "object.npy" for i in (25, 26)]
        self.assertEqual(paths, [str(path) for path in expected])
        for saved_path in expected:
            scene = np.load(saved_path, allow_pickle=True).item()
            self.assertEqual(scene["task"]["obj_name"], "object")
            np.testing.assert_array_equal(scene["scene"]["object"]["pose"], self.poses[0])
        self.assertTrue(any("Regenerate scene 000025" in line for line in logs))
        self.assertFalse(any("Skip scene 000025" in line for line in logs))
        self.assertFalse(any("Retry scene 000025" in line for line in logs))
        first_env.set_spawn_poses.assert_called_once()

    def test_initial_table_penetration_regenerates_same_number(self):
        env = self._stable_env()
        env.set_spawn_poses.return_value = False
        env.get_table_penetration.return_value = self.penetration
        self._assert_regenerated_same_id(env)
        env.settle.assert_not_called()

    def test_settle_table_penetration_regenerates_even_at_attempt_limit(self):
        for attempts in (1, 30):
            with self.subTest(max_scene_attempts=attempts):
                self.cfg.max_scene_attempts = attempts
                env = self._stable_env()
                env.settle.return_value = (False, self.poses, self.penetration)
                self._assert_regenerated_same_id(env)
                env.settle.assert_called_once()

    def test_timeout_can_still_retry_spawn(self):
        env = self._stable_env()
        env.settle.side_effect = [
            (False, self.poses, {"reason": "settle_timeout"}),
            (True, self.poses, {"reason": "stable"}),
        ]
        paths, records, _ = self._generate(env)
        self.assertEqual(len(paths), 2)
        self.assertEqual([record["scene_index"] for record in records], [25, 26])
        self.assertTrue(all(record["settle"]["reason"] == "stable" for record in records))
        self.assertEqual(env.set_spawn_poses.call_count, 2)

    def test_all_stage_writes_manifest_when_no_scenes_requested(self):
        manifest_path = self.root / "manifest.json"
        cfg = SimpleNamespace(
            n_worker=1,
            op=SimpleNamespace(
                stage="all",
                seed=36,
                scene=self.cfg,
                filter=SimpleNamespace(scene_cfg_glob="unused"),
                manifest_path=str(manifest_path),
            ),
        )
        cfg.op.scene.num_scenes = 0
        with (
            patch.object(gen_clutter, "_scene_generation_inputs") as generation_inputs,
            patch.object(gen_clutter, "_filter_scene") as filter_scene,
            self.assertLogs(level="WARNING"),
        ):
            gen_clutter.operate_clutter(cfg)
        generation_inputs.assert_not_called()
        filter_scene.assert_not_called()
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["scene_records"], [])
        self.assertEqual(manifest["filter_records"], [])

    def test_viewer_uses_one_process(self):
        sim_cfg = ClutterSimCfg(debug_view=True)
        with (
            patch.object(gen_clutter, "_scene_sim_cfg", return_value=sim_cfg),
            patch.object(gen_clutter, "_build_catalog", return_value={"object": [self.asset]}),
            patch.object(gen_clutter, "_generate_scene", return_value=([], {})) as generate,
            patch.object(gen_clutter.multiprocessing, "get_context") as get_context,
            self.assertLogs(level="INFO"),
        ):
            gen_clutter._generate_scenes(self.cfg, np.random.default_rng(36), n_worker=8)
        get_context.assert_not_called()
        self.assertEqual(generate.call_count, 2)

    def test_existing_scenes_are_not_submitted_to_workers(self):
        output_dir = Path(self.cfg.output_dir)
        existing = output_dir / "scene_000025" / "object.npy"
        existing.parent.mkdir(parents=True)
        np.save(existing, {"existing": True})
        self.cfg.skip_existing = True
        with (
            patch.object(gen_clutter, "_scene_sim_cfg", return_value=ClutterSimCfg()),
            patch.object(gen_clutter, "_build_catalog", return_value={"object": [self.asset]}),
            patch.object(gen_clutter, "_generate_scene", return_value=([], {"scene_index": 26})) as generate,
            patch.object(gen_clutter.multiprocessing, "get_context") as get_context,
            self.assertLogs(level="INFO"),
        ):
            paths, records = gen_clutter._generate_scenes(
                self.cfg, np.random.default_rng(36), n_worker=8
            )
        get_context.assert_not_called()
        generate.assert_called_once()
        self.assertEqual(generate.call_args.args[0], 26)
        self.assertEqual(paths, [str(existing)])
        self.assertEqual(records, [{"scene_index": 26}])
        self.assertEqual(np.load(existing, allow_pickle=True).item(), {"existing": True})

    def test_worker_thread_limits_restore_parent_environment_on_error(self):
        values = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "4"}
        with patch.dict(os.environ, values):
            original = dict(os.environ)
            with self.assertRaisesRegex(RuntimeError, "test exit"):
                with gen_clutter._scene_worker_environment():
                    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                        self.assertEqual(os.environ[key], "1")
                    raise RuntimeError("test exit")
            self.assertEqual(dict(os.environ), original)

    def test_scene_imports_do_not_load_training_or_rendering_dependencies(self):
        code = (
            "import sys; import dexonomy.main; from dexonomy.op import gen_clutter; "
            "assert not ({'torch', 'trimesh', 'scipy', 'imageio'} & set(sys.modules))"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


class ClutterMultiprocessingIntegrationTests(unittest.TestCase):
    def test_real_mujoco_workers_match_serial_and_resume(self):
        # Like Hydra runtime resolvers, this resolver exists only in the parent.
        OmegaConf.register_new_resolver("clutter_test_targets", lambda: 0)
        self.addCleanup(OmegaConf.clear_resolver, "clutter_test_targets")
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = root / "sources"
            radius = 0.02
            density = 1000.0
            for index in range(3):
                asset_dir = sources / f"object_{index}"
                floating_dir = asset_dir / "floating"
                floating_dir.mkdir(parents=True)
                xml_path = asset_dir / "model.xml"
                xml_path.write_text(
                    '<mujoco><worldbody><geom type="sphere" '
                    f'size="{radius}" density="{density}"/>'
                    '</worldbody></mujoco>'
                )
                info_path = asset_dir / "info.json"
                info_path.write_text(json.dumps({
                    "density": density,
                    "scale": 1.0,
                    "mass": 4 / 3 * np.pi * radius**3 * density,
                    "obb": [radius * 2] * 3,
                }))
                np.save(floating_dir / "scale001.npy", {
                    "scene": {
                        f"object_{index}": {
                            "type": "rigid_object",
                            "xml_path": str(xml_path),
                            "info_path": str(info_path),
                            "scale": np.ones(3),
                            "pose": np.array([0, 0, 0, 1, 0, 0, 0]),
                        },
                    },
                    "task": {"type": "force_closure", "obj_name": f"object_{index}"},
                })

            scene_cfg = {
                **asdict(ClutterSimCfg()),
                "source_scene_dir": str(sources),
                "source_cfg_glob": "${op.scene.source_scene_dir}/**/floating/*.npy",
                "output_dir": str(root / "serial"),
                "start_index": 25,
                "num_scenes": 6,
                "objects_min": 2,
                "objects_max": 2,
                "targets_per_scene": "${clutter_test_targets:}",
                "sample_dataset_scales": False,
                "skip_existing": True,
                "catalog_limit": 0,
                "min_extent": 0.03,
                "max_extent": 0.14,
                "max_scene_attempts": 3,
                "max_spawn_attempts": 3,
                "stable_time": 0.02,
                "settle_timeout": 3.0,
            }
            # Verify runtime interpolation is resolved before workers access it.
            cfg = OmegaConf.create({"op": {"scene": scene_cfg}}).op.scene
            serial_paths, serial_records = gen_clutter._generate_scenes(
                cfg, np.random.default_rng(36), n_worker=1
            )
            cfg.output_dir = str(root / "parallel")
            parallel_paths, parallel_records = gen_clutter._generate_scenes(
                cfg, np.random.default_rng(36), n_worker=2
            )
            self.assertEqual(len(serial_paths), 12)
            self.assertEqual(len(parallel_paths), 12)
            self.assertEqual([r["scene_index"] for r in parallel_records], list(range(25, 31)))
            self.assertEqual(
                [r["object_names"] for r in serial_records],
                [r["object_names"] for r in parallel_records],
            )
            for serial_path, parallel_path in zip(serial_paths, parallel_paths):
                self._assert_same_scene(serial_path, parallel_path)

            cfg.output_dir = str(root / "spawn_fallback")
            with patch.object(
                gen_clutter, "_scene_process_context",
                return_value=multiprocessing.get_context("spawn"),
            ):
                spawn_paths, _ = gen_clutter._generate_scenes(
                    cfg, np.random.default_rng(36), n_worker=2
                )
            self.assertEqual(len(spawn_paths), 12)
            for serial_path, spawn_path in zip(serial_paths, spawn_paths):
                self._assert_same_scene(serial_path, spawn_path)

            cfg.output_dir = str(root / "resumed")
            first_dir = Path(cfg.output_dir) / "scene_000025"
            first_dir.mkdir(parents=True)
            for serial_path in serial_paths:
                source = Path(serial_path)
                if source.parent.name == first_dir.name:
                    (first_dir / source.name).write_bytes(source.read_bytes())
            resumed_paths, resumed_records = gen_clutter._generate_scenes(
                cfg, np.random.default_rng(36), n_worker=2
            )
            self.assertEqual(len(resumed_paths), 12)
            self.assertEqual([r["scene_index"] for r in resumed_records], list(range(26, 31)))
            for serial_path, resumed_path in zip(serial_paths, resumed_paths):
                self._assert_same_scene(serial_path, resumed_path)

            # Mix a below-table asset with valid ones to exercise real rejection
            # and resampling inside worker processes without dropping scene IDs.
            bad_dir = sources / "below_table" / "floating"
            bad_dir.mkdir(parents=True)
            bad_xml = bad_dir.parent / "model.xml"
            bad_xml.write_text(
                '<mujoco><worldbody><geom type="sphere" pos="0 0 -1" '
                f'size="{radius}" density="{density}"/>'
                '</worldbody></mujoco>'
            )
            bad_source = bad_dir / "scale001.npy"
            np.save(bad_source, {
                "scene": {
                    "below_table": {
                        "type": "rigid_object",
                        "xml_path": str(bad_xml),
                        "info_path": str(sources / "object_0" / "info.json"),
                        "scale": np.ones(3),
                    },
                },
                "task": {"type": "force_closure", "obj_name": "below_table"},
            })
            cfg.source_cfg_glob = "${op.scene.source_scene_dir}/**/floating/*.npy"
            cfg.output_dir = str(root / "regenerated")
            cfg.start_index = 50
            cfg.num_scenes = 6
            cfg.objects_min = cfg.objects_max = 1
            regenerated_paths, regenerated_records = gen_clutter._generate_scenes(
                cfg, np.random.default_rng(36), n_worker=2
            )
            self.assertEqual(len(regenerated_paths), 6)
            self.assertEqual([r["scene_index"] for r in regenerated_records], list(range(50, 56)))
            self.assertTrue(all(r["settle"]["reason"] == "stable" for r in regenerated_records))
            self.assertTrue(any(r.get("regeneration_count", 0) > 0 for r in regenerated_records))
            for path in regenerated_paths:
                self.assertTrue(Path(path).is_file())

    def _assert_same_scene(self, first_path, second_path):
        first = np.load(first_path, allow_pickle=True).item()
        second = np.load(second_path, allow_pickle=True).item()
        self.assertEqual(first["scene_id"], second["scene_id"])
        self.assertEqual(first["task"], second["task"])
        self.assertEqual(first["clutter"]["settle"], second["clutter"]["settle"])
        self.assertEqual(list(first["scene"]), list(second["scene"]))
        for name in first["clutter"]["object_names"]:
            np.testing.assert_array_equal(first["scene"][name]["pose"], second["scene"][name]["pose"])


if __name__ == "__main__":
    unittest.main()
