import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dexonomy.op import gen_clutter
from dexonomy.sim.clutter_env import ClutterSimCfg


class ClutterPipelineTests(unittest.TestCase):
    def setUp(self):
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.cfg = SimpleNamespace(
            output_dir=str(self.root / "scenes"), start_index=25, num_scenes=2,
            # Existing scenes must be filtered even when this is false.
            skip_existing=False,
        )
        self.filter_cfg = SimpleNamespace(output_dir=str(self.root / "grasps"))
        self.op_cfg = SimpleNamespace(
            scene=self.cfg, filter=self.filter_cfg, stage="all", seed=36,
            manifest_path=str(self.root / "manifest.json"),
        )

    def _write_scene(self, index, target="object", output_dir=None):
        path = Path(output_dir or self.cfg.output_dir) / f"scene_{index:06d}" / f"{target}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, {
            "scene": {}, "task": {"obj_name": target},
            "clutter": {"object_names": [target], "settle": {"reason": "stable"}},
        })
        return str(path)

    def _generate(self, index, cfg, *args):
        path = self._write_scene(index, output_dir=cfg.output_dir)
        return [path], {"scene_index": index, "target_paths": [path]}

    def _process(self, existing_paths=None):
        return gen_clutter._process_clutter_scene(
            25, self.cfg, ClutterSimCfg(), {}, [], {}, np.random.default_rng(36),
            self.filter_cfg, existing_paths,
        )

    def test_each_scene_is_filtered_before_generating_next(self):
        events = []

        def generate(index, cfg, *args):
            events.append(("generate", index))
            return self._generate(index, cfg)

        def filter_target(path, cfg):
            events.append(("filter", int(Path(path).parent.name.removeprefix("scene_"))))
            return [{"valid_count": 1}]

        with (
            patch.object(gen_clutter, "_scene_generation_inputs", return_value=(ClutterSimCfg(), {}, [])),
            patch.object(gen_clutter, "_generate_scene_until_valid", side_effect=generate),
            patch.object(gen_clutter, "_filter_scene", side_effect=filter_target),
        ):
            _, records, _ = gen_clutter._generate_and_filter_scenes(
                self.op_cfg, np.random.default_rng(36)
            )
        self.assertEqual(events, [("generate", 25), ("filter", 25), ("generate", 26), ("filter", 26)])
        self.assertEqual([r["scene_index"] for r in records], [25, 26])
        self.assertTrue(all(r["valid_grasp_count"] == 1 for r in records))

    def test_existing_positive_scene_goes_directly_to_filter(self):
        existing = self._write_scene(25)
        original = Path(existing).read_bytes()
        self.cfg.num_scenes = 1
        with (
            patch.object(gen_clutter, "_scene_generation_inputs", return_value=(ClutterSimCfg(), {}, [])),
            patch.object(gen_clutter, "_generate_scene_until_valid") as generate,
            patch.object(gen_clutter, "_filter_scene", return_value=[{"valid_count": 2}]) as filter_target,
        ):
            paths, records, _ = gen_clutter._generate_and_filter_scenes(self.op_cfg, np.random.default_rng(36))
        generate.assert_not_called()
        filter_target.assert_called_once_with(existing, self.filter_cfg)
        self.assertEqual(paths, [existing])
        self.assertTrue(records[0]["reused_existing"])
        self.assertEqual(Path(existing).read_bytes(), original)

    def test_zero_scene_cleanup_and_repeated_resampling_keep_id(self):
        existing = self._write_scene(25, "old")
        neighboring = self._write_scene(26, "neighbor")
        neighbor_data = Path(neighboring).read_bytes()
        old_labels = []
        for taxonomy in ("enabled", "previous_taxonomy"):
            path = Path(self.filter_cfg.output_dir) / taxonomy / "scene_000025" / "old.npy"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"stale")
            old_labels.append(path)
        neighboring_label = Path(self.filter_cfg.output_dir) / "enabled" / "scene_000026" / "neighbor.npy"
        neighboring_label.parent.mkdir(parents=True)
        neighboring_label.write_bytes(b"keep")
        attempts = []
        generated_paths = []

        def generate(index, cfg, *args):
            self.assertEqual(index, 25)
            self.assertFalse(Path(existing).parent.exists())
            self.assertTrue(all(not path.exists() for path in old_labels + generated_paths))
            attempts.append(index)
            path = self._write_scene(index, f"new_{len(attempts)}", cfg.output_dir)
            generated_paths.append(Path(path))
            return [path], {"scene_index": index, "target_paths": [path]}

        with (
            patch.object(gen_clutter, "_generate_scene_until_valid", side_effect=generate),
            patch.object(gen_clutter, "_filter_scene", side_effect=[
                [{"valid_count": 0}], [{"status": "missing_source_grasp"}], [{"valid_count": 1}],
            ]),
        ):
            paths, record, filters = self._process([existing])
        self.assertEqual(attempts, [25, 25])
        self.assertEqual(paths, [str(generated_paths[-1])])
        self.assertEqual(list(Path(existing).parent.glob("*.npy")), [generated_paths[-1]])
        self.assertEqual(record["regeneration_failures"], {"zero_valid_grasps": 2})
        self.assertEqual(record["filter_attempts"], 3)
        self.assertFalse(record["reused_existing"])
        self.assertEqual(filters, [{"valid_count": 1}])
        self.assertEqual(Path(neighboring).read_bytes(), neighbor_data)
        self.assertEqual(neighboring_label.read_bytes(), b"keep")

    def test_counts_all_targets_and_taxonomies_without_requiring_each_target(self):
        paths = [self._write_scene(25, target) for target in ("empty", "positive")]
        with (
            patch.object(gen_clutter, "_generate_scene_until_valid") as generate,
            patch.object(gen_clutter, "_filter_scene", side_effect=[
                [{"taxonomy": "a", "valid_count": 0}, {"taxonomy": "b", "valid_count": 0}],
                [{"taxonomy": "a", "valid_count": 0}, {"taxonomy": "b", "valid_count": 3}],
            ]),
        ):
            saved, record, filters = self._process(paths)
        generate.assert_not_called()
        self.assertEqual(saved, paths)
        self.assertEqual(record["valid_grasp_count"], 3)
        self.assertEqual(len(filters), 4)

    def test_physics_and_zero_grasp_regeneration_counts_accumulate(self):
        existing = self._write_scene(25)

        def generate(index, cfg, *args):
            paths, record = self._generate(index, cfg)
            record.update(regeneration_count=2, regeneration_failures={"table_penetration": 2})
            return paths, record

        with (
            patch.object(gen_clutter, "_generate_scene_until_valid", side_effect=generate),
            patch.object(gen_clutter, "_filter_scene", side_effect=[
                [{"valid_count": 0}], [{"valid_count": 0}], [{"valid_count": 1}],
            ]),
        ):
            _, record, _ = self._process([existing])
        self.assertEqual(record["regeneration_count"], 6)
        self.assertEqual(record["regeneration_failures"], {"zero_valid_grasps": 2, "table_penetration": 4})

    def test_filter_stage_keeps_zero_grasp_scenes_and_only_filters_selected_targets(self):
        external_root = self.root / "external_scenes"
        existing = self._write_scene(25, "old", external_root)
        self._write_scene(25, "another", external_root)
        original = Path(existing).read_bytes()
        self.op_cfg.stage = "filter"
        self.filter_cfg.source_grasp_dir = str(self.root / "library")
        Path(self.filter_cfg.source_grasp_dir).mkdir()
        self.filter_cfg.scene_cfg_glob = str(external_root / "*" / "old.npy")
        with (
            patch.object(gen_clutter, "_scene_generation_inputs", side_effect=AssertionError("Filtering must not load generation inputs")),
            patch.object(gen_clutter, "_generate_scene_until_valid", side_effect=AssertionError("Filtering must not regenerate scenes")) as generate,
            patch.object(gen_clutter, "_filter_scene", return_value=[{"valid_count": 0}]) as filter_target,
        ):
            gen_clutter.operate_clutter(SimpleNamespace(op=self.op_cfg, n_worker=1))
        self.assertEqual(filter_target.call_count, 1)
        self.assertEqual(filter_target.call_args.args[0], existing)
        generate.assert_not_called()
        self.assertEqual(Path(existing).read_bytes(), original)
        self.assertFalse(Path(self.cfg.output_dir).exists())
        manifest = json.loads(Path(self.op_cfg.manifest_path).read_text())
        self.assertEqual(manifest["scene_records"], [])
        self.assertEqual(manifest["filter_records"], [{"valid_count": 0}])


if __name__ == "__main__":
    unittest.main()
