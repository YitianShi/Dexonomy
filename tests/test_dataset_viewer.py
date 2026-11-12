from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import socket
import unittest
from unittest.mock import patch

import numpy as np

from scripts.view_clutter_grasps_mujoco import (
    PHASES, collect_scene_groups, configure_display, load_scene_data, parse_args,
)
from scripts.grasp_viewer_panel import build_viewer_command, read_group_details, scale_description


class DatasetViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root / "scene.npy"
        np.save(self.scene, {"scene": {}})

    def sample(self, relative, target="031"):
        path = self.root / "grasps" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        poses = np.zeros((3, 29))
        poses[:, 3] = 1
        poses[:, 0] = [0, .1, .2]
        np.save(path, {"scene_path": str(self.scene), "target_name": target,
                       **{f"{phase}_qpos": poses.copy() for phase in PHASES}})
        return path

    def args(self, *extra):
        return parse_args(["--root", str(self.root / "grasps"), *extra])

    def test_object_scales_stay_separate_across_taxonomies(self):
        for taxonomy in ["1_Large_Diameter", "2_Small_Diameter"]:
            for scale in ["scale_a", "scale_b"]:
                self.sample(f"{taxonomy}/031/floating/{scale}.npy")
        groups, total = collect_scene_groups(self.args("--object", "031"))
        self.assertEqual(total, 2)
        self.assertEqual({g.name for g in groups}, {"031/floating/scale_a", "031/floating/scale_b"})
        for group in groups:
            data = load_scene_data(group, max_grasps=0)
            self.assertEqual(len(data.grasps), 6)
            self.assertEqual(len({g.source_path.stem for g in data.grasps}), 1)

    def test_scene_fps_preserves_each_target(self):
        self.sample("1_Large_Diameter/scene_0100/031.npy", "031")
        self.sample("1_Large_Diameter/scene_0100/046.npy", "046")
        groups, total = collect_scene_groups(self.args("--scene", "100"))
        self.assertEqual(total, 1)
        data = load_scene_data(groups[0], max_grasps=1)
        self.assertEqual({g.target_name for g in data.grasps}, {"031", "046"})
        self.assertEqual(len(data.grasps), 2)

    def test_mismatched_phases_fail_without_truncating(self):
        path = self.sample("1_Large_Diameter/031/floating/scale_a.npy")
        sample = np.load(path, allow_pickle=True).item()
        sample["squeeze_qpos"] = sample["squeeze_qpos"][:1]
        np.save(path, sample)
        groups, _ = collect_scene_groups(self.args())
        with self.assertRaisesRegex(ValueError, "mismatched"):
            load_scene_data(groups[0], max_grasps=1)

    def test_random_sampling_keeps_targets_and_unique_source_rows(self):
        for target in ("031", "046"):
            for taxonomy in ("1_Large_Diameter", "2_Small_Diameter"):
                self.sample(f"{taxonomy}/scene_0100/{target}.npy", target)
        groups, _ = collect_scene_groups(self.args("--scene", "100"))
        first = load_scene_data(groups[0], 2, 12, "random")
        repeated = load_scene_data(groups[0], 2, 12, "random")
        keys = [(g.source_path, g.source_index) for g in first.grasps]
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(set(keys)), 4)
        self.assertEqual(keys, [(g.source_path, g.source_index) for g in repeated.grasps])
        self.assertEqual([g.target_name for g in first.grasps].count("031"), 2)
        self.assertEqual([g.target_name for g in first.grasps].count("046"), 2)

    def test_panel_command_selects_exact_scale_type_and_sampling(self):
        self.sample("1_Large_Diameter/031/floating/scale_a.npy")
        self.sample("1_Large_Diameter/031/floating/scale_b.npy")
        self.sample("2_Small_Diameter/031/floating/scale_a.npy")
        options = self.args()
        command = build_viewer_command(
            options, options.root, "031/floating/scale_a", "1_Large_Diameter",
            "2", "7", "random", "pregrasp", .5,
        )
        selected = parse_args(command[2:])
        groups, total = collect_scene_groups(selected)
        self.assertEqual(total, 1)
        self.assertEqual(groups[0].name, "031/floating/scale_a")
        data = load_scene_data(groups[0], selected.max_grasps, selected.sample_seed, selected.sampling)
        self.assertEqual(len(data.grasps), 2)
        self.assertEqual({g.taxonomy for g in data.grasps}, {"1_Large_Diameter"})
        self.assertEqual(selected.phase, "pregrasp")

    def test_scale_label_uses_numeric_scale(self):
        self.assertEqual(scale_description([1, 1, 1]), "原始尺度 · 1.00×")
        self.assertEqual(scale_description([.5, .5, .5]), "等比缩放 · 0.5×")
        self.assertIn("X 2× / Y 0.5× / Z 1×", scale_description([2, .5, 1]))
        for invalid in ([0, 1, 1], [float('nan'), 1, 1], [-1, -1, -1]):
            with self.assertRaises(ValueError):
                scale_description(invalid)

    def test_dimensions_use_mesh_scale_not_hash_or_world_rotation(self):
        import trimesh
        mesh_path = self.root / "box.obj"
        trimesh.creation.box(extents=[.2, .3, .4]).export(mesh_path)
        np.save(self.scene, {"scene": {"031": {
            "type": "rigid_object", "file_path": str(mesh_path),
            "scale": [2, .5, 1],
            "pose": [5, 6, 7, np.sqrt(.5), 0, 0, np.sqrt(.5)],
        }}})
        self.sample("1_Large_Diameter/031/floating/scale_deadbeef.npy")
        groups, _ = collect_scene_groups(self.args())
        details = read_group_details(groups[0])
        np.testing.assert_allclose(details["dimensions"], [.4, .15, .4])
        np.testing.assert_allclose(details["scale"], [2, .5, 1])
        self.assertNotIn("deadbeef", details["scale_label"])
        vertices, _ = details["geometries"][0]
        np.testing.assert_allclose(vertices.mean(axis=0), [5, 6, 7])

    def test_catalog_paths_are_relative_to_catalog(self):
        self.sample("1_Large_Diameter/031/floating/scale_a.npy")
        catalog = self.root / "catalog.json"
        catalog.write_text(json.dumps({"datasets": [{"id": "example", "data": "grasps"}]}))
        args = parse_args(["--catalog", str(catalog), "--dataset", "example"])
        self.assertEqual(Path(args.root), self.root / "grasps")
        explicit = parse_args(["--root", str(self.root / "grasps"),
                               "--catalog", str(self.root / "missing.json")])
        self.assertEqual(Path(explicit.root), self.root / "grasps")

    def test_default_prefers_dedup_with_legacy_fallback(self):
        catalog = self.root/'catalog.json'
        entries = [{'id': 'graspnet_test_clutter', 'data': 'original'},
                   {'id': 'deduplicated_graspnet_test_clutter', 'data': 'dedup'}]
        catalog.write_text(json.dumps({'datasets': entries}))
        args = parse_args(['--catalog', str(catalog)])
        self.assertEqual(args.dataset, 'deduplicated_graspnet_test_clutter')
        self.assertEqual(Path(args.root), self.root/'dedup')
        catalog.write_text(json.dumps({'datasets': entries[:1]}))
        self.assertEqual(parse_args(['--catalog', str(catalog)]).dataset, 'graspnet_test_clutter')


@unittest.skipUnless(os.name == "posix", "Local X11 sockets require POSIX")
class ViewerDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = []
        self.add_display(1)
        self.env = patch.dict(os.environ, {"DISPLAY": ":0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.glob = patch("scripts.view_clutter_grasps_mujoco.Path.glob",
                          return_value=self.paths)
        self.glob.start()
        self.addCleanup(self.glob.stop)
        self.platform = patch("scripts.view_clutter_grasps_mujoco.sys.platform", "linux")
        self.platform.start()
        self.addCleanup(self.platform.stop)

    def add_display(self, number):
        path = Path(self.temp.name) / f"X{number}"
        server = socket.socket(socket.AF_UNIX)
        self.addCleanup(server.close)
        server.bind(str(path))
        self.paths.append(path)

    def test_inherited_display_switches_to_only_user_owned_desktop(self):
        configure_display()
        self.assertEqual(os.environ["DISPLAY"], ":1")

    def test_explicit_display_is_preserved(self):
        configure_display(":0")
        self.assertEqual(os.environ["DISPLAY"], ":0")

    def test_ssh_forwarding_is_preserved(self):
        os.environ["DISPLAY"] = "localhost:10.0"
        configure_display()
        self.assertEqual(os.environ["DISPLAY"], "localhost:10.0")

    def test_multiple_desktops_do_not_trigger_a_guess(self):
        self.add_display(2)
        configure_display()
        self.assertEqual(os.environ["DISPLAY"], ":0")

    def test_current_owned_display_and_screen_are_preserved(self):
        self.add_display(2)
        os.environ["DISPLAY"] = ":1.0"
        configure_display()
        self.assertEqual(os.environ["DISPLAY"], ":1.0")


if __name__ == "__main__":
    unittest.main()
