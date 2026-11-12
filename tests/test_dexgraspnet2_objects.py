from io import BytesIO
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

from dexonomy.util.file_util import load_scene_cfg
from scripts.prepare_dexgraspnet2_objects import (
    build_benchmark_inventory, extract_full_model_archive, extract_models, mujoco_convex_part, prepare_object,
    read_inventory, read_model_inventory,
)
from scripts.generate_single_object_grasps import DATASET_PRESETS, parse_args


class DexGraspNet2ObjectTests(unittest.TestCase):
    object_id = "a" * 32

    def annotation_archive(self, path, annotations):
        with tarfile.open(path, "w:gz") as bundle:
            for index, annotation in enumerate(annotations):
                data = BytesIO()
                np.savez(data, annotation)
                member = tarfile.TarInfo(f"acronym_test_scenes/test_acronym_dense/scene_dense_{index}.npz")
                member.size = len(data.getvalue())
                bundle.addfile(member, BytesIO(data.getvalue()))

    def test_inventory_deduplicates_shape_ids_without_retaining_clutter_poses(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "scenes.tar.gz"
            body = {"dataset": "acronym", "rest_pose_trans": np.asarray([1, 2, 3])}
            self.annotation_archive(archive, [{self.object_id: body},
                                              {self.object_id: body, "b" * 32: body, "room11": body}])
            inventory = read_inventory(archive)
            self.assertEqual(inventory["object_ids"], [self.object_id, "b" * 32, "room11"])
            self.assertEqual(inventory["object_count"], 3)
            self.assertEqual(inventory["named_object_ids"], ["room11"])
            self.assertEqual(inventory["annotation_count"], 2)
            self.assertNotIn("rest_pose_trans", json.dumps(inventory))

    def test_inventory_rejects_graspnet_objects(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "scenes.tar.gz"
            self.annotation_archive(archive, [{"000": {"dataset": "graspnet"}}])
            with self.assertRaisesRegex(ValueError, "Unexpected ShapeNet"):
                read_inventory(archive)

    def test_model_extraction_selects_only_inventory_geometry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "models.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for object_id in (self.object_id, "b" * 32):
                    for filename in ("scaled.obj", "simplified.obj", "collision.urdf", "texture.png"):
                        member = tarfile.TarInfo(f"acronym/meshes/models/{object_id}/{filename}")
                        member.size = 4
                        bundle.addfile(member, BytesIO(b"data"))
            extract_models(archive, root / "raw", [self.object_id])
            self.assertEqual({p.name for p in (root / "raw" / self.object_id).iterdir()},
                             {"scaled.obj", "simplified.obj", "collision.urdf"})
            self.assertFalse((root / "raw" / ("b" * 32)).exists())

    def test_full_inventory_includes_unannotated_models_and_excludes_missing_scaled_mesh(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "models.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for object_id, filenames in {
                    self.object_id: ["scaled.obj", "simplified.obj", "collision.urdf"],
                    "room11": ["scaled.obj", "simplified.obj", "collision.urdf"],
                    "zapatos^hombre": ["scaled.obj", "simplified.obj", "collision.urdf"],
                    "incomplete": ["simplified.obj", "collision.urdf"],
                    "empty_mesh": ["scaled.obj", "simplified.obj", "collision.urdf"],
                }.items():
                    for filename in filenames:
                        data = b"" if object_id == "empty_mesh" and filename == "scaled.obj" else b"model"
                        member = tarfile.TarInfo(f"acronym/meshes/models/{object_id}/{filename}")
                        member.size = len(data)
                        bundle.addfile(member, BytesIO(data))
            inventory = read_model_inventory(archive)
            self.assertEqual(inventory["object_ids"], [self.object_id, "room11", "zapatos^hombre"])
            self.assertEqual(inventory["archive_model_directory_count"], 5)
            self.assertEqual(inventory["incomplete_geometry_count"], 2)
            self.assertEqual(inventory["incomplete_geometry"]["incomplete"], ["scaled.obj"])
            self.assertEqual(inventory["incomplete_geometry"]["empty_mesh"], ["scaled.obj"])

    def test_full_model_extraction_preserves_files_and_rejects_path_escape(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "models.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("acronym/meshes/models/incomplete/collision.urdf")
                member.size = 5
                bundle.addfile(member, BytesIO(b"model"))
            result = extract_full_model_archive(archive, root / "output", "acronym")
            self.assertEqual(result["files"], 1)
            self.assertEqual((root / "output/acronym/meshes/models/incomplete/collision.urdf").read_bytes(), b"model")
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("acronym/../../escape.obj")
                member.size = 4
                bundle.addfile(member, BytesIO(b"oops"))
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                extract_full_model_archive(archive, root / "output", "acronym")
            self.assertFalse((root / "escape.obj").exists())

    def test_benchmark_inventory_excludes_unannotated_models_in_full_pool(self):
        catalog = [{"dataset": "graspnet", "object_id": f"{i:03d}"} for i in range(88)]
        catalog += [{"dataset": "acronym", "object_id": name}
                    for name in [self.object_id, "room11", "unannotated_extra"]]
        inventory = build_benchmark_inventory(
            {"object_ids": [self.object_id, "room11"], "annotation_count": 2}, catalog)
        self.assertEqual(inventory["confirmed_object_count"], 90)
        self.assertEqual(inventory["confirmed_graspnet_object_count"], 88)
        self.assertEqual(inventory["unresolved_benchmark_id_count"], 1229)
        self.assertEqual(inventory["status"], "incomplete_id_mapping")
        self.assertNotIn("unannotated_extra", [item["object_id"] for item in inventory["objects"]])

    def test_uses_official_scaled_surface_for_sampling_and_convex_geometry(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "raw_source" / self.object_id
            source.mkdir(parents=True)
            physical = trimesh.creation.box(extents=[.1, .2, .3])
            physical.apply_translation([.01, -.02, .03])
            physical.export(source / "scaled.obj")
            unscaled = physical.copy()
            unscaled.apply_scale(100)
            unscaled.export(source / "simplified.obj")
            (source / "collision.urdf").write_text('\n<?xml version="1.0"?>\n<robot name="object"><link name="base">'
                '<collision><geometry><mesh filename="scaled.obj"/></geometry></collision></link></robot>')
            original_bytes = (source / "scaled.obj").read_bytes()
            original_unscaled = (source / "simplified.obj").read_bytes()

            def decompose(vertices, faces, **settings):
                np.testing.assert_allclose(np.ptp(vertices, axis=0), [.1, .2, .3])
                triangles = np.asarray(faces).reshape(-1, 4)
                np.testing.assert_array_equal(triangles[:, 0], np.full(len(physical.faces), 3))
                return [(vertices, triangles[:, 1:])]

            with patch("vhacdx.compute_vhacd", side_effect=decompose):
                record = prepare_object(self.object_id, output)
            scene = load_scene_cfg(record["scene_path"])
            self.assertEqual(set(scene["scene"]), {self.object_id})
            body = scene["scene"][self.object_id]
            np.testing.assert_array_equal(body["scale"], [1, 1, 1])
            np.testing.assert_array_equal(body["pose"], [0, 0, 0, 1, 0, 0, 0])
            self.assertEqual(Path(body["file_path"]).read_bytes(), original_bytes)
            self.assertEqual((source / "simplified.obj").read_bytes(), original_unscaled)
            model = mujoco.MjModel.from_xml_path(body["xml_path"])
            self.assertAlmostEqual(model.body("object").mass[0], .006, places=7)
            np.testing.assert_allclose(model.body("object").ipos, [.01, -.02, .03], atol=1e-7)
            with patch("vhacdx.compute_vhacd", side_effect=AssertionError("Must resume")):
                self.assertEqual(prepare_object(self.object_id, output), record)

    def test_launch_defaults_fix_a_separate_library_and_preserve_existing_defaults(self):
        with patch("sys.argv", ["generate"]):
            remaining = parse_args("dexgraspnet2")
            current = parse_args("graspnet")
        self.assertEqual((remaining.object, remaining.gpu, remaining.workers), ("*", 0, 84))
        self.assertNotEqual(remaining.output, current.output)
        self.assertNotEqual(remaining.object_root, current.object_root)
        self.assertEqual((current.object, current.gpu, current.workers), ("000", 2, 84))
        for key in ("--scene", "--object-root", "--output"):
            with patch("sys.argv", ["generate", key, "unexpected"]), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args("dexgraspnet2")
        self.assertEqual(remaining.output, str(DATASET_PRESETS["dexgraspnet2"]["output"]))

    def test_planar_convex_fragment_is_retained_and_compiles_for_collisions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.asarray([[0, 0, .03], [.02, 0, .03], [.02, .03, .03], [0, .03, .03]])
            mesh, regularized = mujoco_convex_part(points, [[0, 1, 2], [0, 2, 3]])
            self.assertTrue(regularized)
            np.testing.assert_allclose(mesh.bounds[:, :2], [[0, 0], [.02, .03]], atol=1e-12)
            np.testing.assert_allclose(mesh.bounds[:, 2], [.03 - .5e-6, .03 + .5e-6], atol=1e-12)
            mesh.export(root / "part.obj", digits=17)
            (root / "model.xml").write_text('<mujoco><asset><mesh name="part" file="part.obj" '
                'inertia="shell"/></asset><worldbody><body><geom type="mesh" mesh="part" '
                'mass="0"/></body></worldbody></mujoco>')
            model = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
            self.assertEqual(model.ngeom, 1)
            self.assertEqual(model.geom_contype[0], 1)

    def test_real_decomposition_preserves_box_geometry_and_volume(self):
        surface = trimesh.creation.box(extents=[.1, .2, .3])
        surface.apply_translation([.015, -.025, .035])
        parts = trimesh.decomposition.convex_decomposition(surface, asyncACD=False)
        meshes = [trimesh.Trimesh(**part, process=False) for part in parts]
        self.assertTrue(meshes)
        np.testing.assert_allclose(trimesh.util.concatenate(meshes).bounds, surface.bounds, atol=1e-6)
        volume = sum(abs(mesh.volume) for mesh in meshes)
        self.assertAlmostEqual(volume, .006, delta=.006 * .02)


if __name__ == "__main__":
    unittest.main()
