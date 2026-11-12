import copy
from io import BytesIO
import hashlib
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dexonomy.util.file_util import load_scene_cfg
from scripts import prepare_dexgraspnet2_test_scenes as importer


class DexGraspNet2TestSceneTests(unittest.TestCase):
    def annotation(self):
        return {
            "shape_a": {"dataset": "acronym", "rest_pose_trans": np.array([.1, .2, .3]),
                        "rest_pose_quat": np.array([0., 0., np.sqrt(.5), np.sqrt(.5)])},
            "shape_b": {"dataset": "acronym", "rest_pose_trans": np.array([-.1, -.2, .4]),
                        "rest_pose_quat": np.array([0., 0., 0., 1.])},
        }

    def test_official_xyzw_and_table_offset_without_changing_annotation(self):
        annotation = self.annotation()
        before = copy.deepcopy(annotation)
        poses = importer.object_poses(annotation)
        np.testing.assert_allclose(poses["shape_a"], [.1, .2, .25, np.sqrt(.5), 0., 0., np.sqrt(.5)])
        np.testing.assert_allclose(poses["shape_b"], [-.1, -.2, .35, 1., 0., 0., 0.])
        for name in annotation:
            np.testing.assert_array_equal(annotation[name]["rest_pose_trans"], before[name]["rest_pose_trans"])
            np.testing.assert_array_equal(annotation[name]["rest_pose_quat"], before[name]["rest_pose_quat"])

    def test_each_target_keeps_neighbors_original_scale_and_resolvable_mesh_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "model/surface.obj"
            mesh.parent.mkdir()
            mesh.write_text("v 0 0 0\n")
            model = {"type": "rigid_object", "file_path": str(mesh), "scale": np.ones(3),
                     "pose": np.array([0., 0., 0., 1., 0., 0., 0.])}
            models = {name: (mesh.with_suffix(".npy"), model) for name in self.annotation()}
            record = importer.convert_scene("scene_dense_0",
                "acronym_test_scenes/test_acronym_dense/scene_dense_0.npz",
                self.annotation(), root, models)
            self.assertEqual(len(record["target_configs"]), 2)
            for relative in record["target_configs"]:
                saved = load_scene_cfg(str(root / relative))
                self.assertEqual(set(saved["scene"]), {"shape_a", "shape_b", "table"})
                self.assertEqual(saved["task"]["obj_name"], Path(relative).stem)
                np.testing.assert_allclose(saved["scene"]["shape_a"]["pose"][:3], [.1, .2, .25])
                np.testing.assert_array_equal(saved["scene"]["shape_a"]["scale"], [1, 1, 1])
                self.assertEqual(Path(saved["scene"]["shape_b"]["file_path"]).resolve(), mesh)
                self.assertEqual(saved["scene"]["table"]["pose"][2], 0.)
            np.testing.assert_array_equal(model["pose"], [0, 0, 0, 1, 0, 0, 0])

    def test_rejects_invalid_pose_and_wrong_dataset(self):
        for field, value in [("rest_pose_trans", [0, 0, np.nan]),
                             ("rest_pose_quat", [0, 0, 0, 0]),
                             ("rest_pose_quat", [0, 0, 1]), ("dataset", "graspnet")]:
            with self.subTest(field=field, value=value):
                annotation = self.annotation()
                annotation["shape_a"][field] = value
                with self.assertRaises(ValueError):
                    importer.object_poses(annotation)

    def test_reads_only_annotations_and_checks_checksum_before_unpickling(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "test.tar.gz"
            payload = BytesIO()
            np.savez(payload, self.annotation())
            member = "acronym_test_scenes/test_acronym_dense/scene_dense_0.npz"
            with tarfile.open(archive, "w:gz") as bundle:
                for name, data in [(member, payload.getvalue()),
                                   ("acronym_test_scenes/network_input_dense/large.npz", b"ignored")]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    bundle.addfile(info, BytesIO(data))
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch.dict(importer.ARCHIVES, {"inventory": (archive.name, archive.stat().st_size, checksum)}):
                annotations = importer.read_annotations(archive)
                self.assertEqual(list(annotations), ["scene_dense_0"])
                self.assertEqual(annotations["scene_dense_0"][0], member)
            with (patch.dict(importer.ARCHIVES,
                             {"inventory": (archive.name, archive.stat().st_size, "0" * 64)}),
                  patch.object(importer.np, "load") as load):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    importer.read_annotations(archive)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
