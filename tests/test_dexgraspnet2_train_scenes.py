from io import BytesIO
import hashlib
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from dexonomy.util.file_util import load_scene_cfg
from scripts.prepare_dexgraspnet2_train_scenes import (
    build_manifest, convert_scene, download, extract_metadata, object_poses, selected_metadata,
)


class DexGraspNet2TrainSceneTests(unittest.TestCase):
    def annotation(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # A +90 degree rotation about Z in the official wxyz order.
        path.write_text('<scene><obj><obj_id>31</obj_id><pos_in_world>0.1 0.2 0.3</pos_in_world>'
                        '<ori_in_world>0.7071067811865476 0 0 0.7071067811865476</ori_in_world>'
                        '</obj><obj><obj_id>20</obj_id><pos_in_world>0.4 0.5 0.6</pos_in_world>'
                        '<ori_in_world>1 0 0 0</ori_in_world></obj></scene>')

    def test_camera_to_table_composes_translation_and_wxyz_rotation(self):
        with TemporaryDirectory() as directory:
            annotation = Path(directory) / '0000.xml'
            self.annotation(annotation)
            table_from_camera = np.eye(4)
            table_from_camera[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
            table_from_camera[:3, 3] = [1, 2, 3]
            poses = object_poses(annotation, table_from_camera)
            np.testing.assert_allclose(poses['031'][:3], [.8, 2.1, 3.3])
            rotation = Rotation.from_quat(poses['031'][[4, 5, 6, 3]]).as_matrix()
            np.testing.assert_allclose(rotation, np.diag([-1., -1., 1.]), atol=1e-12)

    def test_conversion_keeps_original_scale_and_both_objects_for_each_target(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw/scenes/scene_0000/realsense'
            self.annotation(raw / 'annotations/0000.xml')
            align = np.eye(4)
            align[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
            align[:3, 3] = [1, 2, 3]
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = [.2, -.1, .4]
            np.save(raw / 'cam0_wrt_table.npy', align)
            np.save(raw / 'camera_poses.npy', camera_pose[None])
            asset = root / 'models/surface.obj'
            asset.parent.mkdir()
            asset.write_text('v 0 0 0\nv .1 0 0\nv 0 .2 0\nf 1 2 3\n')
            cfg = {'type': 'rigid_object', 'file_path': str(asset), 'scale': np.ones(3),
                   'pose': np.asarray([0., 0., 0., 1., 0., 0., 0.])}
            models = {name: (asset, cfg) for name in ('020', '031')}
            record = convert_scene('scene_0000', root, 'realsense', 0, models)
            self.assertEqual(len(record['target_configs']), 2)
            np.testing.assert_array_equal(cfg['pose'], [0, 0, 0, 1, 0, 0, 0])
            for relative in record['target_configs']:
                saved = load_scene_cfg(str(root / relative))
                self.assertEqual(set(saved['scene']), {'020', '031', 'table'})
                np.testing.assert_allclose(saved['scene']['031']['pose'][:3], [.9, 2.3, 3.7])
                np.testing.assert_array_equal(saved['scene']['031']['scale'], [1, 1, 1])
                self.assertEqual(Path(saved['scene']['031']['file_path']).resolve(), asset)
                self.assertEqual(saved['clutter']['coordinate_frame'], 'table')
                self.assertEqual(saved['scene']['table']['pose'][2], 0)

    def test_extracts_one_view_of_training_metadata_without_images_or_test_scenes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'scenes_0.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                for index in range(101):
                    for relative in ('annotations/0000.xml', 'annotations/0001.xml',
                                     'cam0_wrt_table.npy', 'camera_poses.npy', 'rgb/0000.png',
                                     'depth/0000.png', 'meta/0000.mat'):
                        member = tarfile.TarInfo(f'scenes/scene_{index:04d}/realsense/{relative}')
                        member.size = 4
                        bundle.addfile(member, BytesIO(b'data'))
            names = extract_metadata(archive, root / 'output', 'realsense', 0)
            self.assertEqual(len(names), 100)
            self.assertNotIn('scene_0100', names)
            files = list((root / 'output/raw').rglob('*'))
            self.assertEqual(sum(path.is_file() for path in files), 300)
            self.assertFalse(any(path.suffix in ('.png', '.mat') for path in files))
            self.assertFalse(selected_metadata('scenes/scene_0000/realsense/../../camera_poses.npy', 'realsense'))
            self.assertFalse(selected_metadata('scenes/scene_8499/kinect/annotations/0000.xml', 'realsense'))
            self.assertTrue(selected_metadata('scenes/scene_8499/realsense/annotations/0000.xml', 'realsense'))
            self.assertFalse(selected_metadata('scenes/scene_8500/realsense/annotations/0000.xml', 'realsense'))

    def test_metadata_symlink_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'scenes_0.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                member = tarfile.TarInfo('scenes/scene_0000/realsense/camera_poses.npy')
                member.type = tarfile.SYMTYPE
                member.linkname = '/etc/passwd'
                bundle.addfile(member)
            with self.assertRaisesRegex(ValueError, 'Unexpected metadata entry'):
                extract_metadata(archive, root / 'output', 'realsense', 0)

    def test_corrupt_archive_is_rejected_even_at_the_expected_size(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scenes_0.tar.gz').write_bytes(b'bad!')
            with patch('scripts.prepare_dexgraspnet2_train_scenes.ARCHIVES', [(4, '0' * 64)]):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    download(0, root, root / 'output', False)

    def test_resume_appends_only_the_correct_range_and_verifies_the_whole_archive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = b'abcdefgh'
            (root / 'scenes_0.tar.gz.part').write_bytes(data[:4])
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code = 206
            response.headers = {'Content-Range': 'bytes 4-7/8'}
            response.iter_content.return_value = iter([data[4:]])
            with (patch('scripts.prepare_dexgraspnet2_train_scenes.ARCHIVES',
                        [(len(data), hashlib.sha256(data).hexdigest())]),
                  patch('scripts.prepare_dexgraspnet2_train_scenes.requests.get', return_value=response) as get):
                archive = download(0, root, root / 'output', True)
            self.assertEqual(get.call_args.kwargs['headers'], {'Range': 'bytes=4-'})
            self.assertEqual(archive.read_bytes(), data)
            self.assertFalse((root / 'scenes_0.tar.gz.part').exists())

    def test_resume_rejects_a_server_that_returns_the_whole_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / 'scenes_0.tar.gz.part'
            partial.write_bytes(b'abcd')
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code = 200
            response.headers = {}
            with (patch('scripts.prepare_dexgraspnet2_train_scenes.ARCHIVES', [(8, '0' * 64)]),
                  patch('scripts.prepare_dexgraspnet2_train_scenes.requests.get', return_value=response)):
                with self.assertRaisesRegex(ValueError, 'download offset'):
                    download(0, root, root / 'output', True)
            self.assertEqual(partial.read_bytes(), b'abcd')

    def test_out_of_range_object_is_rejected(self):
        with TemporaryDirectory() as directory:
            annotation = Path(directory) / '0000.xml'
            self.annotation(annotation)
            annotation.write_text(annotation.read_text().replace('<obj_id>31</obj_id>', '<obj_id>88</obj_id>'))
            with self.assertRaisesRegex(ValueError, 'Unknown GraspNet'):
                object_poses(annotation, np.eye(4))

    def test_manifest_requires_all_scene_ids_and_counts_objects_from_actual_annotations(self):
        records = [{'scene_name': name, 'object_ids': ['031'], 'target_configs': [f'{name}/031.npy']}
                   for name in ('scene_0000', 'scene_1000')]
        with patch('scripts.prepare_dexgraspnet2_train_scenes.TRAIN_IDS', frozenset({0, 1000})):
            manifest = build_manifest([{'part': 0, 'scenes': records}], 'realsense')
            self.assertEqual(manifest['object_count'], 1)
            self.assertEqual(manifest['scene_count'], 2)
            self.assertEqual(manifest['paper_reported_training_objects'], 60)
            with self.assertRaisesRegex(ValueError, 'training split'):
                build_manifest([{'part': 0, 'scenes': records[:1]}], 'realsense')


if __name__ == '__main__':
    unittest.main()
