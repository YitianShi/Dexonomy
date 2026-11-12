import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from dexonomy.util.grasp_dedup import PHASES, PoseThresholds, deduplicate, pose_features, subset_rows
from scripts.deduplicate_grasps import process_group, discover_jobs, parse_args
from scripts.refilter_taxonomy_contacts import discover_jobs as discover_contact_jobs
from scripts.refilter_taxonomy_contacts import parse_args as parse_contact_args


def sample(count):
    poses = np.zeros((count, 29), dtype=np.float32)
    poses[:, 3] = 1
    return {name: poses.copy() for name in PHASES}


class GraspDedupTests(unittest.TestCase):
    def test_catalog_can_contain_only_final_dataset_entries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'datasets/single_object/graspnet'
            path = data/'1_First/031/floating/scale_a.npy'
            path.parent.mkdir(parents=True)
            np.save(path, sample(2))
            catalog = root/'dataset_index.json'
            catalog.write_text(json.dumps({'datasets': [{
                'id': 'deduplicated_graspnet_single', 'kind': 'single_object_grasp',
                'state': 'complete', 'data': 'datasets/single_object/graspnet',
                'grasp_records': 2, 'npy_files': 1,
            }]}))
            options = ['--catalog', str(catalog), '--datasets', 'graspnet_single',
                       '--output', str(root/'work_runs'), '--dry-run']
            jobs, sources = discover_jobs(parse_args(options))
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['root'], str(data))
            self.assertEqual(sources['graspnet_single']['id'], 'deduplicated_graspnet_single')
            jobs, sources = discover_contact_jobs(parse_contact_args(options),
                                                  SimpleNamespace(names=['1_First']))
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['root'], str(data))
            self.assertEqual(sources['graspnet_single']['expected_input_rows'], 2)

    def test_nontransitive_chain_keeps_endpoints(self):
        data = sample(3)
        for phase in PHASES:
            data[phase][:, 0] = [0, .0015, .003]
        mapping, counts = deduplicate(data)
        np.testing.assert_array_equal(mapping, [0, 0, 2])
        self.assertEqual(counts['near_pose_duplicates'], 1)

    def test_antipodal_quaternions_are_same_rotation(self):
        data = sample(3)
        for phase in PHASES:
            data[phase][1, 3] = -1
        mapping, counts = deduplicate(data)
        np.testing.assert_array_equal(mapping, [0, 0, 0])
        self.assertEqual(counts['exact_pose_duplicates'], 1)

    def test_pregrasp_and_squeeze_cannot_be_ignored(self):
        data = sample(3)
        data['pregrasp_qpos'][1, 0] = .02
        data['squeeze_qpos'][2, 8] = np.deg2rad(20)
        np.testing.assert_array_equal(deduplicate(data)[0], [0, 1, 2])

    def test_wrist_rotation_and_single_joint_cap(self):
        data = sample(3)
        for phase in PHASES:
            data[phase][1, 3:7] = [np.cos(np.deg2rad(3)), 0, 0, np.sin(np.deg2rad(3))]
            # RMS < 3 degrees, but one finger differs by > 8 degrees.
            data[phase][2, 9] = np.deg2rad(9)
        np.testing.assert_array_equal(deduplicate(data)[0], [0, 1, 2])

    def test_contact_patterns_and_scales_preserved(self):
        data = sample(4)
        data['contact_body_masks'] = np.array([1 << 60, (1 << 60)+1, 1 << 60, 1 << 60], np.uint64)
        data['scene_scale'] = np.ones((4, 3))
        data['scene_scale'][2] = 2
        np.testing.assert_array_equal(deduplicate(data)[0], [0, 1, 2, 0])

    def test_global_frame_change_preserves_palm_geometry_and_decisions(self):
        from scipy.spatial.transform import Rotation
        data = sample(4)
        for name in PHASES:
            data[name][:, 0] = [0, .001, .006, .0065]
            # A small tilt also moves the palm relative to the stored free root.
            data[name][1, 3:7] = [np.cos(.005), np.sin(.005), 0, 0]
        expected = deduplicate(data)[0]
        global_rotation = Rotation.from_euler('xyz', [.7, -.8, 1.3])
        shift = np.array([.25, -.17, .8])
        changed = {}
        for name in PHASES:
            poses = data[name].astype(float)
            poses[:, :3] = global_rotation.apply(poses[:, :3])+shift
            rotation = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
            poses[:, 3:7] = (global_rotation*rotation).as_quat()[:, [3, 0, 1, 2]]
            changed[name] = poses
        palm, _, _ = pose_features(changed)
        old_palm, _, _ = pose_features(data)
        np.testing.assert_allclose(palm, global_rotation.apply(old_palm.reshape(-1, 3)).reshape(4, 3, 3)+shift,
                                   atol=1e-12)
        np.testing.assert_array_equal(deduplicate(changed)[0], expected)

    def test_subsetting_preserves_labels_provenance_and_global_arrays(self):
        data = sample(33)
        data['taxonomy_names'] = np.arange(33)
        data['original_taxonomy_ids'] = np.arange(33)
        data['source_indices'] = np.arange(33)+100
        output = subset_rows(data, np.array([1, 5]), '/source.npy')
        np.testing.assert_array_equal(output['taxonomy_names'], np.arange(33))
        np.testing.assert_array_equal(output['source_indices'], [101, 105])
        np.testing.assert_array_equal(output['original_taxonomy_ids'], [1, 5])
        np.testing.assert_array_equal(output['dedup_source_indices'], [1, 5])
        for phase in PHASES:
            np.testing.assert_array_equal(output[phase], data[phase][[1, 5]])

    def test_bad_inputs_fail(self):
        with self.assertRaises(ValueError):
            PoseThresholds(position_m=float('nan'))
        data = sample(1)
        data['grasp_qpos'][0, 3] = 0
        with self.assertRaises(ValueError):
            deduplicate(data)
        data = sample(2)
        data['squeeze_qpos'] = data['squeeze_qpos'][:1]
        with self.assertRaises(ValueError):
            deduplicate(data)

    def test_empty_input(self):
        mapping, counts = deduplicate(sample(0))
        self.assertEqual(len(mapping), 0)
        self.assertEqual(counts['removed_rows'], 0)

    def test_disk_outputs_keep_taxonomies_and_source_indices_separate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)/'input'
            output = Path(directory)/'output'
            paths = ['1_First/031/floating/scale_a.npy', '2_Second/031/floating/scale_a.npy']
            data = sample(3)
            data['scene_path'] = 'scene.npy'
            data['source_indices'] = np.array([42, 53, 67])
            data['original_taxonomy_ids'] = np.array([1, 2, 3])
            for relative in paths:
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, data)
            before = [(root/p).read_bytes() for p in paths]
            job = {'dataset': 'example', 'key': '031/floating/scale_a',
                   'root': str(root), 'files': paths}
            result = process_group(job, output, PoseThresholds())
            self.assertEqual(result['counts']['input_rows'], 6)
            self.assertEqual(result['counts']['kept_rows'], 2)
            for relative, original in zip(paths, before):
                saved = np.load(output/'example/filtered_grasps'/relative, allow_pickle=True).item()
                np.testing.assert_array_equal(saved['source_indices'], [42])
                np.testing.assert_array_equal(saved['original_taxonomy_ids'], [1])
                self.assertEqual((root/relative).read_bytes(), original)
            audit = output/'example/records/031/floating/scale_a.npz'
            with np.load(audit) as record:
                np.testing.assert_array_equal(record['file_offsets'], [0, 3, 6])
                np.testing.assert_array_equal(record['representative_indices'], [0, 0, 0, 0, 0, 0])
            # An intact job resumes; a missing output is rebuilt instead of skipped.
            resumed = process_group(job, output, PoseThresholds())
            self.assertEqual(resumed, result)
            (output/'example/filtered_grasps'/paths[0]).unlink()
            rebuilt = process_group(job, output, PoseThresholds())
            self.assertEqual(rebuilt['counts'], result['counts'])


if __name__ == '__main__':
    unittest.main()
