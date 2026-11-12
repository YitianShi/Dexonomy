from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dexonomy.util.taxonomy_contacts import (
    AMBIGUOUS, INVALID_POSE, NO_CONTACT, NO_TAXONOMY, RELABELED, RETAINED,
    ContactReplay, ContactRules,
)
from scripts.refilter_taxonomy_contacts import apply_contact_policy, subset_output

ROOT = Path(__file__).resolve().parents[1]


class TaxonomyContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = ContactRules.load(ROOT/'assets/hand/shadow/init_tmpl')
        cls.tip = cls.rules.names.index('24_Tip_Pinch')
        cls.lateral = cls.rules.names.index('16_Lateral')

    def test_required_contact_groups_and_strict_unexpected_body(self):
        rules = self.rules
        self.assertTrue(rules.missing_groups(rules.body_mask(['rh_ffdistal']), self.tip))
        bodies = rules.body_mask(['rh_ffdistal','rh_thdistal'])
        self.assertEqual(rules.missing_groups(bodies,self.tip),0)
        self.assertEqual(rules.classify(bodies,self.tip)[:2],(RETAINED,self.tip))
        extra = rules.body_mask(['rh_ffdistal','rh_thdistal','rh_palm'])
        self.assertEqual(rules.classify(extra,self.tip)[0],NO_TAXONOMY)
        unannotated = rules.body_mask(['rh_ffdistal','rh_thdistal','rh_forearm'])
        self.assertEqual(rules.matching_mask(unannotated),0)
        self.assertEqual(rules.classify(0,self.tip)[0],NO_CONTACT)

    def test_unique_relabel_and_genuinely_ambiguous_rules(self):
        rules = self.rules
        side = rules.body_mask(['rh_ffmiddle','rh_thdistal'])
        self.assertEqual(rules.classify(side,self.tip)[:2],(RELABELED,self.lateral))
        pinch = rules.body_mask(['rh_ffdistal','rh_thdistal'])
        status, assigned, matches = rules.classify(pinch,self.lateral)
        self.assertEqual((status,assigned),(AMBIGUOUS,-2))
        candidates = [name for index,name in enumerate(rules.names) if matches & (1<<index)]
        self.assertEqual(candidates,['9_Palmar_Pinch','24_Tip_Pinch'])
        palmar = rules.names.index('9_Palmar_Pinch')
        self.assertEqual(rules.classify(pinch,palmar)[:2],(RETAINED,palmar))

    def test_restore_extra_contacts_preserves_existing_relabels_and_missing_groups(self):
        rules = self.rules
        examples = [
            (self.tip, ['rh_ffdistal','rh_thdistal','rh_palm']),
            (rules.names.index('14_Tripod'), ['rh_ffdistal','rh_mfdistal','rh_rfdistal','rh_lfdistal','rh_thdistal']),
            (rules.names.index('7_Prismatic_3_Finger'), ['rh_ffdistal','rh_ffproximal','rh_mfdistal','rh_rfdistal','rh_lfdistal','rh_thdistal']),
            (self.tip, ['rh_ffdistal','rh_palm']),
        ]
        expected_statuses = [NO_TAXONOMY, AMBIGUOUS, RELABELED, NO_TAXONOMY]
        for (taxonomy,names),expected in zip(examples,expected_statuses):
            with self.subTest(taxonomy=taxonomy,bodies=names):
                bodies = rules.body_mask(names)
                status,assigned,matches = rules.classify(bodies,taxonomy)
                self.assertEqual(status,expected)
                missing = rules.missing_groups(bodies,taxonomy)
                audit = {'status':np.array([status],np.uint8),
                         'assigned_taxonomy_ids':np.array([assigned],np.int8),
                         'matching_taxonomy_masks':np.array([matches],np.uint64),
                         'missing_required_group_masks':np.array([missing],np.uint16),
                         'contact_body_masks':np.array([bodies],np.uint64)}
                apply_contact_policy(audit,taxonomy,True)
                restored = expected in (NO_TAXONOMY,AMBIGUOUS) and missing == 0
                self.assertEqual(bool(audit['original_label_restored'][0]),restored)
                self.assertEqual(int(audit['assigned_taxonomy_ids'][0]),taxonomy if restored else assigned)
                self.assertEqual(int(audit['status'][0]),RETAINED if restored else status)
                self.assertEqual(int(audit['strict_matching_taxonomy_masks'][0]),matches)
                apply_contact_policy(audit,taxonomy,False)
                self.assertEqual(int(audit['assigned_taxonomy_ids'][0]),assigned)
                self.assertEqual(int(audit['status'][0]),status)

    def test_subset_keeps_exact_source_rows_and_original_anchor_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'original.npy'
            poses = np.zeros((3,29),np.float32);poses[:,3]=1;poses[:,0]=[.1,.2,.3]
            data = {name:poses.copy() for name in ('pregrasp_qpos','grasp_qpos','squeeze_qpos')}
            data.update(scene_path='scene.npy',target_name='target',hand_contact_indices=np.array([0,1,0]))
            np.save(source,data);before = source.read_bytes()
            audit = {'matching_taxonomy_masks':np.array([1,2,4],np.uint64),
                     'contact_body_masks':np.array([1,2,3],np.uint64),
                     'source_template_max_point_distance_m':np.array([.01,.02,.03],np.float32)}
            destination = root/'derived.npy'
            subset_output([(data,source,self.tip,np.array([0,2]),audit)],destination,self.rules,'assigned')
            output = np.load(destination,allow_pickle=True).item()
            np.testing.assert_array_equal(output['grasp_qpos'],poses[[0,2]])
            np.testing.assert_array_equal(output['source_indices'],[0,2])
            np.testing.assert_array_equal(output['source_hand_contact_indices'],[0,0])
            self.assertNotIn('hand_contact_indices',output)
            self.assertEqual(output['source_grasp_paths'].tolist(),[str(source),str(source)])
            self.assertEqual(source.read_bytes(),before)


class ContactReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = ContactRules.load(ROOT/'assets/hand/shadow/init_tmpl')
        cls.taxonomy = cls.rules.names.index('24_Tip_Pinch')

    def setUp(self):
        self.temporary = TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        xml = Path(self.temporary.name)/'box.xml'
        xml.write_text('<mujoco><worldbody><body name="object"><geom name="collision" '
                       'type="box" size=".08 .08 .08"/></body></worldbody></mujoco>')
        def object_at(x):
            return {'type':'rigid_object','xml_path':str(xml),'scale':np.ones(3),
                    'pose':np.array([x,0,.10,1,0,0,0])}
        scene = {'scene':{'target':object_at(5),'neighbor':object_at(0)},'task':{'obj_name':'target'}}
        self.probe = ContactReplay(scene,ROOT/'assets/hand/shadow/right.xml',self.rules)
        self.pose = self.rules.templates[self.taxonomy]['grasp_qpos'].reshape(-1).copy()

    def test_neighbor_cannot_satisfy_target_and_position_stage_matches_forward(self):
        probe = self.probe;probe.set_target('target')
        self.assertEqual(probe.query(self.pose,self.taxonomy)['status'],NO_CONTACT)
        target,neighbor,_ = probe.environment.get_scene_contacts(.002)
        self.assertEqual(len(target['dist']),0)
        self.assertGreater(len(neighbor['dist']),0)
        probe.set_target('neighbor')
        for offset in (0,.01,-.01):
            pose = self.pose.copy();pose[0]+=offset
            actual = probe.query(pose,self.taxonomy)
            probe.environment._target_body_ids = {
                body for body in probe.environment._object_body_ids
                if probe.model.body(body).name.startswith('obj-neighbor')
            }
            probe.environment.reset_qpos(pose)
            contacts,_,_ = probe.environment.get_scene_contacts(.002)
            expected = self.rules.body_mask([body for body,dist in zip(contacts['bn1'],contacts['dist']) if dist<=.002])
            self.assertEqual(actual['bodies'],expected)
            self.assertNotEqual(actual['bodies'],0)

    def test_invalid_pose_is_distinct_from_a_taxonomy_rejection(self):
        self.probe.set_target('target')
        pose = self.pose.copy();pose[0]=np.nan
        self.assertEqual(self.probe.query(pose,self.taxonomy)['status'],INVALID_POSE)


if __name__ == '__main__':
    unittest.main()
