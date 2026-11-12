"""Replay and classify target contacts using the released hand taxonomy rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


STATUS_NAMES = ("retained", "relabeled", "ambiguous", "no_target_contact", "no_taxonomy", "invalid_pose")
RETAINED, RELABELED, AMBIGUOUS, NO_CONTACT, NO_TAXONOMY, INVALID_POSE = range(6)


@dataclass
class ContactRules:
    names: list[str]
    templates: list[dict]
    bodies: list[str]
    allowed: list[int]
    required: list[list[int]]
    lookup: np.ndarray
    annotated_body_count: int

    @classmethod
    def load(cls, directory, hand_body_names=()):
        paths = sorted(Path(directory).glob('[0-9]*.npy'), key=lambda p:int(p.stem.split('_', 1)[0]))
        templates = [np.load(path, allow_pickle=True).item() for path in paths]
        if len(paths) != 33:
            raise ValueError(f'Expected 33 taxonomy templates; found {len(paths)}')
        annotated = sorted({body for data in templates for body in data['hand_cbody']})
        bodies = annotated + sorted(set(hand_body_names)-set(annotated)) + ['__unannotated__']
        if len(bodies) > 63 or len(annotated) > 20:
            raise ValueError('Contact mask layout exceeds supported size')
        body_indices = {name:index for index,name in enumerate(bodies)}
        def mask(names):
            value = 0
            for name in names:value |= 1 << body_indices[name]
            return value
        allowed = [mask(data['hand_cbody']) for data in templates]
        required = [[mask(group) for group in data['required_cbody']] for data in templates]
        combinations = np.arange(1 << len(annotated), dtype=np.uint64)
        lookup = np.zeros(len(combinations), dtype=np.uint64)
        for index, (permitted, groups) in enumerate(zip(allowed, required)):
            valid = (combinations != 0) & ((combinations & np.uint64(~permitted & ((1 << len(annotated))-1))) == 0)
            for group in groups:valid &= (combinations & np.uint64(group)) != 0
            lookup[valid] |= np.uint64(1 << index)
        return cls([path.stem for path in paths], templates, bodies, allowed, required, lookup, len(annotated))

    def body_mask(self, names):
        indices = {name:index for index,name in enumerate(self.bodies)}
        result = 0
        for name in names:result |= 1 << indices.get(name, len(self.bodies)-1)
        return result

    def matching_mask(self, contacts):
        contacts = int(contacts)
        return int(self.lookup[contacts]) if 0 <= contacts < len(self.lookup) else 0

    def classify(self, contacts, original):
        matches = self.matching_mask(contacts)
        if matches & (1 << original):return RETAINED, original, matches
        if not matches:return (NO_CONTACT if not contacts else NO_TAXONOMY), -1, 0
        if matches & (matches-1):return AMBIGUOUS, -2, matches
        return RELABELED, matches.bit_length()-1, matches

    def missing_groups(self, contacts, taxonomy):
        return sum(1 << i for i,group in enumerate(self.required[taxonomy]) if not int(contacts) & group)

    def manifest(self):
        return {'taxonomies':self.names, 'contact_bodies':self.bodies,
                'allowed_body_masks':self.allowed, 'required_group_masks':self.required,
                'membership':'nonempty target contacts; all required groups covered; no contact outside allowed bodies',
                'ambiguity':'retain valid original label; unique alternative relabeled; multiple alternatives quarantined'}


class ContactReplay:
    """Pin the saved pose and query contacts; no dynamics or grasp optimization."""
    def __init__(self, scene, hand_xml, rules, distance=.002):
        import mujoco
        from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
        self.mujoco, self.rules, self.distance = mujoco, rules, float(distance)
        self.environment = MuJoCo_OptEnv(
            hand_cfg=HandCfg(xml_path=str(hand_xml), freejoint=True), scene_cfg=scene,
            sim_cfg=MuJoCo_OptCfg(obj_margin=self.distance, hand_margin=0, plane_margin=0),
            debug_render=False, debug_view=False,
        )
        self.model, self.data = self.environment._model, self.environment._data
        if self.model.nq-len(self.environment._obj_init_qpos) != 29:
            raise ValueError('Expected a free Shadow hand with 29 qpos')
        self.body_bits = np.zeros(self.model.nbody, dtype=np.uint64)
        self.hand_ids = set(self.environment._hand_body_ids)
        for body_id in self.hand_ids:
            name = self.model.body(body_id).name.removeprefix('hand-')
            self.body_bits[body_id] = np.uint64(rules.body_mask([name]))
        self.geom_body = np.asarray(self.model.geom_bodyid)
        self.geom_hand_bits = self.body_bits[self.geom_body]
        self.target_masks = {}
        self.template_points = []
        for template in rules.templates:
            pose = np.asarray(template['grasp_qpos']).reshape(-1)
            self.data.qpos[:29] = pose
            mujoco.mj_kinematics(self.model, self.data)
            points = {}
            for name, cpn in zip(template['hand_cbody'], template['hand_cpn_w']):
                body = self.model.body('hand-'+name).id
                rotation = self.data.xmat[body].reshape(3, 3)
                local = (np.asarray(cpn[:3])-self.data.xpos[body]) @ rotation
                points.setdefault(body, []).append(local)
            self.template_points.append({body:np.asarray(values) for body,values in points.items()})
        self.object_qpos = np.asarray(self.environment._obj_init_qpos).copy()
        self.target = None

    def set_target(self, target):
        if target not in self.target_masks:
            root = self.model.body('obj-'+target+'world').id
            selected = np.zeros(self.model.nbody, dtype=bool)
            for body in self.environment._object_body_ids:
                ancestor = body
                while ancestor:
                    if ancestor == root:
                        selected[body] = True
                        break
                    ancestor = int(self.model.body_parentid[ancestor])
            if not selected.any():raise ValueError(f'No target bodies for {target}')
            self.target_masks[target] = selected[self.geom_body]
        self.target = self.target_masks[target]

    def query(self, pose, taxonomy, diagnostic=True):
        pose = np.asarray(pose)
        if pose.shape != (29,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:7]) < 1e-8:
            return {'status':INVALID_POSE, 'assigned':-1, 'matches':0, 'bodies':0,
                    'missing':0, 'extra':0, 'max_point_distance':np.nan, 'contact_count':0}
        if self.target is None:raise RuntimeError('Select target before replay')
        self.data.qpos[:29] = pose
        self.data.qpos[29:] = self.object_qpos
        if len(self.data.mocap_pos):
            self.data.mocap_pos[0] = pose[:3]
            self.data.mocap_quat[0] = pose[3:7]
        # Contact geometry uses the position stage only. No simulation steps,
        # object motion, velocities, controller or solver state enter this check.
        self.mujoco.mj_fwdPosition(self.model, self.data)
        contacts = self.data.contact
        g1, g2 = contacts.geom1, contacts.geom2
        hand_first = (self.geom_hand_bits[g1] != 0) & self.target[g2]
        hand_second = (self.geom_hand_bits[g2] != 0) & self.target[g1]
        selected = np.flatnonzero((hand_first | hand_second) & (contacts.dist <= self.distance))
        hand_geoms = np.where(hand_first[selected], g1[selected], g2[selected])
        bits = self.geom_hand_bits[hand_geoms]
        body_mask = int(np.bitwise_or.reduce(bits, initial=np.uint64(0)))
        status, assigned, matches = self.rules.classify(body_mask, taxonomy)
        max_distance = np.nan
        if diagnostic and len(selected):
            max_distance = 0.0
            body_ids = self.geom_body[hand_geoms]
            for body in np.unique(body_ids):
                expected = self.template_points[taxonomy].get(int(body))
                if expected is None:
                    max_distance = np.inf
                    break
                indices = selected[body_ids == body]
                # MuJoCo reports contact midpoints. Move half the signed gap
                # towards the hand surface before expressing it in body frame.
                signs = np.where(hand_first[indices], -1., 1.)
                surface = contacts.pos[indices] + (signs*contacts.dist[indices]/2)[:,None]*contacts.frame[indices,:3]
                local = (surface-self.data.xpos[body]) @ self.data.xmat[body].reshape(3,3)
                nearest = np.sqrt(np.min(np.sum((local[:,None,:]-expected[None,:,:])**2, axis=-1), axis=1))
                max_distance = max(max_distance, float(nearest.max()))
        return {'status':status, 'assigned':assigned, 'matches':matches, 'bodies':body_mask,
                'missing':self.rules.missing_groups(body_mask, taxonomy),
                'extra':body_mask & ~self.rules.allowed[taxonomy],
                'max_point_distance':max_distance, 'contact_count':len(selected)}
