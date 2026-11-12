from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mujoco
import numpy as np

from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
from dexonomy.sim.clutter_env import ClutterSimCfg, MuJoCoClutterEnv
from scripts.view_clutter_grasps_mujoco import AllGraspsModel, GraspRecord, PHASES, SceneData


REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_XML = str(REPO_ROOT / "assets/hand/shadow/right.xml")


def make_object(root):
    path = root / "object.xml"
    path.write_text('''<mujoco><worldbody><body name="object">
      <geom name="visual" type="box" size="0.02 0.03 0.04"
            contype="0" conaffinity="0" mass="0"/>
      <geom name="collision" type="box" size="0.02 0.03 0.04"/>
    </body></worldbody></mujoco>''')
    return {"name": "target", "type": "rigid_object", "xml_path": str(path),
            "scale": [1, 1, 1], "pose": [0, 0, .1, 1, 0, 0, 0], "density": 1000.}


class MuJoCoCompatibilityTests(unittest.TestCase):
    def test_floating_shadow_weld_compiles_and_simulates_with_default_extension_fields(self):
        cfg = MuJoCo_OptCfg()
        env = MuJoCo_OptEnv(HandCfg(HAND_XML, freejoint=True), sim_cfg=cfg)
        self.assertEqual(env._model.nq, 29)
        self.assertEqual(env._model.neq, 1)
        self.assertEqual(env._model.eq_data.shape, (1, mujoco.mjNEQDATA))
        self.assertEqual(env._model.eq_data[0, 10], cfg.hand_mocap_data[10])
        reference = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="mocap" mocap="true"/>
          <body name="hand"><freejoint/><geom type="sphere" size="0.01"/></body>
        </worldbody><equality><weld body1="mocap" body2="hand"/></equality></mujoco>''')
        np.testing.assert_array_equal(env._model.eq_data[0, 11:], reference.eq_data[0, 11:])
        env.set_ctrl(env.get_hand_qpos())
        env.step_sim(20)
        self.assertTrue(np.isfinite(env._data.qpos).all())
        self.assertTrue(np.isfinite(env._data.qvel).all())
        self.assertAlmostEqual(env._data.time, cfg.timestep * 20)

    def test_free_body_retains_linear_damping_without_higher_order_damping(self):
        with TemporaryDirectory() as directory:
            cfg = ClutterSimCfg(freejoint_damping=.003)
            env = MuJoCoClutterEnv([make_object(Path(directory))], cfg)
            np.testing.assert_allclose(env.model.dof_damping, .003)
            if hasattr(env.model, "dof_dampingpoly"):
                np.testing.assert_array_equal(env.model.dof_dampingpoly, 0)
            self.assertTrue(env.set_spawn_poses(np.array([[0, 0, .2, 1, 0, 0, 0]])))
            mujoco.mj_step(env.model, env.data, nstep=10)
            self.assertTrue(np.isfinite(env.data.qpos).all())

    def test_static_viewer_replays_all_phases_without_actuators_or_collisions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            obj = make_object(root)
            qpos = np.zeros(29)
            qpos[:3] = [.2, 0, .15]
            qpos[3] = 1
            phases = {phase: qpos.copy() for phase in PHASES}
            phases["pregrasp"][2] += .1
            phases["squeeze"][8] = .2
            original = {phase: value.copy() for phase, value in phases.items()}
            record = GraspRecord("1_Large_Diameter", "target", root / "unused.npy", 0, phases)
            scene = SceneData("fixture", [], root / "unused_scene.npy",
                              {"scene": {"target": obj}}, [record])
            view = AllGraspsModel(scene, HAND_XML, hand_alpha=.35)
            self.assertEqual(view.model.nu, 0)
            np.testing.assert_array_equal(view.model.geom_contype, 0)
            np.testing.assert_array_equal(view.model.geom_conaffinity, 0)
            for phase in PHASES:
                view.set_phase(phase)
                np.testing.assert_allclose(view.data.qpos, phases[phase])
                np.testing.assert_array_equal(phases[phase], original[phase])


if __name__ == "__main__":
    unittest.main()
