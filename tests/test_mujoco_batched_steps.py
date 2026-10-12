import unittest

import mujoco
import numpy as np

from dexonomy.sim.mujoco_env import MuJoCo_BaseEnv


class BatchedSimulationTests(unittest.TestCase):
    def test_batched_steps_preserve_physics_and_control_callbacks(self):
        model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option timestep="0.002" integrator="implicitfast"/>
          <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <body pos="0 0 0.06">
              <joint name="slide" type="slide" axis="0 0 1"/>
              <geom type="sphere" size="0.05" mass="0.1"/>
            </body>
          </worldbody>
          <actuator><motor joint="slide"/></actuator>
        </mujoco>
        """)
        reference = mujoco.MjData(model)
        env = MuJoCo_BaseEnv.__new__(MuJoCo_BaseEnv)
        env._model, env._data = model, mujoco.MjData(model)
        env._debug_render = env._debug_view = None
        callbacks = []

        def control(_model, data):
            callbacks.append(float(data.time))
            data.ctrl[0] = 0.2 * np.sin(3 * data.time)

        previous_callback = mujoco.get_mjcb_control()
        mujoco.set_mjcb_control(control)
        try:
            for count in (1, 5, 10, 50, 0, 50):
                callbacks.clear()
                for _ in range(count):
                    mujoco.mj_step(model, reference)
                reference_callbacks = callbacks.copy()
                callbacks.clear()
                env.step_sim(count)
                self.assertEqual(callbacks, reference_callbacks)
                for field in ("qpos", "qvel", "qacc", "ctrl", "actuator_force"):
                    np.testing.assert_array_equal(getattr(env._data, field), getattr(reference, field))
                self.assertEqual(env._data.time, reference.time)
                self.assertEqual(env._data.ncon, reference.ncon)
        finally:
            mujoco.set_mjcb_control(previous_callback)


if __name__ == "__main__":
    unittest.main()
