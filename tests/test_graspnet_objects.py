from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import mujoco
import numpy as np
import trimesh

from dexonomy.util.file_util import load_scene_cfg
from scripts.prepare_graspnet_objects import convert_object


class GraspNetObjectImportTests(unittest.TestCase):
    def make_source(self, root):
        source = root / "raw/000"
        coacd = source / "coacd"
        coacd.mkdir(parents=True)
        first = trimesh.creation.box(extents=[.1, .2, .3])
        second = trimesh.creation.box(extents=[.05, .1, .2])
        first.export(coacd / "first.obj")
        second.export(coacd / "second.obj")
        first.apply_translation([.01, .02, .03])
        second.vertices *= [2, 1, .5]
        rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        second.vertices = second.vertices @ rotation.T + [.37, .02, .04]
        trimesh.util.concatenate([first, second]).export(source / "simplified.obj")
        # Child link precedes the parent; fixed-joint and collision origins both matter.
        (coacd / "coacd.urdf").write_text('''<robot name="test">
          <link name="child"><collision><origin xyz="0.02 0.03 0.04"/>
            <geometry><mesh filename="second.obj" scale="2 1 0.5"/></geometry>
          </collision></link>
          <link name="root"><collision><origin xyz="0.01 0.02 0.03"/>
            <geometry><mesh filename="first.obj"/></geometry>
          </collision></link>
          <joint name="fixed" type="fixed"><parent link="root"/><child link="child"/>
            <origin xyz="0.4 0 0" rpy="0 0 1.5707963267948966"/>
          </joint></robot>''')
        return source, first, second

    def test_preserves_original_size_frame_and_all_convex_pieces(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, first, second = self.make_source(root)
            original = (source / "simplified.obj").read_bytes()
            output = root / "assets"
            record = convert_object("000", root / "raw", output)
            self.assertEqual(record["collision_mesh_count"], 2)
            scene = load_scene_cfg(record["scene_path"])
            body = scene["scene"]["000"]
            self.assertEqual(scene["task"], {"type": "force_closure", "obj_name": "000"})
            np.testing.assert_array_equal(body["scale"], [1, 1, 1])
            np.testing.assert_array_equal(body["pose"], [0, 0, 0, 1, 0, 0, 0])
            self.assertEqual(Path(body["file_path"]).read_bytes(), original)
            self.assertEqual((source / "simplified.obj").read_bytes(), original)
            model = mujoco.MjModel.from_xml_path(body["xml_path"])
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            # URDF collisions are emitted in child, parent order.
            for index, expected in enumerate([second, first]):
                geom = model.geom(f"object_collision_{index}").id
                mesh = model.geom_dataid[geom]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                world = model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                np.testing.assert_allclose(world.min(0), expected.vertices.min(0), atol=1e-7)
                np.testing.assert_allclose(world.max(0), expected.vertices.max(0), atol=1e-7)
            info = json.loads(Path(body["info_path"]).read_text())
            self.assertAlmostEqual(info["mass"], .007, places=7)
            np.testing.assert_allclose(info["gravity_center"],
                (.006 * np.asarray([.01, .02, .03]) + .001 * np.asarray([.37, .02, .04])) / .007,
                atol=1e-7)

    def test_rejects_convex_mesh_reference_outside_the_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self.make_source(root)
            urdf = source / "coacd/coacd.urdf"
            urdf.write_text(urdf.read_text().replace('filename="first.obj"', 'filename="../../outside.obj"'))
            trimesh.creation.box().export(root / "raw/outside.obj")
            with self.assertRaisesRegex(ValueError, "Invalid or missing convex mesh"):
                convert_object("000", root / "raw", root / "assets")
            self.assertFalse((root / "assets").exists())

    def test_sub_resolution_piece_keeps_collisions_without_artificial_surface_mass(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self.make_source(root)
            tiny = trimesh.creation.box(extents=[4e-5, 2e-6, 1e-5])
            tiny.apply_translation([.03, .04, .05])
            tiny.export(source / "coacd/tiny.obj")
            urdf = source / "coacd/coacd.urdf"
            urdf.write_text(urdf.read_text().replace('</robot>',
                '<link name="tiny"><collision><geometry><mesh filename="tiny.obj"/>'
                '</geometry></collision></link><joint name="tiny_fixed" type="fixed">'
                '<parent link="root"/><child link="tiny"/></joint></robot>'))
            record = convert_object("000", root / "raw", root / "assets")
            self.assertEqual(record["collision_mesh_count"], 3)
            self.assertEqual(record["collision_only_meshes"], ["convex_002"])
            body = load_scene_cfg(record["scene_path"])["scene"]["000"]
            spec = mujoco.MjSpec.from_file(body["xml_path"])
            for geom in spec.geoms:
                if geom.contype:
                    geom.density = 123.
            model = spec.compile()
            self.assertAlmostEqual(model.body("object").mass[0], .007 * 123., places=6)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            geom = model.geom("object_collision_2").id
            self.assertEqual(model.geom_contype[geom], 1)
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            np.testing.assert_allclose(world.min(0), tiny.bounds[0], atol=1e-8)
            np.testing.assert_allclose(world.max(0), tiny.bounds[1], atol=1e-8)

    def test_rejects_an_articulated_model_in_the_rigid_object_library(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self.make_source(root)
            urdf = source / "coacd/coacd.urdf"
            urdf.write_text(urdf.read_text().replace('type="fixed"', 'type="revolute"'))
            with self.assertRaisesRegex(ValueError, "non-fixed joint"):
                convert_object("000", root / "raw", root / "assets")


if __name__ == "__main__":
    unittest.main()
