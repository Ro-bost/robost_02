"""Model-adapter checks in conda rs02-rl, before expensive training."""
import unittest
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from mjlab.entity import Entity
from robost.rl.base import robot_cfg, PACKAGE, LEGS, make_cfg


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=Entity(robot_cfg()).spec.compile()
        original=ET.parse(PACKAGE/'rs02.xml').getroot()
        original.find('compiler').set('meshdir',str(PACKAGE/'meshes'))
        cls.original=mujoco.MjModel.from_xml_string(ET.tostring(original,encoding='unicode'))

    def test_inertias_and_joint_limits_preserved(self):
        m=self.m;o=self.original
        self.assertAlmostEqual(m.body_mass.sum(),16.940606,places=5)
        self.assertEqual(m.nv,18);self.assertEqual(m.nu,12)
        for name in [l+'_'+j+'_joint' for l in LEGS for j in ('hip','thigh','calf')]:
            np.testing.assert_allclose(m.joint(name).range,o.joint(name).range)
        for i in range(1,m.nbody):
            body=m.body(i);orig=o.body(body.name)
            np.testing.assert_allclose(body.mass,orig.mass)
            np.testing.assert_allclose(body.inertia,orig.inertia)

    def test_torque_limits_and_collision_masks(self):
        m=self.m
        for i in range(m.nu):
            name=m.joint(m.actuator_trnid[i,0]).name
            limit=25.2 if '_calf_' in name else 17.
            np.testing.assert_allclose(m.actuator_forcerange[i],[-limit,limit])
        for i in range(m.ngeom):
            geom=m.geom(i)
            if geom.name.endswith('_collision'):
                self.assertEqual(int(geom.contype[0]),1)
                self.assertEqual(int(geom.conaffinity[0]),1)

    def test_no_crawl_or_artificial_support(self):
        cfg=make_cfg(16,evaluate=True)
        self.assertEqual(cfg.sim.mujoco.timestep,.002)
        self.assertEqual(cfg.decimation,10)
        self.assertNotIn('push_robot',cfg.events)
        self.assertFalse(cfg.auto_reset)
        self.assertEqual(set(cfg.actions),{'joint_pos'})


if __name__=='__main__':unittest.main()
