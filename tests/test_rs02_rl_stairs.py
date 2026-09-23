"""Regression tests: course dimensions and exact zero-weight policy extension."""
import unittest
import torch
import mujoco
import numpy as np
from types import SimpleNamespace
from rs02_rl_stairs import CourseCfg, expand_state, make_cfg, planar_tracking, course_bounds, course_end


class StairTests(unittest.TestCase):
    def test_course_geometry(self):
        for difficulty, rise in ((0., .02), (13/18, .15), (1., .20)):
            spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
            out=CourseCfg(size=(7.,3.)).function(difficulty,spec,None)
            self.assertEqual(len(out.geometries),10)
            tops=[g.geom.pos[2]+g.geom.size[2] for g in out.geometries[1:]]
            np.testing.assert_allclose(tops,np.array([1,2,3,4,5,4,3,2,1])*rise)
            self.assertAlmostEqual(out.geometries[5].geom.size[0]*2,1.)
            self.assertAlmostEqual(out.geometries[1].geom.size[0]*2,.3)
            self.assertAlmostEqual(out.geometries[1].geom.size[1]*2,1.6)
            self.assertTrue(all(g.geom.mass==0 for g in out.geometries))

    def test_policy_extension(self):
        torch.manual_seed(1)
        old={'mlp.0.weight':torch.randn(256,48),
             'obs_normalizer._mean':torch.randn(1,48),
             'obs_normalizer._var':torch.rand(1,48)+.1,
             'obs_normalizer._std':torch.rand(1,48)+.1}
        new={k:torch.empty(v.shape[0],235) for k,v in old.items()}
        result=expand_state(old,new)
        x=torch.randn(10,48);extra=torch.randn(10,187)
        before=(x-old['obs_normalizer._mean'])/old['obs_normalizer._std']
        after=(torch.cat((x,extra),1)-result['obs_normalizer._mean'])/result['obs_normalizer._std']
        torch.testing.assert_close(before@old['mlp.0.weight'].T,after@result['mlp.0.weight'].T)
        self.assertEqual(torch.count_nonzero(result['mlp.0.weight'][:,48:]),0)

    def test_eval_no_curriculum(self):
        cfg=make_cfg(1,evaluate=True)
        self.assertEqual(cfg.scene.terrain.terrain_type,'plane')
        self.assertFalse(cfg.curriculum)
        self.assertEqual(list(cfg.observations['actor'].terms)[-1],'height_scan')
        self.assertEqual(cfg.sim.mujoco.timestep,.002)
        self.assertEqual(len(cfg.actions['joint_pos'].clip),12)
        self.assertAlmostEqual(cfg.actions['joint_pos'].clip['FR_calf_joint'][0],-2.397+.03)
        self.assertAlmostEqual(cfg.actions['joint_pos'].clip['FR_calf_joint'][1],-.87856-.03)

    def test_vertical_motion_not_horizontal_error(self):
        data=SimpleNamespace(root_link_lin_vel_b=torch.tensor([[.25,0.,0.],[.25,0.,.2]]))
        env=SimpleNamespace(scene={'robot':SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(get_command=lambda _:torch.tensor([[.25,0.,0.],[.25,0.,0.]])))
        torch.testing.assert_close(planar_tracking(env),torch.ones(2))

    def test_exit_is_failure_not_success(self):
        class Scene(dict):pass
        scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[1.,.7,0.],[5.4,0.,0.],[1.,0.,0.]]))))
        scene.env_origins=torch.zeros(3,3)
        env=SimpleNamespace(scene=scene)
        self.assertEqual(course_bounds(env).tolist(),[True,False,False])
        self.assertEqual(course_end(env).tolist(),[False,True,False])


if __name__=='__main__':unittest.main()
