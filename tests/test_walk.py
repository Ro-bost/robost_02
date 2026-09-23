import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from robost.simulation.walk import Swing, Crawl, main
from robost.simulation.mpc import Terrain, RADIUS


class WalkTests(unittest.TestCase):
    def test_20cm_descent_waits_for_trailing_pair(self):
        ctl=Crawl.__new__(Crawl)
        ctl.terrain=Terrain('stairs',20)
        feet=np.array([[2.85,.18,1+RADIUS],[2.85,-.18,1+RADIUS],
                       [3.18,.18,.8+RADIUS],[3.18,-.18,.8+RADIUS]])
        ctl.d=SimpleNamespace(xpos=feet,time=0.)
        ctl.wbc=SimpleNamespace(ids=[0,1,2,3])
        ctl.reverse_descent=False;ctl.backward=True;ctl.turning=False
        ctl.order=[2,0,3,1];ctl.index=0;ctl.step_length=.15
        ctl.fault=None
        ctl.choose_step()
        # Leading RR must not go to the second lower tread while both trailing
        # feet are still on the plateau. Choose trailing FR instead.
        self.assertEqual(ctl.active,0)
        # First prepare the trailing foot at the upper edge, not across two
        # levels in one move; the next move can descend one riser.
        self.assertAlmostEqual(ctl.goal[0],2.89)
        self.assertAlmostEqual(ctl.goal[2],1.+RADIUS)
        self.assertIsNone(ctl.fault)

    def test_swing_endpoints_and_stair_clearance(self):
        swing=Swing()
        for h in (.15,.20,-.15,-.20):
            start=np.array([.69,-.18,.52661]);end=start+np.array([.15,0,h])
            p,v,_=swing.sample(start,end,0,2.4)
            np.testing.assert_allclose(p,start);np.testing.assert_allclose(v,0,atol=1e-10)
            p,v,_=swing.sample(start,end,1,2.4)
            np.testing.assert_allclose(p,end);np.testing.assert_allclose(v,0,atol=1e-10)
            for phase in np.linspace(.3,.7,7):
                p,_,_=swing.sample(start,end,phase,2.4)
                self.assertGreaterEqual(p[2],max(start[2],end[2])+.049)

    def test_standing_physics_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            args=['robost.simulation.walk.py','--gait','crawl','--speed','0','--duration','2',
                  '--headless','--output',folder]
            with patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()): main()
            r=json.loads((Path(folder)/'flat_report.json').read_text())
            self.assertEqual(r['qp_failures'],0)
            self.assertIsNone(r['hardware_pass'])
            self.assertGreater(r['final_xyz'][2],.30)
            self.assertTrue(np.all(np.array(r['torque_peak_Nm'])<=np.tile([17,17,25.2],4)+1e-6))
            self.assertEqual(sum(r['warnings']),0)


if __name__=='__main__': unittest.main()
