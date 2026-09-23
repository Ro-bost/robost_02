"""Run with the rs02-mujoco environment: python -m unittest test_rs02_terrain."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import rs02_mpc as sim


class TerrainTests(unittest.TestCase):
    def test_both_courses(self):
        for cm in (15, 20):
            t = sim.Terrain('stairs', cm)
            self.assertEqual(len(t.transitions), 10)
            self.assertAlmostEqual(t.end, 4.15)
            self.assertAlmostEqual(t.height(t.plateau_start + .5), 5 * cm / 100)
            self.assertEqual(t.height(t.end + .01), 0)
            previous = 0
            for edge, top in t.transitions:
                self.assertAlmostEqual(abs(top - previous), cm / 100)
                self.assertAlmostEqual(t.height(edge - 1e-7), previous)
                self.assertAlmostEqual(t.height(edge + 1e-7), top)
                self.assertGreater(t.landing(edge), edge)
                previous = top
            with tempfile.TemporaryDirectory() as folder, patch.object(sim, 'OUT', Path(folder)):
                m, _ = sim.build_scene(t)
                self.assertEqual(m.nu, 12)
                root = ET.parse(Path(folder) / f'{t.output_name}_scene.xml').getroot()
                for name, left, length, top in t.sections:
                    g = root.find(f"worldbody/geom[@name='{name}']")
                    pos = np.fromstring(g.get('pos'), sep=' ')
                    size = np.fromstring(g.get('size'), sep=' ')
                    self.assertAlmostEqual(pos[0] - size[0], left)
                    self.assertAlmostEqual(size[0] * 2, length)
                    self.assertAlmostEqual(pos[2] + size[2], t.height(left + length / 2))

    def test_speed_cli(self):
        for cm, speed in ((15, .1), (20, .3)):
            with tempfile.TemporaryDirectory() as folder, patch.object(sim, 'OUT', Path(folder)):
                args = ['rs02_mpc.py', '--terrain', 'stairs', '--step-height-cm', str(cm),
                        '--speed', str(speed), '--duration', '.01', '--headless']
                with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                    sim.main()
                report = json.loads((Path(folder) / f'stairs_{cm}cm_course_report.json').read_text())
                self.assertEqual(report['command_speed_m_s'], speed)
                self.assertEqual(len(report['all_feet_crossed_each_step_at_s']), 10)


if __name__ == '__main__':
    unittest.main()
