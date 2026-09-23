import unittest
import json
from pathlib import Path
from robost.cli.stairs import select_policy
from robost.paths import ROOT


class LauncherTests(unittest.TestCase):
    def test_defaults_match_release_manifest(self):
        # Source-only installs intentionally omit checkpoint binaries.
        policies={item['height_cm']:item for item in
                  json.loads((ROOT/'configs/policies.json').read_text())}
        adapter,path=select_policy(15)
        self.assertEqual(adapter,'rhythm')
        self.assertEqual(path,ROOT/policies[15]['path'])
        self.assertEqual(adapter,policies[15]['adapter'])
        self.assertEqual(path.name,'15cm.pt')
        adapter20,path20=select_policy(20)
        self.assertEqual(adapter20,'rhythm')
        self.assertEqual(path20.name,'20cm.pt')
        self.assertEqual(path20,ROOT/policies[20]['path'])
        self.assertEqual(adapter20,policies[20]['adapter'])

    def test_custom_checkpoint_requires_explicit_adapter(self):
        with self.assertRaises(ValueError):select_policy(20,Path('custom.pt'))
        with self.assertRaises(ValueError):select_policy(20,adapter='route')
        self.assertEqual(select_policy(20,Path('custom.pt'),'route'),('route',Path('custom.pt')))


if __name__=='__main__':unittest.main()
