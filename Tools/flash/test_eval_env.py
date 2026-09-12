#!/usr/bin/env python3
import sys,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import eval_env

class EvalEnvironmentTests(unittest.TestCase):
 def test_pinned_wheel_and_existing_environment(self):
  identity=eval_env.verify(eval_env.ROOT/'.build/flash/eval-env')
  self.assertEqual(identity['numpy'],'2.3.5');self.assertEqual(identity['wheel']['sha256'],eval_env.EXPECTED)
 def test_wheel_is_verified_before_existing_environment_process(self):
  with mock.patch.object(eval_env,'WHEEL',Path('/missing-wheel')),mock.patch.object(eval_env.subprocess,'run') as run:
   with self.assertRaises(RuntimeError):eval_env.verify(eval_env.ROOT/'.build/flash/eval-env')
   run.assert_not_called()
 def test_environment_must_be_contained(self):
  with tempfile.TemporaryDirectory() as directory:
   with self.assertRaises(RuntimeError):eval_env.verify(Path(directory))
if __name__=='__main__':unittest.main(verbosity=2)
