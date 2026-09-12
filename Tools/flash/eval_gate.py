#!/usr/bin/env python3
"""Explicit Plan 008 host evaluation gate; excluded from ordinary Flash gates."""
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
import eval_env
from common import EvidenceError,atomic_json,fresh_output,sha256

def run(env:Path,output:Path)->int:
 owned=False
 try:
  output=fresh_output(output);owned=True;identity=eval_env.verify(env);python=env.resolve()/"bin/python3"
  script="""import importlib,json,sys,unittest
sys.path.insert(0,'Tools/flash')
def flat(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from flat(x)
  else:yield x
loader=unittest.TestLoader();suite=unittest.TestSuite()
for name in ('test_corpus','test_metrics','test_eval_env'):suite.addTests(loader.loadTestsFromModule(importlib.import_module(name)))
tests=list(flat(suite));result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'testIDs':[x.id() for x in tests],'testClasses':sorted({x.__class__.__name__ for x in tests}),'testsRun':result.testsRun,'failures':[x.id() for x,_ in result.failures],'errors':[x.id() for x,_ in result.errors],'skipped':[x.id() for x,_ in result.skipped],'successful':result.wasSuccessful()}))"""
  command=[str(python),"-I","-c",script]
  result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=300,env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","LC_ALL":"C.UTF-8","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})
  try:structured=json.loads(result.stdout.strip().splitlines()[-1])
  except Exception as error:raise EvidenceError("Plan008 structured test result is missing") from error
  required={"CorpusTests","MetricsTests","EvalEnvironmentTests"}
  if result.returncode or structured.get("successful") is not True or structured.get("testsRun")!=len(structured.get("testIDs",[])) or structured.get("testsRun",0)<1 or not required.issubset(set(structured.get("testClasses",[]))) or structured.get("failures") or structured.get("errors") or structured.get("skipped"):
   raise EvidenceError("Plan008 host evaluation tests failed or were incomplete")
  atomic_json(output/"tests.json",{"command":command,"exitCode":result.returncode,"structured":structured,"stdout":result.stdout,"stderr":result.stderr})
  report={"format":"slotstream-plan008-eval-gate-v1","qualification":False,"hostOnly":True,
          "environment":identity,"testsSHA256":sha256(output/"tests.json"),"modelLoaded":False}
  atomic_json(output/"report.json",report);atomic_json(output/"completion.json",{"format":"slotstream-plan008-eval-gate-completion-v1","reportSHA256":sha256(output/"report.json"),"qualification":False});return 0
 except Exception as error:
  if owned:atomic_json(output/"failure.json",{"error":f"{type(error).__name__}: {error}"})
  return 1

def main():
 p=argparse.ArgumentParser();p.add_argument("--env",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();return run(a.env,a.output)
if __name__=="__main__":raise SystemExit(main())
