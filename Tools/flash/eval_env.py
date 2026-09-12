#!/usr/bin/env python3
"""Create or verify the isolated, hash-pinned Plan 008 evaluation environment."""
import argparse,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'.build/flash/corpora/source-recon';WHEEL=SOURCE/'numpy-2.3.5-cp312-cp312-macosx_14_0_arm64.whl';EXPECTED='612a95a17655e213502f60cfb9bf9408efdc9eb1d5f50535cc6eb365d11b42b5';BASE=Path('/Users/imdad/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_wheel():
 if WHEEL.is_symlink() or not WHEEL.is_file() or WHEEL.stat().st_size!=5088378 or sha(WHEEL)!=EXPECTED:raise RuntimeError('pinned NumPy wheel is missing or changed')
 return {'path':str(WHEEL.resolve()),'bytes':5088378,'sha256':EXPECTED}
def _contained(env):
 allowed=(ROOT/'.build/flash').resolve();target=env.resolve()
 if target==allowed or not target.is_relative_to(allowed):raise RuntimeError('evaluation environment must stay under .build/flash')
 return target
def verify_installation(env):
 site=env/'lib/python3.12/site-packages';checked=0
 with zipfile.ZipFile(WHEEL) as archive:
  for info in archive.infolist():
   if info.is_dir() or not (info.filename.startswith('numpy/') or info.filename.startswith('numpy.libs/')):continue
   path=site/info.filename
   if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=hashlib.sha256(archive.read(info)).hexdigest():raise RuntimeError('installed NumPy bytes differ from pinned wheel: '+info.filename)
   checked+=1
 if checked<100:raise RuntimeError('installed NumPy wheel coverage is incomplete')
 return checked
def verify(env):
 wheel=verify_wheel();env=_contained(env);p=env/'bin/python3'
 if p.is_symlink() or not p.is_file():raise RuntimeError('evaluation interpreter is missing or symlinked')
 installed_files=verify_installation(env)
 code='import importlib.metadata,json,numpy,platform,sys; d=importlib.metadata.distribution("numpy"); print(json.dumps({"python":platform.python_version(),"machine":platform.machine(),"numpy":numpy.__version__,"numpy_path":numpy.__file__,"executable":sys.executable,"installer":d.read_text("INSTALLER"),"backend":numpy.show_config(mode="dicts")}))'
 probe_env=os.environ.copy();probe_env['OPENBLAS_NUM_THREADS']='1';probe_env['OMP_NUM_THREADS']='1'
 r=subprocess.run([str(p),'-I','-c',code],text=True,capture_output=True,timeout=30,env=probe_env)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 if not (d['python']=='3.12.14' and d['machine']=='arm64' and d['numpy']=='2.3.5' and Path(d['numpy_path']).resolve().is_relative_to(env) and Path(d['executable']).resolve()==p.resolve() and d['installer'].strip()=='pip' and isinstance(d['backend'],dict)):raise RuntimeError('evaluation environment identity mismatch')
 d['wheel']=wheel;d['installed_wheel_files_verified']=installed_files;d['interpreter_sha256']=sha(p)
 d['blas_threads']={'OPENBLAS_NUM_THREADS':probe_env['OPENBLAS_NUM_THREADS'],'OMP_NUM_THREADS':probe_env['OMP_NUM_THREADS']}
 return d
def setup(env):
 if env.exists():return verify(env)
 verify_wheel()
 if not BASE.is_file():raise RuntimeError('pinned base interpreter is missing')
 target=_contained(env)
 probe=subprocess.run([str(BASE),'-c','import platform,sys;print(platform.python_version(),platform.machine())'],text=True,capture_output=True,timeout=30)
 if probe.returncode or probe.stdout.strip()!='3.12.14 arm64':raise RuntimeError('pinned base interpreter identity mismatch')
 created=subprocess.run([str(BASE),'-m','venv',str(target)],text=True,capture_output=True,timeout=120)
 if created.returncode:raise RuntimeError('evaluation venv creation failed: '+created.stderr.strip())
 installed=subprocess.run([str(target/'bin/pip'),'install','--no-index','--no-deps',str(WHEEL)],text=True,capture_output=True,timeout=120)
 if installed.returncode:raise RuntimeError('pinned NumPy installation failed: '+installed.stderr.strip())
 return verify(target)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['setup','verify']);p.add_argument('--env',type=Path,required=True);a=p.parse_args();print(json.dumps(setup(a.env) if a.mode=='setup' else verify(a.env),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
