#!/usr/bin/env python3
"""Frozen Plan019 orchestration: inert-hook smoke and matched numerical capture."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'Tools/flash'))
import benchmark, cache_study, widen_study
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import require_terminal_sampling, validate_receipt_file

OUT = ROOT / '.build/flash/runs/prefill-qualification-20260916'
RAW = ROOT / 'db/sources/runs/2026/09/jang-prefill-numerics-20260916.json'
OLD = ROOT / '.build/flash/runs/prefill-chunks-20260916'
FIXTURES = ROOT / '.build/flash/prefill-chunks-20260916'
MODEL = Path('/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream/models/jang-6s').resolve()

def check(value, message):
    if not value: raise EvidenceError(message)

def main():
    check(not RAW.exists(), 'raw evidence already exists')
    check(not any(k.startswith('SLOTSTREAM_') or k.startswith('MLX_') for k in os.environ), 'ambient runtime overrides refused')
    fresh_output(OUT)
    old = read_json(ROOT / 'db/sources/runs/2026/09/jang-prefill-chunks-20260916.json')
    report = {'format':'slotstream-prefill-numerical-cohort-v1', 'completed':False,
        'overall_passed':False, 'timing_eligible':False,
        'scope':'Actual Engine prefill, then bounded common teacher-forced continuation; not a timing study or default admission.',
        'policy':{'memory_gb':14,'context':4096,'chunks':[256,512,1024],
                  'continuation_tokens':8,'prompt_names':['prose','code'],
                  'paging':'diagnostic only for numerical/functional acceptance'},
        'driver_sha256':sha256(Path(__file__)), 'comparator_sha256':sha256(HERE/'compare.py'),
        'smokes':[], 'captures':[], 'comparisons':{}}
    def persist():
        atomic_json(RAW,report)
        atomic_json(OUT/'progress.json',report)
    persist()
    try:
        original=(ROOT/'.build/release/slotstream').resolve()
        report['build_identity'],_=validate_build_identity(original)
        binary,archive_receipt=widen_study._archive_candidate(original,OUT)
        report['archive_receipt']=archive_receipt
        report['binary_sha256']=sha256(binary)
        report['binary_path']=str(binary)
        report['model_verification']=cache_study.verified_model_revision(MODEL)
        geometry=cache_study.derive_source_geometry(MODEL,verification=report['model_verification'])
        report['model_geometry']=geometry
        frozen=harness_hashes()
        for name in ('Tools/prefill_bench.py','Tools/thermal_readiness.py'):
            frozen[name]=sha256(ROOT/name)
        report['harness_hashes']=frozen
        for name,digest in frozen.items():
            dest=OUT/'harness-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest);check(sha256(dest)==digest,'harness snapshot changed')
        for name in ('capture_run.py','compare.py','test_compare.py'):
            shutil.copyfile(HERE/name,OUT/name)
        report['prompts']={}
        for name in ('prose','code'):
            file=FIXTURES/'prompts'/(name+'.txt')
            check(sha256(file)==old['prompts'][name]['sha256'],'frozen prompt changed')
            report['prompts'][name]={'file':str(file),'sha256':sha256(file),
                'expected_ids':old['tokenized_artifacts'][name]['promptIDs']}
        persist()

        def identities():
            check(sha256(Path(__file__))==report['driver_sha256'],'driver changed')
            check(sha256(HERE/'compare.py')==report['comparator_sha256'],'comparator changed')
            check(sha256(binary)==report['binary_sha256'],'binary changed')
            check(all(sha256(ROOT/name)==digest for name,digest in frozen.items()),'harness changed')
            check(validate_build_identity(binary)[0]==report['build_identity'],'build source changed')

        def launch(row,path,command,environment=None):
            identities()
            row.update(status='running',command=command,explicit_environment=environment or {})
            persist()
            os.environ.update(environment or {})
            try:
                code=benchmark.launch(path,14,900,command,run_set_id='prefill-qualification-20260916',
                                      model_hash=sha256(MODEL/'config.json'))
            finally:
                for key in environment or {}:os.environ.pop(key,None)
                if path.exists():
                    row['launcher_artifacts']={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size,
                         'text':p.read_text(errors='replace')} for p in path.iterdir() if p.is_file()}
                persist()
            receipt=validate_receipt_file(path/'receipt.json');row['receipt']=receipt;persist()
            require_terminal_sampling(receipt)
            check(code==0 and receipt['result']['functional_success'],'monitored command failed')
            check(receipt['memory']['qualified'] is True,'memory or terminal evidence incomplete')
            row['status']='complete';persist()
            return receipt

        # Functional predecessor comparison only: no historical timing reused.
        for name in ('prose','code'):
            row={'prompt':name,'chunk':1024,'scope':'ordinary nil-observer output/control smoke'}
            report['smokes'].append(row)
            path=OUT/'smoke'/name
            cmd=[str(binary),'run','--model',str(MODEL),'--memory-gb','14','--max-context','4096',
                 '--mtp','off','--vision','off','--expert-widening','packed4-to6',
                 '--prompt-file',report['prompts'][name]['file'],'--max-tokens','128','--greedy',
                 '--seed','42','--sample-footprint','--stats-json',str(path/'stats.json')]
            print('SMOKE START',name,flush=True)
            receipt=launch(row,path,cmd,{'SLOTSTREAM_PREFIX_CACHE':'0','SLOTSTREAM_PREFILL_CHUNK':'1024'})
            actual=read_json(path/'stats.json');prior=read_json(OLD/name/'0/1024/stats.json')
            prior_receipt=read_json(OLD/name/'0/1024/receipt.json')
            row['stats']=actual
            check(actual['prompt_ids']==report['prompts'][name]['expected_ids'],'smoke prompt mismatch')
            widen_study._exact_work(prior,actual)
            widen_study.compare_runtime_controls(prior,prior_receipt,actual,receipt)
            check(prior['text']==actual['text'],'nil observer changed text')
            row['exact_predecessor_output_and_controls']=True;persist()
            print('SMOKE PASS',name,flush=True)

        for name in ('prose','code'):
            directories=[]
            for chunk in (256,512,1024):
                base=OUT/'numerics'/name/str(chunk);base.mkdir(parents=True,exist_ok=False)
                native=base/'capture';directories.append(native)
                row={'prompt':name,'chunk':chunk,'capture_path':str(native)}
                report['captures'].append(row)
                cmd=[str(binary),'prefill-capture','--model',str(MODEL),
                     '--prompt-file',report['prompts'][name]['file'],'--chunk',str(chunk),'--output',str(native)]
                if chunk!=256:cmd+=['--continuation-file',str(directories[0]/'continuation.json')]
                print('CAPTURE START',name,chunk,flush=True)
                try:launch(row,base/'launcher',cmd)
                finally:
                    if native.exists():
                        row['native_artifacts']={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size}
                                                for p in native.iterdir() if p.is_file()}
                        for filename in ('started.json','failure.json','report.json','completion.json'):
                            if (native/filename).exists():row[filename]=read_json(native/filename)
                        persist()
                capture=read_json(native/'report.json')
                check(capture['prompt_ids']==report['prompts'][name]['expected_ids'],'capture prompt IDs changed')
                check(capture['optimizations']==report['smokes'][0]['stats']['optimizations'],'capture deployment options changed')
                check(capture['identity']['binary_sha256']==report['binary_sha256'],'capture binary binding changed')
                check(capture['effective_expert_widening']=='packed4-to6','widening changed')
                print('CAPTURE PASS',name,chunk,flush=True)
            comparison=OUT/'numerics'/name/'comparison.json'
            result=subprocess.run([sys.executable,str(HERE/'compare.py'),*[str(p) for p in directories],
                                   '--output',str(comparison)],text=True,capture_output=True)
            check(comparison.exists(),'comparator produced no assessment')
            report['comparisons'][name]={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
                'report':read_json(comparison),'report_sha256':sha256(comparison)}
            persist()
            print('COMPARISON',name,'PASS' if report['comparisons'][name]['report']['passed'] else 'FAIL',flush=True)
        identities()
        check(cache_study.derive_source_geometry(MODEL,verification=report['model_verification'])==geometry,'model changed')
        report['completed']=True
        report['overall_passed']=all(v['exit_code']==0 and v['report']['passed'] for v in report['comparisons'].values())
        persist()
        atomic_json(OUT/'numerical-review.json',{'format':'slotstream-prefill-numerical-review-v1',
            'overall_passed':report['overall_passed'],'completed':True,'raw_sha256':sha256(RAW),
            'build_identity':report['build_identity'],'binary_sha256':report['binary_sha256'],
            'harness_hashes':report['harness_hashes'],'comparisons':report['comparisons'],
            'scope':'Two natural prompts, empirical 512 control, 1024 candidate; no default/task/timing admission.'})
        print('NUMERICAL COHORT COMPLETE',report['overall_passed'],flush=True)
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';report['stopped_at_unix']=time.time()
        print('STOP',report['error'],flush=True)
        raise
    finally:
        persist();atomic_json(OUT/'report.json',report)

if __name__=='__main__':main()
