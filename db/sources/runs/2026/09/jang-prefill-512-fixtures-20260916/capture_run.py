#!/usr/bin/env python3
"""Prospectively frozen Plan020: 256 reference, 384 control, 512 candidate."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Tools/flash'))
import benchmark, cache_study, widen_study
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import require_terminal_sampling, validate_receipt_file
OUT=ROOT/'.build/flash/runs/prefill-512-qualification-20260916'
RAW=ROOT/'db/sources/runs/2026/09/jang-prefill-512-numerics-20260916.json'
FIXTURES=ROOT/'.build/flash/prefill-qualification-20260916'
MODEL=Path('/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream/models/jang-6s').resolve()
NAMES=('grounded-document','reader-offsets')
CHUNKS=(256,384,512)
RUN_ID='prefill-512-qualification-20260916'

def check(value,message):
    if not value: raise EvidenceError(message)

def main():
    check(not RAW.exists(),'existing raw cohort cannot be overwritten')
    check(not any(k.startswith('SLOTSTREAM_') or k.startswith('MLX_') for k in os.environ),'ambient runtime overrides refused')
    fresh_output(OUT)
    report={'format':'slotstream-prefill-512-numerical-cohort-v1','run_set_id':RUN_ID,
        'completed':False,'overall_passed':False,'default_admitted':False,
        'reference_chunk':256,'empirical_control_chunk':384,'candidate_chunk':512,
        'control_independently_qualified':False,'timing_eligible':False,
        'scope':'Fresh previously unused prompts; actual Engine numerical evidence only.',
        'protocol_sha256':sha256(ROOT/'plans/020-prefill-512-qualification.md'),
        'protocol_text':(ROOT/'plans/020-prefill-512-qualification.md').read_text(),
        'policy':{'memory_gb':14,'context':4096,'chunks':list(CHUNKS),'continuation_tokens':8,
                  'control_selection':'arithmetic midpoint chosen before observing results',
                  'paging':'diagnostic only for numerical/functional acceptance'},
        'driver_sha256':sha256(Path(__file__)),'comparator_sha256':sha256(HERE/'compare.py'),
        'captures':[],'comparisons':{}}
    def persist():
        atomic_json(RAW,report)
        atomic_json(OUT/'progress.json',report)
    persist()
    try:
        original=(ROOT/'.build/release/slotstream').resolve()
        report['build_identity'],_=validate_build_identity(original)
        prior=read_json(ROOT/'db/sources/runs/2026/09/jang-prefill-numerics-20260916.json')
        before,after=prior['build_identity']['source'],report['build_identity']['source']
        changes=[p for p in sorted(set(before)|set(after)) if before.get(p)!=after.get(p)]
        check(changes==['Sources/SlotstreamDiagnostics/Diagnostics+PrefillCapture.swift'],
              'unexpected source changes beyond diagnostic allowlist/self-check')
        report['source_changes_from_plan019']=changes
        report['prior_inactive_observer_proof']={'raw_sha256':sha256(ROOT/'db/sources/runs/2026/09/jang-prefill-numerics-20260916.json'),
            'scope':'Historical ordinary-output proof remains separate; production runtime source unchanged. No old numerical/timing rows pooled.'}
        binary,report['archive_receipt']=widen_study._archive_candidate(original,OUT)
        report['binary_sha256']=sha256(binary);report['binary_path']=str(binary)
        report['model_verification']=cache_study.verified_model_revision(MODEL)
        geometry=cache_study.derive_source_geometry(MODEL,verification=report['model_verification'])
        report['model_geometry']=geometry
        frozen=harness_hashes()
        for name in ('Tools/prefill_bench.py','Tools/thermal_readiness.py'):frozen[name]=sha256(ROOT/name)
        report['harness_hashes']=frozen
        for name,digest in frozen.items():
            dest=OUT/'harness-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest);check(sha256(dest)==digest,'harness snapshot drift')
        for name in ('capture_run.py','compare.py','test_compare.py','tasks_run.py','test_tasks_run.py'):
            shutil.copyfile(HERE/name,OUT/name)
        task_source=FIXTURES/'task-source.json'
        corpus=read_json(FIXTURES/'tasks-tokenized/prompts.json')
        check(sha256(task_source)==corpus['source_manifest_sha256'],'frozen task source changed')
        tasks={t['id']:t for t in read_json(FIXTURES/'tasks.json')['tasks']}
        native={x['id']:x for x in corpus['documents']}
        report['fixture_hashes']={str(p.relative_to(FIXTURES)):sha256(p) for p in
            (task_source,FIXTURES/'tasks.json',FIXTURES/'tasks-tokenized/prompts.json')}
        report['prompts']={}
        for name in NAMES:
            file=FIXTURES/'tasks'/(name+'.txt');entry=native[name]
            artifact=FIXTURES/'tasks-tokenized'/entry['path']
            check(sha256(file)==tasks[name]['prompt_sha256'],'prompt text drift')
            check(sha256(artifact)==entry['artifact_sha256'],'token artifact drift')
            ids=read_json(artifact)['promptIDs']
            check(len(ids)==entry['prompt_count'] and 1024<len(ids)<3000,'invalid frozen prompt size')
            report['prompts'][name]={'file':str(file),'sha256':sha256(file),'expected_ids':ids,
                                    'token_artifact_sha256':sha256(artifact)}
        expected_opts=prior['smokes'][0]['stats']['optimizations']
        persist()
        def identities():
            check(sha256(Path(__file__))==report['driver_sha256'],'driver changed during cohort')
            check(sha256(HERE/'compare.py')==report['comparator_sha256'],'comparator changed during cohort')
            check(validate_build_identity(binary)[0]==report['build_identity'],'build source drift')
            check(all(sha256(ROOT/p)==h for p,h in frozen.items()),'harness drift')
            check(all(sha256(FIXTURES/p)==h for p,h in report['fixture_hashes'].items()),'fixture drift')

        for name in NAMES:
            directories=[]
            for chunk in CHUNKS:
                identities()
                base=OUT/'numerics'/name/str(chunk);base.mkdir(parents=True,exist_ok=False)
                native_out=base/'capture';directories.append(native_out)
                row={'prompt':name,'chunk':chunk,'capture_path':str(native_out),'status':'running'}
                report['captures'].append(row)
                cmd=[str(binary),'prefill-capture','--model',str(MODEL),'--prompt-file',report['prompts'][name]['file'],
                     '--chunk',str(chunk),'--output',str(native_out)]
                if chunk!=256:cmd+=['--continuation-file',str(directories[0]/'continuation.json')]
                row['command']=cmd;persist();print('CAPTURE START',name,chunk,flush=True)
                path=base/'launcher'
                try:
                    code=benchmark.launch(path,14,900,cmd,run_set_id=RUN_ID,model_hash=sha256(MODEL/'config.json'))
                finally:
                    if path.exists():
                        row['launcher_artifacts']={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size,
                            'text':p.read_text(errors='replace')} for p in path.iterdir() if p.is_file()}
                    if native_out.exists():
                        row['native_artifacts']={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size}
                                                for p in native_out.iterdir() if p.is_file()}
                        for filename in ('started.json','failure.json','report.json','completion.json'):
                            if (native_out/filename).exists():row[filename]=read_json(native_out/filename)
                    persist()
                receipt=validate_receipt_file(path/'receipt.json');row['receipt']=receipt;persist()
                require_terminal_sampling(receipt)
                check(code==0 and receipt['result']['functional_success'],'native capture did not complete')
                check(receipt['memory']['qualified'] is True,'memory/terminal evidence incomplete')
                capture=read_json(native_out/'report.json')
                check(capture['prompt_ids']==report['prompts'][name]['expected_ids'],'prompt ID mismatch')
                check(capture['optimizations']==expected_opts,'deployed settings differ')
                check(capture['identity']['binary_sha256']==report['binary_sha256'],'capture binary differs')
                check(capture['effective_prefill_chunk']==chunk,'effective chunk differs')
                row['status']='complete';persist();print('CAPTURE PASS',name,chunk,flush=True)
            output=OUT/'numerics'/name/'comparison.json'
            result=subprocess.run([sys.executable,str(HERE/'compare.py'),*[str(p) for p in directories],
                                   '--output',str(output)],text=True,capture_output=True)
            check(output.exists(),'comparator emitted no assessment')
            report['comparisons'][name]={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
                'report':read_json(output),'report_sha256':sha256(output)}
            persist();print('COMPARISON',name,'PASS' if report['comparisons'][name]['report']['passed'] else 'FAIL',flush=True)
        identities()
        check(cache_study.derive_source_geometry(MODEL,verification=report['model_verification'])==geometry,'model changed')
        report['completed']=True
        report['overall_passed']=all(x['exit_code']==0 and x['report']['passed'] for x in report['comparisons'].values())
        persist()
        review={'format':'slotstream-prefill-512-numerical-review-v1','run_set_id':RUN_ID,'completed':True,
            'overall_passed':report['overall_passed'],'reference_chunk':256,'empirical_control_chunk':384,'candidate_chunk':512,
            'control_independently_qualified':False,'raw_sha256':sha256(RAW),'build_identity':report['build_identity'],
            'binary_sha256':report['binary_sha256'],'harness_hashes':report['harness_hashes'],'comparisons':report['comparisons'],
            'scope':'Bounded numerical gate only; task/latency admission separate; no default change.'}
        atomic_json(OUT/'numerical-review.json',review)
        print('NUMERICAL COHORT COMPLETE',report['overall_passed'],flush=True)
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';report['stopped_at_unix']=time.time()
        print('STOP',report['error'],flush=True);raise
    finally:
        persist();atomic_json(OUT/'report.json',report)

if __name__=='__main__':main()
