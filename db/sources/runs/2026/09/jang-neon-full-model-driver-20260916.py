import sys,json,statistics,subprocess,hashlib
from pathlib import Path
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'Tools/flash'))
import widen_study as w,cache_study,benchmark
from common import atomic_json,sha256,validate_build_identity,harness_hashes
OUT=ROOT/'.build/flash/runs/neon-full-model-20260916'
RAW=ROOT/'db/sources/runs/2026/09/jang-neon-full-model-20260916.json'
OUT.mkdir(exist_ok=False)
old=ROOT/'.build/flash/runs/pre-simd-widening-20260915/bin/slotstream'
new=ROOT/'.build/release/slotstream'
model=Path('/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream/models/jang-6s')
a,_=validate_build_identity(old,historical=True)
b,_=validate_build_identity(new.resolve())
for path,digest in a['source'].items():
    content=subprocess.check_output(['git','show','926bc10:'+path])
    assert hashlib.sha256(content).hexdigest()==digest, path
verification=cache_study.verified_model_revision(model)
geometry=cache_study.derive_source_geometry(model,verification=verification)
new,_=w._archive_candidate(new.resolve(),OUT)
report={'format':'slotstream-neon-exploratory-benchmark-v1','scope':'two alternating paired rounds per each of three short prompts; not formal qualification','qualification':False,'old_commit':'926bc10','new_commit':'7670f8e','old_binary_sha256':sha256(old),'new_binary_sha256':sha256(new),'model_verification':verification,'options':cache_study.load_fixture()['options'],'widening_policy_both_arms':'packed4-to6','harness_hashes':harness_hashes(),'source_changes':[p for p in sorted(set(a['source'])|set(b['source'])) if a['source'].get(p)!=b['source'].get(p)],'rows':[],'completed':False}
atomic_json(RAW,report)
try:
 for round_id in range(2):
  for prompt_index,prompt in enumerate(cache_study.load_fixture()['prompts']):
   order=['previous','neon'] if (round_id+prompt_index)%2==0 else ['neon','previous']
   row={'prompt_id':prompt['id'],'pair_index':round_id,'order':order,'arms':{}}
   docs={}; receipts={}
   for arm in order:
    print('START',round_id,prompt['id'],arm,flush=True)
    stats,receipt,settled=w._run_arm(old if arm=='previous' else new,model,prompt,'packed4-to6',OUT/prompt['id']/str(round_id)/arm,'neon-exploratory-20260916')
    timings=w.validate_stats(stats,prompt,'packed4-to6')
    docs[arm]=stats; receipts[arm]=receipt
    row['arms'][arm]={'timings':timings,'stats':stats,'receipt':receipt,'settling':settled}
    atomic_json(OUT/'progress.json',row)
    atomic_json(RAW,{**report,'in_progress':row})
    w.validate_timing_eligibility(stats,receipt)
    print('DONE',round_id,prompt['id'],arm,json.dumps(timings),flush=True)
   w._exact_work(docs['previous'],docs['neon'])
   w.compare_runtime_controls(docs['previous'],receipts['previous'],docs['neon'],receipts['neon'])
   row['exact_output_and_work']=True
   row['decode_time_reduction']=1-row['arms']['neon']['timings']['decode_seconds']/row['arms']['previous']['timings']['decode_seconds']
   report['rows'].append(row); atomic_json(RAW,report)
   print('PAIR',prompt['id'],round_id,row['decode_time_reduction'],flush=True)
 assert cache_study.derive_source_geometry(model,verification=verification)==geometry
 report['completed']=True
 report['summary']={}
 for prompt in cache_study.load_fixture()['prompts']:
  rows=[r for r in report['rows'] if r['prompt_id']==prompt['id']]
  report['summary'][prompt['id']]={arm:{k:statistics.median([r['arms'][arm]['timings'][k] for r in rows]) for k in rows[0]['arms'][arm]['timings']} for arm in ['previous','neon']}
  report['summary'][prompt['id']]['paired_decode_time_reduction']=statistics.median(r['decode_time_reduction'] for r in rows)
 print('SUMMARY',json.dumps(report['summary']),flush=True)
except Exception as e:
 report['error']=str(e)
 print('STOP',type(e).__name__,str(e),flush=True)
 raise
finally:
 atomic_json(RAW,report)
 atomic_json(OUT/'report.json',report)
