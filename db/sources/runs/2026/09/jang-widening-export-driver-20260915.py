"""Export approved result fields; original receipts and payloads stay immutable."""
from pathlib import Path
import json
import re
import statistics
import sys

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'Tools/flash'))
from common import read_json, sha256
from receipts import validate_receipt_file, require_terminal_sampling
import widen_study


def completed(root):
    report = read_json(root / 'report.json')
    completion = read_json(root / 'completion.json')
    assert completion['report_sha256'] == sha256(root / 'report.json')
    return report


def export():
    runs = ROOT / '.build/flash/runs'
    session = ROOT / '.build/flash/qualification-20260915'
    root = runs / 'widen-qualify-cooled-20260915'
    assert (root / 'failure.json').exists() and not (root / 'completion.json').exists()
    fixture = widen_study.cache_study.load_fixture()
    prompts = {item['id']:item for item in fixture['prompts']}
    report = {'rows': []}
    for prompt_id in prompts:
        for directory in sorted((root / 'prompts' / prompt_id).glob('pair-*')):
            arms = {}
            for policy in ('scalar','packed4-to6'):
                arm = directory / policy
                if (arm / 'receipt.json').exists():
                    document = read_json(arm / 'stats.json')
                    arms[policy] = {'path':str(arm.relative_to(root)), 'receipt_sha256':sha256(arm / 'receipt.json'),
                        'stats_sha256':sha256(arm / 'stats.json'), 'settling_sha256':sha256(arm / 'settling.json'),
                        'timings':widen_study.validate_stats(document,prompts[prompt_id],policy)}
            if len(arms) != 2: continue
            index = int(directory.name.removeprefix('pair-'))
            doc = read_json(directory / 'scalar/stats.json')
            prompt_index = list(prompts).index(prompt_id)
            report['rows'].append({'prompt_id':prompt_id,'pair_index':index,
                'order':['scalar','packed4-to6'] if (prompt_index+index)%2 == 0 else ['packed4-to6','scalar'],
                'prompt_ids':doc['prompt_ids'],'output_ids':doc['output_ids'],'arms':arms})
    assert len(report['rows']) == 22
    pairs = []
    for row in report['rows']:
        arms = {}
        documents = {}
        receipts = {}
        for policy, evidence in row['arms'].items():
            directory = root / evidence['path']
            assert sha256(directory / 'receipt.json') == evidence['receipt_sha256']
            assert sha256(directory / 'stats.json') == evidence['stats_sha256']
            assert sha256(directory / 'settling.json') == evidence['settling_sha256']
            document = read_json(directory / 'stats.json')
            receipt = validate_receipt_file(directory / 'receipt.json')
            require_terminal_sampling(receipt)
            timing_eligible, ineligible_reason = True, None
            try: widen_study.validate_timing_eligibility(document, receipt)
            except widen_study.EvidenceError as error:
                timing_eligible, ineligible_reason = False, str(error)
            settling = read_json(directory / 'settling.json')
            assert settling['cooldown']['stable_seconds'] == 30
            assert settling['requested_seconds'] == 2
            stats = document['stats']
            arms[policy] = {'stats_sha256': evidence['stats_sha256'], 'receipt_sha256': evidence['receipt_sha256'],
                'settling_sha256': evidence['settling_sha256'], 'timings': evidence['timings'],
                'timing_eligible':timing_eligible, 'ineligible_reason':ineligible_reason, 'first_text_seconds': stats['firstTextSeconds'], 'peak_bytes': receipt['memory']['peak_bytes'],
                'generator_system_before': stats['generatorSystemBefore'], 'generator_system_after': stats['generatorSystemAfter'],
                'cooldown_elapsed_seconds': settling['cooldown']['elapsed_seconds'],
                'launcher_swap_delta': {key:receipt['vm']['after'][key]-receipt['vm']['before'][key] for key in ('swapins','swapouts')},
                'generator_swap_delta': {key:stats['generatorVMAfter'][key]-stats['generatorVMBefore'][key] for key in ('swapins','swapouts')}}
            documents[policy], receipts[policy] = document, receipt
        widen_study._exact_work(documents['scalar'], documents['packed4-to6'])
        widen_study.compare_runtime_controls(documents['scalar'], receipts['scalar'], documents['packed4-to6'], receipts['packed4-to6'])
        pairs.append({'prompt_id':row['prompt_id'], 'pair_index':row['pair_index'], 'order':row['order'],
            'prompt_tokens':len(row['prompt_ids']), 'output_tokens':len(row['output_ids']), 'arms':arms})
    medians = {}
    for prompt in sorted({row['prompt_id'] for row in pairs}):
        selected = [row for row in pairs if row['prompt_id'] == prompt]
        if len(selected) != 10 or not all(arm['timing_eligible'] for row in selected for arm in row['arms'].values()):
            continue
        medians[prompt] = {policy:{key:statistics.median(row['arms'][policy]['timings'][key] for row in selected)
            for key in selected[0]['arms'][policy]['timings']} for policy in ('scalar','packed4-to6')}
    workload_decisions = {}
    for prompt in medians:
        values = [row for row in pairs if row['prompt_id'] == prompt]
        control = [row['arms']['scalar']['timings']['decode_seconds'] for row in values]
        candidate = [row['arms']['packed4-to6']['timings']['decode_seconds'] for row in values]
        boot = widen_study.paired_bootstrap(control,candidate)
        first = statistics.median(row['arms']['packed4-to6']['timings']['first_token_seconds']/row['arms']['scalar']['timings']['first_token_seconds']-1 for row in values)
        workload_decisions[prompt] = {'pairs':10,'decode':boot,'median_first_token_regression_fraction':first,
            'performance_thresholds_met':boot['estimate'] >= .10 and boot['lower_95'] > 0 and first <= .05}
    identity = read_json(root / 'candidate-archive/bin/build-identity.json')
    archive_receipt = read_json(root / 'candidate-archive/receipt.json')
    stopped = runs / 'widen-qualify-20260915'
    bad = stopped / 'prompts/python-deduplicate/pair-09/scalar'
    bad_stats = read_json(bad / 'stats.json')['stats']
    bad_receipt = validate_receipt_file(bad / 'receipt.json')
    native = read_json(session / 'native-checks.json')
    tests = (session / 'all-host-tests.log').read_text()
    assert re.search(r'Ran 228 tests in', tests) and '\nOK\n' in tests
    profile_root = runs / 'jang-stage5-request-profile-quiet-20260915'
    if (profile_root / 'completion.json').exists(): profile = completed(profile_root)
    else:
        profile = read_json(profile_root / 'failure.json')
        profile.update(scope='Planned raw code-prefix timing profile, not run',limits=profile['reason'],median_stats={},rows=[])
    profile_rows = []
    for row in profile['rows']:
        path = profile_root / row['path']
        assert sha256(path / 'stats.json') == row['stats_sha256']
        assert sha256(path / 'receipt.json') == row['receipt_sha256']
        profile_rows.append(row)
    sample_root = runs / 'jang-stage5-cpu-sample-20260915'
    sample = completed(sample_root) if (sample_root / 'completion.json').exists() else read_json(sample_root / 'failure.json')
    output = {
        'format':'slotstream-jang-widening-qualification-session-v1',
        'hardware':read_json(session / 'hardware.json'),
        'native_binary_sha256':identity['binary_sha256'],
        'cooled_failure_sha256':sha256(root / 'failure.json'),
        'parity_report_sha256':sha256(runs / 'widen-parity-qualification-20260915/report.json'),
        'fixture_sha256':sha256(ROOT / 'Tools/fixtures/flash/cache-study.json'),
        'scope':'Current-source fixed short-prompt qualification attempts, thermally inconclusive overall',
        'decision':{'qualified':False,'disposition':'thermally-inconclusive','cohort_complete':False,
            'completed_pairs':22,'eligible_complete_pairs':sum(all(a['timing_eligible'] for a in row['arms'].values()) for row in pairs),
            'required_pairs':30,'failure':read_json(root / 'failure.json'), 'complete_workloads':workload_decisions},
        'medians':medians, 'pairs':pairs,
        'native_checks':{key:native[key] for key in ('passed','failed','skipped')},
        'host_tests':{'passed':228,'log_sha256':sha256(session / 'all-host-tests.log')},
        'harness_hashes':archive_receipt['identities']['harness_hashes'],
        'stopped_attempt':{'failure':read_json(stopped / 'failure.json'), 'failure_sha256':sha256(stopped / 'failure.json'),
            'completed_arms':len(list(stopped.glob('prompts/*/*/*/receipt.json'))),
            'failed_timing_arm':'python-deduplicate/pair-09/scalar',
            'stats_sha256':sha256(bad / 'stats.json'), 'receipt_sha256':sha256(bad / 'receipt.json'),
            'functional_success':bad_receipt['result']['functional_success'],
            'memory_qualified':bad_receipt['memory']['qualified'],
            'before':bad_stats['generatorSystemBefore'], 'after':bad_stats['generatorSystemAfter']},
        'request_profile':{'complete':profile.get('complete',False), 'evidence_sha256':sha256(profile_root / ('report.json' if (profile_root / 'report.json').exists() else 'failure.json')),
            'scope':profile['scope'], 'limits':profile['limits'], 'median_stats':profile['median_stats'], 'rows':profile_rows},
        'cpu_sample':{'complete':sample.get('complete',False), 'limits':sample.get('limits'),
            'sample_sha256':sample.get('sample_sha256'), 'operating_conditions_before':sample.get('operating_conditions_before'), 'report_sha256':sha256(sample_root / 'report.json'), 'exact_output_and_work_vs_untraced':sample.get('exact_output_and_work_vs_untraced'),
            'failure':sample.get('error')},
        'exporter_sha256':sha256(Path(__file__)),
        'limits':'Fixed short-prompt widening qualification and separate raw-code diagnostic. No broad held-out quality, GPU utilization, new kernel speedup, default activation, serving exposure or upstream release.'}
    path = ROOT / 'db/sources/runs/2026/09/jang-widening-qualification-20260915.json'
    assert not path.exists()
    path.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'path':str(path.relative_to(ROOT)), 'sha256':sha256(path), 'decision':output['decision'], 'medians':medians},indent=2))


if __name__ == '__main__': export()
