#!/usr/bin/env python3
"""Frozen Plan019 task screen/extension; never authorizes changing a default."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'Tools/flash'))
import benchmark
import cache_study
import readiness
import thermal_readiness
import widen_study
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity, validate_source_map
from receipts import validate_receipt_file, require_terminal_sampling

CHUNKS = (256, 512, 1024)
TASK_IDS = ('short-json', 'grounded-document', 'reader-offsets', 'long-csv')
EXTEND_IDS = ('short-json', 'long-csv')
DEPLOYED_OPTIMIZATIONS = {'adaptiveSpeculation': False, 'automaticReadScope': True, 'boundedDraftTail': False, 'boundedIndexer': False, 'boundedOutputQueue': True, 'boundedPLE': False, 'boundedSweepRows': False, 'cachedRouterWeights': False, 'compactIndexerRaw': False, 'compactMTPRow': True, 'compactNgramRows': True, 'compactScopeFrontier': False, 'compactStateWindows': True, 'compiledNormFinish': False, 'completePromptCheckpoint': True, 'contiguousSlotWrites': False, 'cpuSlotWrites': False, 'deduplicateImages': False, 'demandedPrefillOutput': False, 'denseExpertLookup': False, 'denseIndexerBypass': False, 'deviceSamplerDraw': True, 'directReadHandles': False, 'disjointSweepOutput': False, 'fusedGDNProjection': False, 'fusedGDNRecording': False, 'fusedRoPE': False, 'incrementalIndexer': False, 'indexerBlockTopK': False, 'layerExpertWorkspace': False, 'layerLocalFloorCache': False, 'ngramLookahead': False, 'ngramRingOrder': False, 'overlapResidentExperts': False, 'overlapSharedExpert': False, 'prefixCheckpointTokens': 256, 'readScopeTokens': 0, 'resolvedRuntimeBudget': False, 'responsiveGovernor': True, 'reuseFirstMTPEntry': False, 'routerTopK': False, 'selectedTextAttention': False, 'sharedRoPE': True, 'skipUnusedFinalForward': True, 'sparsePoolPins': False, 'tailAwarePrefill': False, 'terminalLastQuery': False, 'terminalPrefillPruning': False, 'valueOnlySamplerThreshold': True, 'visionAttentionPadding': 0, 'visionQueryTile': 256, 'wordSlotWrites': False, 'workspacePiecewiseWrites': False, 'workspaceTokenTile': 256}
MODEL = Path('/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream/models/jang-6s')
RUN_BASE = ROOT / '.build/flash/runs/prefill-qualification-20260916'
RAW_BASE = ROOT / 'db/sources/runs/2026/09'


def check(condition, message):
    if not condition:
        raise EvidenceError(message)


def typed_equal(actual, expected):
    """Python's bool==int and float==int are not exact JSON value types."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(typed_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def quality(task, text, finish_reason):
    """Return an objective task verdict; failure does not abort the screen."""
    if finish_reason != 'stop':
        return {'passed': False, 'reason': 'completion did not stop normally'}
    if not isinstance(text, str) or not text.strip():
        return {'passed': False, 'reason': 'empty or missing answer'}
    try:
        if task['kind'] == 'json':
            def reject_constant(value):
                raise ValueError('nonstandard JSON constant: ' + value)
            parsed = json.loads(text.strip(), object_pairs_hook=reject_duplicate_keys,
                                parse_constant=reject_constant)
            passed = typed_equal(parsed, task['expected'])
        elif task['kind'] == 'csv':
            parsed = list(csv.reader(io.StringIO(text.strip()), strict=True))
            passed = parsed == task['expected']
        else:
            raise EvidenceError('unknown fixture kind')
        return {'passed': passed, 'reason': 'exact parsed match' if passed else 'parsed answer differs from frozen expectation', 'parsed': parsed}
    except (ValueError, csv.Error) as error:
        return {'passed': False, 'reason': str(error)}


def schedule(phase):
    for rnd in ([0] if phase == 'screen' else [1, 2]):
        for index, name in enumerate(TASK_IDS):
            if phase == 'extend' and name not in EXTEND_IDS:
                continue
            offset = (rnd + index) % len(CHUNKS)
            order = CHUNKS[offset:] + CHUNKS[:offset]
            for chunk in order:
                yield {'task': name, 'round': rnd, 'chunk': chunk, 'order': list(order)}


def number(value, name, positive=False):
    check(type(value) in (int, float) and math.isfinite(value) and value >= 0
          and (not positive or value > 0), 'invalid ' + name)
    return value


def validate(document, receipt, task, chunk, prompt_ids):
    check(document.get('schema_version') == 1, 'stats schema differs')
    stats, plan = document['stats'], document['plan']
    check(stats.get('runtimeError') is None, 'runtime error')
    check(stats.get('memoryPressureCancelled') is False, 'memory pressure cancellation')
    check(document.get('experimental_memory_family') is False, 'experimental memory family enabled')
    check(document.get('effective_expert_widening') == 'packed4-to6', 'widening differs')
    check(document.get('effective_mtp') is False and document.get('effective_vision') is False, 'effective MTP/vision differs')
    check(plan.get('mtp') is False and plan.get('vision') is False, 'planned MTP/vision differs')
    check(plan.get('checkpoint_format') == 'jang6S', 'checkpoint differs')
    check(plan.get('target_gb') == 14 and plan.get('source') == '--memory-gb', 'target differs')
    check(plan.get('max_context_tokens') == 4096, 'context differs')
    check(plan.get('runtime_prefix_cache_enabled') is False, 'prefix enabled')
    check(plan.get('prefill_chunk') == chunk and document.get('effective_prefill_chunk') == chunk, 'chunk differs')
    check(receipt['environment'].get('SLOTSTREAM_PREFIX_CACHE') == '0', 'prefix environment differs')
    check(document.get('sampling') == {'greedy': True, 'seed': '42', 'requested_max_tokens': str(task['max_tokens'])}, 'sampling differs')
    check(document.get('prompt_ids') == prompt_ids, 'native prompt IDs differ')
    check(stats.get('promptTokens') == len(prompt_ids) and stats.get('prefillTokens') == len(prompt_ids), 'prompt work differs')
    ids = document.get('output_ids')
    check(isinstance(ids, list) and all(type(i) is int for i in ids) and len(ids) <= task['max_tokens'], 'output IDs invalid')
    check(stats.get('decodeTokens') == len(ids), 'output count differs')
    check(stats.get('reusedPrefixTokens') == 0 and stats.get('encodedImages') == 0, 'unexpected prefix/image work')
    check(type(document.get('effective_pool_slots')) is int and document['effective_pool_slots'] > 0, 'invalid slots')
    check(plan.get('pool_slots') == document['effective_pool_slots'], 'planned slots differ')
    check(receipt['memory']['target_gb_decimal'] == 14, 'launcher target differs')
    sampled = stats.get('sampledFootprint', {})
    check(type(sampled.get('samples')) is int and sampled['samples'] > 0 and sampled.get('peakBytes', 0) > 0, 'missing request footprint')
    metrics = {key: number(stats.get(key), key, positive=(key in ('prefillSeconds', 'requestSeconds', 'decodeSeconds')))
               for key in ('prefillSeconds', 'requestSeconds', 'firstTokenSeconds', 'decodeSeconds', 'peakMemoryGB')}
    return metrics | {'decodeTokensPerSecond': len(ids) / metrics['decodeSeconds'],
                      'promptTokens': len(prompt_ids), 'decodeTokens': len(ids),
                      'sampledPeakGB': receipt['memory']['peak_bytes'] / 1e9,
                      'effectivePoolSlots': document['effective_pool_slots']}


def stable_controls(document, receipt):
    # Chunk-dependent pool/ledger allocations may differ; model and arithmetic controls may not.
    values = {key: document.get(key) for key in ('optimizations', 'numerical_environment',
              'effective_expert_widening', 'effective_mtp', 'effective_vision', 'experimental_memory_family')}
    check(values['optimizations'] == DEPLOYED_OPTIMIZATIONS, 'deployed optimization controls differ')
    check(values['numerical_environment'] == {'effective_tf32': True, 'mlx_enable_tf32_raw': None}, 'numerical defaults differ')
    return values | {'launcher_environment': receipt['environment']}


def fixture_binding():
    files = [HERE / 'tasks.json', HERE / 'task-source.json', HERE / 'tasks-tokenized/prompts.json']
    corpus = read_json(files[-1])
    check(sha256(HERE / 'task-source.json') == corpus['source_manifest_sha256'], 'source manifest changed')
    tasks = read_json(HERE / 'tasks.json')['tasks']
    check(tuple(t['id'] for t in tasks) == TASK_IDS, 'task inventory/order differs')
    sources = {d['id']: d for d in read_json(HERE / 'task-source.json')['documents']}
    expected_ids = {}
    artifacts = {}
    for task in tasks:
        check(hashlib.sha256(task['prompt'].encode()).hexdigest() == task['prompt_sha256'], 'task text hash differs')
        check(sources[task['id']]['user'] == task['prompt'] and sources[task['id']]['system'] == '', 'native source prompt differs')
        check(type(task['max_tokens']) is int and 0 < task['max_tokens'] <= 512, 'invalid task output cap')
    for entry in corpus['documents']:
        path = (HERE / 'tasks-tokenized' / entry['path']).resolve(strict=True)
        check(path.is_relative_to(HERE / 'tasks-tokenized'), 'token artifact escaped fixture directory')
        check(sha256(path) == entry['artifact_sha256'], 'native artifact changed')
        artifact = read_json(path)
        ids = artifact['promptIDs']
        check(isinstance(ids, list) and ids and all(type(i) is int for i in ids), 'invalid native IDs')
        check(artifact['promptCount'] == entry['prompt_count'] == len(ids), 'native prompt count differs')
        check(artifact['sourceDocument']['id'] == entry['id'], 'native source identity differs')
        expected_ids[entry['id']] = ids
        artifacts[entry['id']] = artifact
        files.append(path)
    check(set(expected_ids) == set(TASK_IDS), 'native prompt inventory differs')
    for task in tasks:
        check(len(expected_ids[task['id']]) + task['max_tokens'] <= 4096, 'fixture exceeds context')
    hashes = {str(path.relative_to(HERE)): sha256(path) for path in files}
    return {t['id']: t for t in tasks}, expected_ids, hashes, corpus, artifacts


def summarize(rows, extended):
    summary = {'all_quality_passed': all(r.get('quality', {}).get('passed') is True for r in rows),
               'default_admitted': False, 'historical_plan018_rows_pooled': False,
               'output_work_caveat': 'Observed completed-request latency; differing content or token counts are confounded decode work.',
               'latency_qualification_complete': extended, 'candidates': {}}
    if not extended:
        return summary
    for chunk in (512, 1024):
        comparisons = {}
        for task in EXTEND_IDS:
            groups = {c: sorted((r for r in rows if r['task'] == task and r['chunk'] == c), key=lambda r:r['round']) for c in (256, chunk)}
            check(all([r['round'] for r in g] == [0, 1, 2] for g in groups.values()), 'incomplete latency cohort')
            pairs = []
            for base, candidate in zip(groups[256], groups[chunk]):
                b, c = base['metrics'], candidate['metrics']
                check(b['decodeTokensPerSecond'] > 0, 'baseline has no decode throughput')
                pairs.append({'round': base['round'],
                              'request_regression': c['requestSeconds'] / b['requestSeconds'] - 1,
                              'decode_tps_regression': 1 - c['decodeTokensPerSecond'] / b['decodeTokensPerSecond'],
                              'baseline_output_tokens': b['decodeTokens'], 'candidate_output_tokens': c['decodeTokens'],
                              'exact_output_ids': base['stats']['output_ids'] == candidate['stats']['output_ids']})
            comparisons[task] = {'pairs': pairs,
                'median_paired_request_regression': statistics.median(p['request_regression'] for p in pairs),
                'median_paired_decode_tps_regression': statistics.median(p['decode_tps_regression'] for p in pairs),
                'equal_output_work': all(p['exact_output_ids'] for p in pairs)}
        summary['candidates'][str(chunk)] = {'tasks': comparisons,
            'latency_guard_passed': all(c['median_paired_request_regression'] <= .05 for c in comparisons.values())
                                  and comparisons['long-csv']['median_paired_decode_tps_regression'] <= .05,
            'task_quality_passed': all(r['quality']['passed'] for r in rows if r['chunk'] in (256, chunk)),
            'default_admitted': False}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True, help='explicit archived candidate with validated build identity')
    parser.add_argument('--phase', choices=('screen', 'extend'), required=True)
    parser.add_argument('--numeric-report', type=Path, help='required passed composite report for extension')
    args = parser.parse_args()
    check(not any(k.startswith('SLOTSTREAM_') for k in os.environ), 'ambient SLOTSTREAM controls unsupported')
    check('MLX_ENABLE_TF32' not in os.environ, 'ambient TF32 override unsupported')
    check(args.phase == 'extend' or args.numeric_report is None, 'numeric report only applies to extension')
    run_id = 'prefill-tasks-' + args.phase + '-20260916'
    raw = RAW_BASE / ('jang-' + run_id + '.json')
    check(not raw.exists(), 'raw cohort exists; do not replace failed arms')
    binary = args.binary.resolve(strict=True)
    check(binary.parent != (ROOT / '.build/release').resolve(), 'use archived candidate, not mutable release output')
    identity, identity_paths = validate_build_identity(binary)
    tasks, ids, fixtures, corpus, artifacts = fixture_binding()
    frozen = harness_hashes()
    for rel in ('Tools/prefill_bench.py', 'Tools/thermal_readiness.py'):
        frozen[rel] = sha256(ROOT / rel)
    driver_hash = sha256(Path(__file__))
    screen = None
    numeric = None
    if args.phase == 'extend':
        check(args.numeric_report is not None, 'extension requires --numeric-report')
        numeric = read_json(args.numeric_report)
        check(numeric.get('overall_passed') is True, 'numerical review did not pass')
        for key, value in [('build_identity', identity), ('binary_sha256', sha256(binary)), ('harness_hashes', frozen)]:
            check(numeric.get(key) == value, 'numerical report binding differs: ' + key)
        screen_path = RAW_BASE / 'jang-prefill-tasks-screen-20260916.json'
        screen = read_json(screen_path)
        check(screen.get('completed') is True and len(screen.get('rows', [])) == 12, 'screen incomplete')
        check(all(r.get('quality', {}).get('passed') is True and r.get('timing_eligible') is True for r in screen['rows']), 'screen quality/timing did not pass')
        check([{k:r[k] for k in ('task', 'round', 'chunk', 'order')} for r in screen['rows']] == list(schedule('screen')), 'screen schedule differs')
        for key, value in [('build_identity', identity), ('binary_sha256', sha256(binary)), ('harness_hashes', frozen), ('fixture_hashes', fixtures)]:
            check(screen.get(key) == value, 'screen binding differs: ' + key)
        check(screen.get('driver', {}).get('sha256') == driver_hash, 'driver changed since screen')
    output = fresh_output(RUN_BASE / ('tasks-' + args.phase))
    report = {'format': 'slotstream-prefill-task-qualification-v1', 'run_set_id': run_id,
              'phase': args.phase, 'completed': False, 'default_admitted': False,
              'build_identity': identity, 'binary_sha256': sha256(binary), 'harness_hashes': frozen,
              'fixture_hashes': fixtures, 'tokenized_corpus': corpus, 'tokenized_artifacts': artifacts,
              'tasks': list(tasks.values()), 'driver': {'sha256': driver_hash, 'source': Path(__file__).read_text()},
              'policy': {'memory_gb': 14, 'max_context': 4096, 'chunks': list(CHUNKS), 'greedy': True, 'seed': 42,
                         'prefix_cache': False, 'mtp': False, 'vision': False, 'widening': 'packed4-to6',
                         'nominal_seconds': 30, 'maximum_nominal_wait_seconds': 180, 'maximum_model_seconds': 900,
                         'preflight_reclaimable_gb': 17, 'options': 'deployed CLI defaults',
                         'paging_policy': 'Changed global paging excludes performance timing, not product correctness.',
                         'thermal_scope': 'Observed prelaunch window and generator endpoints, not continuous thermometry.',
                         'request_regression_ceiling': .05, 'long_csv_decode_tps_regression_ceiling': .05},
              'schedule': list(schedule(args.phase)), 'rows': []}
    if screen is not None:
        report['screen_binding'] = {'path': str(screen_path), 'sha256': sha256(screen_path)}
        report['numeric_binding'] = {'path': str(args.numeric_report.resolve()), 'sha256': sha256(args.numeric_report), 'report': numeric}
    def persist():
        atomic_json(raw, report)  # Capture first; derived scratch files always follow.
        atomic_json(output / 'progress.json', report)
    persist()
    try:
        model = MODEL.resolve(strict=True)
        report['model_verification'] = cache_study.verified_model_revision(model)
        report['model_geometry'] = cache_study.derive_source_geometry(model, verification=report['model_verification'])
        for item in corpus['tokenizer_identity']['files']:
            check(sha256(model / item['path']) == item['sha256'], 'tokenizer/model identity changed')
        for relative, digest in frozen.items():
            target = output / 'harness-snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            check(sha256(target) == digest, 'harness changed during snapshot')
        shutil.copyfile(Path(__file__), output / 'driver.py')
        reference_controls = screen['rows'][0]['controls'] if screen else None
        identity_hashes = {str(p): sha256(p) for p in identity_paths.values()}
        def unchanged():
            check(sha256(Path(__file__)) == driver_hash, 'driver changed')
            check(all(sha256(ROOT / p) == digest for p, digest in frozen.items()), 'harness changed')
            check(all(sha256(HERE / p) == digest for p, digest in fixtures.items()), 'fixtures changed')
            check(all(sha256(Path(p)) == digest for p, digest in identity_hashes.items()), 'archived build changed')
            validate_source_map(identity)
            if screen:
                check(sha256(screen_path) == report['screen_binding']['sha256'], 'screen evidence changed')
                check(sha256(args.numeric_report) == report['numeric_binding']['sha256'], 'numerical evidence changed')
        persist()
        for item in schedule(args.phase):
            unchanged()
            task = tasks[item['task']]
            row = item | {'status': 'readiness', 'readiness_observations': []}
            report['rows'].append(row)
            persist()
            def observe():
                observed = thermal_readiness.observe()
                row['readiness_observations'].append(observed)
                persist()
                return observed
            row['readiness'] = readiness.wait_for_nominal(stable_seconds=30, max_seconds=180, interval=5, reader=observe)
            row['settling'] = benchmark.settle_before_model_launch()
            path = output / item['task'] / str(item['round']) / str(item['chunk'])
            command = [str(binary), 'run', '--model', str(model), '--memory-gb', '14', '--max-context', '4096',
                       '--mtp', 'off', '--vision', 'off', '--expert-widening', 'packed4-to6', '--prompt', task['prompt'],
                       '--max-tokens', str(task['max_tokens']), '--greedy', '--seed', '42', '--sample-footprint',
                       '--stats-json', str(path / 'stats.json')]
            row.update(status='running', command=command, explicit_environment={'SLOTSTREAM_PREFIX_CACHE': '0', 'SLOTSTREAM_PREFILL_CHUNK': str(item['chunk'])})
            persist()
            print('START', item['task'], item['round'], item['chunk'], flush=True)
            os.environ.update(row['explicit_environment'])
            try:
                code = benchmark.launch(path, 14, 900, command, run_set_id=run_id, model_hash=sha256(model / 'config.json'))
            finally:
                for key in row['explicit_environment']:
                    os.environ.pop(key, None)
                if path.exists():
                    row['artifacts'] = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size, 'text': p.read_text(errors='replace')}
                                        for p in path.iterdir() if p.is_file()}
                persist()
            row['receipt'] = receipt = validate_receipt_file(path / 'receipt.json')
            if (path / 'stats.json').exists():
                row['stats'] = read_json(path / 'stats.json')
            persist()
            require_terminal_sampling(receipt)
            check(code == 0 and receipt['result']['functional_success'], 'monitored generation failed')
            document = row['stats']
            row['metrics'] = validate(document, receipt, task, item['chunk'], ids[item['task']])
            row['controls'] = controls = stable_controls(document, receipt)
            if reference_controls is None:
                reference_controls = controls
            check(controls == reference_controls, 'runtime controls changed')
            widen_study.validate_timing_eligibility(document, receipt)
            row['raw_text'] = document.get('text')
            row['quality'] = quality(task, row['raw_text'], document['stats'].get('finishReason'))
            row.update(status='eligible', timing_eligible=True)
            unchanged()
            persist()
            print('DONE', item['task'], item['round'], item['chunk'], json.dumps(row['quality']), flush=True)
        unchanged()
        check(cache_study.derive_source_geometry(model, verification=report['model_verification']) == report['model_geometry'], 'model geometry changed')
        report['completed'] = True
        persist()
        rows = (screen['rows'] if screen else []) + report['rows']
        report['summary'] = summarize(rows, args.phase == 'extend')
        persist()
        print('SUMMARY', json.dumps(report['summary']), flush=True)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['stopped_at_unix'] = time.time()
        if report['rows']:
            report['rows'][-1]['stop_reason'] = report['error']
        print('STOP', report['error'], flush=True)
        raise
    finally:
        persist()
        atomic_json(output / 'report.json', report)


if __name__ == '__main__':
    main()
