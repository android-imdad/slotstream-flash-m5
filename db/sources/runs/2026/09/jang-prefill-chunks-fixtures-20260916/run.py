#!/usr/bin/env python3
"""Frozen exploratory prefill screen. No default or numerical qualification implied."""
from __future__ import annotations
import copy
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
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import validate_receipt_file, require_terminal_sampling

OUT = ROOT / '.build/flash/runs/prefill-chunks-20260916'
RAW = ROOT / 'db/sources/runs/2026/09/jang-prefill-chunks-20260916.json'
MODEL = Path('/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream/models/jang-6s')
CHUNKS = (256, 512, 1024)
WORKLOADS = ('prose', 'code')
RUN_ID = 'prefill-chunks-20260916'


def check(condition, message):
    if not condition:
        raise EvidenceError(message)


def number(value, name):
    check(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid ' + name)
    return value


def validate(document, receipt, chunk):
    check(document.get('schema_version') == 1, 'stats schema differs')
    stats, plan = document['stats'], document['plan']
    check(stats.get('runtimeError') is None and stats.get('finishReason') in ('stop', 'length'), 'generation failed')
    check(stats.get('memoryPressureCancelled') is False, 'memory pressure cancellation')
    check(document.get('experimental_memory_family') is False, 'experimental memory family enabled')
    check(document.get('effective_expert_widening') == 'packed4-to6', 'widening policy differs')
    check(document.get('effective_mtp') is False and document.get('effective_vision') is False, 'MTP/vision differs')
    check(plan.get('mtp') is False and plan.get('vision') is False, 'plan MTP/vision differs')
    check(plan.get('checkpoint_format') == 'jang6S', 'checkpoint format differs')
    check(plan.get('target_gb') == 14 and plan.get('source') == '--memory-gb', 'fixed target differs')
    check(plan.get('max_context_tokens') == 4096, 'context differs')
    check(plan.get('runtime_prefix_cache_enabled') is False, 'prefix enabled')
    check(plan.get('prefill_chunk') == chunk and document.get('effective_prefill_chunk') == chunk, 'effective chunk differs')
    check(receipt['environment'].get('SLOTSTREAM_PREFIX_CACHE') == '0', 'prefix environment differs')
    check(document.get('sampling') == {'greedy': True, 'seed': '42', 'requested_max_tokens': '128'}, 'sampling differs')
    prompt_ids, output_ids = document.get('prompt_ids'), document.get('output_ids')
    check(isinstance(prompt_ids, list) and 1024 < len(prompt_ids) <= 2500, 'prompt does not exercise chunks or exceeds bound')
    check(isinstance(output_ids, list) and 0 < len(output_ids) <= 128 and bool(document.get('text')), 'output is empty or oversized')
    check(stats.get('promptTokens') == len(prompt_ids) and stats.get('prefillTokens') == len(prompt_ids), 'prompt work differs')
    check(stats.get('decodeTokens') == len(output_ids), 'output token count differs')
    check(stats.get('reusedPrefixTokens') == 0 and stats.get('encodedImages') == 0, 'unexpected reuse/images')
    check(type(document.get('effective_pool_slots')) is int and document['effective_pool_slots'] > 0, 'invalid slots')
    check(plan.get('pool_slots') == document['effective_pool_slots'], 'plan/effective slots differ')
    check(receipt['memory']['target_gb_decimal'] == 14, 'launcher memory budget differs')
    sampled = stats.get('sampledFootprint', {})
    check(type(sampled.get('samples')) is int and sampled['samples'] > 0 and sampled.get('peakBytes', 0) > 0, 'missing request footprint')
    times = {key: number(stats.get(key), key) for key in ('prefillSeconds', 'requestSeconds', 'firstTokenSeconds', 'decodeSeconds', 'peakMemoryGB')}
    check(times['decodeSeconds'] > 0 and times['prefillSeconds'] > 0, 'zero timing')
    return {**times, 'decodeTokensPerSecond': len(output_ids) / times['decodeSeconds'],
            'prefillTokensPerSecond': len(prompt_ids) / times['prefillSeconds'],
            'promptTokens': len(prompt_ids), 'decodeTokens': len(output_ids),
            'sampledPeakGB': receipt['memory']['peak_bytes'] / 1e9,
            'effectivePoolSlots': document['effective_pool_slots']}


def stable_controls(document, receipt):
    # Pool, workspace, expected footprint and expert work intentionally vary with chunk.
    return {key: document.get(key) for key in ('optimizations', 'numerical_environment', 'sampling',
            'effective_expert_widening', 'effective_mtp', 'effective_vision', 'experimental_memory_family')} | {
            'launcher_environment': receipt['environment']}


def schedule():
    for round_index in range(3):
        for workload_index, workload in enumerate(WORKLOADS):
            offset = (round_index + workload_index) % 3
            order = CHUNKS[offset:] + CHUNKS[:offset]
            for chunk in order:
                yield workload, round_index, chunk, order


def summarize(rows):
    result = {}
    for workload in WORKLOADS:
        entries = [row for row in rows if row['workload'] == workload]
        by_chunk = {chunk: sorted([r for r in entries if r['chunk'] == chunk], key=lambda r: r['round']) for chunk in CHUNKS}
        check(all(len(group) == 3 for group in by_chunk.values()), 'incomplete cohort cannot yield summary')
        result[workload] = {'medians_by_chunk': {}, 'matched_rounds_vs_256': {}}
        for chunk, group in by_chunk.items():
            result[workload]['medians_by_chunk'][str(chunk)] = {
                key: statistics.median(row['metrics'][key] for row in group) for key in group[0]['metrics']}
            if chunk == 256:
                continue
            pairs = []
            for baseline, candidate in zip(by_chunk[256], group):
                pairs.append({'round': baseline['round'],
                    'exact_output_ids_diagnostic': baseline['stats']['output_ids'] == candidate['stats']['output_ids'],
                    'same_output_token_count': baseline['metrics']['decodeTokens'] == candidate['metrics']['decodeTokens'],
                    'baseline_output_tokens': baseline['metrics']['decodeTokens'],
                    'candidate_output_tokens': candidate['metrics']['decodeTokens'],
                    'decodeTPSRegression': 1 - candidate['metrics']['decodeTokensPerSecond'] / baseline['metrics']['decodeTokensPerSecond'],
                    **{key + 'Reduction': 1 - candidate['metrics'][key] / baseline['metrics'][key]
                       for key in ('prefillSeconds', 'requestSeconds', 'firstTokenSeconds')}})
            result[workload]['matched_rounds_vs_256'][str(chunk)] = {
                'pairs': pairs,
                'median_prefill_time_reduction': statistics.median(p['prefillSecondsReduction'] for p in pairs),
                'median_ttft_reduction': statistics.median(p['firstTokenSecondsReduction'] for p in pairs),
                'median_observed_request_time_reduction': statistics.median(p['requestSecondsReduction'] for p in pairs),
                'median_paired_decode_tps_regression': statistics.median(p['decodeTPSRegression'] for p in pairs),
                'request_time_interpretation': 'Observed completed-request latency; different output lengths/content are NOT equal decode work.',
                'all_output_token_counts_equal': all(p['same_output_token_count'] for p in pairs)}
    result['timing_screen_only'] = {str(chunk): {
        'passes_timing_shortlist': all(
            result[name]['matched_rounds_vs_256'][str(chunk)]['median_observed_request_time_reduction'] >= 0.10
            and result[name]['matched_rounds_vs_256'][str(chunk)]['median_paired_decode_tps_regression'] <= 0.05
            for name in WORKLOADS),
        'quality_pending': True, 'manual_usefulness_pending': True, 'default_admitted': False,
        'output_length_confounding': any(not result[name]['matched_rounds_vs_256'][str(chunk)]['all_output_token_counts_equal'] for name in WORKLOADS)
    } for chunk in (512, 1024)}
    return result


def main():
    check(not RAW.exists(), 'raw evidence already exists; never overwrite a cohort')
    check(not any(key.startswith('SLOTSTREAM_') for key in os.environ), 'ambient Slotstream controls unsupported')
    output = fresh_output(OUT)
    report = {'format': 'slotstream-prefill-chunk-screen-v1', 'run_set_id': RUN_ID, 'completed': False,
        'qualification': False, 'numerical_quality': 'pending rechunk logit-band qualification',
        'default_qualification': 'pending; no default changes authorized by this screen alone',
        'manual_usefulness_review': 'pending; raw generated text retained for reviewer',
        'timing_screen_rule': 'Shortlist only: >=10% median paired request-time reduction and <=5% median paired decode-TPS regression in EACH workload; every arm timing eligible and <=14 GB; manual usefulness and numerical qualification remain required. Different output lengths are confounded work.',
        'policy': {'memory_gb': 14, 'context_tokens': 4096, 'chunks': CHUNKS, 'rounds': 3,
            'greedy': True, 'seed': 42, 'max_tokens': 128, 'mtp': False, 'vision': False, 'prefix_cache': False,
            'widening': 'packed4-to6', 'nominal_seconds': 30, 'maximum_nominal_wait_seconds': 180,
            'maximum_model_seconds': 900, 'preflight_reclaimable_gb': 17,
            'paging_policy': 'A paging change excludes this performance cohort, never a product correctness verdict.',
            'thermal_scope': '30-second observed prelaunch window and runtime endpoints; not continuous thermometry'},
        'driver': {'sha256': sha256(Path(__file__)), 'source': Path(__file__).read_text()}, 'rows': []}
    def persist():
        atomic_json(RAW, report)  # Raw public capture ALWAYS precedes derived scratch output.
        atomic_json(output / 'progress.json', report)
    persist()
    try:
        model = MODEL.resolve(strict=True)
        report['model_verification'] = cache_study.verified_model_revision(model)
        geometry = cache_study.derive_source_geometry(model, verification=report['model_verification'])
        report['model_geometry'] = geometry
        original = (ROOT / '.build/release/slotstream').resolve(strict=True)
        report['build_identity'], _ = validate_build_identity(original)
        binary, report['archive_receipt'] = widen_study._archive_candidate(original, output)
        report['binary_sha256'] = sha256(binary)
        frozen = harness_hashes()
        for relative in ('Tools/prefill_bench.py', 'Tools/thermal_readiness.py'):
            frozen[relative] = sha256(ROOT / relative)
        report['harness_hashes'] = frozen
        for relative, digest in frozen.items():
            dest = output / 'harness-snapshot' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
            check(sha256(dest) == digest, 'harness changed during snapshot')
        shutil.copyfile(Path(__file__), output / 'driver.py')
        prompts = {name: (HERE / 'prompts' / (name + '.txt')).read_text() for name in WORKLOADS}
        report['prompts'] = {name: {'text': text, 'sha256': sha256(HERE / 'prompts' / (name + '.txt'))} for name, text in prompts.items()}
        corpus = read_json(HERE / 'tokenized/prompts.json')
        check(corpus['executable_identity']['binary_sha256'] == report['binary_sha256'], 'tokenizer binary differs')
        check(sha256(HERE / 'prompt-source.json') == corpus['source_manifest_sha256'], 'prompt source changed')
        expected_ids = {}
        report['tokenized_corpus'] = corpus
        report['tokenized_artifacts'] = {}
        for entry in corpus['documents']:
            artifact_path = HERE / 'tokenized' / entry['path']
            check(sha256(artifact_path) == entry['artifact_sha256'], 'tokenized prompt artifact changed')
            artifact = read_json(artifact_path)
            expected_ids[entry['id']] = artifact['promptIDs']
            report['tokenized_artifacts'][entry['id']] = artifact
        check(set(expected_ids) == set(WORKLOADS), 'tokenized workload set differs')
        report['schedule'] = [{'workload': name, 'round': rnd, 'chunk': chunk, 'order': order} for name, rnd, chunk, order in schedule()]
        persist()
        prompt_ids = {}
        reference_controls = None
        for name, rnd, chunk, order in schedule():
            check(sha256(Path(__file__)) == report['driver']['sha256'], 'driver changed during cohort')
            check(all(sha256(ROOT / relative) == digest for relative, digest in frozen.items()), 'harness changed during cohort')
            check(sha256(binary) == report['binary_sha256'], 'archived binary changed')
            row = {'workload': name, 'round': rnd, 'chunk': chunk, 'order': order, 'status': 'readiness', 'readiness_observations': []}
            report['rows'].append(row)
            persist()
            def observe():
                observed = thermal_readiness.observe()
                row['readiness_observations'].append(observed)
                persist()
                return observed
            row['readiness'] = readiness.wait_for_nominal(stable_seconds=30, max_seconds=180, interval=5, reader=observe)
            row['settling'] = benchmark.settle_before_model_launch()
            path = output / name / str(rnd) / str(chunk)
            command = [str(binary), 'run', '--model', str(model), '--memory-gb', '14', '--max-context', '4096',
                '--mtp', 'off', '--vision', 'off', '--expert-widening', 'packed4-to6', '--prompt', prompts[name],
                '--max-tokens', '128', '--greedy', '--seed', '42', '--sample-footprint', '--stats-json', str(path / 'stats.json')]
            row.update(status='running', command=command, explicit_environment={'SLOTSTREAM_PREFIX_CACHE': '0', 'SLOTSTREAM_PREFILL_CHUNK': str(chunk)})
            persist()
            print('START', name, rnd, chunk, flush=True)
            os.environ.update(row['explicit_environment'])
            try:
                code = benchmark.launch(path, 14, 900, command, run_set_id=RUN_ID, model_hash=sha256(model / 'config.json'))
            finally:
                for key in row['explicit_environment']:
                    os.environ.pop(key, None)
                # Even failed launch evidence is captured before validation can throw.
                if path.exists():
                    row['artifacts'] = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size,
                        'text': p.read_text(errors='replace')} for p in path.iterdir() if p.is_file()}
                persist()
            receipt = validate_receipt_file(path / 'receipt.json')
            row['receipt'] = receipt
            if (path / 'stats.json').exists():
                row['stats'] = read_json(path / 'stats.json')
            persist()
            require_terminal_sampling(receipt)
            check(code == 0 and receipt['result']['functional_success'], 'monitored generation failed')
            stats = row['stats']
            row['metrics'] = validate(stats, receipt, chunk)
            check(stats['prompt_ids'] == expected_ids[name], 'prompt IDs differ from frozen tokenizer artifact')
            if name in prompt_ids:
                check(stats['prompt_ids'] == prompt_ids[name], 'prompt token IDs changed')
            else:
                prompt_ids[name] = stats['prompt_ids']
            controls = stable_controls(stats, receipt)
            if reference_controls is None:
                reference_controls = controls
            check(controls == reference_controls, 'selected runtime controls changed')
            widen_study.validate_timing_eligibility(stats, receipt)
            row.update(status='eligible', timing_eligible=True)
            persist()
            print('DONE', name, rnd, chunk, json.dumps(row['metrics']), flush=True)
        check(cache_study.derive_source_geometry(model, verification=report['model_verification']) == geometry, 'model geometry changed')
        report['completed'] = True
        persist()
        report['summary'] = summarize(report['rows'])
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
    if sys.argv[1:] == ['--self-check']:
        orders = list(schedule())
        assert len(orders) == 18
        for workload in WORKLOADS:
            for chunk in CHUNKS:
                assert sorted(order.index(chunk) for name, rnd, actual, order in orders if name == workload and actual == chunk) == [0, 1, 2]
        fixture = read_json(ROOT / '.build/flash/runs/neon-full-model-20260916/sky-blue/0/neon/stats.json')
        fixture['plan'].update(max_context_tokens=4096, runtime_prefix_cache_enabled=False)
        fixture['sampling']['seed'] = '42'
        fixture['prompt_ids'] = [1] * 1500
        fixture['stats'].update(promptTokens=1500, prefillTokens=1500)
        receipt = read_json(ROOT / '.build/flash/runs/neon-full-model-20260916/sky-blue/0/neon/receipt.json')
        receipt['environment']['SLOTSTREAM_PREFIX_CACHE'] = '0'
        assert validate(fixture, receipt, 256)['promptTokens'] == 1500
        for key, wrong in [('effective_prefill_chunk', 512), ('effective_mtp', True)]:
            changed = copy.deepcopy(fixture)
            changed[key] = wrong
            try:
                validate(changed, receipt, 256)
            except EvidenceError:
                pass
            else:
                raise AssertionError('validator accepted ' + key)
        print('SELF-CHECK PASS: balanced schedule and effective-control rejection; no model launched')
    elif sys.argv[1:]:
        raise SystemExit('Usage: run.py [--self-check]')
    else:
        main()
