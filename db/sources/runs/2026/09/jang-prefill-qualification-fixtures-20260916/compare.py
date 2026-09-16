#!/usr/bin/env python3
"""Fail-closed, full-tensor comparison for the frozen JANG prefill capture.

512 is an empirical rechunk control, not an independently qualified configuration.
No averaging hides a failed layer, tensor, checkpoint or route phase.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import numpy as np

VOCAB, LAYERS, TOPK, EXPERTS = 248320, 48, 10, 512
STAGES = {'prefill': 0, 'step1': 1, 'step8': 8}
LAYER_TYPES = ['full_attention' if (i + 1) % 4 == 0 else 'linear_attention' for i in range(LAYERS)]
TENSOR_KEYS = {'logits', 'tokens', 'ngram', 'ple.1'}
for _i, _t in enumerate(LAYER_TYPES):
    TENSOR_KEYS.update(f'{kind}.{_i}' for kind in
                       (('key', 'value', 'index') if _t == 'full_attention' else ('conv', 'ssm')))
DTYPES = {'float32': np.dtype('<f4'), 'int64': np.dtype('<i8'), 'int32': np.dtype('<i4')}
IDENTITY_KEYS = {'model_config_sha256', 'tokenizer_sha256', 'tokenizer_config_sha256', 'checkpoint_identity',
                 'binary_sha256', 'prompt_sha256', 'metallib_sha256', 'build_identity_sha256', 'source_archive_sha256'}
OPTIMIZATION_KEYS = set('''compactStateWindows compactMTPRow skipUnusedFinalForward tailAwarePrefill
demandedPrefillOutput terminalPrefillPruning terminalLastQuery compactNgramRows incrementalIndexer
compactIndexerRaw valueOnlySamplerThreshold deviceSamplerDraw disjointSweepOutput boundedSweepRows
boundedIndexer sharedRoPE fusedRoPE fusedGDNProjection fusedGDNRecording boundedPLE ngramLookahead
layerExpertWorkspace workspaceTokenTile compactScopeFrontier workspacePiecewiseWrites readScopeTokens
automaticReadScope reuseFirstMTPEntry boundedDraftTail adaptiveSpeculation resolvedRuntimeBudget
layerLocalFloorCache boundedOutputQueue responsiveGovernor routerTopK denseIndexerBypass indexerBlockTopK
overlapSharedExpert overlapResidentExperts deduplicateImages visionAttentionPadding visionQueryTile
cachedRouterWeights directReadHandles compiledNormFinish selectedTextAttention ngramRingOrder
denseExpertLookup sparsePoolPins contiguousSlotWrites wordSlotWrites cpuSlotWrites prefixCheckpointTokens
completePromptCheckpoint'''.split())
PLAN_DIFFERENCES = {'pool_slots', 'pool_gb', 'experts_per_layer_cached', 'prefill_chunk',
                    'runtime_prefill_override', 'expected_peak_gb', 'est_prefill_s_at_max_context',
                    'est_warm_tok_s', 'est_prefill_tok_s', 'memory_ledger', 'notes', 'device_available_gb'}
LEDGER_KEYS = set('''version fixed_bytes pool_bytes active_capacity_bytes additional_active_bytes
retained_capacity_bytes retained_recurrent_bytes prefill_bytes long_context_reserve_bytes mtp_resident_bytes
vision_resident_bytes planning_margin_bytes expected_peak_bytes'''.split())

class EvidenceError(ValueError):
    pass

def require(value, message):
    if not value:
        raise EvidenceError(message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result

def json_read(path):
    require(path.is_file() and not path.is_symlink(), f'missing or symlink JSON file: {path.name}')
    require(path.stat().st_size <= 32 << 20, 'manifest exceeds 32 MiB')
    def invalid_constant(value):
        raise EvidenceError(f'nonfinite JSON constant: {value}')
    return json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=invalid_constant)

def tokens(value, name, count=None):
    require(isinstance(value, list) and (count is None or len(value) == count), f'{name}: wrong token count')
    require(all(type(x) is int and 0 <= x < VOCAB for x in value), f'{name}: invalid token ID')
    return value

def artifact(root, metadata, name, expected_dtype=None):
    require(isinstance(metadata, dict), f'{name}: artifact is not an object')
    require({'file', 'shape', 'original_dtype', 'storage_dtype', 'bytes', 'sha256', 'finite'} <= metadata.keys(),
            f'{name}: incomplete artifact metadata')
    rel = metadata['file']
    require(isinstance(rel, str) and rel and '\\' not in rel, f'{name}: invalid artifact path')
    parts = PurePosixPath(rel)
    require(not parts.is_absolute() and '..' not in parts.parts and '.' not in parts.parts and str(parts) == rel,
            f'{name}: unsafe artifact path')
    path = root / rel
    require(all(not (root.joinpath(*parts.parts[:i])).is_symlink() for i in range(1, len(parts.parts) + 1)),
            f'{name}: symlink artifact path')
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), f'{name}: missing or escaping artifact')
    shape = metadata['shape']
    require(isinstance(shape, list) and all(type(x) is int and 0 < x <= 10000000 for x in shape),
            f'{name}: invalid shape')
    count = math.prod(shape)
    dtype = metadata['storage_dtype']
    require(dtype in DTYPES and (expected_dtype is None or dtype == expected_dtype), f'{name}: invalid storage dtype')
    require(isinstance(metadata['original_dtype'], str) and metadata['original_dtype'] in
            ('float32', 'float16', 'bfloat16', 'int64', 'int32', 'uint32', 'uint64'), f'{name}: invalid original dtype')
    require((dtype == 'float32') == (metadata['original_dtype'] in ('float32', 'float16', 'bfloat16')),
            f'{name}: incompatible original/storage dtype')
    size = count * DTYPES[dtype].itemsize
    require(type(metadata['bytes']) is int and metadata['bytes'] == size and path.stat().st_size == size,
            f'{name}: byte count/shape mismatch')
    require(size <= 32 << 20, f'{name}: artifact exceeds native 32 MiB copy cap')
    require(isinstance(metadata['sha256'], str) and re.fullmatch('[0-9a-f]{64}', metadata['sha256']) is not None,
            f'{name}: invalid artifact hash')
    require(sha256(path) == metadata['sha256'], f'{name}: SHA-256 mismatch')
    require(metadata['finite'] is True, f'{name}: artifact declared nonfinite')
    value = np.memmap(path, dtype=DTYPES[dtype], mode='r', shape=tuple(shape))
    for offset in range(0, count, 1 << 18):
        require(np.isfinite(value.reshape(-1)[offset:offset + (1 << 18)]).all(), f'{name}: nonfinite array')
    return value

def optimizations_valid(opts):
    require(isinstance(opts, dict) and set(opts) == OPTIMIZATION_KEYS, 'incomplete/unknown effective optimizations')
    on = {'compactStateWindows', 'compactMTPRow', 'compactNgramRows', 'automaticReadScope',
          'skipUnusedFinalForward', 'valueOnlySamplerThreshold', 'deviceSamplerDraw',
          'boundedOutputQueue', 'responsiveGovernor', 'completePromptCheckpoint', 'sharedRoPE', 'fusedRoPE'}
    numeric = {'workspaceTokenTile': 256, 'readScopeTokens': 0, 'visionAttentionPadding': 0,
               'visionQueryTile': 256, 'prefixCheckpointTokens': 256}
    for key, value in opts.items():
        expected = True if key in on else numeric.get(key, False)
        require(type(value) is type(expected) and value == expected, f'nondeployed optimization: {key}={value!r}')
    require(on | numeric.keys() <= opts.keys(), 'missing deployed optimization fields')

def load_arm(root, chunk, errors):
    arrays = {}
    report = None
    label = f'chunk{chunk}'
    try:
        report = json_read(root / 'report.json')
        completion = json_read(root / 'completion.json')
        require(isinstance(report, dict) and report.get('format') == 'slotstream-prefill-capture-v1' and
                report.get('schema_version') == 1, 'wrong capture format/version')
        require(isinstance(completion, dict) and completion.get('format') == 'slotstream-prefill-capture-completion-v1'
                and completion.get('schema_version') == 1 and completion.get('passed') is True,
                'missing successful completion marker')
        require(not (root / 'failure.json').exists(), 'failure marker exists')
        require(completion.get('report_sha256') == sha256(root / 'report.json'), 'completion report hash mismatch')
        required = {'identity', 'prompt_ids', 'continuation_ids', 'snapshots', 'routes', 'route_events',
                    'top_k', 'num_layers', 'vocab_size', 'layer_types', 'expected_tensor_keys', 'plan', 'optimizations',
                    'effective_prefill_chunk', 'effective_pool_slots', 'effective_expert_widening',
                    'effective_mtp', 'effective_vision', 'effective_prefix_cache',
                    'numerical_environment', 'runtime_environment', 'actual_engine', 'capture_path', 'raw',
                    'sampling', 'continuation_mode', 'normal_output_ids', 'byte_order'}
        require(required <= report.keys(), f'missing manifest fields: {sorted(required - report.keys())}')
        require(report['actual_engine'] is True, 'not actualEngine capture')
        require(report['capture_path'] == 'actualEngine', 'wrong capture path')
        require(report['byte_order'] == 'little', 'unsupported artifact byte order')
        require(type(report['raw']) is bool, 'missing prompt templating mode')
        require(report['sampling'] == {'greedy': True, 'seed': 42, 'max_tokens': 1}, 'wrong ordinary generation controls')
        require(report['continuation_mode'] == ('baseline-raw-argmax' if chunk == 256 else 'imported-baseline-ids'),
                'wrong continuation source')
        identity = report['identity']
        require(isinstance(identity, dict) and IDENTITY_KEYS <= identity.keys(), 'incomplete model/tokenizer/checkpoint identity')
        for name in IDENTITY_KEYS:
            value = identity[name]
            require(isinstance(value, str) and value, f'missing identity {name}')
            if name.endswith('_sha256'):
                require(re.fullmatch('[0-9a-f]{64}', value) is not None, f'invalid identity hash {name}')
        prompt = tokens(report['prompt_ids'], 'prompt_ids')
        require(1024 < len(prompt) <= 4088, 'prompt does not exercise all chunks or exceeds context')
        continuation = tokens(report['continuation_ids'], 'continuation_ids', 8)
        require(json_read(root / 'continuation.json') == continuation, 'continuation import artifact differs from report')
        tokens(report['normal_output_ids'], 'normal_output_ids', 1)
        require(report['top_k'] == TOPK and report['num_layers'] == LAYERS and report['vocab_size'] == VOCAB,
                'wrong model geometry')
        require(report['layer_types'] == LAYER_TYPES, 'wrong model layer types')
        require(isinstance(report['expected_tensor_keys'], list) and len(report['expected_tensor_keys']) == len(TENSOR_KEYS)
                and set(report['expected_tensor_keys']) == TENSOR_KEYS, 'incomplete architecture tensor catalogue')
        require(report['effective_prefill_chunk'] == chunk, 'wrong effective prefill chunk')
        require(type(report['effective_pool_slots']) is int and report['effective_pool_slots'] > 0, 'invalid pool count')
        require(report['effective_expert_widening'] == 'packed4-to6', 'not deployed packed widening')
        require(all(report[key] is False for key in ('effective_mtp', 'effective_vision', 'effective_prefix_cache')),
                'MTP/vision/prefix-cache must be disabled')
        plan = report['plan']
        require(isinstance(plan, dict) and plan.get('target_gb') == 14 and plan.get('max_context_tokens') == 4096,
                'not fixed 14 GB / 4096 context')
        require(plan.get('prefill_chunk') == chunk and plan.get('pool_slots') == report['effective_pool_slots'],
                'plan/effective runtime mismatch')
        require(plan.get('mtp') is False and plan.get('vision') is False, 'plan enables MTP/vision')
        ledger = plan.get('memory_ledger')
        require(isinstance(ledger, dict) and LEDGER_KEYS <= ledger.keys(), 'incomplete plan memory ledger')
        require(all(type(x) is int and x >= 0 for x in ledger.values()), 'invalid plan memory ledger values')
        require(ledger['version'] == 1 and ledger['expected_peak_bytes'] <= 14_000_000_000, 'plan ledger exceeds target/version')
        optimizations_valid(report['optimizations'])
        require(isinstance(report['numerical_environment'], dict) and report['numerical_environment'], 'missing numerical environment')
        require(isinstance(report['runtime_environment'], dict), 'missing runtime environment')
        for key, value in report['runtime_environment'].items():
            if key.startswith('SLOTSTREAM_'):
                require((key == 'SLOTSTREAM_PREFILL_CHUNK' and value == str(chunk)) or
                        (key == 'SLOTSTREAM_PREFIX_CACHE' and value == '0'), 'unsupported/mismatched runtime environment control')
        tf32_raw = report['runtime_environment'].get('MLX_ENABLE_TF32')
        require(tf32_raw in (None, '0', '1'), 'invalid TF32 runtime control')
        require(report['numerical_environment'] == {'effective_tf32': tf32_raw != '0'}, 'TF32 numerical settings mismatch')
        require(isinstance(report['snapshots'], dict) and set(report['snapshots']) == set(STAGES), 'incomplete checkpoint capture')
        require(isinstance(report['routes'], dict) and set(report['routes']) == {'prefill', 'continuation'}, 'incomplete route phases')
    except Exception as error:
        errors.append({'scope': label, 'error': f'{type(error).__name__}: {error}'})
        return report, arrays

    used_paths = set()
    for stage, extra in STAGES.items():
        try:
            snap = report['snapshots'][stage]
            require(isinstance(snap, dict) and {'consumed_ids', 'greedy_token', 'tensors'} <= snap.keys(), 'incomplete snapshot')
            require(tokens(snap['consumed_ids'], 'consumed_ids') == prompt + continuation[:extra], 'consumed IDs do not match common continuation')
            require(type(snap['greedy_token']) is int and 0 <= snap['greedy_token'] < VOCAB, 'invalid greedy token')
            require(isinstance(snap['tensors'], dict) and set(snap['tensors']) == TENSOR_KEYS, 'missing/extra snapshot tensor')
        except Exception as error:
            errors.append({'scope': f'{label}/{stage}', 'error': f'{type(error).__name__}: {error}'})
            continue
        for name, metadata in snap['tensors'].items():
            key = f'{stage}/{name}'
            try:
                value = artifact(root, metadata, key, 'int64' if name in ('tokens', 'ngram') else 'float32')
                require(metadata['file'] not in used_paths, 'artifact file reused by distinct capture fields')
                used_paths.add(metadata['file'])
                arrays[key] = value
                if name == 'logits':
                    require(value.size == VOCAB and value.shape[-1] == VOCAB, 'not full-vocabulary last logits')
                    require(int(np.argmax(value)) == snap['greedy_token'], 'greedy token differs from full logits argmax')
                    if stage == 'prefill':
                        require(report['normal_output_ids'] == [snap['greedy_token']], 'ordinary generation output differs from raw greedy')
                    if stage == 'prefill' and chunk == 256:
                        require(snap['greedy_token'] == continuation[0], 'reference prefill greedy does not begin continuation')
                    if stage == 'step1' and chunk == 256:
                        require(snap['greedy_token'] == continuation[1], 'step1 greedy does not match continuation')
                if name == 'tokens':
                    require(value.size == 1 and int(value.reshape(-1)[0]) == len(prompt) + extra, 'token state count mismatch')
                if name == 'ngram':
                    require(value.reshape(-1).tolist() == (prompt + continuation[:extra])[-2:], 'ngram state does not equal consumed token suffix')
            except Exception as error:
                errors.append({'scope': f'{label}/{key}', 'error': f'{type(error).__name__}: {error}'})
    for phase, rows in [('prefill', len(prompt)), ('continuation', 8)]:
        fields = report['routes'][phase]
        if not isinstance(fields, dict) or set(fields) != {str(i) for i in range(LAYERS)}:
            errors.append({'scope': f'{label}/routes/{phase}', 'error': 'must capture all 48 layers'})
            continue
        for layer, metadata in fields.items():
            key = f'routes/{phase}/{layer}'
            try:
                value = artifact(root, metadata, key, 'int32')
                require(value.shape == (rows, TOPK), 'route row count/top-K mismatch')
                require(metadata.get('logical_positions') == list(range(rows)), 'missing/inconsistent logical route positions')
                require(np.all((value >= 0) & (value < EXPERTS)), 'expert ID outside 512 experts')
                require(np.all(np.diff(np.sort(value, axis=1), axis=1) != 0), 'duplicate expert in routing keep-set')
                require(metadata['file'] not in used_paths, 'artifact file reused by distinct capture fields')
                used_paths.add(metadata['file'])
                arrays[key] = value
            except Exception as error:
                errors.append({'scope': f'{label}/{key}', 'error': f'{type(error).__name__}: {error}'})
    try:
        events = report['route_events']
        require(isinstance(events, list) and events, 'route callback event sequence missing')
        totals = {(phase, layer): 0 for phase in ('prefill', 'continuation') for layer in range(LAYERS)}
        require(len(events) % LAYERS == 0, 'incomplete routing callback layer sweep')
        seen_continuation = False
        for start in range(0, len(events), LAYERS):
            group = events[start:start + LAYERS]
            first = group[0]
            phase, rows = first.get('phase'), first.get('rows')
            require(phase in ('prefill', 'continuation') and type(rows) is int and rows > 0, 'invalid route event')
            require(not (seen_continuation and phase == 'prefill'), 'prefill event after continuation')
            seen_continuation |= phase == 'continuation'
            require(rows == 1 if phase == 'continuation' else rows <= chunk, 'route event row count exceeds chunk')
            for layer, event in enumerate(group):
                require(event == {'phase': phase, 'layer': layer, 'rows': rows}, 'nonchronological/inconsistent route callback event')
                totals[(phase, layer)] += rows
        require(all(n == (len(prompt) if phase == 'prefill' else 8) for (phase, _), n in totals.items()),
                'route events do not account for every consumed token')
    except Exception as error:
        errors.append({'scope': f'{label}/route_events', 'error': f'{type(error).__name__}: {error}'})
    return report, arrays

def float_metric(reference, other, logits=False):
    r = reference.reshape(-1)
    o = other.reshape(-1)
    maximum_delta, maximum_ref, low, high = 0.0, 0.0, math.inf, -math.inf
    for offset in range(0, r.size, 1 << 18):
        x = np.asarray(r[offset:offset + (1 << 18)], dtype=np.float64)
        y = np.asarray(o[offset:offset + (1 << 18)], dtype=np.float64)
        maximum_delta = max(maximum_delta, float(np.max(np.abs(x - y))))
        maximum_ref = max(maximum_ref, float(np.max(np.abs(x))))
        low, high = min(low, float(np.min(x))), max(high, float(np.max(x)))
    denominator = max(high - low if logits else maximum_ref, 1e-6)
    return {'max_abs_delta': maximum_delta, 'denominator': denominator, 'relative_delta': maximum_delta / denominator}

def route_metric(reference, other):
    # Row order is token order, expert order within a row is semantically irrelevant.
    a, b = np.sort(reference, axis=1), np.sort(other, axis=1)
    matched = (a[:, :, None] == b[:, None, :]).any(axis=2).sum(axis=1)
    missing = TOPK - matched
    return {'keep_set_disagreement': float(missing.sum()) / reference.size,
            'different_rows': int(np.count_nonzero(missing)), 'total_rows': int(len(a)),
            'missing_experts': int(missing.sum()), 'maximum_missing_experts_in_row': int(missing.max())}

def compare(directories):
    errors, measurements = [], []
    arms = [load_arm(Path(path), chunk, errors) for path, chunk in zip(directories, (256, 512, 1024))]
    ref, control, candidate = [arm[0] for arm in arms]
    if all(isinstance(x, dict) for x in (ref, control, candidate)):
        for field in ('identity', 'prompt_ids', 'continuation_ids', 'optimizations', 'numerical_environment',
                      'expected_tensor_keys', 'layer_types', 'raw', 'sampling', 'capture_path'):
            if not (ref.get(field) == control.get(field) == candidate.get(field)):
                errors.append({'scope': 'matching/' + field, 'error': 'reference/control/candidate differ'})
        runtime = [{k: v for k, v in arm.get('runtime_environment', {}).items() if k != 'SLOTSTREAM_PREFILL_CHUNK'}
                   for arm in (ref, control, candidate)]
        if not runtime[0] == runtime[1] == runtime[2]:
            errors.append({'scope': 'matching/runtime_environment', 'error': 'runtime controls differ beyond explicit chunk'})
        try:
            stable_plans = [{k: v for k, v in arm['plan'].items() if k not in PLAN_DIFFERENCES}
                            for arm in (ref, control, candidate)]
            require(stable_plans[0] == stable_plans[1] == stable_plans[2], 'plan differences beyond explicit chunk/pool/derived diagnostics')
            stable_ledgers = [{k: v for k, v in arm['plan']['memory_ledger'].items()
                               if k not in ('pool_bytes', 'prefill_bytes', 'expected_peak_bytes')}
                              for arm in (ref, control, candidate)]
            require(stable_ledgers[0] == stable_ledgers[1] == stable_ledgers[2],
                    'memory ledger differs beyond prefill/pool/derived peak')
        except Exception as error:
            errors.append({'scope': 'matching/plan', 'error': str(error)})
    expected_keys = {f'{stage}/{name}' for stage in STAGES for name in TENSOR_KEYS}
    expected_keys |= {f'routes/{phase}/{layer}' for phase in ('prefill', 'continuation') for layer in range(LAYERS)}
    for key in sorted(expected_keys):
        row = {'field': key, 'passed': False}
        try:
            require(all(key in arrays for _, arrays in arms), 'artifact missing or failed validation in one or more arms')
            values = [arrays[key] for _, arrays in arms]
            require(all(x.shape == values[0].shape and x.dtype == values[0].dtype for x in values), 'shape/dtype mismatch across arms')
            if not key.startswith('routes/'):
                stage, name = key.split('/')
                originals = [arm['snapshots'][stage]['tensors'][name]['original_dtype'] for arm, _ in arms]
                require(len(set(originals)) == 1, 'original tensor dtype differs across arms')
            if key.startswith('routes/'):
                row['metric'] = 'fraction_of_reference_keep_set_replaced'
                row['control'] = route_metric(values[0], values[1])
                row['candidate'] = route_metric(values[0], values[2])
                c, t = row['control']['keep_set_disagreement'], row['candidate']['keep_set_disagreement']
                row['threshold'] = max(3 * c, 0.01)
                row['passed'] = t <= row['threshold']
            elif key.endswith('/tokens') or key.endswith('/ngram'):
                row['metric'] = 'exact_integer_state'
                row['control_exact'] = bool(np.array_equal(values[0], values[1]))
                row['candidate_exact'] = bool(np.array_equal(values[0], values[2]))
                row['passed'] = row['control_exact'] and row['candidate_exact']
            else:
                logits = key.endswith('/logits')
                row['metric'] = 'max_abs_delta_over_reference_spread' if logits else 'max_abs_delta_over_reference_max_abs'
                row['control'] = float_metric(values[0], values[1], logits)
                row['candidate'] = float_metric(values[0], values[2], logits)
                row['threshold'] = max(3 * row['control']['relative_delta'], 0.01)
                row['passed'] = row['candidate']['relative_delta'] <= row['threshold']
                if logits:
                    row['greedy_tokens'] = [int(np.argmax(x)) for x in values]
                    row['greedy_top1_exact'] = row['greedy_tokens'][0] == row['greedy_tokens'][2]
                    row['empirical_control_greedy_top1_exact'] = row['greedy_tokens'][0] == row['greedy_tokens'][1]
                    row['passed'] = row['passed'] and row['greedy_top1_exact']
            if not row['passed']:
                errors.append({'scope': key, 'error': 'numerical/route band or exactness gate failed'})
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            errors.append({'scope': key, 'error': row['error']})
        measurements.append(row)
    return {'format': 'slotstream-prefill-comparison-v1', 'schema_version': 1,
            'passed': not errors, 'reference_chunk': 256, 'empirical_control_chunk': 512, 'candidate_chunk': 1024,
            'control_independently_qualified': False,
            'scope': 'Numerical evidence only; does not establish task quality, latency or default adoption.',
            'comparison_policy': {'relative_band': 'candidate <= max(3 * control, 0.01)', 'denominator_floor': 1e-6,
                                  'greedy_checkpoints': list(STAGES), 'continuation_tokens': 8,
                                  'full_vocabulary': VOCAB, 'route_layers': LAYERS, 'top_k': TOPK},
            'directories': [str(Path(x).resolve()) for x in directories],
            'errors': errors, 'measurements': measurements}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('control', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    directories = [args.reference, args.control, args.candidate]
    try:
        report = compare(directories)
    except Exception as error:
        report = {'format': 'slotstream-prefill-comparison-v1', 'schema_version': 1, 'passed': False,
                  'errors': [{'scope': 'comparison', 'error': f'{type(error).__name__}: {error}'}], 'measurements': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, args.output)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(report['errors'])} failures; full assessment: {args.output}")
    for error in report['errors'][:20]:
        print(f"  {error['scope']}: {error['error']}")
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
