#!/usr/bin/env python3
"""Validate, dry-run and orchestrate bounded Flash diagnostic capture."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, re, stat, struct, sys
from pathlib import Path
from typing import Any
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import benchmark
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import require_terminal_sampling, validate_receipt_file
VOCAB = 248320
LAYERS = 48
TOPK = 10
H = 2560
FF = 640
SUPPORTED_SPLIT = 'development'
POSITION_LIMIT = 64
CONTEXT_LIMIT = 2048

def canonical_manifest_hash(d):
    value = copy.deepcopy(d)
    value.pop('manifestSHA256', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def foundation_manifest_hash(d):
    return canonical_manifest_hash(d)

def read_bounded_manifest(path: Path) -> dict[str, Any]:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_size > 1 << 20:
            raise EvidenceError('capture manifest must be a regular file no larger than 1 MiB')
        with path.open('rb') as stream:
            raw = stream.read((1 << 20) + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > 1 << 20 or len(raw) != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise EvidenceError('capture manifest changed while being read')
        value = json.loads(raw)
    except EvidenceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f'capture manifest cannot be read: {error}') from error
    if not isinstance(value, dict):
        raise EvidenceError('capture manifest must be an object')
    return value

def validate_manifest(path: Path) -> dict[str, Any]:
    d = read_bounded_manifest(path)
    v1 = d.get('format') == 'slotstream-flash-capture-v1'
    if v1:
        if (set(d) != {'format', 'schemaVersion', 'manifestSHA256', 'developmentOnly', 'tokenSource', 'documents'}
                or d.get('schemaVersion') != 1 or d.get('developmentOnly') is not True
                or d.get('tokenSource') != 'arbitrary-valid-diagnostic-ids-not-training-text'
                or d.get('manifestSHA256') != canonical_manifest_hash(d)):
            raise EvidenceError('unknown capture manifest fields, version or hash')
        selected_split = 'development'
    else:
        required = {'format', 'schemaVersion', 'manifestSHA256', 'tokenSource', 'split',
                    'sourceCorpusSHA256', 'tokenizerIdentity', 'shardID', 'documents'}
        if (set(d) != required or d.get('format') != 'slotstream-tokenized-capture-shard-v2'
                or d.get('schemaVersion') != 2 or d.get('manifestSHA256') != foundation_manifest_hash(d)
                or d.get('tokenSource') != 'slotstream-auto-tokenizer-chat-template-v1'
                or d.get('split') not in ('training', 'development', 'qualification')
                or re.fullmatch(r'shard-[0-9]{3,4}', d.get('shardID', '')) is None
                or re.fullmatch(r'[0-9a-f]{64}', d.get('sourceCorpusSHA256', '')) is None
                or not isinstance(d.get('tokenizerIdentity'), dict)):
            raise EvidenceError('unknown capture manifest fields, version or hash')
        selected_split = d['split']
    ids = set()
    total_positions = 0
    for doc in d['documents']:
        expected_keys = ({'id', 'split', 'warmupIDs', 'positions'} if v1 else
                         {'id', 'category', 'sourceID', 'sourceHash', 'split', 'warmupIDs', 'positions'})
        if set(doc) != expected_keys or not isinstance(doc['id'], str) or (not doc['id']) or (len(doc['id']) > 64) or (doc['id'] in ids) or (doc['split'] != selected_split) or (not isinstance(doc['warmupIDs'], list)) or (len(doc['warmupIDs']) > 256) or (not isinstance(doc['positions'], list)) or (re.fullmatch('[A-Za-z0-9_-]+', doc['id']) is None) or any((type(token) is not int or not 0 <= token < VOCAB for token in doc['warmupIDs'])) or (len(doc['warmupIDs']) + len(doc['positions']) > CONTEXT_LIMIT):
            raise EvidenceError('invalid document identity or warmup tokens')
        if not v1 and (re.fullmatch(r'[A-Za-z0-9_-]{1,64}', doc['category']) is None
                or not doc['sourceID'] or len(doc['sourceID'].encode()) > 256
                or any(ord(character) < 32 or ord(character) == 127 for character in doc['sourceID'])
                or re.fullmatch(r'[0-9a-f]{64}', doc['sourceHash']) is None or not doc['positions']):
            raise EvidenceError('invalid tokenized document source metadata')
        ids.add(doc['id'])
        sequence = list(doc['warmupIDs'])
        expected = len(sequence)
        for p in doc['positions']:
            if set(p) != {'position', 'inputID', 'nextTokenID'} or type(p['position']) is not int or p['position'] != expected:
                raise EvidenceError('missing, duplicate or out-of-order capture position')
            if not all((type(p[k]) is int and 0 <= p[k] < VOCAB for k in ('inputID', 'nextTokenID'))):
                raise EvidenceError('capture token ID is out of range')
            if sequence and expected > len(doc['warmupIDs']) and (sequence[-1] != p['inputID']):
                raise EvidenceError('teacher-forced input does not equal prior i+1 target')
            sequence.extend([p['inputID'], p['nextTokenID']])
            expected += 1
            total_positions += 1
    if not 0 < total_positions <= POSITION_LIMIT:
        raise EvidenceError('capture position limit exceeded')
    return d

def documents_for_split(request: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split == 'qualification':
        raise EvidenceError('qualification capture is locked until a later frozen run-set')
    if split not in ('training', 'development'):
        raise EvidenceError('capture split is unsupported')
    request_format = request.get('format', 'slotstream-flash-capture-v1')
    if request_format == 'slotstream-flash-capture-v1' and split != SUPPORTED_SPLIT:
        raise EvidenceError('v1 capture remains development-only')
    if request_format == 'slotstream-tokenized-capture-shard-v2' and request['split'] != split:
        raise EvidenceError('tokenized capture split differs from shard')
    documents = [doc for doc in request['documents'] if doc['split'] == split]
    if not documents or not any((doc['positions'] for doc in documents)):
        raise EvidenceError('requested split is empty')
    return documents

def dry_run(manifest: Path, split: str, output: Path) -> int:
    output = fresh_output(output)
    try:
        d = validate_manifest(manifest)
        docs = documents_for_split(d, split)
        positions = sum((len(x['positions']) for x in docs))
        logits = positions * VOCAB * 4
        activations = positions * LAYERS * (H * 2 + TOPK * FF * 4)
        metadata = positions * (LAYERS * 3 * 1024 + 256 * 1024)
        result = {'format': 'slotstream-flash-capture-dry-run-v1', 'schema_version': 1, 'qualification': False, 'manifest_sha256': sha256(manifest), 'split': split, 'documents': len(docs), 'positions': positions, 'worst_case_logits_bytes': logits, 'worst_case_activation_bytes': activations, 'worst_case_metadata_bytes': metadata, 'live_buffer_limit_bytes': 64 << 20, 'logits_quota_bytes': 8000000000, 'activation_quota_bytes': 32000000000, 'model_loaded': False}
        if logits > result['logits_quota_bytes'] or activations > result['activation_quota_bytes']:
            raise EvidenceError('capture exceeds disk quota')
        atomic_json(output / 'report.json', result)
        atomic_json(output / 'completion.json', {'format': 'slotstream-flash-dry-run-completion-v1', 'report_sha256': sha256(output / 'report.json'), 'qualification': False})
        return 0
    except Exception as e:
        atomic_json(output / 'failure.json', {'format': 'slotstream-flash-capture-failure-v1', 'error': f'{type(e).__name__}: {e}'})
        return 1

def _finite_artifact(path: Path, dtype: str) -> bool:
    raw = path.read_bytes()
    if dtype == 'float32':
        return len(raw) % 4 == 0 and all((math.isfinite(value[0]) for value in struct.iter_unpack('<f', raw)))
    if dtype == 'bfloat16':
        return len(raw) % 2 == 0 and all((value[0] & 32640 != 32640 for value in struct.iter_unpack('<H', raw)))
    return False

def _expected_state_fields(model: Path) -> list[str]:
    text = read_json(model / 'config.json').get('text_config', {})
    layer_types = text.get('layer_types')
    if not isinstance(layer_types, list) or len(layer_types) != LAYERS:
        raise EvidenceError('pinned model layer inventory is missing')
    names = {'ngram', 'tokens', 'lastMulti', 'mtp.key', 'mtp.value', 'mtp.index', 'mtp.offset'}
    for (layer, kind) in enumerate(layer_types):
        if kind == 'linear_attention':
            names.update((f'conv.{layer}', f'ssm.{layer}', f'ple.{layer}'))
        elif kind == 'full_attention':
            names.update((f'key.{layer}', f'value.{layer}', f'index.{layer}'))
        else:
            raise EvidenceError('unsupported pinned model layer type')
    return sorted(names)

def validate_native_report(report, root, manifest, mode, binary, model, request, *, require_split=False):
    report_keys = {'format', 'schema_version', 'qualification', 'mode', 'manifest_sha256', 'binary', 'model', 'source_identity', 'plan', 'memory_ledger', 'documents', 'files', 'activation_bytes', 'logits_bytes', 'invalid_state_reuse_refused', 'optimizations', 'numerical_environment', 'resident_split_evidence'}
    v2 = request.get('format', 'slotstream-flash-capture-v1') == 'slotstream-tokenized-capture-shard-v2'
    if v2:
        report_keys.add('tokenized_corpus')
    if set(report) != report_keys or report.get('format') != 'slotstream-flash-capture-output-v1' or report.get('schema_version') != 1 or (report.get('mode') != mode) or (report.get('qualification') is not False):
        raise EvidenceError('invalid native capture report')
    if report.get('manifest_sha256') != sha256(manifest) or report.get('invalid_state_reuse_refused') is not True:
        raise EvidenceError('native manifest or failed-state evidence mismatch')
    source = report.get('source_identity', {})
    directory = binary.parent
    expected = {'binary_sha256': sha256(binary), 'metallib_sha256': sha256(directory / 'mlx.metallib'), 'build_identity_sha256': sha256(directory / 'build-identity.json'), 'source_archive_sha256': sha256(directory / 'build-source.tar.gz'), 'model_config_sha256': sha256(model / 'config.json'), 'model_index_sha256': sha256(model / 'model.safetensors.index.json')}
    if source != expected or Path(report.get('model', '')).resolve() != model.resolve():
        raise EvidenceError('native source or model identity mismatch')
    if v2:
        identity = request['tokenizerIdentity']
        files = identity.get('files') if isinstance(identity, dict) else None
        if (identity.get('model_revision') != '3781190c6bbdf0a7637beda49ba179822612058a'
                or identity.get('swift_transformers_revision') != '2fa33e1f5e7131a7fc64c28e6d161dcec0d24820'
                or identity.get('text_add_special_tokens') is not False
                or identity.get('thinking') is not False or not isinstance(files, list)):
            raise EvidenceError('tokenized shard tokenizer identity is unsupported')
        for item in files:
            if (set(item) != {'path', 'bytes', 'sha256'} or item['path'] not in
                    ('tokenizer.json', 'tokenizer_config.json', 'config.json')):
                raise EvidenceError('tokenized shard tokenizer file identity is malformed')
            path = model / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                raise EvidenceError('tokenized shard tokenizer file changed')
    if report.get('memory_ledger', {}).get('diagnostic_reserved_bytes') != 128 << 20 or report.get('plan', {}).get('target_gb') != 14:
        raise EvidenceError('native diagnostic reservation or plan mismatch')
    files = report.get('files', [])
    if len({f.get('name') for f in files}) != len(files):
        raise EvidenceError('duplicate native capture files')
    category_bytes = {'activation': 0, 'logits': 0}
    for f in files:
        if f.get('category') not in ('logits', 'input', 'hidden'):
            raise EvidenceError('unknown native capture category')
        required = {'name', 'category', 'document_id', 'token_position', 'input_id', 'dtype', 'shape', 'bytes', 'sha256'}
        if f.get('category') in ('input', 'hidden'):
            required.add('layer')
        if f.get('category') == 'hidden':
            required |= {'router_ranks', 'expert_ids'}
        if set(f) != required or Path(f['name']).name != f['name'] or (not (root / f['name']).is_file()) or (sha256(root / f['name']) != f['sha256']) or ((root / f['name']).stat().st_size != f['bytes']):
            raise EvidenceError('native file metadata mismatch')
        if f['category'] == 'logits' and (f['dtype'] != 'float32' or f['shape'] != [1, 1, VOCAB] or f['bytes'] != VOCAB * 4):
            raise EvidenceError('full-vocabulary logits geometry mismatch')
        if f['category'] == 'input' and (f['dtype'] != 'bfloat16' or f['shape'] != [1, 1, H] or f['bytes'] != H * 2):
            raise EvidenceError('input activation geometry mismatch')
        if f['category'] == 'hidden':
            count = len(f['router_ranks'])
            if f['dtype'] != 'float32' or f['shape'] not in ([1, 1, count, FF], [1, 1, count, 1, FF]) or f['bytes'] != count * FF * 4:
                raise EvidenceError('hidden activation geometry mismatch')
        category_bytes['logits' if f['category'] == 'logits' else 'activation'] += f['bytes']
        if not _finite_artifact(root / f['name'], f['dtype']):
            raise EvidenceError('native capture contains non-finite values')
    if report.get('logits_bytes') != category_bytes['logits'] or report.get('activation_bytes') != category_bytes['activation']:
        raise EvidenceError('native capture byte totals mismatch')
    selected_split = request.get('split', SUPPORTED_SPLIT)
    expected_docs = {d['id']: d for d in documents_for_split(request, selected_split)}
    if v2:
        expected_tokenized = {'format': request['format'], 'split': selected_split,
                              'shard_id': request['shardID'],
                              'source_corpus_sha256': request['sourceCorpusSHA256'],
                              'tokenizer_identity': request['tokenizerIdentity'],
                              'documents': [{'id': item['id'], 'category': item['category'],
                                             'source_id': item['sourceID'],
                                             'source_hash': item['sourceHash']}
                                            for item in request['documents']]}
        if report.get('tokenized_corpus') != expected_tokenized:
            raise EvidenceError('native tokenized corpus provenance mismatch')
    config = read_json(model / 'config.json')
    expected_optimizations = report.get('optimizations')
    if not isinstance(expected_optimizations, dict) or expected_optimizations.get('overlapResidentExperts') is not require_split:
        raise EvidenceError('native effective optimization evidence mismatch')
    numerical = report.get('numerical_environment')
    raw_tf32 = os.environ.get('MLX_ENABLE_TF32')
    if not isinstance(numerical, dict) or set(numerical) != {'mlx_enable_tf32_raw', 'effective_tf32'} or numerical['effective_tf32'] not in (True, False):
        raise EvidenceError('native numerical environment is malformed')
    if numerical != {'mlx_enable_tf32_raw': raw_tf32, 'effective_tf32': raw_tf32 != '0'}:
        raise EvidenceError('native numerical environment differs from the launcher')
    if len(report.get('documents', [])) != len(expected_docs) or {d.get('id') for d in report.get('documents', [])} != set(expected_docs):
        raise EvidenceError('native document coverage mismatch')
    for doc in report.get('documents', []):
        if set(doc) != {'id', 'positions'}:
            raise EvidenceError('native document metadata mismatch')
        if [p['position'] for p in doc['positions']] != [p['position'] for p in expected_docs[doc['id']]['positions']]:
            raise EvidenceError('native position coverage mismatch')
        for p in doc['positions']:
            if set(p) != {'position', 'input_id', 'next_token_id', 'routes', 'state', 'state_sha256', 'continuation_id'}:
                raise EvidenceError('native position metadata mismatch')
            source_position = next((x for x in expected_docs[doc['id']]['positions'] if x['position'] == p['position']))
            if p['input_id'] != source_position['inputID'] or p['next_token_id'] != source_position['nextTokenID'] or type(p['continuation_id']) is not int or (not 0 <= p['continuation_id'] < VOCAB):
                raise EvidenceError('native token identity mismatch')
            routes = p['routes']
            if [r.get('layer') for r in routes] != list(range(LAYERS)) or any((set(r) != {'layer', 'ids'} or len(r['ids']) != TOPK or any((type(x) is not int or not 0 <= x < 512 for x in r['ids'])) for r in routes)):
                raise EvidenceError('ordered route coverage failed')
            selected = [f for f in files if f['document_id'] == doc['id'] and f['token_position'] == p['position']]
            if any((f['input_id'] != p['input_id'] for f in selected)):
                raise EvidenceError('activation input identity mismatch')
            if len([f for f in selected if f['category'] == 'logits']) != 1:
                raise EvidenceError('logits coverage failed')
            if mode == 'reference-on':
                inputs = [f for f in selected if f['category'] == 'input']
                hidden = [f for f in selected if f['category'] == 'hidden']
                if sorted((f['layer'] for f in inputs)) != list(range(LAYERS)):
                    raise EvidenceError('input activation coverage failed')
                for layer in range(LAYERS):
                    parts = [f for f in hidden if f['layer'] == layer]
                    ranks = [r for f in parts for r in f['router_ranks']]
                    if sorted(ranks) != list(range(TOPK)):
                        raise EvidenceError('hidden rank coverage failed')
                    for f in parts:
                        if f['expert_ids'] != [routes[layer]['ids'][r] for r in f['router_ranks']]:
                            raise EvidenceError('hidden expert/rank mismatch')
            elif any((f['category'] in ('input', 'hidden') for f in selected)):
                raise EvidenceError('observer-off activation leak')
            state = p.get('state', {})
            fields = state.get('fields', [])
            if set(state) not in ({'fields', 'indexerBases', 'allocatedSequenceBytes'}, {'fields', 'indexerBases', 'draftIndexerBase', 'allocatedSequenceBytes'}) or not fields or len(p.get('state_sha256', '')) != 64:
                raise EvidenceError('state field identities missing')
            names = []
            for field in fields:
                names.append(field.get('name'))
                expected_keys = {'name', 'present'} if field.get('present') is False else {'name', 'present', 'dtype', 'shape', 'bytes', 'sha256'}
                if set(field) != expected_keys or type(field.get('present')) is not bool or (field.get('present') is True and (not field['dtype'] or not isinstance(field['shape'], list) or type(field['bytes']) is not int or (field['bytes'] < 0) or (len(field['sha256']) != 64))):
                    raise EvidenceError('state field identity is malformed')
            if names != _expected_state_fields(model):
                raise EvidenceError('state field inventory is incomplete')
            state_hash = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            if state_hash != p['state_sha256']:
                raise EvidenceError('state identity hash mismatch')
    expected_positions = {(doc_id, p['position']) for (doc_id, doc) in expected_docs.items() for p in doc['positions']}
    if any(((f['document_id'], f['token_position']) not in expected_positions for f in files)):
        raise EvidenceError('extra native capture artifact')
    split_layers = sorted({f"{f['document_id']}:{f['token_position']}:{f['layer']}" for f in files if f['category'] == 'hidden' and len([g for g in files if g['category'] == 'hidden' and g['document_id'] == f['document_id'] and (g['token_position'] == f['token_position']) and (g['layer'] == f['layer'])]) > 1})
    split_evidence = report.get('resident_split_evidence')
    if not isinstance(split_evidence, dict) or split_evidence.get('required') is not require_split or split_evidence.get('split_layers') != split_layers or (require_split and (not split_layers)):
        raise EvidenceError('resident/miss production split evidence mismatch')

def validate_native_output(root, manifest, mode, binary, model, request, *, require_split=False):
    report_path = root / 'report.json'
    completion_path = root / 'completion.json'
    report = read_json(report_path)
    completion = read_json(completion_path)
    if set(completion) != {'format', 'report_sha256', 'qualification'} or completion.get('format') != 'slotstream-flash-capture-completion-v1' or completion.get('report_sha256') != sha256(report_path) or (completion.get('qualification') is not False):
        raise EvidenceError('native completion does not bind the report')
    validate_native_report(report, root, manifest, mode, binary, model, request, require_split=require_split)
    return report

def _native(binary, model, manifest, split, mode, out, *, require_split=False):
    cmd = [str(binary), 'flash-capture', '--model', str(model), '--manifest', str(manifest), '--split', split, '--mode', mode, '--memory-gb', '14', '--max-context', '2048', '--output', str(out / 'native'), '--live-limit-mb', '64', '--activation-quota-gb', '32', '--logits-quota-gb', '8']
    if require_split:
        cmd.append('--require-resident-split')
    settle = benchmark.settle_before_model_launch()
    prior = os.environ.get('SLOTSTREAM_OPT_RESIDENT_OVERLAP')
    if require_split:
        os.environ['SLOTSTREAM_OPT_RESIDENT_OVERLAP'] = '1'
    try:
        code = benchmark.launch(out, 14, 900, cmd)
    finally:
        if prior is None:
            os.environ.pop('SLOTSTREAM_OPT_RESIDENT_OVERLAP', None)
        else:
            os.environ['SLOTSTREAM_OPT_RESIDENT_OVERLAP'] = prior
    atomic_json(out / 'settling.json', settle)
    receipt = validate_receipt_file(out / 'receipt.json')
    require_terminal_sampling(receipt)
    if code or not receipt['result']['functional_success']:
        raise EvidenceError(f'native {mode} capture failed')
    request = validate_manifest(manifest)
    report = validate_native_output(out / 'native', manifest, mode, binary.resolve(), model.resolve(), request, require_split=require_split)
    return (report, receipt, settle)

def _ordinary(binary, model, out):
    cmd = [str(binary), 'run', '--model', str(model), '--memory-gb', '14', '--max-context', '2048', '--mtp', 'off', '--vision', 'off', '--prompt', 'In one sentence, explain why the sky is blue.', '--max-tokens', '32', '--greedy', '--seed', '7', '--sample-footprint', '--stats-json', str(out / 'stats.json')]
    settle = benchmark.settle_before_model_launch()
    code = benchmark.launch(out, 14, 900, cmd)
    atomic_json(out / 'settling.json', settle)
    receipt = validate_receipt_file(out / 'receipt.json')
    require_terminal_sampling(receipt)
    if code or not receipt['result']['functional_success']:
        raise EvidenceError('ordinary parity generation failed')
    return (read_json(out / 'stats.json'), receipt, settle)

def _stable_plan(plan):
    result = copy.deepcopy(plan)
    for key in ('device_available_gb', 'device_ram_gb', 'device_working_set_gb'):
        result.pop(key, None)
    return result

def _capture_identity(report):
    documents = []
    for document in report['documents']:
        documents.append({'id': document['id'], 'positions': [
            {'position': position['position'], 'routes': position['routes'],
             'state_sha256': position['state_sha256'], 'state': position['state'],
             'continuation_id': position['continuation_id'], 'input_id': position['input_id'],
             'next_token_id': position['next_token_id']} for position in document['positions']]})
    logits = sorted((item['name'], item['sha256'], item['bytes']) for item in report['files']
                    if item['category'] == 'logits')
    return {'documents': documents, 'logits': logits}

def parity(model: Path, reference: Path, manifest: Path, split: str, memory: float, context: int, output: Path) -> int:
    output = fresh_output(output)
    try:
        if memory != 14 or context != 2048:
            raise EvidenceError('parity requires memory 14 and context 2048')
        request = validate_manifest(manifest)
        current = ROOT / '.build/release/slotstream'
        old = reference / 'bin/slotstream'
        documents_for_split(request, split)
        ambient = sorted((key for key in os.environ if key.startswith('SLOTSTREAM_')))
        if ambient:
            raise EvidenceError(f'unsupported ambient Slotstream controls: {ambient}')
        tf32 = os.environ.get('MLX_ENABLE_TF32')
        if tf32 not in (None, '0', '1'):
            raise EvidenceError('MLX_ENABLE_TF32 must be 0 or 1')
        from cache_study import derive_source_geometry, verified_model_revision
        model_verification = verified_model_revision(model)
        geometry = derive_source_geometry(model, verification=model_verification)
        validate_build_identity(current)
        validate_build_identity(old, historical=True)
        (off, off_receipt, off_settle) = _native(current, model, manifest, split, 'reference-off', output / 'observer-off')
        (on, on_receipt, on_settle) = _native(current, model, manifest, split, 'reference-on', output / 'observer-on')
        (split_report, split_receipt, split_settle) = _native(current, model, manifest, split, 'reference-on', output / 'resident-split-control', require_split=True)
        if off['optimizations'] != on['optimizations'] or _stable_plan(off['plan']) != _stable_plan(on['plan']) or off['memory_ledger'] != on['memory_ledger'] or (off['numerical_environment'] != on['numerical_environment']) or (off_receipt['environment'] != on_receipt['environment']):
            raise EvidenceError('observer arms changed effective configuration')
        expected_split_optimizations = copy.deepcopy(on['optimizations'])
        expected_split_optimizations['overlapResidentExperts'] = True
        split_environment = dict(on_receipt['environment'])
        split_environment['SLOTSTREAM_OPT_RESIDENT_OVERLAP'] = '1'
        if split_report['optimizations'] != expected_split_optimizations or _stable_plan(split_report['plan']) != _stable_plan(on['plan']) or split_report['memory_ledger'] != on['memory_ledger'] or (split_report['numerical_environment'] != on['numerical_environment']) or (split_receipt['environment'] != split_environment):
            raise EvidenceError('resident split control changed more than the explicit overlap setting')

        if _capture_identity(off) != _capture_identity(on):
            raise EvidenceError('observer on/off logits, routes, state or continuation parity failed')
        if _capture_identity(on) != _capture_identity(split_report):
            raise EvidenceError('resident split control changed logits, routes, state or continuation')
        (oldstats, old_receipt, old_settle) = _ordinary(old, model, output / 'ordinary-old')
        (newstats, new_receipt, new_settle) = _ordinary(current, model, output / 'ordinary-new')
        if oldstats['output_ids'] != newstats['output_ids'] or oldstats['prompt_ids'] != newstats['prompt_ids']:
            raise EvidenceError('archived/new ordinary output parity failed')
        relevant = ('decodeTokens', 'finishReason', 'prefillTokens', 'decodeForwardPasses', 'prefillRecords', 'decodeRecords', 'prefillReadBytes', 'decodeReadBytes')
        if any((oldstats['stats'].get(k) != newstats['stats'].get(k) for k in relevant)):
            raise EvidenceError('archived/new relevant stats parity failed')

        def ordinary_configuration(stats):
            return {'plan': _stable_plan(stats['plan']), 'optimizations': stats.get('optimizations'), 'effective_pool_slots': stats.get('effective_pool_slots'), 'effective_prefill_chunk': stats.get('effective_prefill_chunk'), 'effective_mtp': stats.get('effective_mtp')}
        if ordinary_configuration(oldstats) != ordinary_configuration(newstats) or old_receipt['environment'] != new_receipt['environment']:
            raise EvidenceError('archived/new ordinary effective configuration differs')
        if newstats.get('optimizations') != on['optimizations']:
            raise EvidenceError('diagnostic and ordinary effective optimizations differ')
        evidence = {}
        for name in ('observer-off', 'observer-on', 'resident-split-control'):
            root = output / name
            evidence[name] = {key: sha256(root / path) for (key, path) in {'launcher_receipt_sha256': 'receipt.json', 'launcher_completion_sha256': 'completion.json', 'native_report_sha256': 'native/report.json', 'native_completion_sha256': 'native/completion.json', 'settling_sha256': 'settling.json'}.items()}
        for name in ('ordinary-old', 'ordinary-new'):
            root = output / name
            evidence[name] = {key: sha256(root / path) for (key, path) in {'launcher_receipt_sha256': 'receipt.json', 'launcher_completion_sha256': 'completion.json', 'stats_sha256': 'stats.json', 'settling_sha256': 'settling.json'}.items()}
        result = {'format': 'slotstream-flash-parity-v1', 'schema_version': 1, 'qualification': False, 'observer_parity': True, 'ordinary_cli_parity': True, 'documents': len([d for d in request['documents'] if d['split'] == split]), 'manifest_sha256': sha256(manifest), 'current_binary_sha256': sha256(current), 'reference_binary_sha256': sha256(old), 'activation_bytes': on['activation_bytes'], 'logits_bytes': on['logits_bytes'], 'model_settling_policy': {'kind': 'fixed-before-full-model-launch', 'seconds': benchmark.MODEL_SETTLE_SECONDS}, 'model_settling_observations': {'observer_off': off_settle, 'observer_on': on_settle, 'resident_split_control': split_settle, 'ordinary_old': old_settle, 'ordinary_new': new_settle}, 'numerical_environment': off['numerical_environment'], 'effective_optimizations': off['optimizations'], 'resident_split_control': {'split_layers': split_report['resident_split_evidence']['split_layers'], 'binary_sha256': split_report['source_identity']['binary_sha256'], 'production_equivalent_build': True}, 'model_verification': model_verification, 'bounded_model_identity': {'config_sha256': geometry['config_sha256'], 'index_sha256': geometry['index_sha256'], 'small_file_identities': geometry['small_file_identities'], 'file_headers': geometry['file_headers']}, 'evidence': evidence, 'harness_hashes': harness_hashes()}
        atomic_json(output / 'report.json', result)
        atomic_json(output / 'completion.json', {'format': 'slotstream-flash-parity-completion-v1', 'report_sha256': sha256(output / 'report.json'), 'qualification': False})
        return 0
    except Exception as e:
        atomic_json(output / 'failure.json', {'format': 'slotstream-flash-parity-failure-v1', 'error': f'{type(e).__name__}: {e}'})
        return 1

def anchor_parity(model: Path, anchor: Path, corpus: Path, output: Path) -> int:
    output = fresh_output(output)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('flash_tokenize_validation', HERE / 'tokenize_corpus.py')
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        corpus_result = module.validate_corpus(corpus, model=model.resolve())
        current = ROOT / '.build/release/slotstream'
        anchor_binary = anchor / 'bin/slotstream'
        validate_build_identity(current)
        anchor_identity, _ = validate_build_identity(anchor_binary, historical=True)
        if anchor_identity['binary_sha256'] != '4ce7f4e05ba3e374dd790f88d05b15587f6339588796188af84218a1f838638b':
            raise EvidenceError('immutable diagnostic reference anchor identity changed')
        v1 = ROOT / 'Tools/fixtures/flash/capture-development.json'
        anchor_off, _, anchor_settle = _native(anchor_binary, model, v1, 'development',
                                                'reference-off', output / 'anchor-v1-off')
        current_off, _, current_off_settle = _native(current, model, v1, 'development',
                                                      'reference-off', output / 'adapter-v1-off')
        current_on, _, current_on_settle = _native(current, model, v1, 'development',
                                                   'reference-on', output / 'adapter-v1-on')
        if (_capture_identity(anchor_off) != _capture_identity(current_off)
                or _capture_identity(current_off) != _capture_identity(current_on)):
            raise EvidenceError('new adapter does not preserve immutable v1 capture identity')
        development = [entry for entry in corpus_result['index']['shards']
                       if entry['split'] == 'development']
        if not development:
            raise EvidenceError('tokenized corpus has no development shard')
        shard = corpus.resolve() / 'corpus' / development[0]['path']
        v2_off, _, v2_off_settle = _native(current, model, shard, 'development',
                                            'reference-off', output / 'adapter-v2-off')
        v2_on, _, v2_on_settle = _native(current, model, shard, 'development',
                                          'reference-on', output / 'adapter-v2-on')
        if _capture_identity(v2_off) != _capture_identity(v2_on):
            raise EvidenceError('tokenized v2 observer changed computational capture identity')
        evidence = {}
        for name in ('anchor-v1-off', 'adapter-v1-off', 'adapter-v1-on',
                     'adapter-v2-off', 'adapter-v2-on'):
            root = output / name
            evidence[name] = {
                'receipt_sha256': sha256(root / 'receipt.json'),
                'native_report_sha256': sha256(root / 'native/report.json'),
                'native_completion_sha256': sha256(root / 'native/completion.json'),
            }
        report = {
            'format': 'slotstream-tokenizer-anchor-parity-v1',
            'schema_version': 1,
            'qualification': False,
            'v1_anchor_parity': True,
            'v2_observer_parity': True,
            'anchor_parsed_v2': False,
            'anchor_binary_sha256': sha256(anchor_binary),
            'adapter_binary_sha256': sha256(current),
            'corpus_sha256': sha256(corpus.resolve() / 'corpus/corpus.json'),
            'development_shard_sha256': sha256(shard),
            'settling': {'anchor_v1_off': anchor_settle, 'adapter_v1_off': current_off_settle,
                         'adapter_v1_on': current_on_settle, 'adapter_v2_off': v2_off_settle,
                         'adapter_v2_on': v2_on_settle},
            'evidence': evidence,
            'harness_hashes': harness_hashes(),
        }
        atomic_json(output / 'report.json', report)
        atomic_json(output / 'completion.json', {
            'format': 'slotstream-tokenizer-anchor-parity-completion-v1',
            'report_sha256': sha256(output / 'report.json'), 'qualification': False})
        return 0
    except Exception as error:
        atomic_json(output / 'failure.json', {'format': 'slotstream-tokenizer-anchor-parity-failure-v1',
                                              'error': f'{type(error).__name__}: {error}'})
        return 1

def self_test():
    import unittest
    r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_capture'))
    return 0 if r.testsRun and (not r.failures) and (not r.errors) and (not r.skipped) else 1

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    s = p.add_subparsers(dest='cmd')
    d = s.add_parser('dry-run')
    d.add_argument('--manifest', type=Path, required=True)
    d.add_argument('--split', required=True)
    d.add_argument('--output', type=Path, required=True)
    q = s.add_parser('parity')
    q.add_argument('--model', type=Path, required=True)
    q.add_argument('--reference', type=Path, required=True)
    q.add_argument('--manifest', type=Path, required=True)
    q.add_argument('--split', default='development')
    q.add_argument('--memory-gb', type=float, required=True)
    q.add_argument('--max-context', type=int, required=True)
    q.add_argument('--output', type=Path, required=True)
    anchor = s.add_parser('anchor-parity')
    anchor.add_argument('--model', type=Path, required=True)
    anchor.add_argument('--anchor', type=Path, required=True)
    anchor.add_argument('--corpus', type=Path, required=True)
    anchor.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == 'dry-run':
        return dry_run(a.manifest, a.split, a.output)
    if a.cmd == 'parity':
        return parity(a.model, a.reference, a.manifest, a.split, a.memory_gb, a.max_context, a.output)
    if a.cmd == 'anchor-parity':
        return anchor_parity(a.model, a.anchor, a.corpus, a.output)
    p.error('choose --self-test, dry-run, parity or anchor-parity')
if __name__ == '__main__':
    raise SystemExit(main())
