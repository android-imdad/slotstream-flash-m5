#!/usr/bin/env python3
import copy, json, struct, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture import (FF, H, LAYERS, TOPK, VOCAB, canonical_manifest_hash, dry_run,
                     documents_for_split, foundation_manifest_hash,
                     validate_manifest, validate_native_output)
from common import EvidenceError, FLASH_ROOT, atomic_json, read_json, sha256

class CaptureTests(unittest.TestCase):

    def setUp(self):
        FLASH_ROOT.mkdir(parents=True, exist_ok=True)
        self.t = tempfile.TemporaryDirectory(dir=FLASH_ROOT)
        self.root = Path(self.t.name)
        self.source = HERE.parent / 'fixtures/flash/capture-development.json'

    def tearDown(self):
        self.t.cleanup()

    def mutated(self, fn):
        d = json.loads(self.source.read_text())
        fn(d)
        d['manifestSHA256'] = canonical_manifest_hash(d)
        p = self.root / 'm.json'
        p.write_text(json.dumps(d))
        return p

    def test_valid_manifest_and_i_plus_one_offset(self):
        d = validate_manifest(self.source)
        self.assertEqual(len(d['documents']), 2)

    def tokenized_shard(self, split='development'):
        value = {'format': 'slotstream-tokenized-capture-shard-v2', 'schemaVersion': 2,
                 'manifestSHA256': '', 'tokenSource': 'slotstream-auto-tokenizer-chat-template-v1',
                 'split': split, 'sourceCorpusSHA256': 'a' * 64,
                 'tokenizerIdentity': {'revision': 'test'}, 'shardID': 'shard-000',
                 'documents': [{'id': 'natural', 'category': 'prose', 'sourceID': 'fixture',
                                'sourceHash': 'b' * 64, 'split': split, 'warmupIDs': [1],
                                'positions': [{'position': 1, 'inputID': 2, 'nextTokenID': 3}]}]}
        value['manifestSHA256'] = foundation_manifest_hash(value)
        path = self.root / f'{split}.json'
        path.write_text(json.dumps(value))
        return path

    def test_v2_manifest_and_qualification_capture_gate(self):
        development = validate_manifest(self.tokenized_shard())
        self.assertEqual(len(documents_for_split(development, 'development')), 1)
        qualification = validate_manifest(self.tokenized_shard('qualification'))
        with self.assertRaisesRegex(EvidenceError, 'locked'):
            documents_for_split(qualification, 'qualification')

    def test_unknown_field_rejected(self):
        p = self.mutated(lambda d: d.update(extra=True))
        with self.assertRaises(EvidenceError):
            validate_manifest(p)

    def test_symlink_and_oversized_manifest_are_rejected_before_json_decode(self):
        link = self.root / 'linked.json'
        link.symlink_to(self.source)
        with self.assertRaisesRegex(EvidenceError, 'regular file'):
            validate_manifest(link)
        large = self.root / 'large.json'
        large.write_bytes(b' ' * ((1 << 20) + 1))
        with self.assertRaisesRegex(EvidenceError, '1 MiB'):
            validate_manifest(large)

    def test_hash_mismatch_rejected(self):
        d = json.loads(self.source.read_text())
        d['documents'][0]['id'] = 'changed'
        p = self.root / 'm.json'
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(EvidenceError, 'hash'):
            validate_manifest(p)

    def test_missing_duplicate_out_of_order_and_wrong_target(self):

        def missing(d):
            d['documents'][0]['positions'][0]['position'] = 3
        with self.assertRaises(EvidenceError):
            validate_manifest(self.mutated(missing))

        def wrong(d):
            d['documents'][0]['positions'][1]['inputID'] = 99
        with self.assertRaisesRegex(EvidenceError, 'i\\+1'):
            validate_manifest(self.mutated(wrong))

    def test_quota_dry_run_without_model(self):
        out = self.root / 'dry'
        self.assertEqual(dry_run(self.source, 'development', out), 0)
        d = read_json(out / 'report.json')
        self.assertFalse(d['model_loaded'])
        self.assertGreater(d['positions'], 0)

    def test_empty_split_preserves_partial_failure(self):
        out = self.root / 'failed'
        self.assertEqual(dry_run(self.source, 'heldout', out), 1)
        self.assertTrue((out / 'failure.json').exists())
        self.assertFalse((out / 'completion.json').exists())

    def test_duplicate_document_identity_rejected(self):

        def duplicate(d):
            d['documents'][1]['id'] = d['documents'][0]['id']
        with self.assertRaisesRegex(EvidenceError, 'identity'):
            validate_manifest(self.mutated(duplicate))

    def test_invalid_warmup_token_is_rejected(self):

        def invalid(d):
            d['documents'][0]['warmupIDs'][0] = VOCAB
        with self.assertRaisesRegex(EvidenceError, 'warmup'):
            validate_manifest(self.mutated(invalid))

    def test_manifest_bounds_and_split_fail_before_model_work(self):

        def unsafe(d):
            d['documents'][0]['id'] = '../escape'
        with self.assertRaises(EvidenceError):
            validate_manifest(self.mutated(unsafe))
        def long_id(d):
            d['documents'][0]['id'] = 'a' * 65
        with self.assertRaises(EvidenceError):
            validate_manifest(self.mutated(long_id))
        def trailing_newline(d):
            d['documents'][0]['id'] = 'safe\n'
        with self.assertRaises(EvidenceError):
            validate_manifest(self.mutated(trailing_newline))
        def foreign_split(d):
            d['documents'][0]['split'] = 'heldout'
        with self.assertRaises(EvidenceError):
            validate_manifest(self.mutated(foreign_split))
        def boolean_position(d):
            d['documents'][0]['warmupIDs'] = []
            d['documents'][0]['positions'][0]['position'] = False
        with self.assertRaises(EvidenceError):
            validate_manifest(self.mutated(boolean_position))
        def long_warmup(d):
            d['documents'][0]['warmupIDs'] = [1] * 257
            d['documents'][0]['positions'] = []
        with self.assertRaises(EvidenceError):
            validate_manifest(self.mutated(long_warmup))

        def too_many(d):
            position = d['documents'][0]['positions'][0]
            d['documents'][0]['warmupIDs'] = []
            d['documents'][0]['positions'] = [{'position': i, 'inputID': 1, 'nextTokenID': 1} for i in range(65)]
        with self.assertRaisesRegex(EvidenceError, 'position limit'):
            validate_manifest(self.mutated(too_many))
        output = self.root / 'unsupported-split'
        self.assertEqual(dry_run(self.source, 'heldout', output), 1)
        self.assertFalse((output / 'completion.json').exists())

    def native_fixture(self, mode='reference-off', suffix='case', widening_policy='scalar'):
        base = self.root / suffix
        base.mkdir()
        binary_dir = base / 'bin'
        binary_dir.mkdir()
        binary = binary_dir / 'slotstream'
        binary.write_bytes(b'binary')
        for (name, data) in (('mlx.metallib', b'metal'), ('build-identity.json', b'identity'), ('build-source.tar.gz', b'source')):
            (binary_dir / name).write_bytes(data)
        model = base / 'model'
        model.mkdir()
        (model / 'config.json').write_text(json.dumps({'text_config': {'layer_types': ['linear_attention' if layer % 4 != 3 else 'full_attention' for layer in range(LAYERS)]}}))
        (model / 'model.safetensors.index.json').write_text('{}')
        manifest = base / 'native-manifest.json'
        manifest.write_text('{}')
        request = {'documents': [{'id': 'doc', 'split': 'development', 'warmupIDs': [], 'positions': [{'position': 0, 'inputID': 1, 'nextTokenID': 2}]}]}
        native = base / 'native'
        native.mkdir()
        files = []

        def artifact(name, category, data, **extra):
            (native / name).write_bytes(data)
            value = {'name': name, 'category': category, 'document_id': 'doc', 'token_position': 0, 'input_id': 1, 'dtype': extra.pop('dtype'), 'shape': extra.pop('shape'), 'bytes': len(data), 'sha256': sha256(native / name)}
            value.update(extra)
            files.append(value)
        artifact('logits.bin', 'logits', bytes(VOCAB * 4), dtype='float32', shape=[1, 1, VOCAB])
        if mode == 'reference-on':
            for layer in range(LAYERS):
                artifact(f'input-{layer}.bin', 'input', bytes(H * 2), dtype='bfloat16', shape=[1, 1, H], layer=layer)
                artifact(f'hidden-{layer}.bin', 'hidden', bytes(TOPK * FF * 4), dtype='float32', shape=[1, 1, TOPK, 1, FF], layer=layer, router_ranks=list(range(TOPK)), expert_ids=list(range(TOPK)))
        routes = [{'layer': layer, 'ids': list(range(TOPK))} for layer in range(LAYERS)]
        state_names = {'ngram', 'tokens', 'lastMulti', 'mtp.key', 'mtp.value', 'mtp.index', 'mtp.offset'}
        for layer in range(LAYERS):
            if layer % 4 == 3:
                state_names.update((f'key.{layer}', f'value.{layer}', f'index.{layer}'))
            else:
                state_names.update((f'conv.{layer}', f'ssm.{layer}', f'ple.{layer}'))
        state_fields = [{'name': name, 'present': False} for name in sorted(state_names)]
        token_index = next((index for (index, field) in enumerate(state_fields) if field['name'] == 'tokens'))
        state_fields[token_index] = {'name': 'tokens', 'present': True, 'dtype': 'int64', 'shape': [], 'bytes': 8, 'sha256': 'a' * 64}
        state = {'fields': state_fields, 'indexerBases': {}, 'allocatedSequenceBytes': 0}
        state_hash = __import__('hashlib').sha256(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        source = {'binary_sha256': sha256(binary), 'metallib_sha256': sha256(binary_dir / 'mlx.metallib'), 'build_identity_sha256': sha256(binary_dir / 'build-identity.json'), 'source_archive_sha256': sha256(binary_dir / 'build-source.tar.gz'), 'model_config_sha256': sha256(model / 'config.json'), 'model_index_sha256': sha256(model / 'model.safetensors.index.json')}
        report = {'format': 'slotstream-flash-capture-output-v1' if widening_policy == 'scalar' else 'slotstream-flash-widening-output-v1', 'schema_version': 1, 'qualification': False, 'mode': mode, 'manifest_sha256': sha256(manifest), 'binary': str(binary), 'model': str(model), 'source_identity': source, 'plan': {'target_gb': 14}, 'memory_ledger': {'diagnostic_reserved_bytes': 128 << 20}, 'documents': [{'id': 'doc', 'positions': [{'position': 0, 'input_id': 1, 'next_token_id': 2, 'routes': routes, 'state': state, 'state_sha256': state_hash, 'continuation_id': 3}]}], 'files': files, 'invalid_state_reuse_refused': True, 'optimizations': {'overlapResidentExperts': False}, 'numerical_environment': {'mlx_enable_tf32_raw': None, 'effective_tf32': True}, 'resident_split_evidence': {'required': False, 'split_layers': []}, 'activation_bytes': sum((f['bytes'] for f in files if f['category'] != 'logits')), 'logits_bytes': sum((f['bytes'] for f in files if f['category'] == 'logits'))}
        if widening_policy == 'packed4-to6':
            report['effective_expert_widening'] = widening_policy
        atomic_json(native / 'report.json', report)
        atomic_json(native / 'completion.json', {'format': 'slotstream-flash-capture-completion-v1' if widening_policy == 'scalar' else 'slotstream-flash-widening-completion-v1', 'report_sha256': sha256(native / 'report.json'), 'qualification': False})
        return (native, manifest, binary, model, request)

    def rewrite_report(self, native, report):
        atomic_json(native / 'report.json', report)
        atomic_json(native / 'completion.json', {'format': 'slotstream-flash-capture-completion-v1', 'report_sha256': sha256(native / 'report.json'), 'qualification': False})

    def test_native_output_validator_accepts_complete_off_and_on_evidence(self):
        (native, manifest, binary, model, request) = self.native_fixture('reference-off', 'off')
        validate_native_output(native, manifest, 'reference-off', binary, model, request)
        (native, manifest, binary, model, request) = self.native_fixture('reference-on', 'on')
        validate_native_output(native, manifest, 'reference-on', binary, model, request)

    def test_native_output_admits_only_explicit_packed_widening_schema(self):
        fixture = self.native_fixture(suffix='packed', widening_policy='packed4-to6')
        (native, manifest, binary, model, request) = fixture
        validate_native_output(native, manifest, 'reference-off', binary, model, request,
                               widening_policy='packed4-to6')
        with self.assertRaisesRegex(EvidenceError, 'native'):
            validate_native_output(native, manifest, 'reference-off', binary, model, request)
        report = read_json(native / 'report.json')
        report['effective_expert_widening'] = 'scalar'
        atomic_json(native / 'report.json', report)
        atomic_json(native / 'completion.json', {
            'format': 'slotstream-flash-widening-completion-v1',
            'report_sha256': sha256(native / 'report.json'), 'qualification': False})
        with self.assertRaisesRegex(EvidenceError, 'native'):
            validate_native_output(native, manifest, 'reference-off', binary, model, request,
                                   widening_policy='packed4-to6')

    def test_native_output_rejects_missing_extra_wrong_dtype_foreign_and_partial(self):
        (native, manifest, binary, model, request) = self.native_fixture(suffix='missing')
        (native / 'logits.bin').unlink()
        with self.assertRaises(EvidenceError):
            validate_native_output(native, manifest, 'reference-off', binary, model, request)
        (native, manifest, binary, model, request) = self.native_fixture(suffix='extra')
        report = read_json(native / 'report.json')
        extra = copy.deepcopy(report['files'][0])
        extra['name'] = 'extra.bin'
        extra['token_position'] = 99
        (native / 'extra.bin').write_bytes(bytes(VOCAB * 4))
        extra['sha256'] = sha256(native / 'extra.bin')
        report['files'].append(extra)
        report['logits_bytes'] += extra['bytes']
        atomic_json(native / 'report.json', report)
        atomic_json(native / 'completion.json', {'format': 'slotstream-flash-capture-completion-v1', 'report_sha256': sha256(native / 'report.json'), 'qualification': False})
        with self.assertRaisesRegex(EvidenceError, 'extra'):
            validate_native_output(native, manifest, 'reference-off', binary, model, request)
        (native, manifest, binary, model, request) = self.native_fixture(suffix='dtype')
        report = read_json(native / 'report.json')
        report['files'][0]['dtype'] = 'float16'
        atomic_json(native / 'report.json', report)
        atomic_json(native / 'completion.json', {'format': 'slotstream-flash-capture-completion-v1', 'report_sha256': sha256(native / 'report.json'), 'qualification': False})
        with self.assertRaisesRegex(EvidenceError, 'geometry'):
            validate_native_output(native, manifest, 'reference-off', binary, model, request)
        (native, manifest, binary, model, request) = self.native_fixture(suffix='foreign')
        report = read_json(native / 'report.json')
        report['model'] = str(self.root / 'foreign-model')
        atomic_json(native / 'report.json', report)
        atomic_json(native / 'completion.json', {'format': 'slotstream-flash-capture-completion-v1', 'report_sha256': sha256(native / 'report.json'), 'qualification': False})
        with self.assertRaisesRegex(EvidenceError, 'identity'):
            validate_native_output(native, manifest, 'reference-off', binary, model, request)
        (native, manifest, binary, model, request) = self.native_fixture(suffix='partial')
        (native / 'completion.json').unlink()
        with self.assertRaises(Exception):
            validate_native_output(native, manifest, 'reference-off', binary, model, request)

    def test_native_output_recomputes_state_hash_and_requires_full_inventory(self):
        (native, manifest, binary, model, request) = self.native_fixture(suffix='state-mutated')
        report = read_json(native / 'report.json')
        report['documents'][0]['positions'][0]['state']['allocatedSequenceBytes'] = 9
        self.rewrite_report(native, report)
        with self.assertRaisesRegex(EvidenceError, 'state identity hash'):
            validate_native_output(native, manifest, 'reference-off', binary, model, request)
        (native, manifest, binary, model, request) = self.native_fixture(suffix='state-missing')
        report = read_json(native / 'report.json')
        state = report['documents'][0]['positions'][0]['state']
        state['fields'].pop(0)
        report['documents'][0]['positions'][0]['state_sha256'] = __import__('hashlib').sha256(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.rewrite_report(native, report)
        with self.assertRaisesRegex(EvidenceError, 'inventory'):
            validate_native_output(native, manifest, 'reference-off', binary, model, request)

    def test_native_output_rejects_nonfinite_logits_hidden_and_bfloat16(self):
        for (suffix, category, payload) in (('nan-logits', 'logits', struct.pack('<f', float('nan'))), ('nan-hidden', 'hidden', struct.pack('<f', float('inf'))), ('nan-input', 'input', struct.pack('<H', 32640))):
            (native, manifest, binary, model, request) = self.native_fixture('reference-on', suffix)
            report = read_json(native / 'report.json')
            entry = next((item for item in report['files'] if item['category'] == category))
            path = native / entry['name']
            data = bytearray(path.read_bytes())
            data[:len(payload)] = payload
            path.write_bytes(data)
            entry['sha256'] = sha256(path)
            self.rewrite_report(native, report)
            with self.subTest(category=category), self.assertRaisesRegex(EvidenceError, 'non-finite'):
                validate_native_output(native, manifest, 'reference-on', binary, model, request)
if __name__ == '__main__':
    unittest.main(verbosity=2)
