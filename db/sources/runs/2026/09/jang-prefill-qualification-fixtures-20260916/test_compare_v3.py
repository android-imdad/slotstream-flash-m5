#!/usr/bin/env python3
"""Synthetic corruption tests; no model allocation, builds or source changes."""
import copy
import contextlib
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location('prefill_compare', Path(__file__).with_name('compare_v3.py'))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')

def seal(root, report):
    write_json(root / 'report.json', report)
    write_json(root / 'completion.json', {'format': 'slotstream-prefill-capture-completion-v1',
               'schema_version': 1, 'passed': True, 'report_sha256': C.sha256(root / 'report.json')})

def put(root, name, value):
    value = np.asarray(value)
    path = root / name
    path.write_bytes(value.tobytes())
    return {'file': name, 'shape': list(value.shape), 'original_dtype': str(value.dtype),
            'storage_dtype': str(value.dtype), 'bytes': value.nbytes,
            'sha256': C.sha256(path), 'finite': True}

def fixture(root, chunk):
    root.mkdir()
    prompt = [20 + (i % 100) for i in range(1100)]
    continuation = [100] * 8
    opts = dict.fromkeys(C.OPTIMIZATION_KEYS, False)
    for key in ('compactStateWindows compactMTPRow compactNgramRows automaticReadScope '
                'skipUnusedFinalForward valueOnlySamplerThreshold deviceSamplerDraw boundedOutputQueue '
                'responsiveGovernor completePromptCheckpoint sharedRoPE').split():
        opts[key] = True
    opts.update(workspaceTokenTile=256, readScopeTokens=0, visionAttentionPadding=0,
                visionQueryTile=256, prefixCheckpointTokens=256)
    report = {'format': 'slotstream-prefill-capture-v1', 'schema_version': 1,
              'identity': {key: 'a' * 64 for key in C.IDENTITY_KEYS}, 'actual_engine': True,
              'capture_path': 'actualEngine', 'byte_order': 'little', 'raw': False, 'sampling': {'greedy': True, 'seed': 42, 'max_tokens': 1},
              'normal_output_ids': [100],
              'continuation_mode': 'baseline-raw-argmax' if chunk == 256 else 'imported-baseline-ids',
              'prompt_ids': prompt, 'continuation_ids': continuation,
              'top_k': C.TOPK, 'num_layers': C.LAYERS, 'vocab_size': C.VOCAB,
              'layer_types': C.LAYER_TYPES, 'expected_tensor_keys': sorted(C.TENSOR_KEYS),
              'optimizations': opts, 'runtime_environment': {'SLOTSTREAM_PREFILL_CHUNK': str(chunk), 'SLOTSTREAM_PREFIX_CACHE': '0'},
              'numerical_environment': {'effective_tf32': True},
              'effective_prefill_chunk': chunk, 'effective_pool_slots': 2000 - chunk,
              'effective_expert_widening': 'packed4-to6', 'effective_mtp': False,
              'effective_vision': False, 'effective_prefix_cache': False,
              'plan': {'target_gb': 14, 'max_context_tokens': 4096, 'prefill_chunk': chunk,
                       'pool_slots': 2000 - chunk, 'mtp': False, 'vision': False, 'source': 'explicit'},
              'snapshots': {}, 'routes': {}, 'route_events': []}
    report['plan']['memory_ledger'] = dict.fromkeys(C.LEDGER_KEYS, 0)
    report['plan']['memory_ledger'].update(version=1, expected_peak_bytes=14000000000)
    for stage, extra in C.STAGES.items():
        consumed = prompt + continuation[:extra]
        fields = {}
        for key in C.TENSOR_KEYS:
            if key == 'logits':
                value = np.linspace(-1, 1, C.VOCAB, dtype=np.float32)
                value[100] = 4
            elif key == 'tokens':
                value = np.array(len(consumed), dtype=np.int64)
            elif key == 'ngram':
                value = np.array(consumed[-2:], dtype=np.int64)
            else:
                value = np.array([1, -2, 3], dtype=np.float32)
            fields[key] = put(root, f'{stage}-{key}.bin', value)
        report['snapshots'][stage] = {'consumed_ids': consumed, 'greedy_token': 100, 'tensors': fields}
    for phase, rows in [('prefill', len(prompt)), ('continuation', 8)]:
        report['routes'][phase] = {}
        for layer in range(C.LAYERS):
            value = np.tile(np.arange(C.TOPK, dtype=np.int32), (rows, 1))
            field = put(root, f'routes-{phase}-{layer}.bin', value)
            field['logical_positions'] = list(range(rows))
            report['routes'][phase][str(layer)] = field
        for start in range(0, rows, chunk if phase == 'prefill' else 1):
            block = min(chunk, rows - start) if phase == 'prefill' else 1
            report['route_events'].extend({'phase': phase, 'layer': i, 'rows': block} for i in range(C.LAYERS))
    write_json(root / 'continuation.json', continuation)
    seal(root, report)

class ComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = tempfile.TemporaryDirectory()
        cls.source = Path(cls.baseline.name)
        for chunk in (256, 512, 1024):
            fixture(cls.source / str(chunk), chunk)

    @classmethod
    def tearDownClass(cls):
        cls.baseline.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for chunk in (256, 512, 1024):
            shutil.copytree(self.source / str(chunk), self.root / str(chunk))
        self.dirs = [self.root / str(x) for x in (256, 512, 1024)]

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, change, chunk=1024):
        root = self.root / str(chunk)
        report = json.loads((root / 'report.json').read_text())
        change(report, root)
        seal(root, report)

    def array_change(self, field_path, transform, chunk=1024):
        def apply(report, root):
            field = report
            for part in field_path:
                field = field[part]
            value = np.fromfile(root / field['file'], dtype=C.DTYPES[field['storage_dtype']]).reshape(field['shape'])
            value = transform(value)
            logical_positions = field.get('logical_positions')
            field.update(put(root, field['file'], value))
            if logical_positions is not None:
                field['logical_positions'] = logical_positions
        self.mutate(apply, chunk)

    def rejected(self, substring):
        result = C.compare(self.dirs)
        self.assertFalse(result['passed'])
        self.assertIn(substring, json.dumps(result['errors']))
        self.assertEqual(len(result['measurements']), len(C.STAGES) * len(C.TENSOR_KEYS) + 96)
        return result

    def test_nominal_pass_and_complete_rows(self):
        result = C.compare(self.dirs)
        self.assertTrue(result['passed'], result['errors'])
        self.assertFalse(result['control_independently_qualified'])
        self.assertEqual(len(result['measurements']), len(C.STAGES) * len(C.TENSOR_KEYS) + 96)

    def test_state_float_band_rejects_and_retains_measurement(self):
        self.array_change(('snapshots', 'step8', 'tensors', 'ssm.0'), lambda x: x + np.float32(.1))
        result = self.rejected('band or exactness')
        row = next(x for x in result['measurements'] if x['field'] == 'step8/ssm.0')
        self.assertGreater(row['candidate']['relative_delta'], .03)

    def test_empirical_control_scales_band(self):
        self.array_change(('snapshots', 'step8', 'tensors', 'ssm.0'), lambda x: x + np.float32(.1), 512)
        self.array_change(('snapshots', 'step8', 'tensors', 'ssm.0'), lambda x: x + np.float32(.2), 1024)
        result = C.compare(self.dirs)
        self.assertTrue(result['passed'], result['errors'])

    def test_full_vocab_not_only_top1_checked(self):
        def change(x):
            x[200000] += .5
            return x
        self.array_change(('snapshots', 'step8', 'tensors', 'logits'), change)
        self.rejected('band or exactness')

    def test_nonfinite_even_with_updated_hash_rejected(self):
        def change(x):
            x[0] = np.nan
            return x
        self.array_change(('snapshots', 'prefill', 'tensors', 'ssm.0'), change)
        self.rejected('nonfinite array')

    def test_hash_corruption_rejected(self):
        path = self.dirs[2] / 'step1-ssm.0.bin'
        path.write_bytes(b'0' * path.stat().st_size)
        self.rejected('SHA-256 mismatch')

    def test_identity_mismatch_rejected(self):
        self.mutate(lambda report, _: report['identity'].update(tokenizer_sha256='b' * 64))
        self.rejected('matching/identity')

    def test_exact_integer_state_rejected(self):
        self.array_change(('snapshots', 'step8', 'tensors', 'ngram'), lambda x: x + 1)
        self.rejected('ngram state')

    def test_route_keep_set_band_rejects(self):
        self.array_change(('routes', 'continuation', '47'), lambda x: x + 20)
        self.rejected('band or exactness')

    def test_single_layer_route_band_is_diagnostic_when_aggregate_passes(self):
        def change(x):
            x[0, 0] = 20
            return x
        self.array_change(('routes', 'continuation', '47'), change)
        result = C.compare(self.dirs)
        self.assertTrue(result['passed'], result['errors'])
        row = next(x for x in result['measurements'] if x['field'] == 'routes/continuation/47')
        self.assertFalse(row['passed'])
        self.assertFalse(row['gating'])
        aggregate = next(x for x in result['route_aggregates'] if x['field'] == 'routes/continuation/all_layers')
        self.assertTrue(aggregate['passed'])
        self.assertEqual(aggregate['candidate']['missing_experts'], 1)
        self.assertEqual(aggregate['candidate']['total_selections'], 48 * 8 * 10)
        self.assertEqual(aggregate['candidate']['keep_set_disagreement'], 1 / (48 * 8 * 10))

    def test_route_order_is_set_invariant(self):
        self.array_change(('routes', 'continuation', '47'), lambda x: x[:, ::-1].copy())
        result = C.compare(self.dirs)
        self.assertTrue(result['passed'], result['errors'])

    def test_route_duplicate_rejected(self):
        self.array_change(('routes', 'continuation', '47'), lambda x: np.zeros_like(x))
        self.rejected('duplicate expert')

    def test_missing_continuation_rejected(self):
        self.mutate(lambda report, _: report['continuation_ids'].pop())
        self.rejected('wrong token count')

    def test_continuation_drift_rejected(self):
        def change(report, root):
            report['continuation_ids'][7] = 101
            write_json(root / 'continuation.json', report['continuation_ids'])
        self.mutate(change)
        self.rejected('matching/continuation_ids')

    def test_shape_corruption_rejected(self):
        self.mutate(lambda report, _: report['snapshots']['step8']['tensors']['ssm.0'].update(shape=[2]))
        self.rejected('byte count/shape mismatch')

    def test_shape_same_count_different_shape_rejected(self):
        self.mutate(lambda report, _: report['snapshots']['step8']['tensors']['ssm.0'].update(shape=[1, 3]))
        self.rejected('shape/dtype mismatch')

    def test_incomplete_common_catalogue_rejected(self):
        for chunk in (256, 512, 1024):
            def change(report, _):
                report['expected_tensor_keys'].remove('ssm.0')
                for snap in report['snapshots'].values():
                    del snap['tensors']['ssm.0']
            self.mutate(change, chunk)
        self.rejected('incomplete architecture tensor catalogue')

    def test_unsafe_artifact_path_rejected(self):
        self.mutate(lambda report, _: report['snapshots']['step8']['tensors']['ssm.0'].update(file='../escape.bin'))
        self.rejected('unsafe artifact path')

    def test_symlink_artifact_rejected(self):
        path = self.dirs[2] / 'step8-ssm.0.bin'
        path.unlink()
        path.symlink_to(self.dirs[0] / path.name)
        self.rejected('symlink artifact path')

    def test_completion_missing_cli_writes_failure_report(self):
        (self.dirs[2] / 'completion.json').unlink()
        output = self.root / 'review.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(C.main([*(str(x) for x in self.dirs), '--output', str(output)]), 1)
        self.assertFalse(json.loads(output.read_text())['passed'])

    def test_route_coverage_rejected(self):
        self.mutate(lambda report, _: report['routes']['prefill']['0']['logical_positions'].pop())
        self.rejected('logical route positions')

    def test_optimization_drift_rejected(self):
        self.mutate(lambda report, _: report['optimizations'].update(terminalPrefillPruning=True))
        self.rejected('nondeployed optimization')

    def test_unqualified_fused_rope_rejected_even_if_all_arms_match(self):
        for chunk in (256, 512, 1024):
            self.mutate(lambda report, _: report['optimizations'].update(fusedRoPE=True), chunk)
        self.rejected('nondeployed optimization: fusedRoPE=True')

    def test_hidden_memory_ledger_policy_drift_rejected(self):
        self.mutate(lambda report, _: report['plan']['memory_ledger'].update(planning_margin_bytes=300000000))
        self.rejected('memory ledger differs')

    def test_greedy_change_rejected_even_inside_band(self):
        def close_logits(x):
            x[101] = np.float32(3.99)
            return x
        for chunk in (256, 512, 1024):
            self.array_change(('snapshots', 'step8', 'tensors', 'logits'), close_logits, chunk)
        def change(x):
            x[101] = np.float32(4.001)
            return x
        self.array_change(('snapshots', 'step8', 'tensors', 'logits'), change)
        self.mutate(lambda report, _: report['snapshots']['step8'].update(greedy_token=101))
        self.rejected('band or exactness')

    def test_control_greedy_drift_is_diagnostic_not_candidate_gate(self):
        def change(x):
            x[101] = 5
            return x
        for stage in C.STAGES:
            self.array_change(('snapshots', stage, 'tensors', 'logits'), change, 512)
        def fix_metadata(report, _):
            for snap in report['snapshots'].values():
                snap['greedy_token'] = 101
            report['normal_output_ids'] = [101]
        self.mutate(fix_metadata, 512)
        result = C.compare(self.dirs)
        self.assertTrue(result['passed'], result['errors'])
        rows = [x for x in result['measurements'] if x['field'].endswith('/logits')]
        self.assertTrue(all(x['greedy_top1_exact'] and not x['empirical_control_greedy_top1_exact'] for x in rows))

if __name__ == '__main__':
    unittest.main(verbosity=2)
