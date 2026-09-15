import copy
import math
from pathlib import Path
import struct
import tempfile
import unittest

from common import EvidenceError, atomic_json, sha256
import calibrate


class NeuronScoringTests(unittest.TestCase):
    def test_column_orientation_and_single_precision_result(self):
        self.assertEqual(calibrate.column_norms([3, 0, -8, 4, -12, 15], 2, 3), [5, 12, 17])
        self.assertEqual(calibrate.column_norms([1, 1], 2, 1), [struct.unpack('<f', struct.pack('<f', math.sqrt(2)))[0]])

    def test_signed_activation_scores_and_ties_match_native_fixture(self):
        scores = calibrate.block_scores([1] * 639 + [-2], [1] * 640)
        self.assertEqual(scores, [64] * 9 + [65])
        self.assertEqual(calibrate.selected_blocks(scores, 2), [True] + [False] * 8 + [True])
        self.assertEqual(calibrate.selected_blocks(scores, 10), [True] * 10)
        self.assertEqual(calibrate.selected_blocks([0] * 10, 2), [True] * 2 + [False] * 8)

    def test_nonfinite_negative_and_invalid_geometry_rejected(self):
        for bad in (float('nan'), float('inf')):
            with self.assertRaises(EvidenceError):
                calibrate.column_norms([bad], 1, 1)
            with self.assertRaises(EvidenceError):
                calibrate.block_scores([bad] + [0] * 639, [1] * 640)
        with self.assertRaises(EvidenceError):
            calibrate.block_scores([1] * 640, [-1] * 640)
        with self.assertRaises(EvidenceError):
            calibrate.column_norms([], 1, 2)
        for count in (3, 0, True, 2.0):
            with self.assertRaises(EvidenceError):
                calibrate.selected_blocks([1] * 10, count)

    def fixture(self, root):
        payload = root / 'layer-00.f32'
        payload.write_bytes(struct.pack('<640f', *([1] * 640)))
        manifest = {'format': 'slotstream-column-norms-v1', 'schema_version': 1,
            'algorithm': calibrate.ALGORITHM, 'scope': 'sample', 'dtype': 'float32-le',
            'shape': [48, 512, 640], 'expert_count': 1, 'model_path': '/model',
            'model_metadata': {}, 'source_identities': {}, 'provenance': {},
            'source_layouts': [{'layer': i, 'bits': 6, 'group_size': 64, 'source_record_bytes': 100} for i in range(48)],
            'model_loaded': False, 'qualification': False,
            'artifacts': [{'path': payload.name, 'layer': 0, 'experts': [0], 'shape': [1, 640],
                           'bytes': payload.stat().st_size, 'sha256': sha256(payload)}]}
        self.save(root, manifest)
        return manifest

    def save(self, root, manifest):
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(root / 'completion.json', {'format': 'slotstream-column-norms-completion-v1',
                    'manifest_sha256': sha256(root / 'manifest.json')})

    def test_sample_cannot_be_misrepresented_as_full_and_mutations_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original = self.fixture(root)
            calibrate.validate_norms(root, require_full=False)
            with self.assertRaises(EvidenceError):
                calibrate.validate_norms(root)
            for change in ('scope', 'duplicate', 'path', 'count'):
                manifest = copy.deepcopy(original)
                if change == 'scope': manifest['scope'] = 'full'
                elif change == 'duplicate': manifest['artifacts'] *= 2
                elif change == 'path': manifest['artifacts'][0]['path'] = '../layer-00.f32'
                elif change == 'count': manifest['expert_count'] += 1
                self.save(root, manifest)
                with self.assertRaises(EvidenceError):
                    calibrate.validate_norms(root, require_full=False)
            self.save(root, original)
            (root / 'layer-00.f32').write_bytes(bytes(2560))
            with self.assertRaisesRegex(EvidenceError, 'bytes changed'):
                calibrate.validate_norms(root, require_full=False)


if __name__ == '__main__':
    unittest.main()
