#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import EvidenceError, FLASH_ROOT

SPEC = importlib.util.spec_from_file_location("flash_tokenize_tool", HERE / "tokenize_corpus.py")
tokenize_tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tokenize_tool)


class TokenizeTests(unittest.TestCase):
    def setUp(self):
        FLASH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="tokenize-test-", dir=FLASH_ROOT)
        self.root = Path(self.temporary.name)
        self.fixture = HERE.parent / "fixtures/flash/tokenize-development.json"

    def tearDown(self):
        self.temporary.cleanup()

    def mutated(self, mutation):
        value = json.loads(self.fixture.read_text())
        mutation(value)
        value["manifestSHA256"] = tokenize_tool.canonical_hash(value, "manifestSHA256")
        path = self.root / "source.json"
        path.write_text(json.dumps(value))
        return path

    def test_authored_fixture_is_strict_and_split_complete(self):
        value = tokenize_tool.validate_source(self.fixture)
        self.assertEqual({item["split"] for item in value["documents"]},
                         {"training", "development", "qualification"})
        self.assertEqual({item["kind"] for item in value["documents"]}, {"raw", "chat"})

    def test_first_assistant_target_and_i_plus_one_chain(self):
        prompt = [10, 11, 12]
        full = prompt + [20, 21, 22]
        window = tokenize_tool.build_chat_capture_window(prompt, full, 2)
        self.assertEqual(window["warmupIDs"], [10, 11])
        self.assertEqual(window["positions"], [
            {"position": 2, "inputID": 12, "nextTokenID": 20},
            {"position": 3, "inputID": 20, "nextTokenID": 21},
        ])

    def test_prefix_mismatch_and_insufficient_text_fail(self):
        with self.assertRaisesRegex(EvidenceError, "prefix"):
            tokenize_tool.build_chat_capture_window([1, 2], [1, 3, 4], 1)
        with self.assertRaises(EvidenceError):
            tokenize_tool.build_capture_window([1, 2], warmup_count=1, score_tokens=1)

    def test_duplicate_cross_split_unknown_and_bounds_fail(self):
        def duplicate(value):
            value["documents"][1]["id"] = value["documents"][0]["id"]
        with self.assertRaises(EvidenceError):
            tokenize_tool.validate_source(self.mutated(duplicate))

        def cross_split(value):
            value["documents"][1]["partitionKey"] = value["documents"][0]["partitionKey"]
        with self.assertRaises(EvidenceError):
            tokenize_tool.validate_source(self.mutated(cross_split))

        def unknown(value):
            value["documents"][0]["extra"] = True
        with self.assertRaises(EvidenceError):
            tokenize_tool.validate_source(self.mutated(unknown))

        def long_name(value):
            value["documents"][0]["id"] = "x" * 65
        with self.assertRaises(EvidenceError):
            tokenize_tool.validate_source(self.mutated(long_name))

    def test_text_and_manifest_hash_mutation_fail(self):
        value = json.loads(self.fixture.read_text())
        value["documents"][0]["text"] += " changed"
        path = self.root / "hash.json"
        path.write_text(json.dumps(value))
        with self.assertRaises(EvidenceError):
            tokenize_tool.validate_source(path)

    def test_shard_mutation_and_qualification_identity(self):
        identity = {"model_revision": "r"}
        source_hash = "a" * 64
        shard = {
            "format": "slotstream-tokenized-capture-shard-v2",
            "schemaVersion": 2,
            "manifestSHA256": "",
            "tokenSource": "slotstream-auto-tokenizer-chat-template-v1",
            "split": "qualification",
            "sourceCorpusSHA256": source_hash,
            "tokenizerIdentity": identity,
            "shardID": "shard-000",
            "documents": [{"id": "doc", "category": "prose", "sourceID": "source",
                           "sourceHash": "b" * 64, "split": "qualification",
                           "warmupIDs": [1], "positions": [
                               {"position": 1, "inputID": 2, "nextTokenID": 3}]}],
        }
        shard["manifestSHA256"] = tokenize_tool.foundation_canonical_hash(shard, "manifestSHA256")
        path = self.root / "shard.json"
        path.write_text(json.dumps(shard))
        _, count = tokenize_tool._validate_shard(path, identity, source_hash)
        self.assertEqual(count, 1)
        shard["documents"][0]["positions"][0]["nextTokenID"] = 4
        path.write_text(json.dumps(shard))
        with self.assertRaises(EvidenceError):
            tokenize_tool._validate_shard(path, identity, source_hash)

    def test_three_and_four_digit_shard_ordinals_are_consistent(self):
        identity={'model_revision':'r'};source_hash='a'*64
        for ordinal in (999, 1000, 2047):
            shard={'format':'slotstream-tokenized-capture-shard-v2','schemaVersion':2,
                'manifestSHA256':'','tokenSource':'slotstream-auto-tokenizer-chat-template-v1',
                'split':'development','sourceCorpusSHA256':source_hash,
                'tokenizerIdentity':identity,'shardID':f'shard-{ordinal:03d}',
                'documents':[{'id':'doc','category':'prose','sourceID':'source/id','sourceHash':'b'*64,
                    'split':'development','warmupIDs':[1],
                    'positions':[{'position':1,'inputID':2,'nextTokenID':3}]}]}
            shard['manifestSHA256']=tokenize_tool.foundation_canonical_hash(shard,'manifestSHA256')
            path=self.root/f'{ordinal}.json';path.write_text(json.dumps(shard))
            self.assertEqual(tokenize_tool._validate_shard(path,identity,source_hash)[1],1)

    def test_new_schemas_declare_every_required_property(self):
        for name in ('authored-corpus-v1.json', 'capture-v2.json', 'tokenized-corpus-v1.json'):
            schema = json.loads((HERE / 'schemas' / name).read_text())
            self.assertLessEqual(set(schema['required']), set(schema['properties']))
        capture = json.loads((HERE / 'schemas/capture-v2.json').read_text())
        document = capture['properties']['documents']['items']
        self.assertLessEqual(set(document['required']), set(document['properties']))

    def test_flash_tool_path_does_not_shadow_stdlib_tokenize(self):
        source = ("import sys;sys.path.insert(0,'Tools/flash');"
                  "import linecache,tokenize;"
                  "assert 'Tools/flash/tokenize.py' not in str(tokenize.__file__);"
                  "assert linecache.getlines('Tools/flash/common.py')")
        result = subprocess.run([sys.executable, '-c', source], cwd=HERE.parents[1],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def fake_corpus(self):
        evidence = self.root / 'evidence'; native = evidence / 'corpus'; shards = native / 'shards'
        shards.mkdir(parents=True)
        source = {'format': 'slotstream-authored-corpus-v1', 'schemaVersion': 1,
                  'manifestSHA256': '', 'documents': [{
                      'id': 'doc', 'kind': 'raw', 'category': 'prose', 'split': 'development',
                      'sourceID': 'source/id', 'license': 'CC0', 'partitionKey': 'doc-v1',
                      'scoreTokens': 1, 'text': 'hello world', 'textSHA256': hashlib.sha256(b'hello world').hexdigest(),
                      'warmupTokens': 1}]}
        source['manifestSHA256'] = tokenize_tool.canonical_hash(source, 'manifestSHA256')
        source_path = self.root / 'source.json'; source_path.write_text(json.dumps(source))
        identity = {'model_revision': 'revision'}
        capture = {'id': 'doc', 'category': 'prose', 'sourceID': 'source/id',
                   'sourceHash': source['documents'][0]['textSHA256'], 'split': 'development',
                   'warmupIDs': [1], 'positions': [{'position': 1, 'inputID': 2, 'nextTokenID': 3}]}
        shard = {'format': 'slotstream-tokenized-capture-shard-v2', 'schemaVersion': 2,
                 'manifestSHA256': '', 'tokenSource': 'slotstream-auto-tokenizer-chat-template-v1',
                 'split': 'development', 'sourceCorpusSHA256': tokenize_tool.sha256(source_path),
                 'tokenizerIdentity': identity, 'shardID': 'shard-000', 'documents': [capture]}
        shard['manifestSHA256'] = tokenize_tool.foundation_canonical_hash(shard, 'manifestSHA256')
        shard_path = shards / 'shard-000.json'; shard_path.write_text(json.dumps(shard))
        vector_value = {'camelKey': 'a/b\n"\\', 'snake_key': 'é e\u0301'}
        vector_text = json.dumps(vector_value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        binary = self.root / 'bin/slotstream'; binary.parent.mkdir(); binary.write_bytes(b'binary')
        build_identity = self.root / 'identity'; build_identity.write_bytes(b'identity')
        source_archive = self.root / 'source.tar'; source_archive.write_bytes(b'source')
        index = {'format': 'slotstream-tokenized-corpus-v1', 'schema_version': 1,
                 'qualification': False, 'model_loaded': False, 'tokenizer_only': True,
                 'source_manifest_path': str(source_path), 'source_manifest_sha256': tokenize_tool.sha256(source_path),
                 'source_provenance_scope': 'unverified', 'tokenizer_identity': identity,
                 'executable_identity': {'path': str(binary), 'binary_sha256': 'bin',
                     'metallib_sha256': 'metal', 'build_identity_sha256': tokenize_tool.sha256(build_identity),
                     'source_archive_sha256': tokenize_tool.sha256(source_archive), 'historical': False},
                 'canonical_test_vector': {'value': vector_value, 'canonical_json': vector_text,
                     'sha256': hashlib.sha256(vector_text.encode()).hexdigest()},
                 'documents': [{'id': 'doc', 'kind': 'raw', 'category': 'prose', 'split': 'development',
                     'sourceID': 'source/id', 'license': 'CC0', 'partitionKey': 'doc-v1',
                     'source_hash': source['documents'][0]['textSHA256'], 'full_ids': [1,2,3],
                     'full_count': 3, 'warmup_count': 1, 'scored_range': [2,3],
                     'unscored_range': [3,3], 'capture': capture}],
                 'shards': [{'path': 'shards/shard-000.json', 'split': 'development',
                     'positions': 1, 'sha256': tokenize_tool.sha256(shard_path)}],
                 'artifacts': {'shards/shard-000.json': {'bytes': shard_path.stat().st_size,
                     'sha256': tokenize_tool.sha256(shard_path)}}}
        index_path = native / 'corpus.json'; index_path.write_text(json.dumps(index))
        (native / 'completion.json').write_text(json.dumps({'format': 'slotstream-tokenized-corpus-completion-v1',
            'corpus_sha256': tokenize_tool.sha256(index_path), 'qualification': False}))
        receipt = {'result': {'functional_success': True}, 'memory': {'target_gb_decimal': 1},
                   'command': [str(binary), 'flash-tokenize', '--model', str((self.root/'model').resolve())]}
        build = {'binary_sha256': 'bin', 'metallib_sha256': 'metal'}
        paths = {'identity': build_identity, 'source_archive': source_archive}
        return evidence, index_path, shard_path, identity, receipt, build, paths

    def test_freeze_refuses_existing_outputs_without_mutation(self):
        existing=self.root/'existing';existing.mkdir();marker=existing/'marker';marker.write_text('keep')
        before={path.name:path.read_bytes() for path in existing.iterdir()}
        self.assertEqual(tokenize_tool.freeze(Path('/missing/binary'),Path('/missing/model'),self.fixture,existing),1)
        self.assertEqual(before,{path.name:path.read_bytes() for path in existing.iterdir()})
        outside=Path(tempfile.gettempdir())/f'flash-existing-{id(self)}';outside.mkdir();outside_marker=outside/'marker';outside_marker.write_text('keep')
        try:
            snapshot=outside_marker.read_bytes()
            self.assertEqual(tokenize_tool.freeze(Path('/missing/binary'),Path('/missing/model'),self.fixture,outside),1)
            self.assertEqual(outside_marker.read_bytes(),snapshot)
            self.assertEqual(sorted(path.name for path in outside.iterdir()),['marker'])
        finally:
            outside_marker.unlink();outside.rmdir()

    def test_validate_corpus_joins_completion_shards_tokenizer_and_receipt(self):
        evidence,index_path,shard_path,identity,receipt,build,paths=self.fake_corpus()
        with mock.patch.object(tokenize_tool,'expected_tokenizer_identity',return_value=identity), \
             mock.patch.object(tokenize_tool,'validate_build_identity',return_value=(build,paths)), \
             mock.patch.object(tokenize_tool,'validate_receipt_file',return_value=receipt), \
             mock.patch.object(tokenize_tool,'require_terminal_sampling'):
            self.assertEqual(tokenize_tool.validate_corpus(evidence,model=self.root/'model')['positions'],1)
            completion=evidence/'corpus/completion.json';original_completion=completion.read_text()
            completion.write_text(json.dumps({'format':'slotstream-tokenized-corpus-completion-v1','corpus_sha256':'0'*64,'qualification':False}))
            with self.assertRaisesRegex(EvidenceError,'completion'):
                tokenize_tool.validate_corpus(evidence,model=self.root/'model')
            completion.write_text(original_completion)
            original_shard=shard_path.read_text();shard_path.write_text(original_shard+'\n')
            with self.assertRaisesRegex(EvidenceError,'order or hash'):
                tokenize_tool.validate_corpus(evidence,model=self.root/'model')
            shard_path.write_text(original_shard)
            index=json.loads(index_path.read_text());index['tokenizer_identity']={'foreign':True};index_path.write_text(json.dumps(index))
            (evidence/'corpus/completion.json').write_text(json.dumps({'format':'slotstream-tokenized-corpus-completion-v1','corpus_sha256':tokenize_tool.sha256(index_path),'qualification':False}))
            with self.assertRaisesRegex(EvidenceError,'tokenizer identity'):
                tokenize_tool.validate_corpus(evidence,model=self.root/'model')

    def test_validate_corpus_rejects_missing_reordered_shards_and_bad_receipt(self):
        evidence,index_path,shard_path,identity,receipt,build,paths=self.fake_corpus()
        def validate(current_receipt=receipt):
            with mock.patch.object(tokenize_tool,'expected_tokenizer_identity',return_value=identity), \
                 mock.patch.object(tokenize_tool,'validate_build_identity',return_value=(build,paths)), \
                 mock.patch.object(tokenize_tool,'validate_receipt_file',return_value=current_receipt), \
                 mock.patch.object(tokenize_tool,'require_terminal_sampling'):
                return tokenize_tool.validate_corpus(evidence,model=self.root/'model')
        original=shard_path.read_bytes();shard_path.unlink()
        with self.assertRaises(Exception):validate()
        shard_path.write_bytes(original)
        index=json.loads(index_path.read_text());index['shards'][0]['path']='shards/shard-001.json';index_path.write_text(json.dumps(index))
        completion=evidence/'corpus/completion.json';completion.write_text(json.dumps({'format':'slotstream-tokenized-corpus-completion-v1','corpus_sha256':tokenize_tool.sha256(index_path),'qualification':False}))
        with self.assertRaisesRegex(EvidenceError,'order or hash'):validate()
        index['shards'][0]['path']='shards/shard-000.json';index_path.write_text(json.dumps(index));completion.write_text(json.dumps({'format':'slotstream-tokenized-corpus-completion-v1','corpus_sha256':tokenize_tool.sha256(index_path),'qualification':False}))
        bad=copy.deepcopy(receipt);bad['memory']['target_gb_decimal']=2
        with self.assertRaisesRegex(EvidenceError,'launcher receipt'):validate(bad)

    def test_archived_corpus_survives_candidate_drift_but_not_archive_mutation(self):
        evidence,index_path,_,identity,launcher,build,paths=self.fake_corpus()
        index=json.loads(index_path.read_text());binary=Path(index['executable_identity']['path'])
        archive_path=binary.parent.parent/'receipt.json';archive_path.write_text('{}')
        index['executable_identity'].update({'historical':True,'archive_receipt_path':str(archive_path),
            'archive_receipt_sha256':tokenize_tool.sha256(archive_path)})
        index_path.write_text(json.dumps(index));completion=evidence/'corpus/completion.json'
        completion.write_text(json.dumps({'format':'slotstream-tokenized-corpus-completion-v1','corpus_sha256':tokenize_tool.sha256(index_path),'qualification':False}))
        original_binary=tokenize_tool.sha256(binary)
        archive={'artifacts':{'bin/slotstream':{'sha256':original_binary}}}
        def build_validator(path,historical=False):
            self.assertTrue(historical)
            if tokenize_tool.sha256(path)!=original_binary:raise EvidenceError('archive changed')
            return build,paths
        def receipt_validator(path,require_qualified=False):
            return archive if Path(path)==archive_path else launcher
        with mock.patch.object(tokenize_tool,'expected_tokenizer_identity',return_value=identity), \
             mock.patch.object(tokenize_tool,'validate_build_identity',side_effect=build_validator), \
             mock.patch.object(tokenize_tool,'validate_receipt_file',side_effect=receipt_validator), \
             mock.patch.object(tokenize_tool,'require_terminal_sampling'):
            (self.root/'candidate-source.swift').write_text('drift')
            self.assertEqual(tokenize_tool.validate_corpus(evidence,model=self.root/'model')['positions'],1)
            binary.write_bytes(b'mutated')
            with self.assertRaisesRegex(EvidenceError,'archive changed'):
                tokenize_tool.validate_corpus(evidence,model=self.root/'model')


if __name__ == "__main__":
    unittest.main(verbosity=2)
