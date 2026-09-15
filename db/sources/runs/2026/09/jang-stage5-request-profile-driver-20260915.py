"""Frozen, bounded stage-five request profile; no kernel speedup claim."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'Tools/flash'))
import benchmark
import cache_study
import engine_bench
import widen_study
from common import atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import require_terminal_sampling, validate_receipt_file


def prepare(output):
    from tokenizers import Tokenizer
    import importlib.metadata
    output = fresh_output(output)
    model = (ROOT / 'models/jang-6s').resolve()
    tokenizer_path = model / 'tokenizer.json'
    assert sha256(tokenizer_path) == cache_study.JANG6S_SMALL_PINS['tokenizer.json'][1]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source = ROOT / 'Tools/fixtures/optimization/code.txt'
    ids = tokenizer.encode(source.read_text(), add_special_tokens=False).ids
    assert len(ids) >= 1024, 'source must contain enough real tokens; no padding/repetition'
    prompts = []
    for length in (256, 1024):
        selected = ids[:length]
        text = tokenizer.decode(selected, skip_special_tokens=False)
        assert tokenizer.encode(text, add_special_tokens=False).ids == selected
        file = output / f'code-{length}.txt'
        file.write_text(text)
        prompts.append({'id': f'code-{length}', 'file': file.name, 'sha256': sha256(file), 'ids': selected})
    atomic_json(output / 'protocol.json', {
        'format': 'jang-stage5-request-profile-v1', 'scope': 'raw code-prefix profiling, not task quality or custom-kernel qualification',
        'model': str(model), 'tokenizer_sha256': sha256(tokenizer_path),
        'tokenizers_version': importlib.metadata.version('tokenizers'), 'source': str(source.relative_to(ROOT)),
        'source_sha256': sha256(source), 'prompts': prompts, 'rounds': 3,
        'memory_gb': 14, 'max_context': 2048, 'max_output': 64, 'seed': 7,
        'widening': 'packed4-to6', 'mtp': False, 'vision': False,
        'limits': 'Host I/O intervals include conversion/staging and may overlap GPU work. No remainder is identified as GPU time. Short raw continuations are not scored tasks.',
        'driver_sha256': sha256(Path(__file__)), 'harness_hashes': harness_hashes()})
    print('PROFILE PREPARED', output, flush=True)


def run(protocol_path, output):
    protocol = read_json(protocol_path)
    assert protocol['driver_sha256'] == sha256(Path(__file__))
    assert protocol['harness_hashes'] == harness_hashes()
    output = fresh_output(output)
    try:
        binary, _ = widen_study._archive_candidate(ROOT / '.build/release/slotstream', output)
        identity = validate_build_identity(binary)[0]
        model = Path(protocol['model'])
        verification = cache_study.verified_model_revision(model)
        geometry = cache_study.derive_source_geometry(model, verification=verification)
        atomic_json(output / 'protocol.json', protocol)
        rows = []
        first = {}
        for round_index in range(protocol['rounds']):
            prompts = protocol['prompts'] if round_index % 2 == 0 else list(reversed(protocol['prompts']))
            for prompt in prompts:
                prompt_file = protocol_path.parent / prompt['file']
                assert sha256(prompt_file) == prompt['sha256']
                arm = output / prompt['id'] / f'round-{round_index}'
                readiness = engine_bench.wait_for_nominal()
                settling = benchmark.settle_before_model_launch()
                command = [str(binary), 'run', '--model', str(model), '--memory-gb', '14',
                    '--max-context', '2048', '--mtp', 'off', '--vision', 'off', '--raw',
                    '--prompt-file', str(prompt_file), '--max-tokens', '64', '--greedy', '--seed', '7',
                    '--expert-widening', 'packed4-to6', '--sample-footprint', '--stats-json', str(arm / 'stats.json')]
                code = benchmark.launch(arm, 14, 900, command, run_set_id='jang-stage5-request-profile-v1', model_hash=sha256(model / 'config.json'))
                atomic_json(arm / 'readiness.json', readiness)
                atomic_json(arm / 'settling.json', settling)
                receipt = validate_receipt_file(arm / 'receipt.json')
                require_terminal_sampling(receipt)
                assert code == 0 and receipt['result']['functional_success']
                document = read_json(arm / 'stats.json')
                widen_study.validate_timing_eligibility(document, receipt)
                stats = document['stats']
                assert document['prompt_ids'] == prompt['ids'], 'native tokenizer must match frozen input IDs'
                assert stats['prefillTokens'] == len(prompt['ids']) and stats['reusedPrefixTokens'] == 0
                assert stats['decodeTokens'] >= 32 and stats['finishReason'] in ('stop', 'length')
                assert document['effective_expert_widening'] == 'packed4-to6'
                assert document['effective_mtp'] is False and document['effective_vision'] is False
                assert document['plan']['target_gb'] == 14 and document['plan']['max_context_tokens'] == 2048
                assert document['effective_prefill_chunk'] == 256
                if prompt['id'] in first:
                    old, old_receipt = first[prompt['id']]
                    widen_study._exact_work(old, document)
                    widen_study.compare_runtime_controls(old, old_receipt, document, receipt)
                else:
                    first[prompt['id']] = (document, receipt)
                row = {'prompt': prompt['id'], 'round': round_index, 'path': str(arm.relative_to(output)),
                    'receipt_sha256': sha256(arm / 'receipt.json'), 'stats_sha256': sha256(arm / 'stats.json'),
                    'peak_bytes': receipt['memory']['peak_bytes'], 'load_seconds': document['load_seconds'],
                    'stats': {key: stats[key] for key in ('prefillTokens', 'decodeTokens', 'prefillSeconds', 'decodeSeconds',
                        'prefillIOSeconds', 'prefillScatterSeconds', 'prefillRecords', 'prefillReadBytes',
                        'decodeIOSeconds', 'decodeScatterSeconds', 'decodeRecords', 'decodeReadBytes', 'prefillPasses',
                        'prefillGPUWaitSeconds', 'prefillRowSortSeconds', 'firstTokenSeconds', 'requestSeconds')}}
                rows.append(row)
                atomic_json(output / 'progress.json', {'rows': rows})
                print('PROFILE COMPLETE ARM', prompt['id'], round_index, row['stats'], flush=True)
                assert validate_build_identity(binary)[0] == identity
                assert protocol['harness_hashes'] == harness_hashes()
                assert protocol['driver_sha256'] == sha256(Path(__file__))
        assert cache_study.derive_source_geometry(model, verification=verification) == geometry
        summary = {}
        for prompt in protocol['prompts']:
            selected = [row for row in rows if row['prompt'] == prompt['id']]
            assert len(selected) == 3
            summary[prompt['id']] = {key: statistics.median(row['stats'][key] for row in selected)
                for key in selected[0]['stats'] if isinstance(selected[0]['stats'][key], (int, float))}
        atomic_json(output / 'report.json', {'format': protocol['format'], 'complete': True, 'qualification': False,
            'scope': protocol['scope'], 'limits': protocol['limits'], 'source_identity': identity,
            'protocol_sha256': sha256(protocol_path), 'rows': rows, 'median_stats': summary})
        atomic_json(output / 'completion.json', {'report_sha256': sha256(output / 'report.json'), 'complete': True})
    except Exception as error:
        atomic_json(output / 'failure.json', {'complete': False, 'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.output)
    else: run(args.protocol.resolve(strict=True), args.output)
