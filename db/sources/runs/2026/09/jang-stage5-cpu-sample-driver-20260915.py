"""One owned-process CPU sample, isolated from all qualification timings."""
import argparse
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'Tools/flash'))
import benchmark
import engine_bench
import widen_study
import thermal_readiness
from common import atomic_json, fresh_output, harness_hashes, read_json, sha256
from observe import DarwinSampler
from receipts import require_terminal_sampling, validate_receipt_file


def run(protocol_path, comparison_path, output):
    protocol = read_json(protocol_path)
    prompt = next(p for p in widen_study.cache_study.load_fixture()['prompts'] if p['id'] == 'python-deduplicate')
    output = fresh_output(output)
    binary, _ = widen_study._archive_candidate(ROOT / '.build/release/slotstream', output)
    before_harness = harness_hashes()
    model = Path(protocol['model'])
    reference = read_json(comparison_path / 'stats.json')
    reference_receipt = validate_receipt_file(comparison_path / 'receipt.json')
    finished = threading.Event()
    collectors = []
    sample_result = {}

    def sampler_factory(pid):
        sampler = DarwinSampler(pid)
        identity = sampler.start_identity

        def collect():
            try:
                if finished.wait(5):
                    sample_result.update(status='request-ended-before-sampling')
                    return
                check = DarwinSampler(pid)
                assert check.start_identity == identity, 'owned process identity changed'
                check.sample()
                command = ['/usr/bin/sample', str(pid), '5', '10', '-mayDie', '-file', str(output / 'cpu-sample.txt')]
                started = time.time()
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                (output / 'sample.stdout.txt').write_text(result.stdout)
                (output / 'sample.stderr.txt').write_text(result.stderr)
                sample_result.update(status='complete' if result.returncode == 0 else 'failed',
                    command=command, pid=pid, process_start_identity=identity, requested_delay_seconds=5,
                    requested_duration_seconds=5, sampling_interval_ms=10, started_unix=started,
                    elapsed_seconds=time.time()-started, exit_code=result.returncode)
            except Exception as error:
                sample_result.update(status='failed', error=f'{type(error).__name__}: {error}')

        collector = threading.Thread(target=collect)
        collector.start()
        collectors.append(collector)
        return sampler

    try:
        readiness = thermal_readiness.observe()
        assert readiness['conditions']['thermalState'] in ('nominal', 'fair')
        assert readiness['conditions']['lowPowerModeEnabled'] is False
        settling = benchmark.settle_before_model_launch()
        arm = output / 'instrumented-request'
        command = [str(binary), 'run', '--model', str(model), '--memory-gb', '14', '--max-context', '2048',
            '--mtp', 'off', '--vision', 'off', '--prompt', prompt['text'], '--max-tokens', '128',
            '--greedy', '--seed', '7', '--expert-widening', 'packed4-to6', '--sample-footprint',
            '--stats-json', str(arm / 'stats.json')]
        code = benchmark.launch(arm, 14, 900, command, sampler_factory=sampler_factory,
            run_set_id='jang-stage5-cpu-sample-v1', model_hash=sha256(model / 'config.json'))
        finished.set()
        for collector in collectors: collector.join(timeout=25)
        assert not any(t.is_alive() for t in collectors), 'owned collector did not stop'
        atomic_json(output / 'sample-result.json', sample_result)
        atomic_json(output / 'readiness.json', readiness)
        atomic_json(output / 'settling.json', settling)
        assert code == 0
        receipt = validate_receipt_file(arm / 'receipt.json')
        require_terminal_sampling(receipt)
        assert receipt['result']['functional_success'] and receipt['memory']['qualified']
        document = read_json(arm / 'stats.json')
        widen_study._exact_work(reference, document)
        widen_study.compare_runtime_controls(reference, reference_receipt, document, receipt)
        assert sample_result['status'] == 'complete'
        sample = output / 'cpu-sample.txt'
        assert 0 < sample.stat().st_size <= 16 << 20
        assert harness_hashes() == before_harness
        atomic_json(output / 'report.json', {'format': 'jang-stage5-cpu-sample-v1', 'complete': True,
            'qualification': False, 'timing_eligible': False, 'exact_output_and_work_vs_untraced': True,
            'driver_sha256': sha256(Path(__file__)), 'profile_input_protocol_sha256': sha256(protocol_path),
            'fixture_sha256': sha256(widen_study.FIXTURE), 'prompt_id': prompt['id'], 'operating_conditions_before': readiness,
            'comparison_stats_sha256': sha256(comparison_path / 'stats.json'),
            'request_receipt_sha256': sha256(arm / 'receipt.json'), 'sample_sha256': sha256(sample),
            'sample_result': sample_result, 'harness_hashes': before_harness,
            'limits': 'CPU thread stack samples from an early short-prompt request interval under the recorded nominal or fair regime; not GPU kernel durations, utilization, or whole-request CPU percentages. Instrumented timings cannot support performance claims.'})
        atomic_json(output / 'completion.json', {'complete': True, 'report_sha256': sha256(output / 'report.json')})
    except Exception as error:
        atomic_json(output / 'failure.json', {'complete': False, 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        finished.set()
        for collector in collectors: collector.join(timeout=25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(strict=True), args.comparison.resolve(strict=True), args.output)
