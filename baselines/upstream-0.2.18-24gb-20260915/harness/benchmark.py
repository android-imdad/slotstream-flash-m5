#!/usr/bin/env python3
"""Benchmark an unmodified upstream Slotstream server; preserve raw responses."""
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'upstream'
BIN = REPO / '.build/release/slotstream'
MODEL = ROOT / 'models/qwen38-flash-next-mlx-4bit'
OUT = ROOT / 'results' / datetime.datetime.now().strftime('benchmark-%Y%m%d-%H%M%S')
PORT = 18086
MEMORY = 24
MODEL_NAME = 'qwen3.8-flash-next:4bit'
LATEST_POINTER = 'latest-benchmark.txt'
PROMPTS = {
    'explanation': 'Explain how a database index speeds up queries. Include a concrete example, the difference between a table scan and an index lookup, and the costs of maintaining indexes. Write about 400 words.',
    'coding': 'Write a Python function that merges two sorted lists of integers into one sorted list without using sort or sorted. Include type hints, a docstring, examples with duplicates and empty lists, and an explanation of time and space complexity.',
    'reasoning': 'A warehouse has 120 boxes. On Monday it ships one quarter of them and receives 18 new boxes. On Tuesday it ships one third of the boxes then present and receives 12 new boxes. Explain each calculation, find the final number of boxes, and check the answer by accounting for all arrivals and departures.',
}

def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def vm():
    raw = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+)', raw)[1])
    values = {k.strip(): int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)', raw, re.M)}
    return {'raw': raw, 'reclaimable_gb': page * sum(values.get(k, 0) for k in ['Pages free', 'Pages purgeable', 'File-backed pages']) / 1e9,
            'swapins': values['Swapins'], 'swapouts': values['Swapouts']}

def competing_models():
    processes = subprocess.check_output(['ps', '-axo', 'pid=,args='], text=True)
    patterns = [r'\bedge0\s+serve\b', r'\bllama-server\b', r'\bmlx_lm[. ]server\b', r'\bmlx_vlm[. ]server\b', r'\bslotstream\s+(serve|run|.*-check)\b']
    return [int(line.strip().split(None, 1)[0]) for line in processes.splitlines()
            if any(re.search(p, line) for p in patterns)]

def request(label, prompt, warmup=False):
    before = vm()
    body = {'model': MODEL_NAME, 'messages': [{'role': 'user', 'content': prompt}],
            'think': False, 'stream': True, 'options': {'temperature': 0, 'seed': 42, 'num_predict': 128}}
    conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=600)
    start = time.monotonic()
    first = None
    frames = []
    try:
        conn.request('POST', '/api/chat', json.dumps(body), {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status}: {resp.read().decode()}')
        with (OUT / (label + '.ndjson')).open('wb') as raw:
            for line in resp:
                raw.write(line)
                raw.flush()
                item = json.loads(line)
                frames.append(item)
                if 'error' in item:
                    raise RuntimeError(item['error'])
                msg = item.get('message', {})
                if first is None and (msg.get('content') or msg.get('thinking')):
                    first = time.monotonic() - start
    finally:
        conn.close()
    elapsed = time.monotonic() - start
    after = vm()
    final = frames[-1]
    if not final.get('done') or not final.get('eval_count') or not final.get('slotstream_benchmark'):
        raise RuntimeError('Incomplete generation or missing instrumentation')
    details = final['slotstream_benchmark']
    row = {'label': label, 'warmup': warmup, 'request': body, 'response': final,
           'wall_seconds': elapsed, 'first_visible_token_seconds': first,
           'decode_tps': final['eval_count'] / (final['eval_duration'] / 1e9),
           'prefill_tps': final['prompt_eval_count'] / (final['prompt_eval_duration'] / 1e9) if final['prompt_eval_duration'] else None,
           'text': ''.join(f.get('message', {}).get('content', '') for f in frames),
           'before_vm': before, 'after_vm': after,
           'swap_clean': before['swapins'] == after['swapins'] and before['swapouts'] == after['swapouts']}
    write(label + '.json', row)
    print(json.dumps({k: row[k] for k in ['label', 'decode_tps', 'first_visible_token_seconds', 'wall_seconds', 'swap_clean']}), flush=True)
    return row

def main(prepared_output=False):
    OUT.mkdir(parents=True, exist_ok=prepared_output)
    (ROOT / 'results' / LATEST_POINTER).write_text(str(OUT) + '\n')
    before = vm()
    if competing_models():
        raise RuntimeError('Another model process is running; benchmark deferred')
    if before['reclaimable_gb'] < MEMORY + 3:
        raise RuntimeError(f'Insufficient headroom: {before["reclaimable_gb"]:.1f} GB')
    command = [str(BIN), 'serve', '--model', str(MODEL), '--memory-gb', str(MEMORY), '--max-context', '32768', '--port', str(PORT), '--no-prefix-cache']
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLOTSTREAM_')}
    env['SLOTSTREAM_BENCH_DETAILS'] = '1'
    write('manifest.json', {'command': command, 'instrumentation': {'SLOTSTREAM_BENCH_DETAILS': '1'},
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'binary_sha256': hashlib.sha256(BIN.read_bytes()).hexdigest(), 'initial_vm': before,
        'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'model': MODEL_NAME, 'memory_target_decimal_gb': MEMORY, 'rounds': 3, 'max_output_tokens': 128,
        'method': 'One fresh server, distinct warmup, three prompts in three rotated rounds. No forced disk cache purge. Physical-footprint sampling enabled. Conversation prefix cache disabled so each prompt is fully processed; other upstream optimizations retain defaults.'})
    print('Output: ' + str(OUT), flush=True)
    child = None
    rows = []
    started = time.monotonic()
    try:
        with (OUT / 'server.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=log)
            write('process.json', {'pid': child.pid})
            while True:
                if child.poll() is not None:
                    raise RuntimeError('Server exited during loading; see server.log')
                if time.monotonic() - started > 600:
                    raise TimeoutError('Server did not become ready in 10 minutes')
                conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=2)
                try:
                    conn.request('GET', '/api/tags')
                    resp = conn.getresponse()
                    if resp.status == 200:
                        write('models.json', json.loads(resp.read()))
                        break
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    conn.close()
                time.sleep(1)
            write('startup.json', {'ready_seconds': time.monotonic() - started, 'vm': vm()})
            request('warmup', 'Describe the water cycle in detail, explaining evaporation, condensation, precipitation, and the role of the sun. Write about 400 words.', True)
            names = list(PROMPTS)
            for r in range(3):
                for name in names[r:] + names[:r]:
                    rows.append(request(f'round{r + 1}-{name}', PROMPTS[name]))
            summary = {'complete': True, 'requests': len(rows), 'groups': {}}
            for name in names:
                group = [x for x in rows if x['label'].endswith(name)]
                eligible = [x for x in group if x['swap_clean']]
                summary['groups'][name] = {'eligible': len(eligible),
                    'decode_tps': [x['decode_tps'] for x in group],
                    'median_decode_tps': statistics.median(x['decode_tps'] for x in eligible) if eligible else None,
                    'median_first_visible_token_seconds': statistics.median(x['first_visible_token_seconds'] for x in eligible) if eligible else None,
                    'output_tokens': [x['response']['eval_count'] for x in group],
                    'same_output_ids': all(x['response']['slotstream_benchmark']['output_ids'] == group[0]['response']['slotstream_benchmark']['output_ids'] for x in group)}
            write('summary.json', summary)
            print(json.dumps(summary, indent=2), flush=True)
    except BaseException as e:
        write('failure.json', {'error': repr(e), 'completed_requests': len(rows)})
        raise
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        write('cleanup.json', {'server_pid': child.pid if child else None, 'server_returncode': child.returncode if child else None, 'final_vm': vm()})

if __name__ == '__main__':
    main()
