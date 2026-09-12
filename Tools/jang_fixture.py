#!/usr/bin/env python3
"""Create a fresh sparse test fixture; never a usable language-model checkpoint."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import struct

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'jang'


def create(tier, output):
    # Refuse existing directories, particularly a user's real checkpoint.
    output.mkdir(parents=True, exist_ok=False)
    headers = json.loads(gzip.decompress((FIXTURES / ('headers-' + tier + '.json.gz')).read_bytes()))
    shutil.copyfile(FIXTURES / ('config-' + tier + '.json'), output / 'config.json')
    lookup = {}
    for name, header in headers.items():
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError('unsafe fixture filename')
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        with (output / name).open('wb') as stream:
            stream.write(struct.pack('<Q', len(raw)))
            stream.write(raw)
            stream.truncate(8 + len(raw) + end)
        for key, value in header.items():
            if key != '__metadata__':
                lookup[key] = (name, value, 8 + len(raw))
    # Distinct records catch layer/expert/stride mixups. The rest is sparse zeros.
    for layer, expert in [(0, 0), (0, 1), (1, 0)]:
        for pi, projection in enumerate(['gate_proj', 'up_proj', 'down_proj']):
            for piece in ['weight', 'scales', 'biases']:
                key = f'language_model.layers.{layer}.mlp.switch_mlp.{projection}.{piece}'
                name, tensor, base = lookup[key]
                start, end = tensor['data_offsets']
                size = (end - start) // tensor['shape'][0]
                if piece == 'weight':
                    pattern = bytes((i * 13 + 17 * expert + 31 * layer + 7 * pi) % 256 for i in range(256))
                    data = (pattern * ((size + 255) // 256))[:size]
                else:
                    data = struct.pack('<e', 0.015625 if piece == 'scales' else -0.25) * (size // 2)
                with (output / name).open('r+b') as stream:
                    stream.seek(base + start + expert * size)
                    stream.write(data)
    (output / 'NOT_MODEL_WEIGHTS.txt').write_text(
        'Synthetic sparse test fixture with original checkpoint headers.\n'
        'Use only with slotstream jang-check --model; never run generation.\n')
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=['4M', '6S'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    create(args.tier, args.output)


if __name__ == '__main__':
    main()
