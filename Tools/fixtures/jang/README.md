# JANG test evidence

The configuration files and small packed weight-row samples come from the
public JANGQ-AI Qwen3.8-Flash-Next repositories, under the accompanying
Qwen Community License. They are test data, not complete model weights.

- JANG_4M: `94ce0e3e868dc8c54c9a0eeca435bea3889d9d0d`
- JANG_6S: `3781190c6bbdf0a7637beda49ba179822612058a`

`rows.json` records the repository, immutable revision, tensor name, shape,
packing, scale type and exact hexadecimal payload for each sample. Samples
were fetched with bounded HTTP ranges from the original safetensors files.
The numerical diagnostic compares the production CPU decoder and integer
code widening against native MLX, preserving the original FP16 scale/bias values, including exact expansion
to FP32 for the expert cache.

These fixtures do not establish full-model accuracy, KL divergence,
performance or physical-memory qualification.

`headers-*.json.gz` retain the original tensor names, shapes, dtypes and offsets
for every indexed shard at those revisions. `Tools/jang_fixture.py` uses them
to construct sparse files with deterministic synthetic expert payloads. The
output is suitable only for the bounded streaming diagnostic, not generation.
It refuses to overwrite an existing directory.
