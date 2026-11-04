"""Move the traps results off a machine that can only send text.

    python -m yieldengine.studies.pack results/traps/*.json > chunks.txt   # on the GPU box
    python -m yieldengine.studies.pack --unpack chunks.txt results/traps    # in the cloud

Bundles the per-model JSON files, rounds floats to 4 places, lzma + base64, and prints
6000-character chunks, each with its own sha256 prefix so a damaged chunk is caught.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

N = 6000


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def pack(paths):
    bundle = {Path(p).stem: _round(json.loads(Path(p).read_text())) for p in paths}
    raw = json.dumps(bundle, separators=(",", ":")).encode()
    b = base64.b64encode(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)).decode()
    parts = [b[i:i + N] for i in range(0, len(b), N)]
    print(f"TOTAL raw={len(raw)} b64={len(b)} chunks={len(parts)} sha256={hashlib.sha256(raw).hexdigest()[:16]}")
    for i, p in enumerate(parts, 1):
        print(f"CHUNK {i}/{len(parts)} sha8={hashlib.sha256(p.encode()).hexdigest()[:8]}")
        print(p)


def unpack(text_path, out_dir):
    text = Path(text_path).read_text()
    total = re.search(r"TOTAL raw=(\d+) b64=(\d+) chunks=(\d+) sha256=([0-9a-f]+)", text)
    n = int(total.group(3))
    chunks, bad = {}, []
    for m in re.finditer(r"CHUNK (\d+)/(\d+) sha8=([0-9a-f]{8})\s+(?:```\s*)?([A-Za-z0-9+/=\s]+?)(?=CHUNK |\Z|```)", text):
        i, sha, body = int(m.group(1)), m.group(3), re.sub(r"\s", "", m.group(4))
        if hashlib.sha256(body.encode()).hexdigest()[:8] == sha:
            chunks[i] = body
        elif i not in chunks:
            bad.append(i)
    missing = [i for i in range(1, n + 1) if i not in chunks]
    if missing:
        sys.exit(f"missing or corrupt chunks: {missing} (bad sha: {sorted(set(bad))})")
    raw = lzma.decompress(base64.b64decode("".join(chunks[i] for i in range(1, n + 1))))
    assert hashlib.sha256(raw).hexdigest()[:16] == total.group(4), "whole-payload hash mismatch"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, res in json.loads(raw).items():
        (out / f"{name}.json").write_text(json.dumps(res))
        print("wrote", out / f"{name}.json")


if __name__ == "__main__":
    if sys.argv[1] == "--unpack":
        unpack(sys.argv[2], sys.argv[3])
    else:
        pack(sys.argv[1:])
