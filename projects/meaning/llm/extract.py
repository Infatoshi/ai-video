"""Copy Llama 3.1 8B Instruct's input embedding table out of its safetensors shard, CPU only (run on the GPU host,
where the weights are in the HF cache; score/ venv for the tokenizer):

  uv run --project ~/dev/cuda-song/score python extract.py <outdir>

Writes <outdir>/emb_bf16.npy (uint16 [128256, 4096]: the raw bf16 bits, exact; fp32 = bits << 16) and
<outdir>/vocab.json (the decoded string of every token id). No GPU and no model load: the safetensors header
gives the tensor's byte range in model-00001-of-00004.safetensors, and the bytes are read as they are.
"""
import glob, hashlib, json, struct, sys
from pathlib import Path

import numpy as np

SNAP = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*"))[0]
NAME = "model.embed_tokens.weight"


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shard = json.loads(Path(SNAP, "model.safetensors.index.json").read_text())["weight_map"][NAME]
    path = Path(SNAP, shard)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr[NAME]
        assert meta["dtype"] == "BF16", meta
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        raw = f.read(b - a)
    w = np.frombuffer(raw, dtype="<u2").reshape(meta["shape"])
    np.save(out / "emb_bf16.npy", w)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    vocab = [tok.decode([i]) for i in range(w.shape[0])]
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    info = {"model": "meta-llama/Llama-3.1-8B-Instruct", "snapshot": Path(SNAP).name, "shard": shard, "tensor": NAME,
            "shape": meta["shape"], "dtype": meta["dtype"], "sha256_bytes": hashlib.sha256(raw).hexdigest()}
    (out / "emb_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main(sys.argv[1])
