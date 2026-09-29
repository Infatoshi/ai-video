"""Generate song candidates with ACE-Step 1.5 (run on the other GPU host from the ACE-Step checkout's venv).

  uv run --project ace python gen.py understand <audio>          # caption/BPM/key of a reference
  uv run --project ace python gen.py gen <run.json>               # a batch of candidates

run.json: {"name", "dit", "lm", "caption", "lyrics_file", "bpm", "key", "duration", "steps",
           "guidance", "shift", "seeds": [...], "batch": 4, "thinking": true, "lm_temperature",
           "reference_audio": null, "offload": false}
Writes out/<name>/<seed>.wav plus out/<name>/<seed>.json (params and the LM's chain-of-thought
metadata).
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACE = HERE / "ace"
os.environ["HF_HOME"] = str(HERE / ".hf")  # the other GPU host's shared /data/hf is not writable
os.environ["HF_HUB_CACHE"] = str(HERE / ".hf" / "hub")
os.chdir(ACE)  # ACE-Step resolves checkpoints/ from the working directory
sys.path.insert(0, str(ACE))

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import GenerationParams, GenerationConfig, generate_music, understand_music  # noqa: E402


def init(dit, lm, offload=False):
    """offload: keep models in CPU RAM and move each to the GPU only while it runs (24 GB cards)."""
    d = AceStepHandler()
    msg = d.initialize_service(project_root=str(ACE), config_path=dit, device="cuda",
                               use_flash_attention=True, offload_to_cpu=offload)
    print("dit:", str(msg)[:300], flush=True)
    l = LLMHandler()
    if lm:
        msg = l.initialize(checkpoint_dir=str(ACE / "checkpoints"), lm_model_path=lm, backend="vllm",
                           device="cuda", offload_to_cpu=offload)
        print("lm:", str(msg)[:300], flush=True)
    return d, l


def understand(audio):
    d, l = init("acestep-v15-turbo", "acestep-5Hz-lm-4B")
    codes = d.convert_src_audio_to_codes(audio)
    for i in range(3):
        r = understand_music(l, codes, temperature=0.7)
        print(json.dumps({k: getattr(r, k) for k in
                          ("caption", "bpm", "keyscale", "timesignature", "language", "duration")}, indent=1))


def gen(cfg_path):
    cfg = json.loads((HERE / cfg_path).read_text())
    out = HERE / "out" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    d, l = init(cfg["dit"], cfg.get("lm"), cfg.get("offload", False))
    lyrics = (HERE / cfg["lyrics_file"]).read_text()
    seeds = cfg["seeds"]
    b = cfg.get("batch", 4)
    for i in range(0, len(seeds), b):
        chunk = seeds[i:i + b]
        p = GenerationParams(
            caption=cfg["caption"], lyrics=lyrics, vocal_language="en",
            bpm=cfg.get("bpm"), keyscale=cfg.get("key", ""), timesignature="4",
            duration=cfg.get("duration", -1), inference_steps=cfg.get("steps", 50),
            guidance_scale=cfg.get("guidance", 7.0), shift=cfg.get("shift", 1.0),
            use_adg=cfg.get("adg", False), thinking=cfg.get("thinking", True),
            lm_temperature=cfg.get("lm_temperature", 0.85), lm_cfg_scale=cfg.get("lm_cfg", 2.0),
            reference_audio=cfg.get("reference_audio"),
        )
        c = GenerationConfig(batch_size=len(chunk), use_random_seed=False, seeds=chunk,
                             audio_format="wav", allow_lm_batch=True)
        t0 = time.time()
        r = generate_music(d, l, p, c, save_dir=str(out / "_raw"))
        if not r.success:
            print("FAILED", chunk, r.error, flush=True)
            continue
        for a in r.audios:
            seed = a["params"]["seed"]
            src = Path(a["path"])
            dst = out / f"{seed}.wav"
            src.rename(dst)
            meta = {k: v for k, v in a["params"].items() if k != "audio_codes"}
            meta["lm"] = {k: v for k, v in (r.extra_outputs.get("lm_metadata") or {}).items()
                          if isinstance(v, (str, int, float, bool, type(None)))} \
                if isinstance(r.extra_outputs.get("lm_metadata"), dict) else None
            (out / f"{seed}.json").write_text(json.dumps(meta, indent=1, default=str))
            print("wrote", dst, flush=True)
        tc = r.extra_outputs.get("time_costs", {})
        print(f"batch {chunk} {time.time() - t0:.1f}s", {k: round(v, 1) for k, v in tc.items()
                                                          if isinstance(v, (int, float))}, flush=True)


if __name__ == "__main__":
    {"understand": understand, "gen": gen}[sys.argv[1]](sys.argv[2])
    sys.stdout.flush()
    os._exit(0)  # nano-vllm's worker threads otherwise keep the process (and its GPU memory) alive
