"""Run mlx-whisper (word timestamps) on the time-corrected vocal stem.

Writes work/whisper_<tag>.json. Used as an independent cross-check for the
CTC forced alignment in align.py.
"""
import common  # noqa: F401  (sets cache dirs)
import json, sys
try:
    import mlx_whisper  # macOS
except ImportError:  # Linux/CUDA (the GPU host): transformers Whisper, same output shape
    mlx_whisper = None

MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large": "mlx-community/whisper-large-v3-mlx",
}

def run_hf(tag, model, prompt=None):
    """transformers Whisper with word timestamps -> the mlx_whisper result shape (segments[].words[])."""
    import librosa, torch
    from transformers import pipeline
    hf = {"mlx-community/whisper-large-v3-turbo": "openai/whisper-large-v3",   # turbo hallucinates intros
          "mlx-community/whisper-large-v3-mlx": "openai/whisper-large-v3"}[model]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"   # tokens: CPU on the GPU host when the GPU lease is busy
    asr = pipeline("automatic-speech-recognition", model=hf, dtype=torch.float16 if dev != "cpu" else torch.float32, device=dev)
    y, _ = librosa.load(str(common.WORK / "vocals16k.wav"), sr=16000)
    kw = {"language": "en", "task": "transcribe"}
    if prompt:
        kw["prompt_ids"] = asr.tokenizer.get_prompt_ids(prompt, return_tensors="pt").to(dev)
    r = asr(y, return_timestamps="word", chunk_length_s=30, batch_size=1, generate_kwargs=kw)
    words = [{"word": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1] or c["timestamp"][0] + 0.2}
             for c in r["chunks"] if c["timestamp"][0] is not None]
    res = {"text": r["text"], "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                                            "text": r["text"], "words": words}]}
    out = common.WORK / f"whisper_{tag}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    return res


def run(tag, model, prompt=None):
    if mlx_whisper is None:
        return run_hf(tag, model, prompt)
    res = mlx_whisper.transcribe(
        str(common.WORK / "vocals16k.wav"), path_or_hf_repo=model, language="en",
        word_timestamps=True, condition_on_previous_text=False, initial_prompt=prompt,
        temperature=0.0, no_speech_threshold=None, hallucination_silence_threshold=None,
    )
    out = common.WORK / f"whisper_{tag}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    for seg in res["segments"]:
        print(f"{seg['start']:7.2f} {seg['end']:7.2f} {seg['text']}")
    return res

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "turbo"
    lyr = " ".join(t for _, _, t in common.load_lyrics_src())
    prompt = {"pdoom": ("Song lyrics about AI doom: P(doom), FOOM, shoggoth, shinigami, Chinchilla, "
                        "basilisk, Omega Point, RLHF, GPU, CDR, MLP, NVDA, Gato, Sydney, Ilya, Loom."),
              "roofline": ("Song lyrics about CUDA programming: PyTorch, cudaMemcpy, GPU, roofline, MNIST, "
                           "cuBLAS, KV cache, coalesce, float4, teraflops, tensor cores, MMA sync, "
                           "NormalFloat4, NVLink, all-reduce, micro-batch, CUTLASS."),
              "token": ("Drum and bass song lyrics about how an LLM writes: tokens, strawberry, vector, "
                        "attention, feed-forward, softmax, temperature, KV cache, roofline, gigs."),
              "tokens": ("Motown song lyrics about tokens: strawberry, r's, AI, chunk of text, numbers, letters, "
                         "E and R, a hundred twenty-eight thousand, seven three seven oh oh, S-T-R, A-W, berry.")}[common.SONG]
    run(which, MODELS[which])
    run(which + "_prompt", MODELS[which], prompt)
