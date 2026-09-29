"""Fill the sung lyrics from the training run: the chorus excerpts are the model's own writing, verbatim.

  python3 song/make_sing.py <run.json> <template> <out lyrics.sing.txt> <out excerpts.json>

Each chorus ends with the start of what the model wrote after "ROMEO:" at that chorus's checkpoint (CHORUS_STEPS,
the same as app/src/clock.ts): from its first character, whole words, at least MIN and at most MAX characters,
stopping at a blank line once MIN is reached. It is sung as two lines of about equal length (the page shows
the newlines). Also fills
{N} (the run's wall-clock minutes, rounded), {LT1} (training loss at the last step, one decimal, in words),
{LVMIN} (the lowest validation loss) and {VSTEP} (the step it was reached at, in words). excerpts.json records the character spans for the page highlight and the exact numbers.
"""
import json, sys

CHORUS_STEPS = [100, 500, 5000]
MIN, MAX = 26, 50
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def words_num(x: float) -> str:
    """4.3 -> 'four point three' (x rounded to one decimal)."""
    a, b = f"{x:.1f}".split(".")
    return f"{ONES[int(a)]} point {ONES[int(b)]}"


def thousands(n: int) -> str:
    """2000 -> 'two thousand' (whole thousands only; otherwise the digits)."""
    return f"{ONES[n // 1000]} thousand" if n % 1000 == 0 and n // 1000 <= 20 else str(n)


def excerpt(text: str):
    a = 0
    while a < len(text) and text[a] in "\n ":
        a += 1
    b = a
    i = a
    while i < len(text):
        if text[i] in " \n" or i == len(text) - 1:
            end = i if text[i] in " \n" else i + 1
            if end - a > MAX:
                break
            b = end
            if end - a >= MIN and text[end:end + 2] == "\n\n":
                break
        i += 1
    if b == a:  # one long run with no spaces: cut hard
        b = a + MAX
    return a, b, text[a:b]


def two_lines(s: str):
    """The excerpt's words (newlines are just breaks here; the page shows them) in two sung lines of about equal length."""
    w = s.split()
    best = min(range(1, len(w)), key=lambda k: abs(len(" ".join(w[:k])) - len(" ".join(w[k:])))) if len(w) > 1 else 1
    return [" ".join(w[:best]), " ".join(w[best:])] if len(w) > 1 else [" ".join(w)]


def main(run_json, tmpl, out_txt, out_json):
    r = json.load(open(run_json))
    samples = {s["step"]: s for s in r["samples"]}
    steps = sorted(samples)
    ex = []
    for want in CHORUS_STEPS:
        st = max(s for s in steps if s <= want)
        a, b, s = excerpt(samples[st]["text"])
        ex.append(dict(step=st, a=a, b=b, text=s, lines=two_lines(s)))
    N = round(r["wall_s"] / 60)
    L0, L1 = r["evals"][0]["val"], r["evals"][-1]["val"]
    LT1 = r["evals"][-1]["train"]
    best = min(r["evals"], key=lambda e: e["val"])
    t = open(tmpl).read()
    t = t.replace("{N} minutes ago", f"{ONES[N].capitalize()} minutes ago").replace("{N} minutes on", f"{ONES[N].capitalize()} minutes on")
    t = t.replace("{LT1}", words_num(LT1).replace("zero point", "point")).replace("{LVMIN}", words_num(best["val"]))
    t = t.replace("{VSTEP}", thousands(best["step"]))
    for k, e in enumerate(ex):
        t = t.replace("{EX%d}" % (k + 1), "\n".join(e["lines"]))
    open(out_txt, "w").write(t)
    json.dump(dict(minutes=N, wall_s=r["wall_s"], loss0=L0, loss1=L1, train1=LT1, val_min=best["val"], val_min_step=best["step"],
                   excerpts=ex), open(out_json, "w"), indent=1)
    print(t)
    print(json.dumps(ex, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
