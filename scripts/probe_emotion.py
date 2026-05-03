"""Quick emotion2vec probe — measure emotion signal of given audio files."""
import sys
from pathlib import Path
from funasr import AutoModel

def main():
    files = sys.argv[1:]
    em = AutoModel(model="iic/emotion2vec_plus_base", hub="hf", disable_update=True)
    print(f"{'file':<60} {'top':<10} {'p':<6} {'2nd':<10} {'p2'}")
    print("-" * 100)
    for f in files:
        out = em.generate(f, granularity="utterance", extract_embedding=False)[0]
        # out has 'labels' (list of str) and 'scores' (list of float)
        labels = out["labels"]
        scores = out["scores"]
        pairs = sorted(zip(labels, scores), key=lambda x: -x[1])
        top, p = pairs[0]
        sec, p2 = pairs[1]
        print(f"{Path(f).name:<60} {top:<10} {p:.3f}  {sec:<10} {p2:.3f}")

if __name__ == "__main__":
    main()
