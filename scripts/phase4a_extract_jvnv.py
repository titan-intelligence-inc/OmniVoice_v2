"""Extract one sample per (speaker, emotion) from JVNV.

Output: /workspace/OmniVoice_v2/baseline/jvnv_samples_multi/{speaker}_{emotion}.wav
"""
import io
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

OUT = Path("/workspace/OmniVoice_v2/baseline/jvnv_samples_multi")
OUT.mkdir(parents=True, exist_ok=True)

SPEAKERS = ["F1", "F2", "M1", "M2"]
EMOTIONS = ["anger", "sad", "fear"]   # emotion2vec confidently detects these


def main():
    found = {}
    for shard in range(5):
        path = hf_hub_download(
            repo_id="asahi417/jvnv-emotional-speech-corpus",
            filename=f"data/test-0000{shard}-of-00005.parquet",
            repo_type="dataset",
        )
        df = pq.read_table(path).to_pandas()
        for sp in SPEAKERS:
            for emo in EMOTIONS:
                key = (sp, emo)
                if key in found:
                    continue
                rows = df[(df["speaker_id"] == sp) & (df["style"] == emo)]
                for _, row in rows.iterrows():
                    audio = row["audio"]
                    wav, sr = sf.read(io.BytesIO(audio["bytes"]))
                    duration = len(wav) / sr
                    if 3.0 <= duration <= 10.0:
                        outp = OUT / f"jvnv_{sp}_{emo}.wav"
                        sf.write(outp, wav, sr)
                        found[key] = (str(outp), duration, sr)
                        break

    for k in sorted(found):
        print(k, found[k])
    missing = [(sp, emo) for sp in SPEAKERS for emo in EMOTIONS if (sp, emo) not in found]
    if missing:
        print("MISSING:", missing)


if __name__ == "__main__":
    main()
