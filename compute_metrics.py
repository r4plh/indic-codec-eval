import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pesq import pesq
from pystoi import stoi

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LANGUAGES = ["hindi", "tamil", "telugu", "bengali", "kannada"]
CODECS = ["encodec", "dac", "snac"]
SAMPLES_PER_LANG = 10

# Monkey-patch torchaudio for speechbrain compatibility
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

# Load speaker similarity model
print("Loading ECAPA-TDNN for speaker similarity...")
HAS_SPK_MODEL = False
try:
    from speechbrain.inference.speaker import EncoderClassifier
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    HAS_SPK_MODEL = True
    print("  ✅ SpeechBrain loaded")
except Exception as e:
    print(f"  ⚠️ SpeechBrain failed: {e}")
    print("  Skipping speaker similarity")


def compute_pesq_safe(ref, deg, sr):
    try:
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            ref_16k = resampler(torch.from_numpy(ref).unsqueeze(0)).squeeze().numpy()
            deg_16k = resampler(torch.from_numpy(deg).unsqueeze(0)).squeeze().numpy()
        else:
            ref_16k, deg_16k = ref, deg
        min_len = min(len(ref_16k), len(deg_16k))
        ref_16k = ref_16k[:min_len]
        deg_16k = deg_16k[:min_len]
        return pesq(16000, ref_16k, deg_16k, 'wb')
    except Exception as e:
        return None


def compute_stoi_safe(ref, deg, sr):
    try:
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            ref_16k = resampler(torch.from_numpy(ref).unsqueeze(0)).squeeze().numpy()
            deg_16k = resampler(torch.from_numpy(deg).unsqueeze(0)).squeeze().numpy()
        else:
            ref_16k, deg_16k = ref, deg
        min_len = min(len(ref_16k), len(deg_16k))
        ref_16k = ref_16k[:min_len]
        deg_16k = deg_16k[:min_len]
        return stoi(ref_16k, deg_16k, 16000, extended=False)
    except Exception as e:
        return None


def compute_spk_sim_safe(ref_path, deg_path):
    if not HAS_SPK_MODEL:
        return None
    try:
        emb1 = spk_model.encode_batch(spk_model.load_audio(ref_path).unsqueeze(0))
        emb2 = spk_model.encode_batch(spk_model.load_audio(deg_path).unsqueeze(0))
        sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        return sim
    except Exception as e:
        return None


# Compute all metrics
all_results = []

for codec in CODECS:
    for lang in LANGUAGES:
        for i in range(SAMPLES_PER_LANG):
            ref_path = f"data/{lang}/sample_{i}_24k.wav"
            deg_path = f"reconstructed/{codec}/{lang}/sample_{i}.wav"

            if not os.path.exists(ref_path) or not os.path.exists(deg_path):
                continue

            ref_wav, ref_sr = torchaudio.load(ref_path)
            deg_wav, deg_sr = torchaudio.load(deg_path)

            ref_np = ref_wav.squeeze().numpy()
            deg_np = deg_wav.squeeze().numpy()

            pesq_score = compute_pesq_safe(ref_np, deg_np, ref_sr)
            stoi_score = compute_stoi_safe(ref_np, deg_np, ref_sr)
            spk_sim = compute_spk_sim_safe(ref_path, deg_path)

            all_results.append({
                "codec": codec,
                "language": lang,
                "sample": i,
                "pesq": pesq_score,
                "stoi": stoi_score,
                "spk_sim": spk_sim
            })

            print(f"  {codec} | {lang} | sample_{i} | PESQ={pesq_score} | STOI={stoi_score} | SPK={spk_sim}")

# Save and display results
df = pd.DataFrame(all_results)
df.to_csv("results/raw_results.csv", index=False)

for metric in ["pesq", "stoi", "spk_sim"]:
    metric_df = df.dropna(subset=[metric])
    if len(metric_df) == 0:
        print(f"\n⚠️ No {metric} data — skipping")
        continue

    table = metric_df.pivot_table(
        values=metric,
        index="codec",
        columns="language",
        aggfunc="mean"
    ).round(3)

    table["MEAN"] = table.mean(axis=1).round(3)

    print(f"\n{'='*70}")
    print(f"  {metric.upper()} (higher = better)")
    print(f"{'='*70}")
    print(table.to_string())

    table.to_csv(f"results/{metric}_results.csv")

print("\n\n=== ALL RESULTS SAVED TO results/ ===")
