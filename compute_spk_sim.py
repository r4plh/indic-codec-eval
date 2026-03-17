"""Compute speaker similarity separately (fix SpeechBrain audio loading)."""
import os
import torch
import torchaudio
import pandas as pd

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

device = 'cpu'
LANGUAGES = ["hindi", "tamil", "telugu", "bengali", "kannada"]
CODECS = ["encodec", "dac", "snac"]
SAMPLES_PER_LANG = 10

from speechbrain.inference.speaker import EncoderClassifier
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

def get_embedding(wav_path):
    wav, sr = torchaudio.load(wav_path)
    # SpeechBrain expects 16kHz mono, shape [batch, time]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    # wav is [1, T], encode_batch expects [batch, time] — squeeze channel dim
    wav = wav.squeeze(0)  # [T]
    emb = spk_model.encode_batch(wav.unsqueeze(0))  # [1, 1, emb_dim]
    return emb.squeeze()

# Load existing results
df = pd.read_csv("results/raw_results.csv")

spk_sims = []
for idx, row in df.iterrows():
    codec = row["codec"]
    lang = row["language"]
    i = int(row["sample"])

    ref_path = f"data/{lang}/sample_{i}_24k.wav"
    deg_path = f"reconstructed/{codec}/{lang}/sample_{i}.wav"

    try:
        emb1 = get_embedding(ref_path)
        emb2 = get_embedding(deg_path)
        sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        spk_sims.append(sim)
        print(f"  {codec} | {lang} | sample_{i} | SPK_SIM={sim:.4f}")
    except Exception as e:
        spk_sims.append(None)
        print(f"  ❌ {codec} | {lang} | sample_{i}: {e}")

df["spk_sim"] = spk_sims
df.to_csv("results/raw_results.csv", index=False)

# Create spk_sim table
metric_df = df.dropna(subset=["spk_sim"])
table = metric_df.pivot_table(
    values="spk_sim",
    index="codec",
    columns="language",
    aggfunc="mean"
).round(3)
table["MEAN"] = table.mean(axis=1).round(3)

print(f"\n{'='*70}")
print(f"  SPEAKER SIMILARITY (higher = better, 1.0 = identical)")
print(f"{'='*70}")
print(table.to_string())
table.to_csv("results/spk_sim_results.csv")
