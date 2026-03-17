"""Download Indic speech data from public HuggingFace datasets."""
import os
import torch
import torchaudio
from datasets import load_dataset

SAMPLES_PER_LANG = 10
TARGET_SR = 24000
DATA_DIR = "data"

DATASETS = {
    "hindi": "En1gma02/hindi_speech_male_5hr",
    "tamil": "shunyalabs/tamil-speech-dataset",
    "telugu": "shunyalabs/telugu-speech-dataset",
    "bengali": "shunyalabs/bengali-speech-dataset",
    "kannada": "shunyalabs/kannada-speech-dataset",
}

for lang, ds_id in DATASETS.items():
    os.makedirs(f"{DATA_DIR}/{lang}", exist_ok=True)
    print(f"\n=== {lang.upper()} ({ds_id}) ===")

    try:
        ds = load_dataset(ds_id, split="train", streaming=True)
        count = 0
        for sample in ds:
            if count >= SAMPLES_PER_LANG:
                break

            audio_obj = sample["audio"]

            # New datasets API: AudioDecoder with get_all_samples()
            result = audio_obj.get_all_samples()
            wav = result.data  # [channels, samples]
            sr = result.sample_rate

            # Convert to mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # Skip short clips (< 1s)
            if wav.shape[1] / sr < 1.0:
                continue

            # Truncate to 10s
            max_samples = sr * 10
            if wav.shape[1] > max_samples:
                wav = wav[:, :max_samples]

            # Save original
            orig_path = f"{DATA_DIR}/{lang}/sample_{count}_orig.wav"
            torchaudio.save(orig_path, wav, sr)

            # Resample to 24kHz
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                wav_24k = resampler(wav)
            else:
                wav_24k = wav

            path_24k = f"{DATA_DIR}/{lang}/sample_{count}_24k.wav"
            torchaudio.save(path_24k, wav_24k, TARGET_SR)

            count += 1
            print(f"  ✅ sample_{count-1} ({wav.shape[1]/sr:.1f}s, {sr}Hz)")

        print(f"  Total: {count}/{SAMPLES_PER_LANG} samples")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

# Verify
print("\n=== VERIFICATION ===")
total = 0
for lang in DATASETS.keys():
    n = len([f for f in os.listdir(f"{DATA_DIR}/{lang}") if f.endswith("_24k.wav")])
    total += n
    print(f"  {lang}: {n} samples")
print(f"  TOTAL: {total}")
