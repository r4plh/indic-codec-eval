"""
WER Evaluation: Measure ASR degradation from codec reconstruction.
Uses language-specific fine-tuned Indic Whisper models (vasista22).

- Hindi: vasista22/whisper-hindi-large-v2
- Tamil: vasista22/whisper-tamil-large-v2

Approach:
  1. Run Indic ASR on original audio → reference transcripts
  2. Run Indic ASR on reconstructed audio → hypothesis transcripts
  3. WER = word error rate between reference and hypothesis
  4. A perfect codec → WER = 0% (identical transcripts)
"""
import os
import json
import torch
import pandas as pd
from transformers import pipeline
from jiwer import wer as compute_wer

LANGUAGES = ["hindi", "tamil", "telugu", "kannada"]
CODECS = ["encodec", "dac", "snac"]
SAMPLES_PER_LANG = 10

# Language-specific fine-tuned Indic Whisper models
MODELS = {
    "hindi": "vasista22/whisper-hindi-large-v2",
    "tamil": "vasista22/whisper-tamil-large-v2",
    "telugu": "vasista22/whisper-telugu-large-v2",
    "kannada": "vasista22/whisper-kannada-medium",
}

# Load one pipeline per language
pipes = {}
for lang, model_id in MODELS.items():
    print(f"Loading {lang} ASR: {model_id}")
    pipes[lang] = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device="cpu",
        torch_dtype=torch.float32,
    )
    print(f"  ✅ {lang} model loaded")


def transcribe(audio_path, language):
    """Transcribe audio file with language-specific Indic Whisper."""
    try:
        result = pipes[language](
            audio_path,
            generate_kwargs={
                "language": {"hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn"}[language],
                "task": "transcribe",
                "no_repeat_ngram_size": 3,
            },
        )
        return result["text"].strip()
    except Exception as e:
        print(f"    ⚠️ Transcription failed for {audio_path}: {e}")
        return None


# ============================================================
# Step 1: Transcribe all original audio
# ============================================================
print("\n=== Step 1: Transcribing ORIGINAL audio ===")
orig_transcripts = {}

for lang in LANGUAGES:
    orig_transcripts[lang] = {}
    for i in range(SAMPLES_PER_LANG):
        path = f"data/{lang}/sample_{i}_24k.wav"
        if not os.path.exists(path):
            continue
        text = transcribe(path, lang)
        orig_transcripts[lang][i] = text
        preview = text[:60] if text else "NONE"
        print(f"  {lang} sample_{i}: \"{preview}...\"")

with open("results/original_transcripts.json", "w", encoding="utf-8") as f:
    json.dump(orig_transcripts, f, ensure_ascii=False, indent=2)

# ============================================================
# Step 2: Transcribe all reconstructed audio
# ============================================================
print("\n=== Step 2: Transcribing RECONSTRUCTED audio ===")
recon_transcripts = {}

for codec in CODECS:
    recon_transcripts[codec] = {}
    for lang in LANGUAGES:
        recon_transcripts[codec][lang] = {}
        for i in range(SAMPLES_PER_LANG):
            path = f"reconstructed/{codec}/{lang}/sample_{i}.wav"
            if not os.path.exists(path):
                continue
            text = transcribe(path, lang)
            recon_transcripts[codec][lang][i] = text
            print(f"  {codec} | {lang} sample_{i}: done")

with open("results/reconstructed_transcripts.json", "w", encoding="utf-8") as f:
    json.dump(recon_transcripts, f, ensure_ascii=False, indent=2)

# ============================================================
# Step 3: Compute WER
# ============================================================
print("\n=== Step 3: Computing WER ===")
all_wer_results = []

for codec in CODECS:
    for lang in LANGUAGES:
        for i in range(SAMPLES_PER_LANG):
            ref = orig_transcripts.get(lang, {}).get(i)
            hyp = recon_transcripts.get(codec, {}).get(lang, {}).get(i)

            if ref is None or hyp is None:
                continue
            if ref == "" and hyp == "":
                w = 0.0
            elif ref == "" or hyp == "":
                w = 1.0
            else:
                try:
                    w = compute_wer(ref, hyp)
                except Exception as e:
                    print(f"  ⚠️ WER failed for {codec}/{lang}/sample_{i}: {e}")
                    continue

            all_wer_results.append({
                "codec": codec,
                "language": lang,
                "sample": i,
                "wer": w,
                "ref_text": ref,
                "hyp_text": hyp,
                "exact_match": 1 if ref == hyp else 0
            })

            print(f"  {codec} | {lang} | sample_{i} | WER={w:.3f} | exact={ref == hyp}")

# ============================================================
# Step 4: Create results tables
# ============================================================
df = pd.DataFrame(all_wer_results)
df.to_csv("results/wer_raw_results.csv", index=False)

wer_table = df.pivot_table(
    values="wer", index="codec", columns="language", aggfunc="mean"
).round(3)
wer_table["MEAN"] = wer_table.mean(axis=1).round(3)

print(f"\n{'='*70}")
print(f"  WER (Word Error Rate) — lower is better (0 = perfect preservation)")
print(f"  Hindi ASR: {MODELS['hindi']}")
print(f"  Tamil ASR: {MODELS['tamil']}")
print(f"{'='*70}")
print(wer_table.to_string())
wer_table.to_csv("results/wer_results.csv")

em_table = df.pivot_table(
    values="exact_match", index="codec", columns="language", aggfunc="mean"
).round(3)
em_table["MEAN"] = em_table.mean(axis=1).round(3)

print(f"\n{'='*70}")
print(f"  Exact Match Rate — higher is better (1.0 = identical transcript)")
print(f"{'='*70}")
print(em_table.to_string())
em_table.to_csv("results/exact_match_results.csv")

print("\n\n=== WER EVALUATION COMPLETE ===")
print(f"Hindi ASR: {MODELS['hindi']}")
print(f"Tamil ASR: {MODELS['tamil']}")
print(f"Total comparisons: {len(df)}")
