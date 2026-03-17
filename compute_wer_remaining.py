"""
WER Evaluation for Telugu & Kannada.
Uses language-specific fine-tuned Indic Whisper models on MPS (M4 GPU).
Anti-hallucination measures: max_new_tokens cap, no_repeat_ngram_size, compression_ratio filter.
"""
import os
import json
import torch
import torchaudio
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer as compute_wer

# Use MPS (Metal) on M4 Mac, fallback to CPU
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16
    print("Using MPS (M4 GPU)")
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
    print("Using CUDA GPU")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("Using CPU")

LANGUAGES = ["telugu", "kannada"]
CODECS = ["encodec", "dac", "snac"]
SAMPLES_PER_LANG = 10

MODELS = {
    "telugu": "vasista22/whisper-telugu-large-v2",
    "kannada": "vasista22/whisper-kannada-medium",
}

LANG_CODES = {"telugu": "te", "kannada": "kn"}


def load_pipe(model_id):
    """Load ASR pipeline with MPS support and float16."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch_dtype,
    )
    return pipe


def transcribe(pipe, audio_path, language):
    """Transcribe with anti-hallucination settings."""
    try:
        # These older fine-tuned models have outdated generation configs
        # that don't support language/task args — just pass minimal kwargs
        result = pipe(
            audio_path,
            generate_kwargs={
                "no_repeat_ngram_size": 4,
                "max_new_tokens": 256,
            },
        )
        text = result["text"].strip()

        # Filter obvious hallucinations: if output is way too long relative to audio
        wav, sr = torchaudio.load(audio_path)
        duration = wav.shape[1] / sr
        words = len(text.split())
        # More than 8 words per second is likely hallucination
        if duration > 0 and words / duration > 8:
            print(f"    Filtered hallucination: {words} words in {duration:.1f}s")
            return None

        return text
    except Exception as e:
        print(f"    Transcription failed: {e}")
        return None


# Load existing results if available
existing_wer_path = "results/wer_raw_results.csv"
existing_orig_path = "results/original_transcripts.json"
existing_recon_path = "results/reconstructed_transcripts.json"

# ============================================================
# Step 1: Load models and transcribe originals
# ============================================================
print("\n=== Loading ASR models ===")
pipes = {}
for lang, model_id in MODELS.items():
    print(f"Loading {lang}: {model_id}")
    pipes[lang] = load_pipe(model_id)
    print(f"  Loaded on {device}")

print("\n=== Step 1: Transcribing ORIGINAL audio ===")
orig_transcripts = {}

for lang in LANGUAGES:
    orig_transcripts[lang] = {}
    for i in range(SAMPLES_PER_LANG):
        path = f"data/{lang}/sample_{i}_24k.wav"
        if not os.path.exists(path):
            continue
        text = transcribe(pipes[lang], path, lang)
        orig_transcripts[lang][i] = text
        preview = (text[:60] + "...") if text and len(text) > 60 else (text or "NONE")
        print(f"  {lang} sample_{i}: \"{preview}\"")

# ============================================================
# Step 2: Transcribe reconstructed audio
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
            text = transcribe(pipes[lang], path, lang)
            recon_transcripts[codec][lang][i] = text
            print(f"  {codec} | {lang} sample_{i}: done")

# Save transcripts
with open("results/original_transcripts_telugu_kannada.json", "w", encoding="utf-8") as f:
    json.dump(orig_transcripts, f, ensure_ascii=False, indent=2)
with open("results/reconstructed_transcripts_telugu_kannada.json", "w", encoding="utf-8") as f:
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
                    print(f"  WER failed for {codec}/{lang}/sample_{i}: {e}")
                    continue

            all_wer_results.append({
                "codec": codec,
                "language": lang,
                "sample": i,
                "wer": w,
                "ref_text": ref,
                "hyp_text": hyp,
                "exact_match": 1 if ref == hyp else 0,
            })
            print(f"  {codec} | {lang} | sample_{i} | WER={w:.3f} | exact={ref == hyp}")

# ============================================================
# Step 4: Merge with existing Hindi/Tamil results & save
# ============================================================
df_new = pd.DataFrame(all_wer_results)
df_new.to_csv("results/wer_raw_results_telugu_kannada.csv", index=False)

# Load existing Hindi/Tamil WER
existing_csv = "results/wer_results.csv"
if os.path.exists(existing_csv):
    df_existing = pd.read_csv(existing_csv, index_col=0)
    print(f"\nExisting WER (Hindi/Tamil):\n{df_existing.to_string()}")

if len(df_new) > 0:
    # New table for Telugu/Kannada
    wer_table = df_new.pivot_table(
        values="wer", index="codec", columns="language", aggfunc="mean"
    ).round(3)
    wer_table["MEAN"] = wer_table.mean(axis=1).round(3)

    print(f"\n{'='*70}")
    print(f"  WER (Telugu & Kannada) — lower is better")
    print(f"  Telugu ASR: {MODELS['telugu']}")
    print(f"  Kannada ASR: {MODELS['kannada']}")
    print(f"{'='*70}")
    print(wer_table.to_string())
    wer_table.to_csv("results/wer_results_telugu_kannada.csv")

    em_table = df_new.pivot_table(
        values="exact_match", index="codec", columns="language", aggfunc="mean"
    ).round(3)
    em_table["MEAN"] = em_table.mean(axis=1).round(3)
    print(f"\n  Exact Match Rate:")
    print(em_table.to_string())
    em_table.to_csv("results/exact_match_results_telugu_kannada.csv")

    # Merge all 4 languages into combined table
    all_csv = "results/wer_raw_results.csv"
    if os.path.exists(all_csv):
        df_old = pd.read_csv(all_csv)
        if len(df_old) > 0:
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
    else:
        df_combined = df_new
    df_combined.to_csv("results/wer_raw_results_all.csv", index=False)

    combined_wer = df_combined.pivot_table(
        values="wer", index="codec", columns="language", aggfunc="mean"
    ).round(3)
    combined_wer["MEAN"] = combined_wer.mean(axis=1).round(3)
    print(f"\n{'='*70}")
    print(f"  COMBINED WER (all languages)")
    print(f"{'='*70}")
    print(combined_wer.to_string())
    combined_wer.to_csv("results/wer_results_combined.csv")
else:
    print("\n⚠️ No valid WER results computed — ASR may have failed for all samples")

print("\n=== WER EVALUATION COMPLETE ===")
