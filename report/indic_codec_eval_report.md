# Neural Audio Codec Evaluation on Indian Language Speech
## First Systematic Benchmark on Indian Language Data in codecs

**Date:** March 17, 2026
**Author:** Aman Agrawal
**Datasets:** Public HuggingFace datasets (En1gma02/hindi_speech_male_5hr, shunyalabs/tamil-speech-dataset, shunyalabs/telugu-speech-dataset, shunyalabs/bengali-speech-dataset, shunyalabs/kannada-speech-dataset)

---

## Motivation

Neural audio codecs (EnCodec, DAC, SNAC, etc.) are the backbone of modern speech generation systems like VALL-E, VoiceCraft, and Koel-TTS. However, these codecs were trained predominantly on English speech data. As Indian language TTS and voice AI systems grow rapidly, a critical question remains unanswered: **how well do these codecs preserve speech quality, intelligibility, and speaker identity for Indian languages?**

This evaluation provides the first systematic answer to that question.

---

## Setup

- **Languages:** Hindi, Tamil, Telugu, Bengali, Kannada
- **Samples per language:** 10 (duration: 1–10 seconds each)
- **Source data:** Public HuggingFace speech datasets (Hindi at 48kHz studio quality; Tamil, Telugu, Bengali, Kannada at 16kHz)
- **Compute:** Apple MacBook Air M4 — inference ran on the M4's integrated GPU via MPS (Metal Performance Shaders) backend in PyTorch
- **Codecs evaluated:**

| Codec | Architecture | Sample Rate | Bitrate | Source |
|-------|-------------|-------------|---------|--------|
| EnCodec | RVQ | 24kHz | 6 kbps | Meta |
| DAC | RVQ | 24kHz | 8 kbps | Descript |
| SNAC | Multi-scale RVQ | 24kHz | variable | Hubertsiuzdak |

- **Metrics:**
  - **PESQ** — Perceptual Evaluation of Speech Quality (range -0.5 to 4.5, higher = better)
  - **STOI** — Short-Time Objective Intelligibility (range 0 to 1, higher = better)
  - **Speaker Similarity** — ECAPA-TDNN cosine similarity (range -1 to 1, higher = better)
  - **WER** — Word Error Rate via language-specific ASR (lower = better, 0 = no degradation)
  - **Exact Match Rate** — Fraction of samples with identical ASR transcript before/after codec (higher = better)

---

## Results

### PESQ (Perceptual Quality — higher is better)

| Codec   | Bengali | Hindi | Kannada | Tamil | Telugu | MEAN  |
|---------|---------|-------|---------|-------|--------|-------|
| DAC     | 4.458   | 4.554 | 4.454   | 4.460 | 4.440  | 4.473 |
| EnCodec | 2.536   | 3.188 | 2.725   | 2.815 | 2.720  | 2.797 |
| SNAC    | 1.645   | 2.374 | 1.809   | 1.995 | 1.784  | 1.921 |

### STOI (Intelligibility — higher is better)

| Codec   | Bengali | Hindi | Kannada | Tamil | Telugu | MEAN  |
|---------|---------|-------|---------|-------|--------|-------|
| DAC     | 0.989   | 0.997 | 0.992   | 0.992 | 0.989  | 0.992 |
| EnCodec | 0.875   | 0.953 | 0.911   | 0.920 | 0.890  | 0.910 |
| SNAC    | 0.752   | 0.931 | 0.836   | 0.846 | 0.773  | 0.828 |

### Speaker Similarity (Voice Preservation — higher is better)

| Codec   | Bengali | Hindi | Kannada | Tamil | Telugu | MEAN  |
|---------|---------|-------|---------|-------|--------|-------|
| DAC     | 0.985   | 0.983 | 0.988   | 0.990 | 0.988  | 0.987 |
| EnCodec | 0.877   | 0.898 | 0.903   | 0.908 | 0.925  | 0.902 |
| SNAC    | 0.632   | 0.755 | 0.725   | 0.756 | 0.730  | 0.720 |

### WER — ASR Degradation (lower is better, 0 = no degradation)

*Evaluated on Hindi and Tamil only. See [Why only 2 languages for WER?](#why-only-2-languages-for-wer) below.*

**ASR Models used:**
- Hindi: `vasista22/whisper-hindi-large-v2`
- Tamil: `vasista22/whisper-tamil-large-v2`

| Codec   | Hindi | Tamil | MEAN  |
|---------|-------|-------|-------|
| DAC     | 0.178 | 0.359 | 0.268 |
| EnCodec | 0.184 | 0.508 | 0.346 |
| SNAC    | 0.340 | 0.829 | 0.584 |

### Exact Match Rate (higher is better, 1.0 = identical ASR transcript)

| Codec   | Hindi | Tamil | MEAN |
|---------|-------|-------|------|
| DAC     | 0.30  | 0.10  | 0.20 |
| EnCodec | 0.30  | 0.00  | 0.15 |
| SNAC    | 0.00  | 0.00  | 0.00 |

---

## Why Only 2 Languages for WER?

WER evaluation requires a reliable ASR model that produces accurate, stable transcriptions in the target language. We attempted WER for all 5 languages using language-specific fine-tuned Whisper models from the `vasista22` family:

- `vasista22/whisper-hindi-large-v2` — **worked reliably**
- `vasista22/whisper-tamil-large-v2` — **worked reliably**
- `vasista22/whisper-telugu-large-v2` — produced hallucinated/repetitive outputs
- `vasista22/whisper-kannada-medium` — produced unreliable transcriptions with wrong scripts

For Telugu, Bengali, and Kannada, the ASR models generated hallucinated text, infinite repetition loops, or outputs in the wrong script — making WER computation meaningless. Rather than reporting misleading numbers, we chose to **report WER only for the two languages where ASR was reliable** (Hindi and Tamil), and clearly flag this as a limitation.

Extending WER to all languages is a priority for future work, pending availability of better Indic ASR models or fine-tuning existing ones.

---

## Key Observations

### 1. DAC dominates across all metrics
DAC achieves near-perfect reconstruction with PESQ=4.473, STOI=0.992, and speaker similarity=0.987. This is remarkable given it was trained primarily on English data. DAC's performance is consistent across all five Indian languages, showing strong cross-lingual generalization.

### 2. Clear codec ranking: DAC >> EnCodec >> SNAC
This ranking holds across all five languages and all metrics — PESQ, STOI, speaker similarity, and WER. The gap between DAC and EnCodec is larger than the gap between EnCodec and SNAC for perceptual quality (PESQ), but SNAC falls dramatically behind on speaker preservation.

### 3. Hindi scores highest, Bengali/Telugu score lowest
Hindi consistently scores highest across all codecs (e.g., EnCodec PESQ: 3.188 vs Bengali: 2.536). This may reflect that Hindi's phonological features are closer to English (the dominant training language), or that the Hindi dataset had higher recording quality (48kHz studio vs 16kHz for others).

### 4. SNAC struggles significantly on Indic languages
SNAC's multi-scale architecture shows the largest quality degradation, with mean PESQ of only 1.921 and speaker similarity of 0.720. Bengali is especially poor (PESQ=1.645, speaker similarity=0.632). SNAC reconstructions for Tamil are largely unintelligible to downstream ASR (WER=82.9%).

### 5. Speaker identity preservation varies dramatically
DAC preserves speaker identity almost perfectly (0.987), EnCodec is good (0.902), but SNAC loses significant speaker characteristics (0.720). This makes SNAC unsuitable for voice cloning or speaker-conditioned TTS pipelines on Indian languages.

### 6. WER confirms signal metrics
ASR-based evaluation on Hindi and Tamil independently confirms the codec ranking. DAC introduces only 17.8% WER on Hindi, while SNAC degrades Tamil ASR output by 82.9%. This validates that signal-level metrics (PESQ/STOI) are meaningful proxies for downstream task performance.

### 7. Tamil is harder than Hindi across all codecs
Tamil WER is consistently 2–2.4x worse than Hindi for every codec. This likely reflects a combination of: (a) Tamil's agglutinative morphology creating longer word units, (b) less Tamil data in codec training, and (c) Tamil ASR model being less robust than the Hindi one.

---

## Significance

This is the **first systematic evaluation of neural audio codecs on Indian language speech data**. While these codecs are increasingly being used in Indian language TTS systems (Koel-TTS, MagpieTTS, IndicVoices), there has been no published benchmark quantifying their performance on Indian languages.

Our results show that:
- **DAC generalizes well** to Indian languages and can be used confidently in production pipelines
- **EnCodec is acceptable** but shows noticeable degradation, especially on Dravidian languages
- **SNAC is not recommended** for Indian language applications in its current form — it introduces significant artifacts that degrade both perceived quality and downstream ASR performance
- **Language matters** — codec performance varies meaningfully across Indian languages, and evaluating only on English is insufficient

---

## Limitations

- **Small sample size** — 10 samples per language; results may not generalize to all speakers, domains, or recording conditions
- **WER limited to Hindi and Tamil** — ASR models for Telugu, Bengali, and Kannada produced unreliable outputs (hallucinations, wrong scripts). WER for these languages is not reported rather than reporting misleading numbers
- **PESQ not validated for Indian languages** — PESQ was designed and calibrated for English/European speech; absolute scores may not perfectly reflect perceived quality for native speakers
- **Different source datasets per language** — Hindi (48kHz studio) vs Tamil/Telugu/Bengali/Kannada (16kHz) — varying recording quality may influence cross-language comparisons
- **NanoCodec (NVIDIA, FSQ) not evaluated** — Could not be included due to protobuf/onnx dependency conflicts. This is notable since NanoCodec uses FSQ (Finite Scalar Quantization) rather than RVQ, and is the codec behind Koel-TTS/MagpieTTS
- **Mimi codec (Kyutai) not evaluated** — moshi package was not included
- **No phoneme-class-specific analysis** — No minimal pair tests for Indian-language-specific contrasts (retroflex vs dental, aspirated vs unaspirated, gemination)
- **No 8kHz telephony evaluation** — Many Indian voice applications operate over telephony networks
- **Apple Silicon only** — All codecs were run on M4 GPU (MPS backend); results on CUDA GPUs may differ slightly due to floating-point precision differences

---

## Future Work

### Immediate priorities
- **Extend WER to all 5 languages** — Fine-tune or source reliable ASR models for Telugu, Bengali, and Kannada (e.g., AI4Bharat IndicWhisper, or fine-tune Whisper on IndicVoices data)
- **Include NanoCodec/LFSC** — Resolve dependency issues and evaluate NVIDIA's FSQ-based codec, which is directly relevant to Koel-TTS/MagpieTTS systems
- **Scale to more samples** — Increase from 10 to 50–100 samples per language for statistically robust conclusions

### Medium-term
- **Scale to all 22 scheduled Indian languages** — Leverage IndicVoices-R dataset for comprehensive coverage
- **Minimal pair tests** — Design phoneme-specific evaluation: retroflex vs dental (/ɖ/ vs /d/), aspirated vs unaspirated (/kʰ/ vs /k/), nasalization, gemination
- **8kHz telephony evaluation** — Test codec performance at telephony sample rates, critical for Indian voice AI deployed over phone networks
- **Evaluate additional codecs** — Expand benchmark to include emerging codecs gaining traction in the speech community:
  - **Mimi** (Kyutai) — streaming codec powering Moshi, low-latency conversational AI
  - **WavTokenizer** — single-codebook codec achieving high quality at extreme compression
  - **FACodec** (NaturalSpeech3) — factorized codec separating content, prosody, and speaker into disentangled codes
  - **SpeechTokenizer** (Fudan) — unified semantic + acoustic tokenizer, used in speech LLMs
  - **SemantiCodec** — semantically-structured codec for language model integration
  - **Single-Codec** (ByteDance) — single-codebook approach for simplified LM-based TTS
  - **TiCodec** — time-invariant codec designed for better speaker disentanglement

### Long-term
- **Fine-tune codecs on Indic data** — Train or fine-tune EnCodec/DAC/SNAC on Indian language speech and measure improvement
- **Subjective listening tests** — Conduct MOS (Mean Opinion Score) evaluation with native speakers of each language
- **Downstream task evaluation** — Measure impact on full TTS pipeline quality, not just codec reconstruction
- **Publish benchmark** — Release as an open benchmark for the Indian language speech community
