# Indic Codec Eval

**First systematic benchmark of neural audio codecs on Indian language speech.**

[![Report](https://img.shields.io/badge/Report-Live-6c63ff)](report/index.html)

Nobody has benchmarked how well neural audio codecs (EnCodec, DAC, SNAC) work on Indian languages. These codecs are trained on English — but they're being used in Indian language TTS systems (Koel-TTS, MagpieTTS, etc.) without any Indic-specific evaluation.

This repo fills that gap.

## Key Results

| Codec | PESQ (↑) | STOI (↑) | Speaker Sim (↑) | WER Hindi (↓) | WER Tamil (↓) | Verdict |
|-------|----------|----------|-----------------|---------------|---------------|---------|
| **DAC** | **4.47** | **0.992** | **0.987** | **0.178** | **0.359** | ✅ Recommended |
| EnCodec | 2.80 | 0.910 | 0.902 | 0.184 | 0.508 | ⚠️ Acceptable |
| SNAC | 1.92 | 0.828 | 0.720 | 0.340 | 0.829 | ❌ Not Recommended |

- **DAC** generalizes remarkably well — near-perfect reconstruction across all 5 Indian languages
- **SNAC** loses 28% speaker identity and produces 82.9% WER on Tamil — broken for Indic
- **Hindi** scores highest everywhere; **Bengali** and **Telugu** score lowest
- Full interactive report with charts: [`report/index.html`](report/index.html)

## Languages

Hindi, Tamil, Telugu, Bengali, Kannada (10 samples each, from public HuggingFace datasets)

## Metrics

- **PESQ** — Perceptual speech quality (-0.5 to 4.5)
- **STOI** — Speech intelligibility (0 to 1)
- **Speaker Similarity** — ECAPA-TDNN cosine similarity (-1 to 1)
- **WER** — Word error rate via language-specific fine-tuned Whisper (Hindi & Tamil)
- **Exact Match** — Fraction of identical ASR transcripts before/after codec

## Project Structure

```
indic_codec_eval/
├── data/                    # Source audio (5 languages × 10 samples)
├── reconstructed/           # Codec outputs (encodec, dac, snac)
├── results/                 # CSV results (PESQ, STOI, SPK_SIM, WER)
├── report/
│   ├── index.html           # Interactive HTML report with charts
│   └── indic_codec_eval_report.md
├── nbs/
│   └── waveform_comparison.ipynb
├── download_data.py         # Download & preprocess Indic speech data
├── run_codecs.py            # Encode-decode through all codecs
├── compute_metrics.py       # PESQ, STOI, Speaker Similarity
├── compute_spk_sim.py       # Speaker similarity (standalone)
├── compute_wer.py           # WER evaluation (Hindi & Tamil)
├── compute_wer_remaining.py # WER for Telugu & Kannada (WIP)
└── generate_report.py       # Generate markdown report
```

## Quickstart

```bash
# Clone
git clone https://github.com/r4plh/indic-codec-eval.git
cd indic-codec-eval

# Install (using uv)
uv sync

# Or using pip
pip install -r requirements.txt

# Run the full pipeline
python download_data.py
python run_codecs.py
python compute_metrics.py
python compute_wer.py
```

## Why This Matters

Neural audio codecs are the backbone of modern speech generation (VALL-E, VoiceCraft, Koel-TTS). If the codec introduces artifacts, loses speaker identity, or degrades intelligibility — everything downstream breaks.

All major codecs were trained on English. Indian languages have distinct phonological properties: retroflex consonants, aspiration contrasts, gemination, agglutinative morphology (Dravidian). **Evaluating only on English is not enough.**

## Future Work

- Scale to 22 Indian languages (via IndicVoices-R)
- Add codecs: WavTokenizer, FACodec, SpeechTokenizer, SemantiCodec, Mimi, NanoCodec, Single-Codec, TiCodec
- Extend WER to all languages with reliable Indic ASR
- Minimal pair tests (retroflex vs dental, aspirated vs unaspirated)
- Fine-tune codecs on Indic data and measure improvement
- 8kHz telephony evaluation
- MOS listening tests with native speakers
- Open benchmark release

## Limitations

- Small sample size (10/language)
- WER only for Hindi & Tamil (other Indic ASR models hallucinate)
- PESQ not calibrated for Indian languages
- Uneven source data quality (Hindi 48kHz vs others 16kHz)
- NanoCodec & Mimi not evaluated (dependency issues)
- CPU-only inference on Apple M4

## Citation

If you use this work, please cite:

```
@misc{agrawal2026indiccodeceval,
  title={Indic Codec Eval: Neural Audio Codec Benchmark on Indian Language Speech},
  author={Aman Agrawal},
  year={2026},
  url={https://github.com/r4plh/indic-codec-eval}
}
```

## License

MIT
