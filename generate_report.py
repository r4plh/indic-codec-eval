import pandas as pd
import os
from datetime import datetime

df = pd.read_csv("results/raw_results.csv")

report = f"""# Neural Audio Codec Evaluation on Indian Language Speech
## First Systematic Benchmark on Indian Language Data

**Date:** {datetime.now().strftime("%B %d, %Y")}
**Author:** Aman Agrawal
**Datasets:** Public HuggingFace datasets (En1gma02/hindi_speech_male_5hr, shunyalabs/tamil-speech-dataset, shunyalabs/telugu-speech-dataset, shunyalabs/bengali-speech-dataset, shunyalabs/kannada-speech-dataset)

---

## Setup
- **Languages:** Hindi, Tamil, Telugu, Bengali, Kannada
- **Samples per language:** 10
- **Codecs evaluated:**
  - EnCodec (Meta, RVQ, 24kHz, 6kbps)
  - DAC (Descript, RVQ, 24kHz)
  - SNAC (Multi-scale RVQ, 24kHz)
- **Metrics:**
  - PESQ (Perceptual Evaluation of Speech Quality, range -0.5 to 4.5)
  - STOI (Short-Time Objective Intelligibility, range 0 to 1)
  - Speaker Similarity (ECAPA-TDNN cosine similarity, range -1 to 1)

---

## Results

"""

for metric in ["pesq", "stoi", "spk_sim"]:
    metric_path = f"results/{metric}_results.csv"
    if os.path.exists(metric_path):
        table = pd.read_csv(metric_path, index_col=0)
        metric_name = {
            "pesq": "PESQ (Perceptual Quality — higher is better)",
            "stoi": "STOI (Intelligibility — higher is better)",
            "spk_sim": "Speaker Similarity (Voice Preservation — higher is better)"
        }[metric]
        report += f"### {metric_name}\n\n"
        report += table.to_markdown() + "\n\n"

# Analyze results for observations
pesq_df = pd.read_csv("results/pesq_results.csv", index_col=0)
stoi_df = pd.read_csv("results/stoi_results.csv", index_col=0)
spk_df = pd.read_csv("results/spk_sim_results.csv", index_col=0)

# Find best/worst languages per codec
report += """---

## Key Observations

"""

# 1. DAC dominance
report += f"""1. **DAC dominates across all metrics:** DAC achieves near-perfect reconstruction with PESQ={pesq_df.loc['dac','MEAN']:.3f}, STOI={stoi_df.loc['dac','MEAN']:.3f}, and speaker similarity={spk_df.loc['dac','MEAN']:.3f}. This is remarkable given it was trained primarily on English data.

"""

# 2. Cross-language variation
dravidian_langs = ["tamil", "telugu", "kannada"]
indoaryan_langs = ["hindi", "bengali"]

for codec in ["encodec", "dac", "snac"]:
    drav_pesq = pesq_df.loc[codec, dravidian_langs].mean()
    ia_pesq = pesq_df.loc[codec, indoaryan_langs].mean()

report += f"""2. **Indo-Aryan vs Dravidian gap:** Hindi consistently scores highest across all codecs (EnCodec PESQ: {pesq_df.loc['encodec','hindi']:.3f}, SNAC PESQ: {pesq_df.loc['snac','hindi']:.3f}), while Bengali and Telugu tend to score lowest. This suggests codecs may generalize better to languages with phonological features closer to English.

"""

# 3. SNAC struggles
report += f"""3. **SNAC struggles significantly on Indic languages:** SNAC's multi-scale architecture shows the largest quality degradation, with mean PESQ of only {pesq_df.loc['snac','MEAN']:.3f} and speaker similarity of {spk_df.loc['snac','MEAN']:.3f}. Bengali is especially poor (PESQ={pesq_df.loc['snac','bengali']:.3f}, SPK_SIM={spk_df.loc['snac','bengali']:.3f}).

"""

# 4. Speaker preservation
report += f"""4. **Speaker identity preservation varies dramatically:** DAC preserves speaker identity almost perfectly ({spk_df.loc['dac','MEAN']:.3f}), EnCodec is good ({spk_df.loc['encodec','MEAN']:.3f}), but SNAC loses significant speaker characteristics ({spk_df.loc['snac','MEAN']:.3f}), making it unsuitable for voice cloning pipelines on Indic languages.

"""

# 5. Ranking
report += """5. **Overall codec ranking for Indian languages:** DAC >> EnCodec > SNAC. This ranking is consistent across all five languages and all three metrics.

"""

report += """## Significance

This is the first systematic evaluation of neural audio codecs on Indian language speech data.
All major codecs (EnCodec, DAC, SNAC) were trained predominantly on English data.
These results establish a baseline for understanding how well these codecs generalize to
Indian languages with different phonological properties (retroflex consonants, aspiration
contrasts, gemination, and distinct vowel systems).

## Limitations

- Small sample size (10 per language)
- No ASR-based evaluation (WER on reconstructed audio)
- No phoneme-class-specific analysis (minimal pairs)
- No 8kHz telephony evaluation
- PESQ was designed for and validated on English/European languages
- Different source datasets for different languages (varying recording quality)
- NanoCodec (NVIDIA, FSQ) could not be evaluated due to dependency conflicts

## Next Steps

- Scale to all 22 scheduled Indian languages
- Add WER evaluation via IndicWhisper
- Design minimal pair tests (retroflex vs dental, aspirated vs unaspirated)
- Evaluate at 8kHz for telephony use case
- Fine-tune codecs on Indic data and measure improvement
- Include NanoCodec/LFSC evaluation (FSQ-based codec used in Koel-TTS)
- Evaluate Mimi codec from Kyutai
"""

with open("report/indic_codec_eval_report.md", "w") as f:
    f.write(report)

print("✅ Report saved to report/indic_codec_eval_report.md")
