import os
import torch
import torchaudio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

LANGUAGES = ["hindi", "tamil", "telugu", "bengali", "kannada"]
SAMPLES_PER_LANG = 10

# ============================================================
# CODEC 1: EnCodec (24kHz)
# ============================================================
def run_encodec():
    print("\n=== Running EnCodec ===")
    try:
        from encodec import EncodecModel

        model = EncodecModel.encodec_model_24khz().to(device)
        model.set_target_bandwidth(6.0)

        for lang in LANGUAGES:
            os.makedirs(f"reconstructed/encodec/{lang}", exist_ok=True)
            for i in range(SAMPLES_PER_LANG):
                input_path = f"data/{lang}/sample_{i}_24k.wav"
                if not os.path.exists(input_path):
                    continue
                try:
                    wav, sr = torchaudio.load(input_path)
                    wav = wav.unsqueeze(0).to(device)  # [1, 1, T]

                    with torch.no_grad():
                        encoded = model.encode(wav)
                        decoded = model.decode(encoded)

                    output_path = f"reconstructed/encodec/{lang}/sample_{i}.wav"
                    torchaudio.save(output_path, decoded.squeeze(0).cpu(), 24000)
                    print(f"  ✅ encodec {lang} sample_{i}")
                except Exception as e:
                    print(f"  ❌ encodec {lang} sample_{i}: {e}")
        return True
    except Exception as e:
        print(f"  ❌ EnCodec FAILED entirely: {e}")
        import traceback; traceback.print_exc()
        return False

# ============================================================
# CODEC 2: DAC (24kHz)
# ============================================================
def run_dac():
    print("\n=== Running DAC ===")
    try:
        import dac
        from audiotools import AudioSignal

        model_path = dac.utils.download(model_type="24khz")
        model = dac.DAC.load(model_path).to(device)

        for lang in LANGUAGES:
            os.makedirs(f"reconstructed/dac/{lang}", exist_ok=True)
            for i in range(SAMPLES_PER_LANG):
                input_path = f"data/{lang}/sample_{i}_24k.wav"
                if not os.path.exists(input_path):
                    continue
                try:
                    signal = AudioSignal(input_path)
                    signal = signal.to(device)

                    with torch.no_grad():
                        x = model.preprocess(signal.audio_data, signal.sample_rate)
                        z, codes, latents, _, _ = model.encode(x)
                        y = model.decode(z)

                    output_path = f"reconstructed/dac/{lang}/sample_{i}.wav"
                    torchaudio.save(output_path, y.squeeze(0).cpu(), 24000)
                    print(f"  ✅ dac {lang} sample_{i}")
                except Exception as e:
                    print(f"  ❌ dac {lang} sample_{i}: {e}")
        return True
    except Exception as e:
        print(f"  ❌ DAC FAILED entirely: {e}")
        import traceback; traceback.print_exc()
        return False

# ============================================================
# CODEC 3: SNAC (24kHz)
# ============================================================
def run_snac():
    print("\n=== Running SNAC ===")
    try:
        from snac import SNAC

        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

        for lang in LANGUAGES:
            os.makedirs(f"reconstructed/snac/{lang}", exist_ok=True)
            for i in range(SAMPLES_PER_LANG):
                input_path = f"data/{lang}/sample_{i}_24k.wav"
                if not os.path.exists(input_path):
                    continue
                try:
                    wav, sr = torchaudio.load(input_path)
                    wav = wav.unsqueeze(0).to(device)  # [1, 1, T]

                    with torch.no_grad():
                        codes = model.encode(wav)
                        decoded = model.decode(codes)

                    output_path = f"reconstructed/snac/{lang}/sample_{i}.wav"
                    torchaudio.save(output_path, decoded.squeeze(0).cpu(), 24000)
                    print(f"  ✅ snac {lang} sample_{i}")
                except Exception as e:
                    print(f"  ❌ snac {lang} sample_{i}: {e}")
        return True
    except Exception as e:
        print(f"  ❌ SNAC FAILED entirely: {e}")
        import traceback; traceback.print_exc()
        return False

# ============================================================
# RUN ALL
# ============================================================
if __name__ == "__main__":
    results = {}
    results["encodec"] = run_encodec()
    results["dac"] = run_dac()
    results["snac"] = run_snac()

    print("\n=== CODEC RUN SUMMARY ===")
    for codec, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {codec}: {status}")
