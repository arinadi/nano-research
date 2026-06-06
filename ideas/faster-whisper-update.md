# Faster Whisper: Update dari large-v2

> **Bagian dari:** [[ocr-transcript-pipeline-overview]]
> **Target:** Transkripsi audio Bahasa Indonesia di Kaggle Free GPU

---

## Apa yang Berubah Sejak large-v2?

### Model: large-v3 (November 2023)
Whisper large-v3 dilatih pada 1 juta jam audio weakly labeled + 4 juta jam pseudo-labeled dari v2. Hasilnya: penurunan error rate 10–20% dibanding large-v2 di berbagai bahasa. Arsitektur sama persis, hanya Mel frequency bins naik dari 80 ke 128. **Ini upgrade paling straightforward dan paling sepadan untuk Bahasa Indonesia.**

### Model: large-v3-turbo (September 2024)
Turbo mengurangi jumlah decoder layer dari 32 menjadi 4, menghasilkan peningkatan kecepatan signifikan sambil mempertahankan akurasi setara large-v2. Ukuran 809M parameter (vs 1.54B di v3), dengan degradasi akurasi sedikit lebih besar di bahasa seperti Thai dan Cantonese. Untuk kecepatan maksimum dengan akurasi "cukup baik" — pilihan yang valid.

### distil-large-v3.5 (Maret 2025)
Distil-Whisper hanya tersedia untuk English speech recognition. Untuk multilingual termasuk Bahasa Indonesia, disarankan menggunakan Whisper Turbo.

**Rekomendasi untuk Bahasa Indonesia: tetap pakai `large-v3`**, bukan turbo atau distil.

## Fitur Baru Engine Faster-Whisper

Rilis terbaru faster-whisper membawa batched inference yang **4× lebih cepat dan akurat**, support model large-v3-turbo, VAD filter 3× lebih cepat di CPU, dan feature extraction 3× lebih cepat.

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

# Drop-in replacement dari .transcribe() biasa
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### Fitur tambahan yang berguna:
- `language_detection_segments` + `language_detection_threshold` — deteksi bahasa lebih baik
- `multilingual=True` pada model medium ke bawah (large sudah code-switch otomatis)
- VAD filter built-in semakin stabil untuk deteksi silence
- `log_progress=True` untuk tracking progress transkripsi panjang

## WhisperX — Kalau Butuh Speaker Diarization

WhisperX menambahkan precise word-level timestamps dan speaker diarization menggunakan pyannote-audio. Sangat berguna untuk konten multi-speaker atau rekaman panjang seperti wawancara dan meeting.

```python
import whisperx

model = whisperx.load_model("large-v3", device="cuda", compute_type="float16")
result = model.transcribe("audio.mp3", batch_size=16)

# Alignment untuk word-level timestamps
model_a, metadata = whisperx.load_align_model(language_code="id", device="cuda")
result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda")

# Diarization (butuh HuggingFace token + pyannote)
diarize_model = whisperx.DiarizationPipeline(use_auth_token="HF_TOKEN", device="cuda")
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)
```

## Ringkasan Pilihan Whisper

| Situasi | Rekomendasi |
|---|---|
| Upgrade dari v2, akurasi terbaik (Bahasa Indonesia) | `large-v3` + `BatchedInferencePipeline` |
| Kecepatan prioritas, akurasi "cukup" | `large-v3-turbo` + batched |
| Multi-speaker, perlu siapa ngomong apa | WhisperX + `large-v3` |
| English only, ekstrem cepat | `distil-large-v3.5` |

---

*Ref: [[gemma4-model-selection]] | [[gemini-api-providers]] | [[pipeline-kaggle-full]]*
