# Pipeline Lengkap: Transcript + OCR + Summarization di Kaggle

> **Bagian dari:** [[ocr-transcript-pipeline-overview]]
> **Stack:** Faster-Whisper large-v3 + Gemma 4 E4B
> **Target:** Kaggle Free GPU (T4 16GB)

---

## Arsitektur Pipeline

```
Audio File ──► Whisper (transcript) ──► Teks ──► Gemma E4B (summary)
                                                      │
Foto/Gambar ─────────────────────────────────────► Gemma E4B (OCR)
                                                      │
Dokumen Teks ───────────────────────────────────► Gemma E4B (summary)
```

Karena Whisper dan Gemma adalah model terpisah, Anda bisa load keduanya sekaligus di T4 (Whisper FP16 ~3GB + E4B bfloat16 ~8GB = ~11GB, masih dalam 16GB). Tidak perlu unload/reload antar task.

## VRAM Budget

| Component | VRAM |
|---|---|
| Faster-Whisper large-v3 (FP16) | ~3 GB |
| Gemma 4 E4B (bfloat16) | ~8 GB |
| OS + overhead | ~2 GB |
| **Total** | **~13 GB** (muat di T4 16GB) |

---

## Script Lengkap

```python
# ============================================================
# PIPELINE: Transcript + OCR + Summarization
# Stack: Faster-Whisper large-v3 | Gemma 4 E4B
# Target: Kaggle Free GPU (T4 16GB)
# ============================================================

!pip install -q --upgrade faster-whisper "transformers>=5.10.1" accelerate "pillow<12.0" librosa soundfile

import os, time, torch, gc
import numpy as np
from PIL import Image
from io import BytesIO

print("=" * 60)
print("PIPELINE: Whisper large-v3 + Gemma 4 E4B")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# ============================================================
# MODULE 1: FASTER-WHISPER TRANSCRIPT
# ============================================================

from faster_whisper import WhisperModel, BatchedInferencePipeline

print("\n[1/3] Loading Faster-Whisper large-v3...")
t0 = time.time()

whisper_model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    download_root="/kaggle/working/whisper_cache"
)
batched_whisper = BatchedInferencePipeline(model=whisper_model)

print(f"  ✅ Whisper loaded in {time.time()-t0:.1f}s")


def transcribe_audio(audio_path: str, language: str = "id") -> dict:
    """
    Transcript audio dengan Faster-Whisper large-v3.

    Args:
        audio_path: Path ke file audio (WAV, MP3, FLAC, OGG, M4A)
        language: Kode bahasa. "id" untuk Bahasa Indonesia, None untuk auto-detect

    Returns:
        dict dengan keys: text, segments, language, duration
    """
    segments, info = batched_whisper.transcribe(
        audio_path,
        language=language,
        batch_size=16,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        log_progress=True,
    )

    all_segments = []
    full_text = []

    for seg in segments:
        all_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        full_text.append(seg.text.strip())

    return {
        "text": " ".join(full_text),
        "segments": all_segments,
        "language": info.language,
        "duration": info.duration,
    }


def transcribe_long_audio(audio_path: str, language: str = "id",
                           chunk_minutes: int = 30) -> str:
    """Transcript audio panjang dengan chunking."""
    import librosa, soundfile as sf

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    chunk_samples = chunk_minutes * 60 * sr
    chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]

    all_text = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"/tmp/chunk_{i}.wav"
        sf.write(chunk_path, chunk, sr)
        result = transcribe_audio(chunk_path, language=language)
        all_text.append(result["text"])
        print(f"  Chunk {i+1}/{len(chunks)} done: {len(result['text'])} chars")

    return " ".join(all_text)


# ============================================================
# MODULE 2: GEMMA 4 E4B — OCR + SUMMARIZATION
# ============================================================

from transformers import AutoProcessor, AutoModelForImageTextToText

print("\n[2/3] Loading Gemma 4 E4B...")
t0 = time.time()

GEMMA_MODEL_ID = "google/gemma-4-E4B-it"

processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)

gemma_model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
gemma_model.eval()

print(f"  ✅ Gemma 4 E4B loaded in {time.time()-t0:.1f}s")
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")


def gemma_generate(messages: list, max_new_tokens: int = 512) -> str:
    """Helper untuk generate dari Gemma 4 E4B."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(gemma_model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    ).strip()


def ocr_image(image_input, language_hint: str = "Indonesia") -> str:
    """OCR + baca teks dari gambar."""
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            import requests
            img = Image.open(BytesIO(requests.get(image_input).content))
        else:
            img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("image_input harus path, URL, atau PIL.Image")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": (
                f"Baca semua teks dalam gambar ini secara lengkap dan akurat. "
                f"Bahasa dominan: {language_hint}. "
                f"Pertahankan struktur asli (baris, spasi, tabel). "
                f"Jika ada angka atau harga, jangan ada yang terlewat. "
                f"Output hanya teks yang kamu baca, tanpa komentar tambahan."
            )}
        ]
    }]

    return gemma_generate(messages, max_new_tokens=800)


def summarize_text(text: str, output_language: str = "Indonesia",
                   style: str = "ringkas") -> str:
    """Summarize teks panjang."""
    if style == "ringkas":
        format_instruction = (
            "Format: 3-4 kalimat ringkasan utama, lalu 3-5 poin kunci (bullet), "
            "lalu 1 kalimat kesimpulan."
        )
    else:
        format_instruction = (
            "Format: Ringkasan komprehensif dengan konteks, poin-poin penting, "
            "dan implikasi atau kesimpulan."
        )

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": (
                f"Buat ringkasan dalam Bahasa {output_language}.\n"
                f"{format_instruction}\n\n"
                f"TEKS:\n{text}"
            )
        }]
    }]

    return gemma_generate(messages, max_new_tokens=600)


def summarize_transcript(transcript: str, context: str = "") -> str:
    """Summarize hasil transcript audio menjadi notulensi/ringkasan."""
    context_str = f"Konteks: {context}\n" if context else ""

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": (
                f"{context_str}"
                f"Berikut adalah transcript audio. Buat ringkasan/notulensi dalam Bahasa Indonesia:\n"
                f"- Poin-poin utama yang dibahas\n"
                f"- Keputusan atau action item (jika ada)\n"
                f"- Kesimpulan singkat\n\n"
                f"TRANSCRIPT:\n{transcript}"
            )
        }]
    }]

    return gemma_generate(messages, max_new_tokens=800)


# ============================================================
# MODULE 3: FULL PIPELINE
# ============================================================

def full_pipeline(audio_path: str, image_paths: list = None) -> dict:
    """
    Pipeline lengkap: transcript → summarize, OCR → summarize.

    Args:
        audio_path: Path ke file audio
        image_paths: List path ke gambar (untuk OCR)

    Returns:
        dict dengan keys: transcript, summary, ocr_results
    """
    output = {}

    # Step 1: Transcript
    print("🎙️ Transcribing audio...")
    transcript = transcribe_audio(audio_path, language="id")
    output["transcript"] = transcript["text"]

    # Step 2: Summarize transcript
    print("📋 Summarizing transcript...")
    output["summary"] = summarize_transcript(
        transcript["text"],
        context="rekaman audio"
    )

    # Step 3: OCR gambar (jika ada)
    if image_paths:
        output["ocr_results"] = []
        for img_path in image_paths:
            print(f"📄 OCR: {img_path}")
            ocr = ocr_image(img_path)
            output["ocr_results"].append(ocr)

    return output


# ============================================================
# TEST
# ============================================================

print("\n[3/3] Running pipeline tests...")
print("-" * 40)

# --- TEST A: OCR ---
print("\n📄 TEST A: OCR dari gambar teks")

from PIL import ImageDraw
test_img = Image.new("RGB", (500, 250), color=(248, 245, 235))
draw = ImageDraw.Draw(test_img)
lines = [
    "INVOICE #2026-0601",
    "Tanggal: 5 Juni 2026",
    "",
    "Produk A    x2    Rp  500.000",
    "Produk B    x1    Rp  350.000",
    "Ongkos Kirim      Rp   50.000",
    "─────────────────────────────",
    "TOTAL             Rp  900.000",
]
y = 20
for line in lines:
    draw.text((30, y), line, fill=(20, 20, 20))
    y += 28

t0 = time.time()
ocr_result = ocr_image(test_img, language_hint="Indonesia")
print(f"  Hasil OCR ({time.time()-t0:.1f}s):")
print(f"  {ocr_result}")

# --- TEST B: Summarization ---
print("\n📋 TEST B: Summarization teks panjang")

sample_text = """
Transformasi digital di sektor kesehatan Indonesia mengalami percepatan signifikan
sejak 2024. Kementerian Kesehatan meluncurkan platform SATUSEHAT yang mengintegrasikan
rekam medis elektronik dari lebih dari 3.000 fasilitas kesehatan.
"""

t0 = time.time()
summary_result = summarize_text(sample_text, output_language="Indonesia", style="ringkas")
print(f"  Hasil Summary ({time.time()-t0:.1f}s):")
print(f"  {summary_result}")

# --- CLEANUP ---
print("\n🧹 Membersihkan memory GPU...")
torch.cuda.empty_cache()
gc.collect()
print(f"  VRAM setelah cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n" + "=" * 60)
print("✅ Semua test selesai!")
print("=" * 60)
```

## Async untuk Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_batch(audio_files: list) -> list:
    results = []
    for f in audio_files:
        r = transcribe_audio(f, language="id")
        results.append(r)
    return results
```

---

*Ref: [[gemma4-model-selection]] | [[faster-whisper-update]] | [[mimo-v25-evaluation]]*
