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
| Gemma 4 E4B (8-bit quantized) | ~5 GB |
| OS + overhead | ~2 GB |
| **Total** | **~10 GB** (muat di T4 16GB) |

> **Note:** Sequential loading — Whisper dan Gemma load bergantian, tidak bersamaan.

---

## Script Lengkap

```python
# ============================================================
# PIPELINE: Transcript + OCR + Summarization
# Stack: Faster-Whisper large-v3 | Gemma 4 E4B
# Target: Kaggle Free GPU (T4 16GB)
# ============================================================

!pip install -q faster-whisper accelerate librosa soundfile bitsandbytes
!pip install -q git+https://github.com/huggingface/transformers.git  # butuh versi terbaru untuk Gemma 4
!pip install -q "pillow==11.1.0" --force-reinstall --no-deps  # fix pillow conflict dengan torchvision

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
# NOTE: Whisper dan Gemma TIDAK bisa load bersamaan di T4 (16GB).
# Strategy: load Whisper → transcribe → unload → load Gemma.

from faster_whisper import WhisperModel, BatchedInferencePipeline

def load_whisper():
    print("\n[1/3] Loading Faster-Whisper large-v3...")
    t0 = time.time()
    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
        download_root="/kaggle/working/whisper_cache"
    )
    batched = BatchedInferencePipeline(model=model)
    print(f"  ✅ Whisper loaded in {time.time()-t0:.1f}s")
    return model, batched

def unload_whisper(model):
    """Unload Whisper dari VRAM."""
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  🧹 Whisper unloaded from VRAM")

# Load Whisper
whisper_model, batched_whisper = load_whisper()


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

GEMMA_MODEL_ID = "google/gemma-4-E4B-it"

def load_gemma():
    print("\n[2/3] Loading Gemma 4 E4B...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, trust_remote_code=True)

    # Cari GPU dengan VRAM paling banyak
    free_mem = [(i, torch.cuda.mem_get_info(i)[0]) for i in range(torch.cuda.device_count())]
    best_gpu = max(free_mem, key=lambda x: x[1])
    print(f"  📊 GPU options: {[(i, f'{m/1e9:.1f}GB free') for i, m in free_mem]}")
    print(f"  🎯 Using GPU {best_gpu[0]} ({best_gpu[1]/1e9:.1f}GB free)")

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_MODEL_ID,
        quantization_config=bnb_config,
        device_map=f"cuda:{best_gpu[0]}",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  ✅ Gemma loaded in {time.time()-t0:.1f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return proc, model

def unload_gemma(proc, model):
    """Unload Gemma dari VRAM."""
    import gc
    del model, proc
    gc.collect()
    torch.cuda.empty_cache()
    print("  🧹 Gemma unloaded from VRAM")


def gemma_generate(proc, model, messages: list, max_new_tokens: int = 512) -> str:
    """Helper untuk generate dari Gemma 4 E4B."""
    inputs = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return proc.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    ).strip()


def ocr_image(proc, model, image_input, language_hint: str = "Indonesia") -> str:
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

    return gemma_generate(proc, model, messages, max_new_tokens=800)


def summarize_text(proc, model, text: str, output_language: str = "Indonesia",
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

    return gemma_generate(proc, model, messages, max_new_tokens=600)


def summarize_transcript(proc, model, transcript: str, context: str = "") -> str:
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

    return gemma_generate(proc, model, messages, max_new_tokens=800)


# ============================================================
# MODULE 2.5: AUTO-DETECT & JALANKAN PIPELINE
# ============================================================
# Upload file ke Kaggle (drag & drop), lalu run cell ini.
# Script akan auto-detect audio & foto di /kaggle/input/

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}

def scan_kaggle_input():
    """Scan /kaggle/input/ untuk audio dan gambar."""
    input_dir = "/kaggle/input"
    audio_files = []
    image_files = []

    if not os.path.exists(input_dir):
        return audio_files, image_files

    for root, dirs, files_list in os.walk(input_dir):
        for f in files_list:
            ext = os.path.splitext(f)[1].lower()
            full_path = os.path.join(root, f)
            if ext in AUDIO_EXTS:
                audio_files.append(full_path)
            elif ext in IMAGE_EXTS:
                image_files.append(full_path)

    return audio_files, image_files

# ── Scan ──
print("=" * 50)
print("🔍 SCANNING /kaggle/input/...")
print("=" * 50)

audio_files, image_files = scan_kaggle_input()

# ── Tampilkan hasil scan ──
print(f"\n🎙️ Audio ditemukan: {len(audio_files)}")
for i, f in enumerate(audio_files):
    size = os.path.getsize(f) / 1e6
    print(f"  [{i}] {os.path.basename(f)} ({size:.1f} MB)  ← {f}")

print(f"\n📷 Foto ditemukan: {len(image_files)}")
for i, f in enumerate(image_files):
    size = os.path.getsize(f) / 1e6
    print(f"  [{i}] {os.path.basename(f)} ({size:.1f} MB)  ← {f}")

print("=" * 50)

# ── Pilih file ──
if not audio_files and not image_files:
    print("\n⚠️ Tidak ada file ditemukan di /kaggle/input/")
    print("   Upload file via Kaggle UI (drag & drop), lalu run cell ini lagi.")
else:
    # Pilih audio
    AUDIO_PATH = ""
    if audio_files:
        if len(audio_files) == 1:
            AUDIO_PATH = audio_files[0]
            print(f"\n🎙️ Auto-pilih audio: {os.path.basename(AUDIO_PATH)}")
        else:
            idx = input(f"\nPilih audio (0-{len(audio_files)-1}, atau Enter untuk skip): ").strip()
            if idx.isdigit() and 0 <= int(idx) < len(audio_files):
                AUDIO_PATH = audio_files[int(idx)]

    # Pilih foto
    IMAGE_PATHS = []
    if image_files:
        if len(image_files) == 1:
            IMAGE_PATHS = image_files
            print(f"📷 Auto-pilih foto: {os.path.basename(image_files[0])}")
        else:
            idx = input(f"Pilih foto (0-{len(image_files)-1}, atau Enter untuk semua): ").strip()
            if idx.isdigit() and 0 <= int(idx) < len(image_files):
                IMAGE_PATHS = [image_files[int(idx)]]
            elif not idx:
                IMAGE_PATHS = image_files  # semua

    print("\n" + "=" * 50)
    print("🚀 MENJALANKAN PIPELINE")
    print("=" * 50)

    result = {}

    # ── Phase 1: Transcript ──
    if AUDIO_PATH:
        print("\n🎙️ [1/2] Transcribing audio...")
        whisper_m, batched = load_whisper()
        transcript = transcribe_audio(AUDIO_PATH, language="id")
        result["transcript"] = transcript["text"]
        print(f"  ✅ Transcript: {len(transcript['text'])} chars")
        unload_whisper(whisper_m)

    # ── Phase 2: OCR + Summarize ──
    print("\n🤖 [2/2] Loading Gemma 4 E4B...")
    proc, gemma = load_gemma()

    if AUDIO_PATH and result.get("transcript"):
        print("📋 Summarizing...")
        result["summary"] = summarize_transcript(
            proc, gemma, result["transcript"], context="rekaman audio"
        )
        print("  ✅ Summary done")

    if IMAGE_PATHS:
        result["ocr"] = []
        for img_path in IMAGE_PATHS:
            print(f"📄 OCR: {os.path.basename(img_path)}")
            ocr = ocr_image(proc, gemma, img_path)
            result["ocr"].append({"file": os.path.basename(img_path), "text": ocr})

    unload_gemma(proc, gemma)

    # ── Hasil ──
    print("\n" + "=" * 50)
    print("📊 HASIL")
    print("=" * 50)

    if result.get("transcript"):
        print(f"\n🎙️ TRANSCRIPT:")
        print(result["transcript"][:500] + ("..." if len(result["transcript"]) > 500 else ""))

    if result.get("summary"):
        print(f"\n📋 RINGKASAN:")
        print(result["summary"])

    if result.get("ocr"):
        for item in result["ocr"]:
            print(f"\n📄 OCR — {item['file']}:")
            print(item["text"][:300] + ("..." if len(item["text"]) > 300 else ""))

    print("\n" + "=" * 50)
    print("✅ Selesai!")
    print("=" * 50)


# ============================================================
# MODULE 3: FULL PIPELINE (Sequential Loading)
# ============================================================
# Whisper dan Gemma load bergantian — tidak muat di VRAM bersamaan.

def full_pipeline(audio_path: str, image_paths: list = None) -> dict:
    """
    Pipeline: Whisper (transcribe) → unload → Gemma (OCR + summarize).

    Args:
        audio_path: Path ke file audio
        image_paths: List path ke gambar (untuk OCR)

    Returns:
        dict dengan keys: transcript, summary, ocr_results
    """
    output = {}

    # ── Phase 1: Transcript (Whisper) ──
    print("🎙️ [Phase 1] Transcribing with Whisper...")
    transcript = transcribe_audio(audio_path, language="id")
    output["transcript"] = transcript["text"]
    print(f"  ✅ Transcript: {len(transcript['text'])} chars")

    # Unload Whisper, free VRAM
    unload_whisper(whisper_model)

    # ── Phase 2: OCR + Summarize (Gemma) ──
    print("\n🤖 [Phase 2] Loading Gemma 4 E4B...")
    proc, gemma = load_gemma()

    print("📋 Summarizing transcript...")
    output["summary"] = summarize_transcript(
        proc, gemma, transcript["text"], context="rekaman audio"
    )
    print(f"  ✅ Summary done")

    if image_paths:
        output["ocr_results"] = []
        for img_path in image_paths:
            print(f"📄 OCR: {img_path}")
            ocr = ocr_image(proc, gemma, img_path)
            output["ocr_results"].append(ocr)

    # Unload Gemma
    unload_gemma(proc, gemma)

    return output


# ============================================================
# RUN: Widget sudah ditampilkan di atas.
# Klik "🎙️ Upload Audio" atau "📷 Upload Foto Nota",
# lalu klik "🚀 Run Pipeline".
# ============================================================
print("✅ Semua module loaded. Gunakan widget di atas untuk upload & proses.")
```

---

*Ref: [[gemma4-model-selection]] | [[faster-whisper-update]] | [[mimo-v25-evaluation]]*
