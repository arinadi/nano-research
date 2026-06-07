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
| Gemma 4 E4B (bfloat16, 1 GPU) | ~8 GB |
| MTP Drafter (~277 MB) | ~0.3 GB |
| OS + overhead | ~2 GB |
| **Total** | **~13 GB** (1 GPU, sequential) |

> **Note:** Sequential loading — Whisper dan Gemma load bergantian.
> MTP drafter memberikan ~2× speedup tanpa tambahan VRAM signifikan.

---

## Script Lengkap

```python
# ============================================================
# PIPELINE: Transcript + OCR + Summarization
# Stack: Faster-Whisper large-v3 | Gemma 4 E4B
# Target: Kaggle Free GPU (T4 16GB)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os, sys
# Suppress pip dependency warnings (dask-cuda, cuml, cudf conflicts — tidak dipakai pipeline ini)
!pip install -q faster-whisper accelerate librosa soundfile bitsandbytes 2>&1 | grep -v "ERROR\|requires\|incompatible"
!pip install -q git+https://github.com/huggingface/transformers.git 2>&1 | grep -v "ERROR\|requires\|incompatible"
!pip install -q "pillow==11.1.0" --force-reinstall --no-deps 2>&1 | grep -v "ERROR\|requires\|incompatible"

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

GEMMA_DRAFT_ID = "google/gemma-4-E4B-it-assistant"  # MTP drafter (~277 MB)

def load_gemma():
    print("\n[2/3] Loading Gemma 4 E4B + MTP drafter...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, trust_remote_code=True)

    # Tampilkan info GPU
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  📊 GPU {i}: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")

    # Target model — satu GPU saja, hindari tensor parallel overhead
    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # MTP drafter — prediksi beberapa token sekaligus, kualitas identik
    drafter = AutoModelForImageTextToText.from_pretrained(
        GEMMA_DRAFT_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    drafter.eval()

    print(f"  ✅ Gemma + MTP loaded in {time.time()-t0:.1f}s")
    used = torch.cuda.memory_allocated(0) / 1e9
    print(f"  📊 GPU 0 VRAM used: {used:.2f} GB")
    return proc, model, drafter

def unload_gemma(proc, model, drafter=None):
    """Unload Gemma + drafter dari VRAM."""
    import gc
    if drafter is not None:
        del drafter
    del model, proc
    gc.collect()
    torch.cuda.empty_cache()
    print("  🧹 Gemma + MTP unloaded from VRAM")


def gemma_generate(proc, model, messages: list, max_new_tokens: int = 512, drafter=None) -> dict:
    """Helper untuk generate dari Gemma 4 E4B. Return dict dengan text + metrics."""
    inputs = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda:0")

    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    # MTP drafter untuk speculative decoding (~2× lebih cepat)
    if drafter is not None:
        gen_kwargs["assistant_model"] = drafter

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - t0

    output_ids = outputs[0][input_len:]
    text = proc.decode(output_ids, skip_special_tokens=True).strip()
    tokens_out = len(output_ids)
    tps = tokens_out / elapsed if elapsed > 0 else 0

    return {
        "text": text,
        "input_tokens": input_len,
        "output_tokens": tokens_out,
        "time_s": round(elapsed, 2),
        "tps": round(tps, 1),
    }


def ocr_image(proc, model, image_input, language_hint: str = "Indonesia", drafter=None) -> dict:
    """OCR + baca teks dari gambar. Return dict dengan text + metrics."""
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
                f"Output hanya teks yang terlihat di gambar, tanpa penjelasan atau analisis."
            )}
        ]
    }]

    return gemma_generate(proc, model, messages, max_new_tokens=1000, drafter=drafter)


def summarize_text(proc, model, text: str, output_language: str = "Indonesia",
                   style: str = "ringkas", drafter=None) -> dict:
    """Summarize teks panjang. Return dict dengan summary + metrics."""
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

    return gemma_generate(proc, model, messages, max_new_tokens=600, drafter=drafter)


def summarize_transcript(proc, model, transcript: str, context: str = "", drafter=None) -> dict:
    """Summarize hasil transcript audio. Return dict dengan summary + metrics."""
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

    return gemma_generate(proc, model, messages, max_new_tokens=800, drafter=drafter)


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
    metrics = []

    # ── Phase 1: Transcript ──
    if AUDIO_PATH:
        print("\n🎙️ [1/2] Transcribing audio...")
        whisper_m, batched = load_whisper()
        t0 = time.time()
        transcript = transcribe_audio(AUDIO_PATH, language="id")
        whisper_time = time.time() - t0
        result["transcript"] = transcript["text"]
        print(f"  ✅ Transcript: {len(transcript['text'])} chars ({whisper_time:.1f}s)")
        metrics.append(f"Whisper transcript: {whisper_time:.1f}s | {len(transcript['text'])} chars")
        unload_whisper(whisper_m)

    # ── Phase 2: OCR + Summarize ──
    print("\n🤖 [2/2] Loading Gemma 4 E4B + MTP drafter...")
    proc, gemma, drafter = load_gemma()

    if AUDIO_PATH and result.get("transcript"):
        # ── Bonus: Gemma Transcript Test (per 30s chunk) ──
        print("\n🎙️ [Bonus] Testing Gemma transcript (per 30s chunk)...")
        try:
            import librosa
            import soundfile as sf

            audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
            duration = len(audio) / sr
            chunk_sec = 30
            chunk_samples = chunk_sec * sr
            chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]

            print(f"  📊 Audio: {duration:.1f}s → {len(chunks)} chunk(s) à {chunk_sec}s")

            gemma_transcript_parts = []
            gemma_transcript_total_time = 0
            gemma_transcript_total_tokens = 0
            for ci, chunk in enumerate(chunks):
                chunk_path = f"/tmp/gemma_chunk_{ci}.wav"
                sf.write(chunk_path, chunk, sr)

                # Kirim audio ke Gemma — pakai numpy array langsung
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": chunk,  # numpy array langsung
                        },
                        {
                            "type": "text",
                            "text": (
                                "Transkrip audio ini secara lengkap dan akurat dalam Bahasa Indonesia. "
                                "Output hanya teks transkrip, tanpa komentar."
                            )
                        }
                    ]
                }]

                print(f"  🎙️ Chunk {ci+1}/{len(chunks)}...", end=" ", flush=True)
                chunk_result = gemma_generate(proc, gemma, messages, max_new_tokens=800, drafter=drafter)
                gemma_transcript_parts.append(chunk_result["text"])
                gemma_transcript_total_time += chunk_result["time_s"]
                gemma_transcript_total_tokens += chunk_result["output_tokens"]
                print(f"✅ {chunk_result['output_tokens']} tokens | {chunk_result['time_s']}s | {chunk_result['tps']} tok/s")

                os.remove(chunk_path)

            gemma_full_transcript = " ".join(gemma_transcript_parts)
            result["gemma_transcript"] = gemma_full_transcript
            gemma_tps_avg = gemma_transcript_total_tokens / gemma_transcript_total_time if gemma_transcript_total_time > 0 else 0
            print(f"  ✅ Gemma transcript total: {len(gemma_full_transcript)} chars | {gemma_transcript_total_time:.1f}s | {gemma_tps_avg:.1f} tok/s")

            metrics.append(f"Gemma transcript ({len(chunks)} chunks): {len(gemma_full_transcript)} chars | {gemma_transcript_total_time:.1f}s | {gemma_tps_avg:.1f} tok/s")

        except Exception as e:
            print(f"  ⚠️ Gemma transcript error: {e}")

        print("📋 Summarizing...")
        summary_result = summarize_transcript(
            proc, gemma, result["transcript"], context="rekaman audio", drafter=drafter
        )
        result["summary"] = summary_result["text"]
        print(f"  ✅ Summary: {summary_result['output_tokens']} tokens | {summary_result['time_s']}s | {summary_result['tps']} tok/s")
        metrics.append(f"Summarize: {summary_result['time_s']}s | {summary_result['output_tokens']} tokens | {summary_result['tps']} tok/s")

    if IMAGE_PATHS:
        result["ocr"] = []
        for img_path in IMAGE_PATHS:
            print(f"📄 OCR: {os.path.basename(img_path)}")

            # Tampilkan foto asli
            try:
                from IPython.display import display as ipy_display
                img_preview = Image.open(img_path)
                if max(img_preview.size) > 600:
                    img_preview.thumbnail((600, 600), Image.LANCZOS)
                ipy_display(img_preview)
            except Exception:
                pass

            ocr = ocr_image(proc, gemma, img_path, drafter=drafter)
            result["ocr"].append({"file": os.path.basename(img_path), "text": ocr["text"]})
            print(f"  ✅ OCR: {ocr['output_tokens']} tokens | {ocr['time_s']}s | {ocr['tps']} tok/s")
            metrics.append(f"OCR {os.path.basename(img_path)}: {ocr['time_s']}s | {ocr['output_tokens']} tokens | {ocr['tps']} tok/s")

    unload_gemma(proc, gemma, drafter)

    # ── Hasil ──
    print("\n" + "=" * 50)
    print("📊 HASIL")
    print("=" * 50)

    if result.get("transcript"):
        print(f"\n🎙️ TRANSCRIPT (Whisper):")
        print(result["transcript"][:500] + ("..." if len(result["transcript"]) > 500 else ""))

    if result.get("gemma_transcript"):
        print(f"\n🎙️ TRANSCRIPT (Gemma 4 E4B):")
        print(result["gemma_transcript"][:500] + ("..." if len(result["gemma_transcript"]) > 500 else ""))

    if result.get("summary"):
        print(f"\n📋 RINGKASAN:")
        print(result["summary"])

    if result.get("ocr"):
        for item in result["ocr"]:
            print(f"\n📄 OCR — {item['file']}:")
            print(item["text"][:500] + ("..." if len(item["text"]) > 500 else ""))

    # ── Metrics ──
    print("\n" + "=" * 50)
    print("⏱️  PERFORMANCE METRICS")
    print("=" * 50)
    for m in metrics:
        print(f"  • {m}")

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
