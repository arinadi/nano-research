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
| vLLM Gemma 4 E4B + MTP (FP16, 80% util) | ~11.6 GB |
| OS + overhead | ~1.5 GB |
| **Total** | **~13 GB** (1 GPU, 14.56 GB available) |

> **Note:** Whisper di-load SEBELUM vLLM start, lalu di-unload.
> T4 tidak support bfloat16 → pakai float16 (`--dtype half`).
> Estimasi throughput: **~15–25 tok/s** (vs ~6 tok/s HF Transformers).

---

## Script Lengkap

```python
# ============================================================
# PIPELINE: Transcript + OCR + Summarization
# Stack: Faster-Whisper large-v3 | vLLM + MTP (Gemma 4 E4B)
# Target: Kaggle Free GPU (T4 16GB)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os, sys
# Suppress pip dependency warnings
!pip install -q faster-whisper librosa soundfile 2>&1 | grep -v "ERROR\|requires\|incompatible"
!pip install -q vllm "vllm[audio]" 2>&1 | grep -v "ERROR\|requires\|incompatible"

import os, time, torch, gc, base64, pathlib, subprocess, signal
import numpy as np
from PIL import Image
from io import BytesIO

print("=" * 60)
print("PIPELINE: Whisper large-v3 + vLLM + MTP (Gemma 4 E4B)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# ============================================================
# MODULE 1: FASTER-WHISPER TRANSCRIPT
# ============================================================

import subprocess as _sp
import json as _json

def transcribe_audio(audio_path: str, language: str = "id") -> dict:
    """
    Transcript audio via Whisper di subprocess terpisah.
    VRAM benar-benar dibebaskan oleh OS setelah selesai.
    """
    print(f"  🎙️ Whisper subprocess: {os.path.basename(audio_path)}")
    t0 = time.time()

    script = f'''
import json, sys, warnings
warnings.filterwarnings("ignore")
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16",
                     download_root="/kaggle/working/whisper_cache")
segments, info = model.transcribe(
    "{audio_path.replace(chr(92), "/")}",
    language={"None" if not language else repr(language)},
    beam_size=5, vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
result = {{"text": " ".join(s.text.strip() for s in segments),
           "language": info.language, "duration": info.duration}}
print(json.dumps(result))
'''

    proc = _sp.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"Whisper failed:\n{proc.stderr[-500:]}")

    last_line = proc.stdout.strip().split("\n")[-1]
    result = _json.loads(last_line)
    print(f"  ✅ Transcript: {len(result['text'])} chars ({elapsed:.1f}s)")
    return result


# ============================================================
# MODULE 2: vLLM + MTP SERVER (Gemma 4 E4B)
# ============================================================
# vLLM serve Gemma 4 E4B + MTP drafter via OpenAI-compatible API.
# Throughput ~25–35 tok/s di T4 (vs ~6 tok/s HF Transformers).

from openai import OpenAI

VLLM_PORT = 8000
VLLM_MODEL = "google/gemma-4-E4B-it"
VLLM_DRAFTER = "google/gemma-4-E4B-it-assistant"

def start_vllm_server():
    """
    Start vLLM server di SUBPROCESS terpisah.
    VRAM benar-benar bebas — tidak ada PyTorch allocator dari Jupyter kernel.
    """
    print("\n[2/3] Starting vLLM + MTP server (subprocess)...")
    t0 = time.time()

    # Cari GPU dengan VRAM paling banyak
    best_gpu = 0
    best_free = 0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  📊 GPU {i}: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")
        if free > best_free:
            best_free = free
            best_gpu = i

    print(f"  🎯 Target: GPU {best_gpu} ({best_free/1e9:.1f}GB free)")

    # Tulis vLLM server script ke file (hindari nested subprocess issues)
    vllm_script_path = "/tmp/vllm_server.py"
    with open(vllm_script_path, "w") as f:
        f.write(f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{best_gpu}"
os.system("python -m vllm.entrypoints.openai.api_server "
          "--model {VLLM_MODEL} "
          "--dtype half "
          "--max-model-len 16384 "
          "--gpu-memory-utilization 0.92 "
          '--limit-mm-per-prompt \'{{"image": 2, "audio": 1}}\' '
          "--spec-method mtp "
          "--spec-model {VLLM_DRAFTER} "
          "--spec-tokens 3 "
          "--host 0.0.0.0 "
          "--port {VLLM_PORT} "
          "--enforce-eager")
''')

    vllm_proc = subprocess.Popen(
        [sys.executable, vllm_script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Tunggu server ready (max 180s — model download butuh waktu)
    client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="EMPTY")
    for i in range(180):
        try:
            client.models.list()
            break
        except Exception:
            if vllm_proc.poll() is not None:
                output = vllm_proc.stdout.read()
                raise RuntimeError(f"vLLM server crashed:\n{output[-2000:]}")
            time.sleep(1)
    else:
        raise RuntimeError("vLLM server timeout (180s)")

    elapsed = time.time() - t0
    print(f"  ✅ vLLM + MTP server ready in {elapsed:.1f}s (port {VLLM_PORT})")
    return vllm_proc, client


def stop_vllm_server(vllm_proc):
    """Stop vLLM server."""
    if vllm_proc and vllm_proc.poll() is None:
        vllm_proc.terminate()
        vllm_proc.wait(timeout=10)
        print("  🧹 vLLM server stopped")


def vllm_generate(client, messages: list, max_new_tokens: int = 4096) -> dict:
    """Generate via vLLM OpenAI-compatible API. Return dict dengan text + metrics."""
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0,
    )
    elapsed = time.perf_counter() - t0

    text = response.choices[0].message.content or ""
    usage = response.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0
    tps = tokens_out / elapsed if elapsed > 0 else 0

    return {
        "text": text.strip(),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "time_s": round(elapsed, 2),
        "tps": round(tps, 1),
    }


def encode_image_base64(image_input) -> str:
    """Encode image ke base64 untuk vLLM API."""
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            import requests
            return base64.b64encode(requests.get(image_input).content).decode()
        else:
            return base64.b64encode(pathlib.Path(image_input).read_bytes()).decode()
    elif isinstance(image_input, Image.Image):
        buf = BytesIO()
        image_input.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    else:
        raise ValueError("image_input harus path, URL, atau PIL.Image")


def encode_audio_base64(audio_path: str) -> tuple[str, str]:
    """Encode audio ke base64. Return (base64_data, mime_type)."""
    ext = pathlib.Path(audio_path).suffix.lstrip(".").lower()
    mime = {"m4a": "audio/mp4", "mp3": "audio/mpeg", "wav": "audio/wav",
            "flac": "audio/flac", "ogg": "audio/ogg"}.get(ext, "audio/wav")
    data = base64.b64encode(pathlib.Path(audio_path).read_bytes()).decode()
    return data, mime


def ocr_image(client, image_input, language_hint: str = "Indonesia") -> dict:
    """OCR + baca teks dari gambar via vLLM. Return dict dengan text + metrics."""
    img_b64 = encode_image_base64(image_input)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": (
                    f"Baca semua teks dalam gambar ini secara lengkap dan akurat. "
                    f"Bahasa dominan: {language_hint}. "
                    f"Pertahankan struktur asli (baris, spasi, tabel). "
                    f"Jika ada angka atau harga, jangan ada yang terlewat. "
                    f"Output hanya teks yang terlihat di gambar, tanpa penjelasan atau analisis."
                )
            }
        ]
    }]

    return vllm_generate(client, messages, max_new_tokens=4096)


def summarize_transcript(client, transcript: str, context: str = "") -> dict:
    """Summarize hasil transcript audio. Return dict dengan summary + metrics."""
    context_str = f"Konteks: {context}\n" if context else ""

    messages = [{
        "role": "user",
        "content": (
            f"{context_str}"
            f"Berikut adalah transcript audio. Buat ringkasan/notulensi dalam Bahasa Indonesia:\n"
            f"- Poin-poin utama yang dibahas\n"
            f"- Keputusan atau action item (jika ada)\n"
            f"- Kesimpulan singkat\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )
    }]

    return vllm_generate(client, messages, max_new_tokens=4096)


def transcribe_audio_gemma(client, audio_path: str) -> dict:
    """Transcribe audio via Gemma 4 E4B (native audio support)."""
    audio_b64, mime = encode_audio_base64(audio_path)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:{mime};base64,{audio_b64}"},
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

    return vllm_generate(client, messages, max_new_tokens=4096)


def transcribe_audio_gemma_chunked(client, audio_path: str, chunk_sec: int = 30) -> dict:
    """Transcribe audio panjang per chunk via Gemma 4 E4B."""
    import librosa
    import soundfile as sf

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    chunk_samples = chunk_sec * sr
    chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]

    print(f"  📊 Audio: {duration:.1f}s → {len(chunks)} chunk(s) à {chunk_sec}s")

    parts = []
    total_time = 0
    total_tokens = 0

    for ci, chunk in enumerate(chunks):
        chunk_path = f"/tmp/gemma_chunk_{ci}.wav"
        sf.write(chunk_path, chunk, sr)

        print(f"  🎙️ Chunk {ci+1}/{len(chunks)}...", end=" ", flush=True)
        result = transcribe_audio_gemma(client, chunk_path)
        parts.append(result["text"])
        total_time += result["time_s"]
        total_tokens += result["output_tokens"]
        print(f"✅ {result['output_tokens']} tokens | {result['time_s']}s | {result['tps']} tok/s")

        os.remove(chunk_path)

    full_text = " ".join(parts)
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    return {
        "text": full_text,
        "output_tokens": total_tokens,
        "time_s": round(total_time, 2),
        "tps": round(avg_tps, 1),
    }


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

    # ── Phase 1: Transcript (via subprocess — VRAM bebas setelah selesai) ──
    if AUDIO_PATH:
        print("\n🎙️ [1/2] Transcribing audio (subprocess)...")
        transcript = transcribe_audio(AUDIO_PATH, language="id")
        result["transcript"] = transcript["text"]
        metrics.append(f"Whisper transcript: {len(transcript['text'])} chars")

    # ── Phase 2: Start vLLM + MTP Server ──
    print("\n🤖 [2/2] Starting vLLM + MTP server...")
    vllm_proc, client = start_vllm_server()

    if AUDIO_PATH and result.get("transcript"):
        # ── Bonus: Gemma Transcript Test ──
        print("\n🎙️ [Bonus] Testing Gemma transcript via vLLM...")
        try:
            gemma_result = transcribe_audio_gemma_chunked(client, AUDIO_PATH, chunk_sec=30)
            result["gemma_transcript"] = gemma_result["text"]
            print(f"  ✅ Gemma transcript: {len(gemma_result['text'])} chars | {gemma_result['time_s']}s | {gemma_result['tps']} tok/s")
            metrics.append(f"Gemma transcript: {gemma_result['time_s']}s | {gemma_result['output_tokens']} tokens | {gemma_result['tps']} tok/s")
        except Exception as e:
            print(f"  ⚠️ Gemma transcript error: {e}")

        print("📋 Summarizing...")
        summary_result = summarize_transcript(
            client, result["transcript"], context="rekaman audio"
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

            ocr = ocr_image(client, img_path)
            result["ocr"].append({"file": os.path.basename(img_path), "text": ocr["text"]})
            print(f"  ✅ OCR: {ocr['output_tokens']} tokens | {ocr['time_s']}s | {ocr['tps']} tok/s")
            metrics.append(f"OCR {os.path.basename(img_path)}: {ocr['time_s']}s | {ocr['output_tokens']} tokens | {ocr['tps']} tok/s")

    # ── Cleanup ──
    stop_vllm_server(vllm_proc)

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
