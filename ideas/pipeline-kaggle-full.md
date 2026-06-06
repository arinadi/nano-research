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
# MODULE 2.5: UPLOAD WIDGET
# ============================================================

import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import files
import shutil

# ── State ──
upload_state = {"audio": None, "images": []}

# ── Widget: Audio Upload ──
audio_output = widgets.Output()
audio_btn = widgets.Button(description="🎙️ Upload Audio", button_style="primary",
                           layout=widgets.Layout(width="200px"))

def on_audio_upload(btn):
    with audio_output:
        clear_output()
        print("📂 Pilih file audio (WAV, MP3, FLAC, M4A)...")
        uploaded = files.upload()
        if uploaded:
            fname = list(uploaded.keys())[0]
            dest = f"/tmp/upload_{fname}"
            shutil.move(fname, dest)
            upload_state["audio"] = dest
            print(f"✅ Audio: {fname} ({os.path.getsize(dest)/1e6:.1f} MB)")

audio_btn.on_click(on_audio_upload)

# ── Widget: Image Upload (multi) ──
image_output = widgets.Output()
image_btn = widgets.Button(description="📷 Upload Foto Nota", button_style="success",
                           layout=widgets.Layout(width="200px"))
image_list_label = widgets.Label(value="Belum ada foto")

def on_image_upload(btn):
    with image_output:
        clear_output()
        print("📂 Pilih foto nota/invoice (bisa multiple)...")
        uploaded = files.upload()
        for fname in uploaded.keys():
            dest = f"/tmp/upload_{fname}"
            shutil.move(fname, dest)
            upload_state["images"].append(dest)
            print(f"  ✅ {fname} ({os.path.getsize(dest)/1e6:.1f} MB)")
        image_list_label.value = f"{len(upload_state['images'])} foto siap"

image_btn.on_click(on_image_upload)

# ── Widget: Reset ──
reset_btn = widgets.Button(description="🗑️ Reset", button_style="danger",
                           layout=widgets.Layout(width="100px"))

def on_reset(btn):
    upload_state["audio"] = None
    upload_state["images"] = []
    image_list_label.value = "Belum ada foto"
    with audio_output:
        clear_output()
        print("Audio: belum ada")
    with image_output:
        clear_output()
        print("Foto: belum ada")

reset_btn.on_click(on_reset)

# ── Widget: Run Pipeline ──
run_btn = widgets.Button(description="🚀 Run Pipeline", button_style="warning",
                         layout=widgets.Layout(width="200px"))
run_output = widgets.Output()

def on_run(btn):
    if not upload_state["audio"] and not upload_state["images"]:
        with run_output:
            clear_output()
            print("⚠️ Upload audio atau foto dulu!")
        return

    with run_output:
        clear_output()
        print("=" * 50)
        print("🚀 MEMPROSES...")
        print("=" * 50)

        result = {}

        # Transcript
        if upload_state["audio"]:
            print("\n🎙️ [1] Transcribing audio...")
            whisper_m, batched = load_whisper()
            transcript = transcribe_audio(upload_state["audio"], language="id")
            result["transcript"] = transcript["text"]
            print(f"  ✅ {len(transcript['text'])} chars")
            unload_whisper(whisper_m)

        # OCR + Summary
        if upload_state["audio"] or upload_state["images"]:
            print("\n🤖 [2] Loading Gemma 4 E4B...")
            proc, gemma = load_gemma()

            if upload_state["audio"] and result.get("transcript"):
                print("📋 Summarizing...")
                result["summary"] = summarize_transcript(
                    proc, gemma, result["transcript"], context="rekaman audio"
                )
                print("  ✅ Summary done")

            if upload_state["images"]:
                result["ocr"] = []
                for img_path in upload_state["images"]:
                    print(f"📄 OCR: {os.path.basename(img_path)}")
                    ocr = ocr_image(proc, gemma, img_path)
                    result["ocr"].append({"file": os.path.basename(img_path), "text": ocr})
                    print(f"  ✅ Done")

            unload_gemma(proc, gemma)

        # ── Tampilkan Hasil ──
        print("\n" + "=" * 50)
        print("📊 HASIL")
        print("=" * 50)

        if result.get("transcript"):
            print(f"\n🎙️ TRANSCRIPT ({len(result['transcript'])} chars):")
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

run_btn.on_click(on_run)

# ── Layout ──
header = widgets.HTML("<h3>📁 Upload Files</h3>")
audio_section = widgets.VBox([
    widgets.HTML("<b>Audio (untuk transkripsi + ringkasan):</b>"),
    audio_btn, audio_output
])
image_section = widgets.VBox([
    widgets.HTML("<b>Foto Nota (untuk OCR):</b>"),
    image_btn, image_list_label, image_output
])
actions = widgets.HBox([run_btn, reset_btn])

display(widgets.VBox([
    header,
    audio_section,
    image_section,
    widgets.HBox([actions]),
    run_output,
]))


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
# TEST — Sequential: Whisper dulu, lalu Gemma
# ============================================================

print("\n[3/3] Running pipeline tests (sequential)...")
print("-" * 40)

# --- TEST A: Whisper Transcript ---
print("\n🎙️ TEST A: Whisper Transcript")
# (Skip jika tidak ada file audio — uncomment untuk test)
# result = transcribe_audio("/kaggle/input/your-audio/recording.mp3", language="id")
# print(f"  Hasil: {result['text'][:200]}...")
print("  (Skip — tidak ada file audio untuk test)")

# Unload Whisper
unload_whisper(whisper_model)

# --- Load Gemma ---
print("\n🤖 Loading Gemma 4 E4B...")
proc, gemma = load_gemma()

# --- TEST B: OCR ---
print("\n📄 TEST B: OCR dari gambar teks")

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
ocr_result = ocr_image(proc, gemma, test_img, language_hint="Indonesia")
print(f"  Hasil OCR ({time.time()-t0:.1f}s):")
print(f"  {ocr_result}")

# --- TEST C: Summarization ---
print("\n📋 TEST C: Summarization teks panjang")

sample_text = """
Transformasi digital di sektor kesehatan Indonesia mengalami percepatan signifikan
sejak 2024. Kementerian Kesehatan meluncurkan platform SATUSEHAT yang mengintegrasikan
rekam medis elektronik dari lebih dari 3.000 fasilitas kesehatan.
"""

t0 = time.time()
summary_result = summarize_text(proc, gemma, sample_text, output_language="Indonesia", style="ringkas")
print(f"  Hasil Summary ({time.time()-t0:.1f}s):")
print(f"  {summary_result}")

# --- CLEANUP ---
unload_gemma(proc, gemma)

print("\n" + "=" * 60)
print("✅ Semua test selesai!")
print("=" * 60)
```

---

*Ref: [[gemma4-model-selection]] | [[faster-whisper-update]] | [[mimo-v25-evaluation]]*
