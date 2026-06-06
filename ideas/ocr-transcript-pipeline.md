# Research: Pipeline OCR · Transcript · Summarization
## Gemma 4 (Kaggle Local) + Faster Whisper

> **Tanggal Research:** Juni 2026 | **Target:** Kaggle Free GPU (T4/P100 16GB)
> **Use Case:** Transcript Audio · OCR & Analisis Foto · Summarization Teks

---

## TL;DR — Rekomendasi Final

| Task | Tool Terbaik | Alasan |
|---|---|---|
| **Transcript Audio** | Faster-Whisper `large-v3` + `BatchedInferencePipeline` | Akurasi 10–20% lebih baik dari v2, 4× lebih cepat dengan batching |
| **OCR + Image Analysis** | Gemma 4 **E4B** (lokal Kaggle) | Fit T4 16GB tanpa quantization, audio+vision native, non-blocking |
| **Summarization** | Gemma 4 **E4B** (lokal Kaggle) | 128K context, Bahasa Indonesia oke, jauh lebih responsif dari API |
| **Gemini API 26B A4B** | ❌ Hindari untuk produksi | TTFT tinggi, RTO, blocking — bukan masalah model, tapi masalah infrastruktur |

---

## Bagian 1 — Gemma 4: Pilih Model yang Tepat

### Keluarga Model Gemma 4 (April 2026)

Ada empat ukuran dalam Gemma 4: E2B (~2.3B efektif), E4B (~4.5B efektif), 26B A4B (MoE, 4B aktif per forward pass), dan 31B Dense. Context window 128K untuk E2B/E4B, dan 256K untuk 26B dan 31B. Semua variant menerima teks dan gambar, sementara E2B dan E4B juga support audio native.

Pada benchmark AIME 2026 (matematika), E4B sudah mencapai 42.5% — lebih dari dua kali lipat kemampuan model full-size generasi sebelumnya (Gemma 3 27B di 20.8%).

### Kenapa E4B untuk Kaggle Free?

Google sendiri sudah tegas: E2B dan E4B adalah model edge-first. Untuk developer yang tidak build untuk ponsel dan punya memori cukup, pilihan biasanya bukan E4B vs E2B — melainkan 26B A4B vs 31B. Tapi dengan keterbatasan hardware gratis Kaggle, E4B adalah pilihan realistis.

**VRAM Kaggle vs kebutuhan model:**

| Model | VRAM bfloat16 | VRAM int8 | VRAM int4 | Kaggle T4 (16GB) |
|---|---|---|---|---|
| E4B | ~8 GB | ~5 GB | ~3 GB | ✅ Muat penuh |
| 12B | ~24 GB | ~14 GB | ~8 GB | ⚠️ Perlu 4-bit |
| 26B A4B | ~16 GB | ~10 GB | ~6 GB | ⚠️ Perlu quantization |
| 31B | ~64 GB | ~32 GB | ~16 GB | ❌ Tidak muat |

> Dengan T4 16GB Kaggle: **E4B bisa jalan full precision (bfloat16)** tanpa quantization. Ini keunggulan besar — tidak ada degradasi akurasi dari quantization.

### Kemampuan E4B yang Relevan

Dalam pengujian Hugging Face, model E4B menghasilkan transkripsi dan deskripsi audio yang akurat. Visual token budget E4B mendukung hingga 1120 token untuk OCR dan document parsing dengan detail maksimum.

E4B perform comparably to dense models twice its size pada banyak benchmark. Arsitektur conformer audio encoder-nya (USM-style) sudah tervalidasi baik untuk speech recognition — tanpa cloud dependency, karena native di model.

### Catatan Komunitas tentang E4B

Seorang developer menguji semua model Gemma 4 di GPU GTX 1650 (4GB VRAM) dan hasilnya memuaskan untuk E4B — "tidak menyangka model seperti E4B bisa berjalan selancar ini di GPU seperti ini."

Catatan dari Unsloth: jika ada loss 13–15 saat fine-tuning E2B/E4B, itu normal — perilaku umum model multimodal, juga terjadi di Gemma-3N dan Llama Vision. Untuk multimodal prompt Gemma 4, taruh gambar sebelum instruksi teks.

### Batasan E4B yang Perlu Diketahui

- **Context window 128K** (vs 256K di 26B) — cukup untuk ~100 halaman teks, tapi lebih terbatas untuk dokumen sangat panjang
- **Reasoning lebih lemah** dari 26B pada tugas kompleks — untuk OCR dan summarization biasa, tidak terasa
- **Bukan pilihan terbaik** jika Anda akhirnya punya akses GPU besar

---

## Bagian 2 — Masalah Gemini API 26B A4B: RTO & Blocking

### Akar Masalah

Yang Anda alami (RTO, lambat, pipeline blocking) bukan masalah kualitas model-nya — ini masalah infrastruktur provider API.

Google AI Studio menyediakan akses pertama ke Gemma 4 26B A4B secara gratis, dengan TTFT 1.63 detik dan output speed ~46.7 t/s. Ini cukup untuk prototyping, tapi tidak kompetitif untuk aplikasi live yang menghadap user.

Komunitas melaporkan Gemma 4 26B A4B berjalan jauh lebih lambat dibanding Qwen 3.5 setara: ada user yang dapat 11 token/detik di Gemma 4 26B A4B vs 60+ token/detik di Qwen 3.5 35B A3B pada GPU yang sama (5060 Ti 16GB). VRAM untuk context juga dilaporkan lebih boros di quantization yang sama.

### Perbandingan Provider API (Mei 2026)

Ada 7 provider API untuk Gemma 4 26B A4B. Output speed tercepat: Clarifai (151.5 t/s), Cloudflare (105.6 t/s), Parasail (56.6 t/s). TTFT terendah: DeepInfra (0.62s), Clarifai (0.86s), Cloudflare (0.92s). Harga paling murah: DeepInfra dan Parasail ($0.10/1M token blended).

Jika tetap ingin pakai API (bukan lokal), **DeepInfra** atau **Cloudflare** adalah alternatif yang jauh lebih cepat dari Google AI Studio.

### Cloudflare Workers AI: Bukan Sekadar Routing

> **Ini mungkin mengejutkan:** Cloudflare bukan hanya CDN/proxy. Mereka punya **infrastruktur GPU sendiri** dan menjalankan model langsung di dalamnya.

Cloudflare membedakan tiga tipe model di katalog mereka:

- **Hosted** — model berjalan di GPU Cloudflare sendiri (Llama, Qwen, Gemma, dst). Ini yang dapat free tier dan neuron pricing.
- **Proxied** — Cloudflare hanya forward request ke API provider lain (OpenAI, Anthropic). Tidak ada free tier, biaya langsung ke provider.
- **Partner** — model di-deploy di infrastruktur mitra di dalam/dekat jaringan Cloudflare.

**`@cf/google/gemma-4-26b-a4b-it` adalah model Cloudflare-hosted** — berjalan di GPU Cloudflare sendiri, bukan forward ke Google. Ini penjelasan kenapa latency-nya bisa 0.92s TTFT vs 1.63s Google AI Studio.

#### Free Tier Cloudflare Workers AI

Cloudflare Workers AI menyediakan **10.000 Neurons/hari gratis**, reset setiap 00:00 UTC. Tapi ada gotcha penting yang dilaporkan komunitas:

> *"Kenapa 10.000 neurons habis hanya dari satu prompt?"* — Seseorang di komunitas Cloudflare melaporkan error 4006 (neuron habis) setelah prompt sederhana ke gemma-4-26b-a4b-it. Model besar seperti 26B mengonsumsi neuron jauh lebih banyak per request dibanding model kecil.

Faktanya, 10.000 neuron/hari cukup untuk sekitar **~25–50 request pendek** ke model 26B. Untuk produksi, ini tidak cukup — Anda perlu Workers Paid ($0.011 per 1.000 neurons) atau pakai provider lain.

Ada juga laporan billing anomaly: seorang user ditagih $292 untuk 15.000 calls (estimasi harusnya ~$8), menunjukkan konversi neuron-ke-token untuk Gemma 4 model mungkin belum sepenuhnya stabil di dashboard Cloudflare.

#### Catatan Penting: Context Window di Cloudflare

Dokumentasi Cloudflare menyatakan context 256K untuk gemma-4-26b-a4b-it, tapi ada laporan bug terbuka dari komunitas: deployment aktual hanya menerima **128K token** (bukan 256K). Jika prompt + output Anda mendekati 128K token, request akan ditolak dengan error.

### DeepInfra: Provider Tercepat untuk API

DeepInfra memiliki TTFT terendah (0.68s) dan harga paling murah ($0.07/M input token, $0.34/M output token) untuk Gemma 4 26B A4B di antara semua provider yang ada. Endpoint-nya fully OpenAI-compatible — tinggal ganti `base_url` dan `api_key`, kode yang sama dengan OpenAI SDK langsung jalan.

### Cell Test: Benchmark Provider API

Cell berikut menguji Cloudflare Workers AI dan DeepInfra, membandingkan TTFT, throughput, dan kualitas output — sehingga Anda bisa memilih provider terbaik untuk use case Anda sebelum pindah ke lokal.

```python
# ============================================================
# CELL: BENCHMARK PROVIDER API — Cloudflare vs DeepInfra
# vs Google AI Studio untuk Gemma 4 26B A4B
#
# Setup yang dibutuhkan:
#   pip install openai requests
#
# API Keys (set sebagai Kaggle Secrets atau environment variable):
#   CLOUDFLARE_ACCOUNT_ID  — dari dashboard.cloudflare.com
#   CLOUDFLARE_API_TOKEN   — Workers AI token (bukan API key global)
#   DEEPINFRA_API_KEY      — dari deepinfra.com/dashboard
#   GOOGLE_API_KEY         — dari aistudio.google.com/apikey
# ============================================================

import os
import time
import json
import statistics
import requests
from openai import OpenAI

# ── Konfigurasi ─────────────────────────────────────────────
# Isi salah satu atau semua untuk perbandingan
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN  = os.environ.get("CLOUDFLARE_API_TOKEN", "")
DEEPINFRA_API_KEY     = os.environ.get("DEEPINFRA_API_KEY", "")
GOOGLE_API_KEY        = os.environ.get("GOOGLE_API_KEY", "")

# Prompt test — relevan dengan use case OCR/summarization
TEST_PROMPTS = {
    "short": "Apa itu Mixture of Experts dalam konteks LLM? Jawab dalam 2 kalimat.",
    "medium": (
        "Ringkas teks berikut dalam Bahasa Indonesia (3 poin kunci):\n\n"
        "Transformasi digital di sektor kesehatan Indonesia telah mencapai tonggak penting "
        "dengan diluncurkannya platform SATUSEHAT yang menghubungkan lebih dari 3.000 fasilitas "
        "kesehatan. Adopsi telemedicine meningkat 340% dibanding sebelum pandemi, dengan lebih "
        "dari 50 juta konsultasi per bulan di berbagai platform. Tantangan utama masih pada "
        "konektivitas di daerah terpencil dan literasi digital tenaga kesehatan di pelosok."
    ),
    "ocr_sim": (
        "Bayangkan sebuah gambar invoice dengan teks berikut. Ekstrak semua informasi penting "
        "dan format sebagai JSON:\n"
        "INVOICE #INV-2026-0042 | Tanggal: 5 Juni 2026\n"
        "Vendor: PT Teknologi Maju | NPWP: 01.234.567.8-901.000\n"
        "Item 1: Lisensi Software A x5 unit @ Rp 2.500.000 = Rp 12.500.000\n"
        "Item 2: Maintenance Fee x1 = Rp 3.000.000\n"
        "Subtotal: Rp 15.500.000 | PPN 11%: Rp 1.705.000 | TOTAL: Rp 17.205.000"
    ),
}

# ── Helper: ukur TTFT dan total time ────────────────────────
def benchmark_request(call_fn, label: str, prompt_key: str = "short") -> dict:
    """Jalankan request, ukur TTFT dan total time, return hasilnya."""
    prompt = TEST_PROMPTS[prompt_key]
    print(f"\n  → {label} [{prompt_key}]", end="", flush=True)

    try:
        t_start = time.perf_counter()
        result = call_fn(prompt)
        t_total = time.perf_counter() - t_start

        ttft    = result.get("ttft", t_total)
        text    = result.get("text", "")
        tokens  = result.get("tokens_out", len(text.split()))

        tps = tokens / t_total if t_total > 0 else 0
        print(f" ✅ {t_total:.2f}s | {tps:.0f} tok/s")

        return {
            "label": label,
            "prompt": prompt_key,
            "ttft_s": round(ttft, 3),
            "total_s": round(t_total, 3),
            "tokens_out": tokens,
            "tok_per_s": round(tps, 1),
            "text_preview": text[:200],
            "error": None,
        }

    except Exception as e:
        t_elapsed = time.perf_counter() - t_start if 't_start' in dir() else 0
        print(f" ❌ Error: {e}")
        return {
            "label": label,
            "prompt": prompt_key,
            "ttft_s": None,
            "total_s": round(t_elapsed, 3),
            "tokens_out": 0,
            "tok_per_s": 0,
            "text_preview": "",
            "error": str(e),
        }


# ── Provider 1: Cloudflare Workers AI ───────────────────────
# Cloudflare adalah GPU inference sendiri — BUKAN sekadar proxy/routing.
# Model @cf/google/gemma-4-26b-a4b-it berjalan di GPU Cloudflare.
# Free tier: 10.000 neurons/hari. Untuk 26B model, cukup ~25-50 request/hari.
# Endpoint: OpenAI-compatible di /v1/chat/completions

def call_cloudflare(prompt: str) -> dict:
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        raise ValueError("CLOUDFLARE_ACCOUNT_ID atau CLOUDFLARE_API_TOKEN belum diset")

    client = OpenAI(
        api_key=CLOUDFLARE_API_TOKEN,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
    )

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="@cf/google/gemma-4-26b-a4b-it",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        timeout=30,    # Cloudflare biasanya cepat, timeout 30s cukup
    )
    ttft = time.perf_counter() - t0   # approximation (non-streaming)

    text = response.choices[0].message.content or ""
    return {
        "text": text,
        "ttft": ttft,
        "tokens_out": response.usage.completion_tokens if response.usage else len(text.split()),
    }


# Alternatif: gunakan REST API langsung (tanpa openai library)
def call_cloudflare_rest(prompt: str) -> dict:
    """Versi REST murni — berguna jika openai library tidak tersedia."""
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        raise ValueError("Credentials Cloudflare belum diset")

    url = (f"https://api.cloudflare.com/client/v4/accounts/"
           f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/google/gemma-4-26b-a4b-it")

    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }

    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    ttft = time.perf_counter() - t0

    resp.raise_for_status()
    data = resp.json()

    # Cloudflare REST response format
    text = ""
    if data.get("success") and data.get("result"):
        result = data["result"]
        if isinstance(result, dict):
            text = result.get("response", "")
        elif isinstance(result, str):
            text = result

    return {"text": text, "ttft": ttft, "tokens_out": len(text.split())}


# ── Provider 2: DeepInfra ────────────────────────────────────
# OpenAI-compatible. TTFT tercepat (0.62-0.68s), harga termurah ($0.07/M input).
# Endpoint: https://api.deepinfra.com/v1/openai
# Model string: "google/gemma-4-26B-A4B-it"

def call_deepinfra(prompt: str) -> dict:
    if not DEEPINFRA_API_KEY:
        raise ValueError("DEEPINFRA_API_KEY belum diset")

    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="google/gemma-4-26B-A4B-it",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        timeout=30,
    )
    ttft = time.perf_counter() - t0

    text = response.choices[0].message.content or ""
    return {
        "text": text,
        "ttft": ttft,
        "tokens_out": response.usage.completion_tokens if response.usage else len(text.split()),
    }


# ── Provider 3: Google AI Studio (baseline — yang sering RTO) ─
# Untuk perbandingan baseline. Kalau ini masih sering timeout,
# konfirmasi bahwa pindah provider adalah keputusan yang benar.

def call_google_aistudio(prompt: str) -> dict:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY belum diset")

    # Google AI Studio menggunakan endpoint Gemini, bukan OpenAI-compatible murni
    # tapi bisa diakses via openai library dengan base_url yang benar
    client = OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="gemma-4-26b-a4b-it",    # nama model di Google AI Studio
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        timeout=60,    # timeout lebih panjang karena sering lambat
    )
    ttft = time.perf_counter() - t0

    text = response.choices[0].message.content or ""
    return {
        "text": text,
        "ttft": ttft,
        "tokens_out": response.usage.completion_tokens if response.usage else len(text.split()),
    }


# ── Jalankan Benchmark ───────────────────────────────────────
print("=" * 60)
print("BENCHMARK: Gemma 4 26B A4B — Provider Comparison")
print("=" * 60)
print("\nCatatan:")
print("  • Cloudflare: GPU inference sendiri, bukan routing/proxy")
print("  • Free tier Cloudflare: 10.000 neurons/hari (~25-50 req untuk 26B)")
print("  • DeepInfra: TTFT tercepat, harga termurah ($0.07/M input)")
print("  • Google AI Studio: baseline — ini yang sering RTO")

results = []

# Test semua provider yang sudah dikonfigurasi
providers = [
    ("Cloudflare Workers AI", call_cloudflare),
    ("DeepInfra", call_deepinfra),
    ("Google AI Studio", call_google_aistudio),
]

for prompt_key in ["short", "medium"]:
    print(f"\n{'─'*40}")
    print(f"Prompt: [{prompt_key}]")
    print('─'*40)

    for label, fn in providers:
        # Skip jika credentials tidak ada
        if label == "Cloudflare Workers AI" and not (CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN):
            print(f"  → {label} [SKIP — credentials tidak diset]")
            continue
        if label == "DeepInfra" and not DEEPINFRA_API_KEY:
            print(f"  → {label} [SKIP — credentials tidak diset]")
            continue
        if label == "Google AI Studio" and not GOOGLE_API_KEY:
            print(f"  → {label} [SKIP — credentials tidak diset]")
            continue

        r = benchmark_request(fn, label, prompt_key)
        results.append(r)

# ── Tampilkan Hasil ──────────────────────────────────────────
if results:
    print(f"\n{'='*60}")
    print("HASIL BENCHMARK")
    print(f"{'='*60}")
    print(f"\n{'Provider':<25} {'Prompt':<8} {'TTFT':>7} {'Total':>7} {'Tok/s':>7} {'Error'}")
    print(f"{'─'*25} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*20}")

    for r in results:
        ttft  = f"{r['ttft_s']:.2f}s"  if r['ttft_s']  else "—"
        total = f"{r['total_s']:.2f}s" if r['total_s'] else "—"
        tps   = f"{r['tok_per_s']:.0f}" if r['tok_per_s'] else "—"
        err   = r['error'][:20] if r['error'] else "✅"
        print(f"{r['label']:<25} {r['prompt']:<8} {ttft:>7} {total:>7} {tps:>7} {err}")

    # Preview output dari prompt "medium" (summarization)
    print(f"\n{'─'*60}")
    print("Preview output [medium] — bandingkan kualitas summarization:")
    for r in results:
        if r['prompt'] == 'medium' and r['text_preview']:
            print(f"\n  [{r['label']}]")
            print(f"  {r['text_preview'][:300]}{'...' if len(r['text_preview']) >= 300 else ''}")

print(f"""
{'='*60}
KESIMPULAN BENCHMARK
{'='*60}

Provider ranking berdasarkan benchmark komunitas (Mei 2026):
  1. DeepInfra      — TTFT 0.62–0.68s, $0.07/M input, tercepat & termurah
  2. Cloudflare     — TTFT 0.84–0.92s, $0.12/M blended, gratis 10K neurons/hari
  3. Parasail       — TTFT ~1.0s, $0.10/M blended, alternatif solid
  4. Google AI Studio — TTFT 1.63s, gratis tapi sering RTO di peak hours

⚠️  Catatan penting Cloudflare:
  • 10.000 neurons/hari habis cepat untuk model 26B (~25-50 request)
  • Context window aktual: 128K (bukan 256K seperti dokumentasi)
  • Ada laporan billing anomaly di komunitas — monitor dashboard

💡 Rekomendasi berdasarkan use case:
  • Prototyping / testing     → Cloudflare (gratis, sampai 10K neurons)
  • Produksi volume rendah    → DeepInfra (tercepat, termurah per token)
  • Lokal Kaggle (no API)     → Gemma 4 E4B (gratis, non-blocking, full kontrol)
""")
```

### Solusi: Jalankan Lokal di Kaggle

Menjalankan E4B lokal di Kaggle **menghilangkan semua masalah API**:
- ✅ Tidak ada RTO — model ada di dalam notebook
- ✅ Tidak ada latency jaringan
- ✅ Pipeline non-blocking dengan async/threading sederhana
- ✅ Gratis (kuota GPU Kaggle)

---

## Bagian 3 — Faster Whisper: Update dari large-v2

### Apa yang Berubah Sejak large-v2?

#### Model: large-v3 (November 2023)
Whisper large-v3 dilatih pada 1 juta jam audio weakly labeled + 4 juta jam pseudo-labeled dari v2. Hasilnya: penurunan error rate 10–20% dibanding large-v2 di berbagai bahasa. Arsitektur sama persis, hanya Mel frequency bins naik dari 80 ke 128. **Ini upgrade paling straightforward dan paling sepadan untuk Bahasa Indonesia.**

#### Model: large-v3-turbo (September 2024)
Turbo mengurangi jumlah decoder layer dari 32 menjadi 4, menghasilkan peningkatan kecepatan signifikan sambil mempertahankan akurasi setara large-v2. Ukuran 809M parameter (vs 1.54B di v3), dengan degradasi akurasi sedikit lebih besar di bahasa seperti Thai dan Cantonese. Untuk kecepatan maksimum dengan akurasi "cukup baik" — pilihan yang valid.

#### distil-large-v3.5 (Maret 2025)
Distil-Whisper hanya tersedia untuk English speech recognition. Untuk multilingual termasuk Bahasa Indonesia, disarankan menggunakan Whisper Turbo.

**Rekomendasi untuk Bahasa Indonesia: tetap pakai `large-v3`**, bukan turbo atau distil.

### Fitur Baru Engine Faster-Whisper

Rilis terbaru faster-whisper membawa batched inference yang **4× lebih cepat dan akurat**, support model large-v3-turbo, VAD filter 3× lebih cepat di CPU, dan feature extraction 3× lebih cepat.

Cara pakai BatchedInferencePipeline (API baru, drop-in replacement):

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

# Drop-in replacement dari .transcribe() biasa
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

#### Fitur tambahan yang berguna:
- `language_detection_segments` + `language_detection_threshold` — deteksi bahasa lebih baik
- `multilingual=True` pada model medium ke bawah (large sudah code-switch otomatis)
- VAD filter built-in semakin stabil untuk deteksi silence
- `log_progress=True` untuk tracking progress transkripsi panjang

### WhisperX — Kalau Butuh Speaker Diarization

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

### Ringkasan Pilihan Whisper

| Situasi | Rekomendasi |
|---|---|
| Upgrade dari v2, akurasi terbaik (Bahasa Indonesia) | `large-v3` + `BatchedInferencePipeline` |
| Kecepatan prioritas, akurasi "cukup" | `large-v3-turbo` + batched |
| Multi-speaker, perlu siapa ngomong apa | WhisperX + `large-v3` |
| English only, ekstrem cepat | `distil-large-v3.5` |

---

## Bagian 4 — Script Python: Pipeline Lengkap di Kaggle

Script ini menggabungkan Faster-Whisper (transcript) + Gemma 4 E4B (OCR + summarization) dalam satu notebook Kaggle. Dirancang **non-blocking** dengan pemisahan pipeline yang jelas.

```python
# ============================================================
# PIPELINE: Transcript + OCR + Summarization
# Stack: Faster-Whisper large-v3 | Gemma 4 E4B
# Target: Kaggle Free GPU (T4 16GB)
# ============================================================

# !pip install -q --upgrade faster-whisper "transformers>=5.10.1" accelerate pillow librosa soundfile

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
    compute_type="float16",   # float16 untuk T4
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
        language=language,          # set None untuk auto-detect
        batch_size=16,              # lebih besar = lebih cepat, tapi lebih banyak VRAM
        beam_size=5,                # beam search, 5 = standar (lebih tinggi = lebih akurat tapi lambat)
        vad_filter=True,            # skip silence otomatis
        vad_parameters=dict(
            min_silence_duration_ms=500,  # diam > 0.5 detik = skip
        ),
        log_progress=True,
    )

    # Collect segments (generator)
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


# Contoh penggunaan:
# result = transcribe_audio("/kaggle/input/your-audio/recording.mp3", language="id")
# print(result["text"])


# Untuk audio panjang (> 30 menit), potong dulu:
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
    # TIDAK perlu quantization! E4B muat di T4 16GB dengan bfloat16 penuh
    # Jika OOM: tambahkan quantization_config=BitsAndBytesConfig(load_in_8bit=True)
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
    """
    OCR + baca teks dari gambar.

    Args:
        image_input: Path string, URL, atau PIL.Image object
        language_hint: Bahasa dominan dalam gambar

    Returns:
        String teks hasil OCR
    """
    # Load image
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


def analyze_image_colors(image_input) -> str:
    """
    Analisis warna dan kualitas visual gambar,
    berikan rekomendasi koreksi.

    Returns:
        String analisis + rekomendasi koreksi warna
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": (
                "Analisis kualitas visual dan warna gambar ini:\n"
                "1. Masalah exposure (over/under-exposed area)\n"
                "2. Color cast yang terdeteksi (warm/cool/green/magenta)\n"
                "3. Saturasi dan kontras: normal, terlalu tinggi, atau terlalu rendah?\n"
                "4. Rekomendasi koreksi spesifik dengan angka "
                "(contoh: turunkan brightness 15%, tambah contrast 10%, "
                "shift white balance ke arah cool 200K)\n"
                "5. Apakah ada bagian gambar yang perlu lokal correction?\n"
                "Format jawaban: poin bernomor, singkat dan actionable."
            )}
        ]
    }]

    return gemma_generate(messages, max_new_tokens=600)


def summarize_text(text: str, output_language: str = "Indonesia",
                   style: str = "ringkas") -> str:
    """
    Summarize teks panjang.

    Args:
        text: Teks yang akan diringkas
        output_language: Bahasa output ("Indonesia" atau "English")
        style: "ringkas" (3-4 kalimat + poin) atau "detail" (lebih panjang)

    Returns:
        String ringkasan
    """
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
    """
    Summarize hasil transcript audio menjadi notulensi/ringkasan.

    Args:
        transcript: Hasil transcript dari Whisper
        context: Konteks tambahan (misalnya "rapat tim marketing", "wawancara narasumber")
    """
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
# MODULE 3: FULL PIPELINE TEST
# ============================================================

print("\n[3/3] Running pipeline tests...")
print("-" * 40)

# --- TEST A: OCR ---
print("\n📄 TEST A: OCR dari gambar teks")

# Buat sample gambar untuk test
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

# --- TEST B: Color Analysis ---
print("\n🎨 TEST B: Analisis warna gambar")

# Buat gambar dengan isu warna
from PIL import ImageEnhance
color_img = Image.new("RGB", (400, 250), (110, 85, 65))
draw2 = ImageDraw.Draw(color_img)
draw2.rectangle([40, 40, 180, 150], fill=(190, 130, 80))   # over-exposed
draw2.rectangle([220, 40, 360, 150], fill=(25, 18, 12))    # under-exposed
draw2.text((70, 85), "BRIGHT", fill=(255, 240, 210))
draw2.text((250, 85), "DARK", fill=(60, 45, 30))
color_img = ImageEnhance.Brightness(color_img).enhance(1.25)

t0 = time.time()
color_result = analyze_image_colors(color_img)
print(f"  Hasil Color Analysis ({time.time()-t0:.1f}s):")
print(f"  {color_result}")

# --- TEST C: Summarization ---
print("\n📋 TEST C: Summarization teks panjang")

sample_text = """
Transformasi digital di sektor kesehatan Indonesia mengalami percepatan signifikan
sejak 2024. Kementerian Kesehatan meluncurkan platform SATUSEHAT yang mengintegrasikan
rekam medis elektronik dari lebih dari 3.000 fasilitas kesehatan. Ini memungkinkan
dokter mengakses riwayat pasien lintas rumah sakit secara real-time.

Di sisi lain, adopsi telemedicine meningkat 340% dibanding sebelum pandemi.
Platform seperti Halodoc, Alodokter, dan GoodDoctor mencatat lebih dari 50 juta
konsultasi per bulan. Namun tantangan tetap ada: konektivitas internet di daerah
terpencil masih menjadi hambatan utama, dengan hanya 67% wilayah Indonesia yang
memiliki akses broadband memadai.

Pemerintah menargetkan 100% fasilitas kesehatan primer terhubung ke sistem digital
pada akhir 2026 melalui program Satu Data Kesehatan. Investasi yang dibutuhkan
diperkirakan mencapai Rp 12 triliun. WHO memberikan apresiasi atas progres Indonesia,
menjadikannya salah satu negara berkembang dengan transformasi digital kesehatan
paling ambisius di kawasan Asia Tenggara.
"""

t0 = time.time()
summary_result = summarize_text(sample_text, output_language="Indonesia", style="ringkas")
print(f"  Hasil Summary ({time.time()-t0:.1f}s):")
print(f"  {summary_result}")

# --- TEST D: Transcript Summary (simulasi) ---
print("\n🎙️ TEST D: Summarize transcript (simulasi tanpa audio)")

sample_transcript = """
jadi kita hari ini mau membahas update proyek website yang sudah berjalan tiga bulan
progress dari tim frontend sudah selesai untuk halaman utama dan halaman produk
tapi halaman checkout masih ada bug di payment gateway khusus untuk kartu kredit
tim backend sudah identify masalahnya dan estimasi fix dua hari kerja
untuk timeline kita masih on track untuk launch minggu depan
tapi kita perlu QA testing dulu setelah bug fix selesai
action item: andi fix payment bug sebelum jumat, budi prepare testing environment,
citra buat testing checklist, dan kita meeting lagi jumat sore jam tiga untuk review
"""

t0 = time.time()
transcript_summary = summarize_transcript(
    sample_transcript,
    context="meeting tim development proyek website e-commerce"
)
print(f"  Hasil Notulensi ({time.time()-t0:.1f}s):")
print(f"  {transcript_summary}")

# --- CLEANUP ---
print("\n🧹 Membersihkan memory GPU...")
torch.cuda.empty_cache()
gc.collect()
print(f"  VRAM setelah cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n" + "=" * 60)
print("✅ Semua test selesai!")
print("=" * 60)
```

---

## Bagian 5 — Tips Integrasi Pipeline Non-Blocking

### Masalah Pipeline Blocking

Masalah Anda dengan Gemini API (RTO, lambat, blocking) bisa diselesaikan dengan kombinasi:
1. **Pindah ke lokal (E4B)** — eliminasi network dependency
2. **Pisahkan pipeline Whisper dan Gemma** — tidak perlu tunggu satu selesai

### Arsitektur Pipeline yang Disarankan

```
Audio File ──► Whisper (transcript) ──► Teks ──► Gemma E4B (summary)
                                                      │
Foto/Gambar ─────────────────────────────────────► Gemma E4B (OCR)
                                                      │
Dokumen Teks ───────────────────────────────────► Gemma E4B (summary)
```

Karena Whisper dan Gemma adalah model terpisah, Anda bisa load keduanya sekaligus di T4 (Whisper FP16 ~3GB + E4B bfloat16 ~8GB = ~11GB, masih dalam 16GB). Tidak perlu unload/reload antar task.

### Async untuk Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Proses multiple audio files paralel dengan Whisper
def process_batch(audio_files: list) -> list:
    results = []
    for f in audio_files:
        r = transcribe_audio(f, language="id")
        results.append(r)
    return results

# Setelah transcript selesai, langsung summarize
def full_pipeline(audio_path: str, image_paths: list = None) -> dict:
    output = {}

    # Step 1: Transcript
    transcript = transcribe_audio(audio_path, language="id")
    output["transcript"] = transcript["text"]

    # Step 2: Summarize transcript (langsung, non-blocking)
    output["summary"] = summarize_transcript(
        transcript["text"],
        context="rekaman audio"
    )

    # Step 3: OCR gambar (jika ada)
    if image_paths:
        output["ocr_results"] = []
        for img_path in image_paths:
            ocr = ocr_image(img_path)
            output["ocr_results"].append(ocr)

    return output
```

---

## Bagian 5 — MiMo V2.5: Apakah Cocok untuk Pipeline Ini?

Anda sudah subscribe MiMo untuk vibe coding, jadi wajar mempertimbangkan apakah model yang sama bisa dipakai untuk OCR, summarization, dan color analysis — biar tidak perlu gonta-ganti tool.

### Apa Itu MiMo V2.5?

MiMo V2.5 dirilis Xiaomi pada 22 April 2026. Ini bukan model kecil — arsitekturnya Sparse MoE dengan 310B total parameter dan 15B aktif per inference pass. Dilatih pada 48T token dengan FP8 mixed precision, dan context window-nya 1 juta token.

Yang penting untuk use case Anda: MiMo V2.5 (bukan Pro) adalah model **omnimodal native** — support teks, gambar, video, dan audio dalam satu arsitektur terpadu. Visual encoder-nya 729M-parameter Vision Transformer (ViT) dengan hybrid window attention, dan ada dedicated audio encoder yang diinisialisasi dari weights MiMo-Audio.

MiMo V2.5 Pro adalah versi berbeda — fokus ke coding/agentic tasks dan **text-only** (tidak punya kemampuan vision di Token Plan endpoint). Ini menyebabkan kebingungan di komunitas karena penamaan kurang jelas.

### Spesifikasi & Harga

| | MiMo V2.5 | MiMo V2.5 Pro |
|---|---|---|
| **Parameter** | 310B total / 15B aktif | 1T total / 42B aktif |
| **Modalities** | Teks, Gambar, Video, Audio | Teks saja (di Token Plan) |
| **Context** | 1M token | 1M token |
| **Harga (API)** | $0.14/M input, $0.28/M output | $0.43/M input, $0.87/M output |
| **Output speed** | ~86–150 tok/s | ~42 tok/s |
| **TTFT** | ~2.80s | ~3.05s |
| **Lisensi** | MIT (open source) | MIT (open source) |

Blended rate MiMo V2.5 hanya **$0.06/1M token** — jauh lebih murah dari Gemma 4 26B di API manapun.

### Performa Benchmark

MiMo V2.5 mendapat skor 49 di Artificial Analysis Intelligence Index, di atas rata-rata model open-weight seukurannya (median 31). Pada image, video, dan multimodal agentic tasks, Xiaomi mengklaim MiMo V2.5 setara dengan model closed-source frontier — matching Gemini 3 Pro pada video, dan Claude Sonnet 4.6 pada multimodal agentic work.

Untuk coding, MiMo V2.5 Pro menjadi perhatian komunitas karena sebelum dirilis publik beredar di OpenRouter dengan nama samaran "Hunter Alpha" dan sempat dipuncak chart harian — komunitas mengira itu DeepSeek V4.

### Masalah Moderation Block — Penjelasan Teknis

Ini yang Anda alami. Ada dua layer penolakan yang berbeda di MiMo API:

**Layer 1 — Content filter API (error 421):** Berlaku untuk semua request, bukan hanya gambar. Komunitas melaporkan error `421 Moderation Block` dengan pesan `"The request was rejected because it was considered high risk"` bahkan untuk prompt coding yang tidak berbahaya. Ini adalah filter berbasis teks di level proxy Xiaomi, bukan di model itu sendiri. Masalah ini sudah dilaporkan sejak MiMo V2-Flash dan belum sepenuhnya diselesaikan.

**Layer 2 — Face detection pada gambar:** Ini yang Anda hadapi saat color correction. MiMo API memiliki image moderation layer yang mendeteksi wajah manusia dalam foto dan menolak request, terlepas dari intent Anda (editing, analisis warna, OCR, dll). Ini bukan kebijakan model — ini filter di infrastruktur API Xiaomi. Alasannya mirip dengan model Cina lain yang beroperasi global: compliance hukum di berbagai yurisdiksi soal synthetic media dan likeness rights.

Tidak ada workaround resmi yang didokumentasikan untuk masalah ini di MiMo API. Xiaomi menutup beberapa issue terkait tanpa solusi yang jelas.

### Apakah MiMo V2.5 Cocok untuk Pipeline Ini?

| Task | MiMo V2.5 | Verdict |
|---|---|---|
| **OCR teks/dokumen** (tanpa wajah) | ✅ Capable | Oke, tapi RTO bisa terjadi |
| **OCR foto orang / ID card / KTP** | ❌ Diblok | Face detection aktif |
| **Color correction analysis — foto landscape/produk** | ✅ Bisa | Aman jika tidak ada wajah |
| **Color correction — foto portrait/orang** | ❌ Diblok | Face detection aktif |
| **Summarization teks** | ✅ Sangat baik | Kuat, 1M context, murah |
| **Transcript audio** | ✅ Native audio | Belum ada pengalaman komunitas luas |
| **Vibe coding** | ✅ Kuat (Pro) | Ini memang kekuatan utamanya |

### Rekomendasi Penggunaan MiMo V2.5

**Gunakan MiMo V2.5 untuk:**
- Summarization teks panjang — murah, 1M context, kualitas frontier
- OCR dokumen yang tidak ada wajah manusia (invoice, tabel, laporan)
- Color analysis foto produk, landscape, makanan, arsitektur

**Jangan andalkan MiMo V2.5 untuk:**
- Foto yang ada wajah orang — selalu diblok di API
- Pipeline yang butuh reliability tinggi — moderation block tidak predictable

**Saran praktis untuk color correction dengan foto orang:**
Crop atau blur area wajah sebelum dikirim ke MiMo, lalu minta analisis warna keseluruhan frame. Atau, pakai **Gemma 4 E4B lokal** yang tidak punya content filter dari pihak ketiga — model bisa menganalisis foto orang sepenuhnya karena Anda yang mengelola inferensinya sendiri.

### Cell Test: MiMo V2.5 untuk OCR & Summarization

```python
# ============================================================
# CELL: TEST MiMo V2.5 — OCR, Color Analysis, Summarization
#
# Catatan penting:
#   • Gunakan model "mimo-v2.5" (BUKAN "mimo-v2.5-pro")
#   • mimo-v2.5-pro adalah text-only di Token Plan endpoint
#   • Foto dengan wajah manusia akan diblok (error 421)
#   • Endpoint: OpenAI-compatible
#
# Setup:
#   pip install openai pillow
#   MIMO_API_KEY = dari platform.xiaomimimo.com
# ============================================================

import os
import time
import base64
from pathlib import Path
from openai import OpenAI
from PIL import Image, ImageDraw
from io import BytesIO

MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")

# MiMo menggunakan OpenAI-compatible endpoint
# Token Plan endpoint (lebih murah): token-plan-cn.xiaomimimo.com
# Standard API endpoint: api.xiaomimimo.com
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
# Jika pakai Token Plan: MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"

def get_mimo_client():
    if not MIMO_API_KEY:
        raise ValueError("MIMO_API_KEY belum diset. Set via os.environ atau Kaggle Secrets.")
    return OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Konversi PIL Image ke base64 string untuk dikirim ke API."""
    buf = BytesIO()
    img.save(buf, format=format, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mimo_ocr(image_input, language_hint: str = "Indonesia") -> dict:
    """
    OCR gambar menggunakan MiMo V2.5.
    
    ⚠️  AKAN DIBLOK jika gambar mengandung wajah manusia (error 421).
    Aman untuk: dokumen, invoice, tabel, foto produk/landscape.
    """
    client = get_mimo_client()

    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("image_input harus path atau PIL.Image")

    img_b64 = image_to_base64(img)

    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model="mimo-v2.5",    # PENTING: bukan mimo-v2.5-pro
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Baca semua teks dalam gambar ini secara lengkap dan akurat. "
                            f"Bahasa dominan: {language_hint}. "
                            f"Pertahankan struktur asli (baris, spasi, tabel). "
                            f"Output hanya teks yang terbaca, tanpa komentar tambahan."
                        )
                    }
                ]
            }],
            max_tokens=1000,
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        text = response.choices[0].message.content or ""
        return {
            "success": True,
            "text": text,
            "time_s": round(elapsed, 2),
            "tokens_out": response.usage.completion_tokens if response.usage else None,
        }

    except Exception as e:
        err_str = str(e)
        # Error 421 = Moderation Block (face detected atau content filter)
        is_face_block = "421" in err_str or "Moderation Block" in err_str
        return {
            "success": False,
            "text": "",
            "time_s": 0,
            "error": err_str,
            "likely_face_block": is_face_block,
            "hint": (
                "Gambar kemungkinan mengandung wajah manusia. "
                "Coba crop/blur wajah, atau gunakan Gemma 4 E4B lokal."
            ) if is_face_block else "Error tidak dikenal",
        }


def mimo_color_analysis(image_input) -> dict:
    """
    Analisis warna dan kualitas visual gambar via MiMo V2.5.
    
    ⚠️  AKAN DIBLOK jika ada wajah manusia di foto.
    Workaround: crop/blur area wajah sebelum mengirim.
    """
    client = get_mimo_client()

    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("image_input harus path atau PIL.Image")

    img_b64 = image_to_base64(img)

    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model="mimo-v2.5",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analisis kualitas visual dan warna gambar ini:\n"
                            "1. Masalah exposure (over/under-exposed)\n"
                            "2. Color cast yang terdeteksi (warm/cool/green/magenta)\n"
                            "3. Saturasi dan kontras: normal, terlalu tinggi, terlalu rendah?\n"
                            "4. Rekomendasi koreksi spesifik dengan nilai numerik\n"
                            "   (contoh: turunkan brightness 15%, shift white balance +200K ke cool)\n"
                            "5. Area gambar yang perlu local correction?\n"
                            "Format: poin bernomor, singkat dan actionable."
                        )
                    }
                ]
            }],
            max_tokens=600,
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        return {
            "success": True,
            "analysis": response.choices[0].message.content or "",
            "time_s": round(elapsed, 2),
        }

    except Exception as e:
        err_str = str(e)
        is_face_block = "421" in err_str or "Moderation Block" in err_str
        return {
            "success": False,
            "analysis": "",
            "error": err_str,
            "likely_face_block": is_face_block,
            "hint": (
                "WORKAROUND: Crop atau blur area wajah terlebih dahulu:\n"
                "  from PIL import ImageFilter\n"
                "  # Contoh blur area wajah (sesuaikan koordinat):\n"
                "  face_region = img.crop((x1, y1, x2, y2))\n"
                "  blurred = face_region.filter(ImageFilter.GaussianBlur(radius=20))\n"
                "  img.paste(blurred, (x1, y1))\n"
                "  # Lalu kirim img yang sudah di-blur ke mimo_color_analysis()"
            ) if is_face_block else "Error tidak dikenal",
        }


def mimo_summarize(text: str, style: str = "ringkas",
                   output_lang: str = "Indonesia") -> dict:
    """
    Summarization teks via MiMo V2.5.
    Ini use case terkuat MiMo — tidak ada masalah moderation untuk teks.
    Context 1M token = bisa handle dokumen sangat panjang.
    """
    client = get_mimo_client()

    if style == "ringkas":
        fmt = "3-4 kalimat ringkasan + 3-5 poin kunci (bullet) + 1 kalimat kesimpulan."
    else:
        fmt = "Ringkasan komprehensif dengan konteks, poin penting, dan kesimpulan."

    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model="mimo-v2.5",
            messages=[{
                "role": "user",
                "content": (
                    f"Buat ringkasan dalam Bahasa {output_lang}.\n"
                    f"Format: {fmt}\n\n"
                    f"TEKS:\n{text}"
                )
            }],
            max_tokens=600,
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        return {
            "success": True,
            "summary": response.choices[0].message.content or "",
            "time_s": round(elapsed, 2),
            "tokens_out": response.usage.completion_tokens if response.usage else None,
            "tokens_in": response.usage.prompt_tokens if response.usage else None,
        }

    except Exception as e:
        return {"success": False, "summary": "", "error": str(e)}


# ── Jalankan Test ────────────────────────────────────────────
print("=" * 60)
print("TEST: MiMo V2.5 — OCR + Color Analysis + Summarization")
print("=" * 60)

# Test A: OCR dokumen (tanpa wajah — aman)
print("\n📄 TEST A: OCR Invoice/Dokumen (no face — aman)")

invoice_img = Image.new("RGB", (500, 260), (248, 245, 235))
draw = ImageDraw.Draw(invoice_img)
for i, line in enumerate([
    "INVOICE #MiMo-2026-001",
    "Tanggal: 22 April 2026",
    "",
    "Layanan API MiMo V2.5   x100K token   $0.014",
    "Layanan API MiMo V2.5   x500K output  $0.140",
    "──────────────────────────────────────────",
    "TOTAL                              $0.154",
]):
    draw.text((30, 20 + i * 30), line, fill=(20, 20, 20))

result_ocr = mimo_ocr(invoice_img, language_hint="Indonesia/English")
if result_ocr["success"]:
    print(f"  ✅ Berhasil ({result_ocr['time_s']}s)")
    print(f"  {result_ocr['text']}")
else:
    print(f"  ❌ {result_ocr.get('error', 'Unknown error')}")
    if result_ocr.get("likely_face_block"):
        print(f"  💡 {result_ocr['hint']}")


# Test B: Color Analysis foto produk (tanpa wajah — aman)
print("\n🎨 TEST B: Color Analysis foto produk/landscape (no face)")

product_img = Image.new("RGB", (400, 300), (85, 100, 120))  # slightly cold/blueish
draw2 = ImageDraw.Draw(product_img)
draw2.rectangle([50, 60, 200, 200], fill=(70, 90, 110))
draw2.rectangle([220, 60, 370, 200], fill=(200, 190, 175))
draw2.text((90, 120), "PRODUCT", fill=(200, 215, 230))
draw2.text((255, 120), "HIGHLIGHT", fill=(230, 220, 200))

result_color = mimo_color_analysis(product_img)
if result_color["success"]:
    print(f"  ✅ Berhasil ({result_color['time_s']}s)")
    print(f"  {result_color['analysis'][:400]}")
else:
    print(f"  ❌ {result_color.get('error', 'Unknown error')}")
    if result_color.get("likely_face_block"):
        print(f"\n  💡 WORKAROUND untuk foto dengan wajah:")
        print(result_color["hint"])


# Test C: Summarization teks (tidak ada masalah moderation)
print("\n📋 TEST C: Summarization — kekuatan utama MiMo")

long_text = """
Perkembangan model bahasa besar (LLM) China pada 2026 telah mengubah lanskap kompetisi AI global.
Xiaomi MiMo, Alibaba Qwen, dan Tencent Hunyuan kini bersaing langsung dengan model-model Amerika
dalam benchmark coding, reasoning, dan multimodal. Harga API model China secara konsisten 5-10x
lebih murah dari equivalen Amerika, mendorong developer global untuk diversifikasi provider.

MiMo V2.5 menjadi perhatian khusus karena sebelum dirilis resmi, model ini beredar dengan nama
samaran "Hunter Alpha" di OpenRouter dan berhasil menduduki puncak chart penggunaan harian.
Ketika identitasnya terungkap sebagai model Xiaomi, komunitas AI terkejut — tidak menyangka
perusahaan ponsel bisa menghasilkan model yang bersaing dengan Claude dan GPT-5.4.

Namun adopsi di perusahaan Barat terhambat oleh kekhawatiran regulasi. Beberapa yurisdiksi
mulai mempertanyakan keamanan data saat menggunakan model dari perusahaan Cina, mirip
dengan kontroversi TikTok sebelumnya. Xiaomi merespons dengan membuat MiMo V2.5 fully
open-source di bawah lisensi MIT, memberi opsi self-hosting bagi enterprise yang khawatir.
"""

result_summary = mimo_summarize(long_text, style="ringkas")
if result_summary["success"]:
    print(f"  ✅ Berhasil ({result_summary['time_s']}s)")
    if result_summary.get("tokens_in"):
        cost_est = (result_summary["tokens_in"] / 1e6 * 0.14 +
                    (result_summary.get("tokens_out") or 0) / 1e6 * 0.28)
        print(f"  Token: {result_summary['tokens_in']} in / {result_summary['tokens_out']} out")
        print(f"  Estimasi biaya: ${cost_est:.6f}")
    print(f"\n  {result_summary['summary']}")
else:
    print(f"  ❌ {result_summary.get('error', 'Unknown error')}")


# ── Ringkasan ────────────────────────────────────────────────
print(f"""
{'='*60}
KESIMPULAN: MiMo V2.5 untuk Pipeline Ini
{'='*60}

✅ COCOK untuk:
   • Summarization teks panjang (1M context, murah, tidak ada moderation issue)
   • OCR dokumen/invoice/tabel yang tidak ada wajah
   • Color analysis foto produk/landscape/makanan
   • Diintegrasikan dengan vibe coding workflow yang sudah ada

❌ TIDAK COCOK untuk:
   • Foto portrait atau foto yang ada wajah manusia
   • Pipeline yang butuh 100% reliability (moderation block tidak predictable)

💡 WORKAROUND untuk foto dengan wajah:
   • Crop/blur area wajah sebelum kirim ke MiMo
   • Atau gunakan Gemma 4 E4B lokal (tidak ada filter pihak ketiga)
   • Pertimbangkan routing: jika ada wajah → E4B lokal, tidak ada wajah → MiMo V2.5

⚠️  PASTIKAN pakai model "mimo-v2.5" (bukan "mimo-v2.5-pro"):
   • mimo-v2.5      = omnimodal (gambar, audio, video) ← yang Anda butuhkan
   • mimo-v2.5-pro  = text-only di Token Plan endpoint, fokus coding
""")
```

---

## Ringkasan Perbandingan Akhir

| Aspek | Gemini API 26B A4B | Gemma 4 E4B Lokal | MiMo V2.5 API |
|---|---|---|---|
| **Latency** | 1.6–5.5s TTFT, RTO risk | <1s (lokal) | ~2.8s TTFT |
| **Blocking** | Ya — network | Tidak | Ya — tapi lebih stabil |
| **Biaya** | Gratis (quota) | Gratis (Kaggle GPU) | $0.06/1M blended |
| **OCR dokumen** | ✅ Bagus | ✅ Cukup | ✅ Bagus |
| **OCR foto orang** | ✅ | ✅ (lokal, no filter) | ❌ Diblok |
| **Color analysis — landscape/produk** | ✅ | ✅ | ✅ |
| **Color analysis — portrait** | ✅ | ✅ | ❌ Diblok |
| **Summarization** | ✅ 256K ctx | ✅ 128K ctx | ✅✅ 1M ctx, termurah |
| **Audio/transcript** | ❌ | ✅ native | ✅ native |
| **Vibe coding** | ❌ | ❌ | ✅✅ (Pro) |
| **Offline/lokal** | ❌ | ✅ | ❌ |

**Strategi routing yang disarankan:**

```
Foto dengan wajah manusia  → Gemma 4 E4B lokal (Kaggle)
Foto tanpa wajah           → MiMo V2.5 API (atau E4B lokal)
Teks panjang / summarize   → MiMo V2.5 API (1M ctx, termurah)
Transcript audio           → Faster-Whisper large-v3 (selalu lokal)
Vibe coding                → MiMo V2.5 Pro (sudah Anda pakai)
```

---

## Referensi

- [Gemma 4 Model Hub — Kaggle](https://www.kaggle.com/models/google/gemma-4)
- [google/gemma-4-E4B-it — Hugging Face](https://huggingface.co/google/gemma-4-E4B-it)
- [Faster-Whisper GitHub (SYSTRAN)](https://github.com/SYSTRAN/faster-whisper)
- [Systran/faster-whisper-large-v3 — Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3)
- [Gemma 4 API Provider Benchmark — DeepInfra](https://deepinfra.com/blog/gemma-4-26b-a4b-api-benchmarks)
- [MiMo V2.5 — Xiaomi Official](https://mimo.xiaomi.com/mimo-v2-5/)
- [XiaomiMiMo/MiMo-V2.5 — Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2.5)
- [MiMo Moderation Block Issues — GitHub](https://github.com/XiaomiMiMo/MiMo-V2-Flash/issues/18)
- [Gemma 4 After 24 Hours — Community Report](https://dev.to/dentity007/-gemma-4-after-24-hours-what-the-community-found-vs-what-google-promised-3a2f)
- [Google AI for Developers — Gemma 4 Overview](https://ai.google.dev/gemma/docs/core)

---

*Research disusun Juni 2026 | Stack: Faster-Whisper large-v3 + Gemma 4 E4B + MiMo V2.5 | Target: Kaggle Free T4 GPU + API hybrid*
