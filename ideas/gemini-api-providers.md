# Gemini API 26B A4B: Masalah RTO & Provider Comparison

> **Bagian dari:** [[ocr-transcript-pipeline-overview]]
> **Masalah:** RTO, lambat, pipeline blocking di Google AI Studio

---

## Akar Masalah

Yang Anda alami (RTO, lambat, pipeline blocking) bukan masalah kualitas model-nya — ini masalah infrastruktur provider API.

Google AI Studio menyediakan akses pertama ke Gemma 4 26B A4B secara gratis, dengan TTFT 1.63 detik dan output speed ~46.7 t/s. Ini cukup untuk prototyping, tapi tidak kompetitif untuk aplikasi live yang menghadap user.

Komunitas melaporkan Gemma 4 26B A4B berjalan jauh lebih lambat dibanding Qwen 3.5 setara: ada user yang dapat 11 token/detik di Gemma 4 26B A4B vs 60+ token/detik di Qwen 3.5 35B A3B pada GPU yang sama (5060 Ti 16GB). VRAM untuk context juga dilaporkan lebih boros di quantization yang sama.

## Perbandingan Provider API (Mei 2026)

Ada 7 provider API untuk Gemma 4 26B A4B. Output speed tercepat: Clarifai (151.5 t/s), Cloudflare (105.6 t/s), Parasail (56.6 t/s). TTFT terendah: DeepInfra (0.62s), Clarifai (0.86s), Cloudflare (0.92s). Harga paling murah: DeepInfra dan Parasail ($0.10/1M token blended).

Jika tetap ingin pakai API (bukan lokal), **DeepInfra** atau **Cloudflare** adalah alternatif yang jauh lebih cepat dari Google AI Studio.

## Cloudflare Workers AI: Bukan Sekadar Routing

> **Ini mungkin mengejutkan:** Cloudflare bukan hanya CDN/proxy. Mereka punya **infrastruktur GPU sendiri** dan menjalankan model langsung di dalamnya.

Cloudflare membedakan tiga tipe model di katalog mereka:

- **Hosted** — model berjalan di GPU Cloudflare sendiri (Llama, Qwen, Gemma, dst). Ini yang dapat free tier dan neuron pricing.
- **Proxied** — Cloudflare hanya forward request ke API provider lain (OpenAI, Anthropic). Tidak ada free tier, biaya langsung ke provider.
- **Partner** — model di-deploy di infrastruktur mitra di dalam/dekat jaringan Cloudflare.

**`@cf/google/gemma-4-26b-a4b-it` adalah model Cloudflare-hosted** — berjalan di GPU Cloudflare sendiri, bukan forward ke Google. Ini penjelasan kenapa latency-nya bisa 0.92s TTFT vs 1.63s Google AI Studio.

### Free Tier Cloudflare Workers AI

Cloudflare Workers AI menyediakan **10.000 Neurons/hari gratis**, reset setiap 00:00 UTC. Tapi ada gotcha penting yang dilaporkan komunitas:

> *"Kenapa 10.000 neurons habis hanya dari satu prompt?"* — Seseorang di komunitas Cloudflare melaporkan error 4006 (neuron habis) setelah prompt sederhana ke gemma-4-26b-a4b-it. Model besar seperti 26B mengonsumsi neuron jauh lebih banyak per request dibanding model kecil.

Faktanya, 10.000 neuron/hari cukup untuk sekitar **~25–50 request pendek** ke model 26B. Untuk produksi, ini tidak cukup — Anda perlu Workers Paid ($0.011 per 1.000 neurons) atau pakai provider lain.

Ada juga laporan billing anomaly: seorang user ditagih $292 untuk 15.000 calls (estimasi harusnya ~$8), menunjukkan konversi neuron-ke-token untuk Gemma 4 model mungkin belum sepenuhnya stabil di dashboard Cloudflare.

### Context Window di Cloudflare

Dokumentasi Cloudflare menyatakan context 256K untuk gemma-4-26b-a4b-it, tapi ada laporan bug terbuka dari komunitas: deployment aktual hanya menerima **128K token** (bukan 256K). Jika prompt + output Anda mendekati 128K token, request akan ditolak dengan error.

## DeepInfra: Provider Tercepat untuk API

DeepInfra memiliki TTFT terendah (0.68s) dan harga paling murah ($0.07/M input token, $0.34/M output token) untuk Gemma 4 26B A4B di antara semua provider yang ada. Endpoint-nya fully OpenAI-compatible — tinggal ganti `base_url` dan `api_key`, kode yang sama dengan OpenAI SDK langsung jalan.

## Solusi: Jalankan Lokal di Kaggle

Menjalankan E4B lokal di Kaggle **menghilangkan semua masalah API**:
- ✅ Tidak ada RTO — model ada di dalam notebook
- ✅ Tidak ada latency jaringan
- ✅ Pipeline non-blocking dengan async/threading sederhana
- ✅ Gratis (kuota GPU Kaggle)

---

## Benchmark Provider API

```python
# ============================================================
# BENCHMARK: Gemma 4 26B A4B — Provider Comparison
# Cloudflare vs DeepInfra vs Google AI Studio
#
# Setup:
#   pip install openai requests
# ============================================================

import os, time, json, statistics, requests
from openai import OpenAI

CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN  = os.environ.get("CLOUDFLARE_API_TOKEN", "")
DEEPINFRA_API_KEY     = os.environ.get("DEEPINFRA_API_KEY", "")
GOOGLE_API_KEY        = os.environ.get("GOOGLE_API_KEY", "")

TEST_PROMPTS = {
    "short": "Apa itu Mixture of Experts dalam konteks LLM? Jawab dalam 2 kalimat.",
    "medium": (
        "Ringkas teks berikut dalam Bahasa Indonesia (3 poin kunci):\n\n"
        "Transformasi digital di sektor kesehatan Indonesia telah mencapai tonggak penting "
        "dengan diluncurkannya platform SATUSEHAT yang menghubungkan lebih dari 3.000 fasilitas "
        "kesehatan. Adopsi telemedicine meningkat 340% dibanding sebelum pandemi, dengan lebih "
        "dari 50 juta konsultasi per bulan di berbagai platform."
    ),
}


def benchmark_request(call_fn, label, prompt_key="short"):
    prompt = TEST_PROMPTS[prompt_key]
    print(f"\n  → {label} [{prompt_key}]", end="", flush=True)
    try:
        t_start = time.perf_counter()
        result = call_fn(prompt)
        t_total = time.perf_counter() - t_start
        ttft = result.get("ttft", t_total)
        text = result.get("text", "")
        tokens = result.get("tokens_out", len(text.split()))
        tps = tokens / t_total if t_total > 0 else 0
        print(f" ✅ {t_total:.2f}s | {tps:.0f} tok/s")
        return {"label": label, "prompt": prompt_key, "ttft_s": round(ttft, 3),
                "total_s": round(t_total, 3), "tokens_out": tokens,
                "tok_per_s": round(tps, 1), "text_preview": text[:200], "error": None}
    except Exception as e:
        print(f" ❌ Error: {e}")
        return {"label": label, "prompt": prompt_key, "error": str(e)}


def call_cloudflare(prompt):
    client = OpenAI(api_key=CLOUDFLARE_API_TOKEN,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1")
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="@cf/google/gemma-4-26b-a4b-it",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, timeout=30)
    ttft = time.perf_counter() - t0
    text = response.choices[0].message.content or ""
    return {"text": text, "ttft": ttft,
            "tokens_out": response.usage.completion_tokens if response.usage else len(text.split())}


def call_deepinfra(prompt):
    client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="google/gemma-4-26B-A4B-it",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, timeout=30)
    ttft = time.perf_counter() - t0
    text = response.choices[0].message.content or ""
    return {"text": text, "ttft": ttft,
            "tokens_out": response.usage.completion_tokens if response.usage else len(text.split())}


def call_google_aistudio(prompt):
    client = OpenAI(api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="gemma-4-26b-a4b-it",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, timeout=60)
    ttft = time.perf_counter() - t0
    text = response.choices[0].message.content or ""
    return {"text": text, "ttft": ttft,
            "tokens_out": response.usage.completion_tokens if response.usage else len(text.split())}


# ── Run ──
print("=" * 60)
print("BENCHMARK: Gemma 4 26B A4B — Provider Comparison")
print("=" * 60)

results = []
providers = [
    ("Cloudflare Workers AI", call_cloudflare, CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN),
    ("DeepInfra", call_deepinfra, DEEPINFRA_API_KEY),
    ("Google AI Studio", call_google_aistudio, GOOGLE_API_KEY),
]

for prompt_key in ["short", "medium"]:
    print(f"\n{'─'*40}\nPrompt: [{prompt_key}]\n{'─'*40}")
    for label, fn, has_key in providers:
        if not has_key:
            print(f"  → {label} [SKIP — credentials tidak diset]")
            continue
        results.append(benchmark_request(fn, label, prompt_key))

if results:
    print(f"\n{'='*60}\nHASIL BENCHMARK\n{'='*60}")
    print(f"\n{'Provider':<25} {'Prompt':<8} {'TTFT':>7} {'Total':>7} {'Tok/s':>7} {'Error'}")
    print(f"{'─'*25} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*20}")
    for r in results:
        ttft  = f"{r.get('ttft_s',0):.2f}s" if r.get('ttft_s') else "—"
        total = f"{r.get('total_s',0):.2f}s" if r.get('total_s') else "—"
        tps   = f"{r.get('tok_per_s',0):.0f}" if r.get('tok_per_s') else "—"
        err   = r['error'][:20] if r.get('error') else "✅"
        print(f"{r['label']:<25} {r['prompt']:<8} {ttft:>7} {total:>7} {tps:>7} {err}")
```

---

*Ref: [[gemma4-model-selection]] | [[faster-whisper-update]] | [[mimo-v25-evaluation]]*
