# Pipeline OCR · Transcript · Summarization — Overview

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

## Dokumen Terkait

| Dokumen | Isi |
|---|---|
| [[gemma4-model-selection]] | Kenapa E4B untuk Kaggle Free, VRAM comparison, kemampuan & batasan |
| [[gemini-api-providers]] | Masalah RTO, perbandingan Cloudflare/DeepInfra/Google, benchmark code |
| [[faster-whisper-update]] | Update large-v3, turbo, WhisperX, fitur baru engine |
| [[pipeline-kaggle-full]] | Script Python lengkap: Whisper + Gemma 4 E4B di Kaggle |
| [[mimo-v25-evaluation]] | Apakah MiMo V2.5 cocok untuk pipeline ini (OCR, summary, color) |

---

## Strategi Routing yang Disarankan

```
Foto dengan wajah manusia  → Gemma 4 E4B lokal (Kaggle)
Foto tanpa wajah           → MiMo V2.5 API (atau E4B lokal)
Teks panjang / summarize   → MiMo V2.5 API (1M ctx, termurah)
Transcript audio           → Faster-Whisper large-v3 (selalu lokal)
Vibe coding                → MiMo V2.5 Pro (sudah Anda pakai)
```

## Perbandingan Akhir

| Aspek | Gemini API 26B A4B | Gemma 4 E4B Lokal | MiMo V2.5 API |
|---|---|---|---|
| **Latency** | 1.6–5.5s TTFT, RTO risk | <1s (lokal) | ~2.8s TTFT |
| **Blocking** | Ya — network | Tidak | Ya — tapi lebih stabil |
| **Biaya** | Gratis (quota) | Gratis (Kaggle GPU) | $0.06/1M blended |
| **OCR dokumen** | ✅ Bagus | ✅ Cukup | ✅ Bagus |
| **OCR foto orang** | ✅ | ✅ (lokal, no filter) | ❌ Diblok |
| **Summarization** | ✅ 256K ctx | ✅ 128K ctx | ✅✅ 1M ctx, termurah |
| **Audio/transcript** | ❌ | ✅ native | ✅ native |
| **Offline/lokal** | ❌ | ✅ | ❌ |

---

## Referensi

- [Gemma 4 Model Hub — Kaggle](https://www.kaggle.com/models/google/gemma-4)
- [google/gemma-4-E4B-it — Hugging Face](https://huggingface.co/google/gemma-4-E4B-it)
- [Faster-Whisper GitHub (SYSTRAN)](https://github.com/SYSTRAN/faster-whisper)
- [Gemma 4 API Provider Benchmark — DeepInfra](https://deepinfra.com/blog/gemma-4-26b-a4b-api-benchmarks)
- [MiMo V2.5 — Xiaomi Official](https://mimo.xiaomi.com/mimo-v2-5/)
- [MiMo Moderation Block Issues — GitHub](https://github.com/XiaomiMiMo/MiMo-V2-Flash/issues/18)

---

*Research disusun Juni 2026 | Stack: Faster-Whisper large-v3 + Gemma 4 E4B + MiMo V2.5 | Target: Kaggle Free T4 GPU + API hybrid*
