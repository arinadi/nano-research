# Gemma 4: Pilih Model yang Tepat untuk Kaggle Free

> **Bagian dari:** [[ocr-transcript-pipeline-overview]]
> **Target:** Kaggle Free GPU (T4/P100 16GB)

---

## Keluarga Model Gemma 4 (April 2026)

Ada empat ukuran dalam Gemma 4: E2B (~2.3B efektif), E4B (~4.5B efektif), 26B A4B (MoE, 4B aktif per forward pass), dan 31B Dense. Context window 128K untuk E2B/E4B, dan 256K untuk 26B dan 31B. Semua variant menerima teks dan gambar, sementara E2B dan E4B juga support audio native.

Pada benchmark AIME 2026 (matematika), E4B sudah mencapai 42.5% — lebih dari dua kali lipat kemampuan model full-size generasi sebelumnya (Gemma 3 27B di 20.8%).

## Kenapa E4B untuk Kaggle Free?

Google sendiri sudah tegas: E2B dan E4B adalah model edge-first. Untuk developer yang tidak build untuk ponsel dan punya memori cukup, pilihan biasanya bukan E4B vs E2B — melainkan 26B A4B vs 31B. Tapi dengan keterbatasan hardware gratis Kaggle, E4B adalah pilihan realistis.

**VRAM Kaggle vs kebutuhan model:**

| Model | VRAM bfloat16 | VRAM int8 | VRAM int4 | Kaggle T4 (16GB) |
|---|---|---|---|---|
| E4B | ~8 GB | ~5 GB | ~3 GB | ✅ Muat penuh |
| 12B | ~24 GB | ~14 GB | ~8 GB | ⚠️ Perlu 4-bit |
| 26B A4B | ~16 GB | ~10 GB | ~6 GB | ⚠️ Perlu quantization |
| 31B | ~64 GB | ~32 GB | ~16 GB | ❌ Tidak muat |

> Dengan T4 16GB Kaggle: **E4B bisa jalan full precision (bfloat16)** tanpa quantization. Ini keunggulan besar — tidak ada degradasi akurasi dari quantization.

## Kemampuan E4B yang Relevan

Dalam pengujian Hugging Face, model E4B menghasilkan transkripsi dan deskripsi audio yang akurat. Visual token budget E4B mendukung hingga 1120 token untuk OCR dan document parsing dengan detail maksimum.

E4B perform comparably to dense models twice its size pada banyak benchmark. Arsitektur conformer audio encoder-nya (USM-style) sudah tervalidasi baik untuk speech recognition — tanpa cloud dependency, karena native di model.

## Catatan Komunitas tentang E4B

Seorang developer menguji semua model Gemma 4 di GPU GTX 1650 (4GB VRAM) dan hasilnya memuaskan untuk E4B — "tidak menyangka model seperti E4B bisa berjalan selancar ini di GPU seperti ini."

Catatan dari Unsloth: jika ada loss 13–15 saat fine-tuning E2B/E4B, itu normal — perilaku umum model multimodal, juga terjadi di Gemma-3N dan Llama Vision. Untuk multimodal prompt Gemma 4, taruh gambar sebelum instruksi teks.

## Batasan E4B yang Perlu Diketahui

- **Context window 128K** (vs 256K di 26B) — cukup untuk ~100 halaman teks, tapi lebih terbatas untuk dokumen sangat panjang
- **Reasoning lebih lemah** dari 26B pada tugas kompleks — untuk OCR dan summarization biasa, tidak terasa
- **Bukan pilihan terbaik** jika Anda akhirnya punya akses GPU besar

---

*Ref: [[gemini-api-providers]] | [[faster-whisper-update]] | [[mimo-v25-evaluation]]*
