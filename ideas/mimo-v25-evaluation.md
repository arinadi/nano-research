# MiMo V2.5: Evaluasi untuk Pipeline Ini

> **Bagian dari:** [[ocr-transcript-pipeline-overview]]
> **Konteks:** Sudah subscribe MiMo untuk vibe coding — apakah bisa dipakai juga untuk OCR, summarization, color analysis?

---

## Apa Itu MiMo V2.5?

MiMo V2.5 dirilis Xiaomi pada 22 April 2026. Ini bukan model kecil — arsitekturnya Sparse MoE dengan 310B total parameter dan 15B aktif per inference pass. Dilatih pada 48T token dengan FP8 mixed precision, dan context window-nya 1 juta token.

Yang penting untuk use case Anda: MiMo V2.5 (bukan Pro) adalah model **omnimodal native** — support teks, gambar, video, dan audio dalam satu arsitektur terpadu. Visual encoder-nya 729M-parameter Vision Transformer (ViT) dengan hybrid window attention, dan ada dedicated audio encoder yang diinisialisasi dari weights MiMo-Audio.

MiMo V2.5 Pro adalah versi berbeda — fokus ke coding/agentic tasks dan **text-only** (tidak punya kemampuan vision di Token Plan endpoint). Ini menyebabkan kebingungan di komunitas karena penamaan kurang jelas.

## Spesifikasi & Harga

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

## Performa Benchmark

MiMo V2.5 mendapat skor 49 di Artificial Analysis Intelligence Index, di atas rata-rata model open-weight seukurannya (median 31). Pada image, video, dan multimodal agentic tasks, Xiaomi mengklaim MiMo V2.5 setara dengan model closed-source frontier — matching Gemini 3 Pro pada video, dan Claude Sonnet 4.6 pada multimodal agentic work.

Untuk coding, MiMo V2.5 Pro menjadi perhatian komunitas karena sebelum dirilis publik beredar di OpenRouter dengan nama samaran "Hunter Alpha" dan sempat dipuncak chart harian — komunitas mengira itu DeepSeek V4.

## Masalah Moderation Block — Penjelasan Teknis

Ini yang Anda alami. Ada dua layer penolakan yang berbeda di MiMo API:

**Layer 1 — Content filter API (error 421):** Berlaku untuk semua request, bukan hanya gambar. Komunitas melaporkan error `421 Moderation Block` dengan pesan `"The request was rejected because it was considered high risk"` bahkan untuk prompt coding yang tidak berbahaya. Ini adalah filter berbasis teks di level proxy Xiaomi, bukan di model itu sendiri. Masalah ini sudah dilaporkan sejak MiMo V2-Flash dan belum sepenuhnya diselesaikan.

**Layer 2 — Face detection pada gambar:** Ini yang Anda hadapi saat color correction. MiMo API memiliki image moderation layer yang mendeteksi wajah manusia dalam foto dan menolak request, terlepas dari intent Anda (editing, analisis warna, OCR, dll). Ini bukan kebijakan model — ini filter di infrastruktur API Xiaomi. Alasannya mirip dengan model Cina lain yang beroperasi global: compliance hukum di berbagai yurisdiksi soal synthetic media dan likeness rights.

Tidak ada workaround resmi yang didokumentasikan untuk masalah ini di MiMo API. Xiaomi menutup beberapa issue terkait tanpa solusi yang jelas.

## Apakah MiMo V2.5 Cocok untuk Pipeline Ini?

| Task | MiMo V2.5 | Verdict |
|---|---|---|
| **OCR teks/dokumen** (tanpa wajah) | ✅ Capable | Oke, tapi RTO bisa terjadi |
| **OCR foto orang / ID card / KTP** | ❌ Diblok | Face detection aktif |
| **Color correction — foto landscape/produk** | ✅ Bisa | Aman jika tidak ada wajah |
| **Color correction — foto portrait/orang** | ❌ Diblok | Face detection aktif |
| **Summarization teks** | ✅ Sangat baik | Kuat, 1M context, murah |
| **Transcript audio** | ✅ Native audio | Belum ada pengalaman komunitas luas |
| **Vibe coding** | ✅ Kuat (Pro) | Ini memang kekuatan utamanya |

## Rekomendasi Penggunaan

**Gunakan MiMo V2.5 untuk:**
- Summarization teks panjang — murah, 1M context, kualitas frontier
- OCR dokumen yang tidak ada wajah manusia (invoice, tabel, laporan)
- Color analysis foto produk, landscape, makanan, arsitektur

**Jangan andalkan MiMo V2.5 untuk:**
- Foto yang ada wajah orang — selalu diblok di API
- Pipeline yang butuh reliability tinggi — moderation block tidak predictable

**Saran praktis untuk color correction dengan foto orang:**
Crop atau blur area wajah sebelum dikirim ke MiMo, lalu minta analisis warna keseluruhan frame. Atau, pakai **Gemma 4 E4B lokal** yang tidak punya content filter dari pihak ketiga.

---

*Ref: [[gemma4-model-selection]] | [[gemini-api-providers]] | [[ocr-transcript-pipeline-overview]]*
