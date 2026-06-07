# Gemma 4 E4B — Inference Optimization Reference

> **Target:** Tesla T4 (15.6 GB VRAM), BF16 / INT8 only (no QAT int4)
> **Tanggal:** Juni 2026

---

## Perbandingan Cepat

| Backend | Estimasi tok/s (T4) | Multimodal | Setup | Catatan |
|---|---|---|---|---|
| HF Transformers (sekarang) | ~6 | ✅ image + audio | Termudah | Baseline, tidak optimal |
| HF Transformers + MTP | ~10–13 | ✅ image + audio | +1 model | Drop-in, kualitas identik |
| vLLM BF16 | ~15–22 | ✅ image + audio | Medium | PagedAttention, OpenAI API |
| vLLM + MTP | ~25–35 | ✅ image + audio | Medium | Terbaik untuk T4 |
| llama.cpp Q8_0 | ~12–18 | ✅ image (terbatas) | Paling mudah | Tidak semua fitur audio |

---

## Opsi 1 — HF Transformers + MTP Drafter (Terdekat ke Setup Sekarang)

MTP (Multi-Token Prediction) adalah drafter resmi Google. Model drafter kecil (~277 MB) memprediksi beberapa token sekaligus, lalu target model verifikasi semuanya dalam satu forward pass. **Kualitas output identik** dengan generation biasa — ini bukan approximation.

```python
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

GEMMA_MODEL_ID  = "google/gemma-4-E4B-it"
GEMMA_DRAFT_ID  = "google/gemma-4-E4B-it-assistant"   # MTP drafter

def load_gemma_mtp():
    print("\n[2/3] Loading Gemma 4 E4B + MTP drafter...")
    t0 = time.time()

    proc = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, trust_remote_code=True)

    # Target model — sama persis dengan cara sekarang, tapi pakai 1 GPU saja
    # device_map="auto" pada 1 GPU lebih efisien daripada split 2 GPU
    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",         # satu GPU, hindari tensor parallel overhead
        attn_implementation="eager", # ganti ke "flash_attention_2" jika flash-attn terinstall
    )
    model.eval()

    # Drafter (text-only, sangat ringan ~277 MB)
    drafter = AutoModelForImageTextToText.from_pretrained(
        GEMMA_DRAFT_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",         # harus GPU yang sama dengan target
    )
    drafter.eval()

    print(f"  ✅ Gemma + MTP loaded in {time.time()-t0:.1f}s")
    return proc, model, drafter


def generate_mtp(proc, model, drafter, messages, max_new_tokens=512):
    """Generate dengan speculative decoding MTP."""
    inputs = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to("cuda:0")

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            assistant_model=drafter,   # ← satu-satunya perubahan dari generate() biasa
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# Contoh — OCR
def ocr_image_mtp(proc, model, drafter, image_path: str) -> str:
    from PIL import Image
    img = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Ekstrak semua teks dari gambar ini secara akurat."},
            ],
        }
    ]
    return generate_mtp(proc, model, drafter, messages, max_new_tokens=512)


# Contoh — Summarization (text biasa)
def summarize_mtp(proc, model, drafter, transcript: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"Buat ringkasan/notulensi dari transkrip berikut:\n\n{transcript}",
        }
    ]
    return generate_mtp(proc, model, drafter, messages, max_new_tokens=700)
```

**Catatan penting:**
- `assistant_model` drafter harus di GPU yang sama dengan target.
- Gunakan `device_map="cuda:0"` bukan `"auto"` agar tidak split antar GPU — overhead tensor parallel HF lebih besar dari manfaatnya untuk model 4B.
- Jika punya `flash-attn` terinstall, ganti `attn_implementation="eager"` → `"flash_attention_2"` untuk tambahan ~20% speed.

---

## Opsi 2 — vLLM BF16 (Throughput Tertinggi)

vLLM menggunakan PagedAttention yang jauh lebih efisien dari HF untuk KV cache, plus continuous batching. Diakses via OpenAI-compatible API.

### Install

```bash
pip install vllm "vllm[audio]"   # [audio] wajib untuk input audio E4B
```

### Jalankan server

```bash
vllm serve google/gemma-4-E4B-it \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
  --host 0.0.0.0 \
  --port 8000
```

### Load dan inference dari Python

```python
from openai import OpenAI
import base64, pathlib

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def encode_image(path: str) -> str:
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()

def encode_audio(path: str) -> str:
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()


def ocr_vllm(image_path: str) -> str:
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model="google/gemma-4-E4B-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": "Ekstrak semua teks dari gambar ini secara akurat."},
                ],
            }
        ],
        max_tokens=512,
        temperature=0,
    )
    return response.choices[0].message.content


def transcribe_audio_vllm(audio_path: str) -> str:
    """Audio input via vLLM (hanya E4B / E2B yang mendukung)."""
    b64 = encode_audio(audio_path)
    ext = pathlib.Path(audio_path).suffix.lstrip(".").lower()
    mime = {"m4a": "audio/mp4", "mp3": "audio/mpeg", "wav": "audio/wav"}.get(ext, "audio/wav")

    response = client.chat.completions.create(
        model="google/gemma-4-E4B-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": "Transkripsi audio ini secara verbatim."},
                ],
            }
        ],
        max_tokens=1024,
        temperature=0,
    )
    return response.choices[0].message.content


def summarize_vllm(transcript: str) -> str:
    response = client.chat.completions.create(
        model="google/gemma-4-E4B-it",
        messages=[
            {
                "role": "user",
                "content": f"Buat ringkasan/notulensi dari transkrip berikut:\n\n{transcript}",
            }
        ],
        max_tokens=700,
        temperature=0,
    )
    return response.choices[0].message.content
```

---

## Opsi 3 — vLLM + MTP (Kombinasi Terbaik)

Tambahkan MTP drafter ke vLLM server untuk speculative decoding.

```bash
vllm serve google/gemma-4-E4B-it \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
  --speculative-model google/gemma-4-E4B-it-assistant \
  --speculative-model-uses-vllm-mtp-path \
  --host 0.0.0.0 \
  --port 8000
```

Client code-nya **identik** dengan Opsi 2 di atas — tidak ada perubahan di sisi client.

---

## Opsi 4 — llama.cpp / GGUF Q8_0

Paling mudah di-setup, tidak butuh vLLM server. Cocok untuk eksperimen cepat.
Audio support terbatas di llama.cpp, tapi image (OCR) dan text bekerja baik.

### Build dan jalankan

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Atau build dari source dengan CUDA
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j4
```

```bash
# Download GGUF Q8 (~4.5 GB, kualitas identik BF16 untuk OCR dan summarization)
# Cari di: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q8_0.gguf --local-dir ./models/

# Jalankan server
./build/bin/llama-server \
  -m ./models/gemma-4-E4B-it-Q8_0.gguf \
  --n-gpu-layers 999 \
  --ctx-size 8192 \
  --host 0.0.0.0 \
  --port 8080
```

### Inference dari Python

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/gemma-4-E4B-it-Q8_0.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    chat_format="gemma",
)


def ocr_llama(image_path: str) -> str:
    import base64, pathlib
    b64 = base64.b64encode(pathlib.Path(image_path).read_bytes()).decode()

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Ekstrak semua teks dari gambar ini secara akurat."},
                ],
            }
        ],
        max_tokens=512,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]


def summarize_llama(transcript: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"Buat ringkasan/notulensi dari transkrip berikut:\n\n{transcript}",
            }
        ],
        max_tokens=700,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]
```

---

## Tips Tambahan: Flash Attention 2

Jika tidak ingin ganti backend, setidaknya aktifkan Flash Attention 2 pada setup HF yang sekarang:

```bash
pip install flash-attn --no-build-isolation
```

```python
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)
```

Ini saja bisa memberikan +15–25% speedup tanpa perubahan lain.

---

*Ref: [[gemma4-model-selection]] | [[pipeline-kaggle-full]] | [[ocr-transcript-pipeline-overview]]*
