# 🧪 Groq API — Capability Test Notebook (1 Cell)

Daftar API key gratis di https://console.groq.com · Simpan di Colab Secrets dengan nama `GROQ_API_KEY`

---

## Cell — Full Test

```python
#@title 🧪 Groq API — Full Capability Test
# ============================================================
# Pilih model dari dropdown → semua test otomatis jalan
# Ganti model → test jalan ulang otomatis
# ============================================================

!pip install groq -q

from google.colab import userdata
from groq import Groq
import ipywidgets as widgets
from IPython.display import display
import base64, json, time, urllib.request

client = Groq(api_key=userdata.get("GROQ_API_KEY"))

# --- Retry helper (handle rate limit) ---
def safe_call(fn, retries=3, delay=5, **kwargs):
    """Call fn with retry on rate limit."""
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception as e:
            if "rate_limit" in str(e) and attempt < retries - 1:
                wait = delay * (attempt + 1)
                print(f"⏳ Rate limit, tunggu {wait}s... (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise

def show_payload(payload):
    """Tampilkan payload tanpa base64 (terlalu panjang)."""
    cleaned = {}
    for k, v in payload.items():
        if k == "messages":
            msgs = []
            for m in v:
                if isinstance(m.get("content"), list):
                    for part in m["content"]:
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                msgs.append({**m, "content": [{**part, "image_url": {"url": f"data:{url.split(';')[0]};base64,<TRUNCATED>"}}]})
                            else:
                                msgs.append(m)
                        else:
                            msgs.append(m)
                else:
                    msgs.append(m)
            cleaned[k] = msgs
        else:
            cleaned[k] = v
    print(json.dumps(cleaned, indent=2, ensure_ascii=False))

def show_raw(resp):
    """Tampilkan raw response."""
    print(json.dumps({
        "id": resp.id,
        "model": resp.model,
        "choices": [{"index": c.index, "finish_reason": c.finish_reason, "message": c.message.model_dump()} for c in resp.choices],
        "usage": resp.usage.model_dump() if resp.usage else None,
    }, indent=2, ensure_ascii=False))

# --- List Models (dynamic, no hardcode) ---
allmodels = sorted([m.id for m in client.models.list().data])

# Filter: exclude whisper (audio model, bukan chat)
chatmodels = [m for m in allmodels if "whisper" not in m.lower()]

# Vision: hanya model yang support vision
visionmodels = [m for m in chatmodels if any(x in m for x in ["llama-4", "llava", "vision"])]

# Chat: semua selain vision
chat_only = [m for m in chatmodels if m not in visionmodels]

print(f"Total model  : {len(allmodels)}")
print(f"Chat model   : {len(chat_only)}: {chat_only}")
print(f"Vision model : {len(visionmodels)}: {visionmodels}")

# --- Dropdown: default = item pertama (no hardcode) ---
modeldd = widgets.Dropdown(
    options=chat_only, value=chat_only[0] if chat_only else None,
    description="Chat Model:", layout=widgets.Layout(width="420px")
)

visiondd = widgets.Dropdown(
    options=visionmodels if visionmodels else ["meta-llama/llama-4-scout-17b-16e-instruct"],
    description="Vision:", layout=widgets.Layout(width="420px")
)

ui = widgets.VBox([modeldd, visiondd])
out = widgets.Output()

display(ui, out)

with out:
    print("👆 Pilih model dari dropdown di atas untuk memulai test")

# --- Semua Test ---
def run_all_tests(model_name, vision_name):
    out.clear_output(wait=True)
    with out:
        print(f"🔄 Model dipilih: {model_name}")
        print(f"   Vision model : {vision_name}\n")

        # ---- TEST 1: Basic Chat + Kecepatan ----
        try:
            print("="*60)
            print("TEST 1 — Basic Chat + Kecepatan (tokens/sec)")
            print("="*60)

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Jelaskan cara kerja transformer neural network dalam 3 paragraf."}],
                "max_tokens": 400,
            }
            print("📤 Payload:")
            show_payload(payload)

            t0 = time.time()
            resp = safe_call(client.chat.completions.create, **payload)
            elapsed = time.time() - t0
            u = resp.usage
            tps = u.completion_tokens / elapsed

            print(f"\n📥 Raw Response:")
            show_raw(resp)

            print(f"\n⚡ Model       : {model_name}")
            print(f"   Prompt tok  : {u.prompt_tokens}")
            print(f"   Output tok  : {u.completion_tokens}")
            print(f"   Waktu       : {elapsed:.2f}s  →  {tps:.0f} tok/s")
            print(f"\n{resp.choices[0].message.content}\n")
        except Exception as e:
            print(f"⚠️  TEST 1 gagal: {e}\n")

        time.sleep(15)  # Anti rate limit

        # ---- TEST 2: Structured Output / JSON Mode ----
        try:
            print("="*60)
            print("TEST 2 — Structured Output / JSON Mode")
            print("="*60)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Kamu adalah ekstractor data. Selalu balas dengan JSON valid."},
                    {"role": "user",   "content": (
                        "Ekstrak informasi dari teks ini ke JSON dengan field: nama, kota, pekerjaan, umur.\n\n"
                        "Teks: Budi Santoso, 34 tahun, tinggal di Surabaya dan bekerja sebagai data engineer."
                    )},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 200,
            }
            print("📤 Payload:")
            show_payload(payload)

            resp = safe_call(client.chat.completions.create, **payload)

            print(f"\n📥 Raw Response:")
            show_raw(resp)

            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            print(f"\nParsed dict: {parsed}")
            print(f"Akses field: {parsed.get('nama')} | {parsed.get('kota')}\n")
        except Exception as e:
            print(f"⚠️  TEST 2 gagal: {e}\n")

        time.sleep(15)  # Anti rate limit

        # ---- TEST 3: Function / Tool Calling ----
        try:
            print("="*60)
            print("TEST 3 — Function / Tool Calling")
            print("="*60)

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Bagaimana cuaca di Yogyakarta hari ini?"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_cuaca",
                        "description": "Ambil cuaca terkini untuk kota tertentu",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "kota":   {"type": "string", "description": "Nama kota"},
                                "satuan": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["kota"],
                        },
                    },
                }],
                "tool_choice": "auto",
                "max_tokens": 200,
            }
            print("📤 Payload:")
            show_payload(payload)

            resp = safe_call(client.chat.completions.create, **payload)

            print(f"\n📥 Raw Response:")
            show_raw(resp)

            msg = resp.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                args = json.loads(tc.function.arguments)
                print(f"\n✅ Model memanggil tool : {tc.function.name}")
                print(f"   Argumen             : {args}\n")
            else:
                print(f"\n⚠️  Model tidak memanggil tool, langsung jawab:")
                print(msg.content + "\n")
        except Exception as e:
            print(f"⚠️  TEST 3 gagal — model mungkin tidak support tool calling")
            print(f"   Error: {e}\n")

        time.sleep(15)  # Anti rate limit

        # ---- TEST 4: Vision (Image URL) ----
        try:
            print("="*60)
            print("TEST 4 — Vision (Image URL)")
            print("="*60)

            IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNGtransparencydemonstration1.png/280px-PNGtransparencydemonstration1.png"

            payload = {
                "model": vision_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": "Deskripsikan gambar ini secara detail."},
                        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    ],
                }],
                "max_tokens": 300,
            }
            print("📤 Payload:")
            show_payload(payload)

            resp = safe_call(client.chat.completions.create, **payload)

            print(f"\n📥 Raw Response:")
            show_raw(resp)
            print(f"\nModel   : {vision_name}")
            print(f"Respons : {resp.choices[0].message.content}\n")
        except Exception as e:
            print(f"⚠️  TEST 4 gagal: {e}\n")

        time.sleep(15)  # Anti rate limit

        # ---- TEST 5: Vision (Image Base64) ----
        try:
            print("="*60)
            print("TEST 5 — Vision (Image Base64)")
            print("="*60)

            def url_to_base64(url: str) -> tuple[str, str]:
                with urllib.request.urlopen(url) as r:
                    data = r.read()
                    mime = r.headers.get_content_type()
                return base64.b64encode(data).decode("utf-8"), mime

            b64str, mimetype = url_to_base64(IMAGE_URL)
            print(f"MIME     : {mimetype}")
            print(f"Size     : {len(b64str) / 1024:.1f} KB (base64)")

            payload = {
                "model": vision_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Ada berapa warna utama dalam gambar ini?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mimetype};base64,<TRUNCATED>"
                        }},
                    ],
                }],
                "max_tokens": 200,
            }
            print("📤 Payload:")
            show_payload(payload)

            resp = safe_call(
                client.chat.completions.create,
                model=vision_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Ada berapa warna utama dalam gambar ini?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mimetype};base64,{b64str}"
                        }},
                    ],
                }],
                max_tokens=200,
            )

            print(f"\n📥 Raw Response:")
            show_raw(resp)
            print(f"\nRespons  : {resp.choices[0].message.content}")
            print(f"Tokens   : {resp.usage.prompt_tokens} in / {resp.usage.completion_tokens} out\n")
        except Exception as e:
            print(f"⚠️  TEST 5 gagal: {e}\n")

        time.sleep(15)  # Anti rate limit

        # ---- TEST 6: Prompt Caching ----
        try:
            print("="*60)
            print("TEST 6 — Prompt Caching")
            print("="*60)

            LONG_SYSTEM = "Kamu adalah asisten ahli data science yang sangat detail. " + ("X" * 3000)

            def call_with_cache_check(usermsg):
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": LONG_SYSTEM},
                        {"role": "user",   "content": usermsg},
                    ],
                    "max_tokens": 100,
                }
                resp = safe_call(client.chat.completions.create, **payload)
                u = resp.usage
                cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                print(f"📤 Payload (system): {len(LONG_SYSTEM)} chars")
                print(f"📥 Prompt tok : {u.prompt_tokens}  |  Cached : {cached}  |  Output : {u.completion_tokens}")
                return resp.choices[0].message.content

            print("=== Request 1 (cache miss) ===")
            call_with_cache_check("Apa itu overfitting?")

            time.sleep(2)

            print("\n=== Request 2 (cache hit — prefix identik) ===")
            call_with_cache_check("Apa itu underfitting?")
            print()
        except Exception as e:
            print(f"⚠️  TEST 6 gagal: {e}\n")

        time.sleep(15)  # Anti rate limit

        # ---- TEST 7: Multi-turn Conversation ----
        try:
            print("="*60)
            print("TEST 7 — Multi-turn Conversation")
            print("="*60)

            history = [{"role": "system", "content": "Kamu adalah asisten coding Python yang ringkas."}]

            def chat(usermsg):
                history.append({"role": "user", "content": usermsg})
                payload = {
                    "model": model_name,
                    "messages": history,
                    "max_tokens": 300,
                }
                resp = safe_call(client.chat.completions.create, **payload)
                reply = resp.choices[0].message.content
                history.append({"role": "assistant", "content": reply})
                print(f"📤 User  : {usermsg}")
                print(f"📥 Groq  : {reply}\n")

            chat("Tulis fungsi Python untuk hitung faktorial.")
            chat("Sekarang tambahkan validasi input negatif.")
            chat("Ubah jadi versi iteratif, bukan rekursif.")

            print(f"Total turns  : {len([m for m in history if m['role']=='user'])}")
            print(f"Total history: {len(history)} messages\n")
        except Exception as e:
            print(f"⚠️  TEST 7 gagal: {e}\n")

        # ---- RINGKASAN ----
        print("="*60)
        print("RINGKASAN HASIL TEST")
        print("="*60)
        print(f"Model chat   : {model_name}")
        print(f"Model vision : {vision_name}")
        print(f"Semua test selesai ✓")

# --- Auto-run saat model berubah ---
def on_model_change(change):
    if change["name"] == "value":
        run_all_tests(modeldd.value, visiondd.value)

modeldd.observe(on_model_change, names="value")
visiondd.observe(on_model_change, names="value")
```
