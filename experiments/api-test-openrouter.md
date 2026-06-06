# 🧪 OpenRouter API — Capability Test Notebook (1 Cell)

Daftar API key gratis di https://openrouter.ai · Simpan di Colab Secrets dengan nama `OPENROUTER_API_KEY`
> Free tier: **20 req/menit, 50 req/hari** (naik ke 1.000/hari kalau pernah top-up min $10)
> Model gratis: ID berakhiran `:free` — 29+ model termasuk DeepSeek R1, Llama 4, Qwen3, Gemini Flash

---

## Cell — Full Test

```python
#@title 🧪 OpenRouter API — Full Capability Test
# ============================================================
# Pilih model dari dropdown → semua test otomatis jalan
# Ganti model → test jalan ulang otomatis
# ============================================================

!pip install openai -q

from google.colab import userdata
from openai import OpenAI
import ipywidgets as widgets
from IPython.display import display
import base64, json, time, urllib.request, urllib.parse

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=userdata.get("OPENROUTER_API_KEY"),
)

# --- Retry helper (handle rate limit) ---
def safe_call(fn, retries=3, delay=10, **kwargs):
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                if attempt < retries - 1:
                    wait = delay * (attempt + 1)
                    print(f"⏳ Rate limit, tunggu {wait}s... (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise

def show_payload(payload):
    cleaned = {}
    for k, v in payload.items():
        if k == "messages":
            msgs = []
            for m in v:
                if isinstance(m.get("content"), list):
                    new_content = []
                    for part in m["content"]:
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                new_content.append({**part, "image_url": {"url": f"data:{url.split(';')[0]};base64,<TRUNCATED>"}})
                            else:
                                new_content.append(part)
                        else:
                            new_content.append(part)
                    msgs.append({**m, "content": new_content})
                else:
                    msgs.append(m)
            cleaned[k] = msgs
        else:
            cleaned[k] = v
    print(json.dumps(cleaned, indent=2, ensure_ascii=False))

def show_raw(resp):
    print(json.dumps({
        "id":     resp.id,
        "model":  resp.model,
        "choices": [{
            "index":        c.index,
            "finish_reason": c.finish_reason,
            "message":      c.message.model_dump(),
        } for c in resp.choices],
        "usage": resp.usage.model_dump() if resp.usage else None,
    }, indent=2, ensure_ascii=False))

# --- List Models ---
# Ambil semua model dari API, filter hanya yg gratis (:free)
raw_models = client.models.list().data
all_ids    = sorted([m.id for m in raw_models])
free_ids   = [m for m in all_ids if m.endswith(":free")]
paid_ids   = [m for m in all_ids if not m.endswith(":free")]

# Vision: model yang kemungkinan support image (heuristic)
vision_keywords = ["vision", "llava", "llama-4", "gemini", "gpt-4o", "claude-3", "pixtral", "qwen.*vl", "mistral.*vision"]
import re
vision_free  = [m for m in free_ids if any(re.search(k, m) for k in vision_keywords)]
chat_free    = [m for m in free_ids if m not in vision_free]

# Fallback jika filter vision kosong: ambil semua free
if not vision_free:
    vision_free = free_ids

print(f"Total model  : {len(all_ids)}")
print(f"Free model   : {len(free_ids)}: {free_ids}")
print(f"Vision (free): {len(vision_free)}: {vision_free}")
print(f"Chat (free)  : {len(chat_free)}: {chat_free[:5]} ...")

# --- Dropdown ---
modeldd = widgets.Dropdown(
    options=free_ids, value=free_ids[0] if free_ids else None,
    description="Chat Model:", layout=widgets.Layout(width="500px")
)
visiondd = widgets.Dropdown(
    options=vision_free, value=vision_free[0] if vision_free else None,
    description="Vision:", layout=widgets.Layout(width="500px")
)

ui  = widgets.VBox([modeldd, visiondd])
out = widgets.Output()
display(ui, out)

with out:
    print("👆 Pilih model dari dropdown di atas untuk memulai test")

# --- Run All Tests ---
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

            payload = dict(
                model=model_name,
                messages=[{"role": "user", "content": "Jelaskan cara kerja transformer neural network dalam 3 paragraf."}],
                max_tokens=400,
            )
            print("📤 Payload:")
            show_payload(payload)

            t0   = time.time()
            resp = safe_call(client.chat.completions.create, **payload)
            elapsed = time.time() - t0
            u    = resp.usage
            tps  = u.completion_tokens / elapsed if elapsed > 0 else 0

            print("\n📥 Raw Response:")
            show_raw(resp)
            print(f"\n⚡ Model       : {model_name}")
            print(f"   Prompt tok  : {u.prompt_tokens}")
            print(f"   Output tok  : {u.completion_tokens}")
            print(f"   Waktu       : {elapsed:.2f}s  →  {tps:.0f} tok/s\n")
            print(resp.choices[0].message.content)
        except Exception as e:
            print(f"⚠️  TEST 1 gagal: {e}\n")

        time.sleep(20)

        # ---- TEST 2: Structured Output / JSON Mode ----
        try:
            print("="*60)
            print("TEST 2 — Structured Output / JSON Mode")
            print("="*60)

            payload = dict(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Kamu adalah ekstractor data. Selalu balas dengan JSON valid saja, tanpa markdown."},
                    {"role": "user",   "content": (
                        "Ekstrak ke JSON dengan field: nama, kota, pekerjaan, umur.\n\n"
                        "Teks: Budi Santoso, 34 tahun, tinggal di Surabaya dan bekerja sebagai data engineer."
                    )},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            print("📤 Payload:")
            show_payload(payload)

            resp   = safe_call(client.chat.completions.create, **payload)
            print("\n📥 Raw Response:")
            show_raw(resp)

            raw    = resp.choices[0].message.content
            parsed = json.loads(raw)
            print(f"\nParsed dict: {parsed}")
            print(f"Akses field: {parsed.get('nama')} | {parsed.get('kota')}\n")
        except Exception as e:
            print(f"⚠️  TEST 2 gagal: {e}\n")

        time.sleep(20)

        # ---- TEST 3: Function / Tool Calling ----
        try:
            print("="*60)
            print("TEST 3 — Function / Tool Calling")
            print("="*60)

            payload = dict(
                model=model_name,
                messages=[{"role": "user", "content": "Bagaimana cuaca di Yogyakarta hari ini?"}],
                tools=[{
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
                tool_choice="auto",
                max_tokens=200,
            )
            print("📤 Payload:")
            show_payload(payload)

            resp = safe_call(client.chat.completions.create, **payload)
            print("\n📥 Raw Response:")
            show_raw(resp)

            msg = resp.choices[0].message
            if msg.tool_calls:
                tc   = msg.tool_calls[0]
                args = json.loads(tc.function.arguments)
                print(f"\n✅ Model memanggil tool : {tc.function.name}")
                print(f"   Argumen             : {args}\n")
            else:
                print(f"\n⚠️  Model tidak memanggil tool, langsung jawab:")
                print(msg.content + "\n")
        except Exception as e:
            print(f"⚠️  TEST 3 gagal — model mungkin tidak support tool calling")
            print(f"   Error: {e}\n")

        time.sleep(20)

        # ---- TEST 4: Vision (Image URL) ----
        try:
            print("="*60)
            print("TEST 4 — Vision (Image URL)")
            print("="*60)

            IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

            payload = dict(
                model=vision_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": "Deskripsikan gambar ini secara detail."},
                        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    ],
                }],
                max_tokens=300,
            )
            print("📤 Payload:")
            show_payload(payload)

            resp = safe_call(client.chat.completions.create, **payload)
            print("\n📥 Raw Response:")
            show_raw(resp)
            print(f"\nModel   : {vision_name}")
            print(f"Respons : {resp.choices[0].message.content}\n")
        except Exception as e:
            print(f"⚠️  TEST 4 gagal: {e}\n")

        time.sleep(20)

        # ---- TEST 5: Vision (Image Base64) ----
        try:
            print("="*60)
            print("TEST 5 — Vision (Image Base64)")
            print("="*60)

            def url_to_base64(url):
                with urllib.request.urlopen(url) as r:
                    data = r.read()
                    mime = r.headers.get_content_type()
                return base64.b64encode(data).decode("utf-8"), mime

            b64str, mimetype = url_to_base64(IMAGE_URL)
            print(f"MIME     : {mimetype}")
            print(f"Size     : {len(b64str) / 1024:.1f} KB (base64)")

            payload_display = dict(
                model=vision_name,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Ada berapa warna utama dalam gambar ini?"},
                    {"type": "image_url", "image_url": {"url": f"data:{mimetype};base64,<TRUNCATED>"}},
                ]}],
                max_tokens=200,
            )
            print("📤 Payload:")
            show_payload(payload_display)

            resp = safe_call(
                client.chat.completions.create,
                model=vision_name,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Ada berapa warna utama dalam gambar ini?"},
                    {"type": "image_url", "image_url": {"url": f"data:{mimetype};base64,{b64str}"}},
                ]}],
                max_tokens=200,
            )
            print("\n📥 Raw Response:")
            show_raw(resp)
            print(f"\nRespons  : {resp.choices[0].message.content}")
            print(f"Tokens   : {resp.usage.prompt_tokens} in / {resp.usage.completion_tokens} out\n")
        except Exception as e:
            print(f"⚠️  TEST 5 gagal: {e}\n")

        time.sleep(20)

        # ---- TEST 6: Multi-turn Conversation ----
        try:
            print("="*60)
            print("TEST 6 — Multi-turn Conversation")
            print("="*60)

            history = [{"role": "system", "content": "Kamu adalah asisten coding Python yang ringkas."}]

            def chat(usermsg):
                history.append({"role": "user", "content": usermsg})
                resp = safe_call(
                    client.chat.completions.create,
                    model=model_name,
                    messages=history,
                    max_tokens=300,
                )
                reply = resp.choices[0].message.content
                history.append({"role": "assistant", "content": reply})
                print(f"📤 User  : {usermsg}")
                print(f"📥 Model : {reply}\n")

            chat("Tulis fungsi Python untuk hitung faktorial.")
            time.sleep(5)
            chat("Sekarang tambahkan validasi input negatif.")
            time.sleep(5)
            chat("Ubah jadi versi iteratif, bukan rekursif.")

            print(f"Total turns  : {len([m for m in history if m['role']=='user'])}")
            print(f"Total history: {len(history)} messages\n")
        except Exception as e:
            print(f"⚠️  TEST 6 gagal: {e}\n")

        # ---- TEST 7: Model Fallback ----
        try:
            print("="*60)
            print("TEST 7 — Model Fallback (OpenRouter feature)")
            print("="*60)
            # OpenRouter mendukung fallback via models array
            # Jika model pertama gagal/rate limit, otomatis ke model berikutnya

            fallback_models = [model_name, free_ids[1] if len(free_ids) > 1 else model_name]
            print(f"Fallback chain: {fallback_models}")

            payload = dict(
                model=fallback_models[0],
                messages=[{"role": "user", "content": "Sebutkan 3 bahasa pemrograman populer."}],
                max_tokens=100,
                extra_body={"models": fallback_models},  # OpenRouter-specific
            )
            print("📤 Payload:")
            show_payload(payload)

            resp = safe_call(client.chat.completions.create, **payload)
            print("\n📥 Raw Response:")
            show_raw(resp)
            print(f"\nModel yang dipakai : {resp.model}")
            print(f"Respons            : {resp.choices[0].message.content}\n")
        except Exception as e:
            print(f"⚠️  TEST 7 gagal: {e}\n")

        # ---- RINGKASAN ----
        print("="*60)
        print("RINGKASAN HASIL TEST")
        print("="*60)
        print(f"Model chat   : {model_name}")
        print(f"Model vision : {vision_name}")
        print(f"Free models  : {len(free_ids)} tersedia")
        print(f"Total models : {len(all_ids)} (berbayar + gratis)")
        print("""
Fitur OpenRouter vs Groq:
  + 315+ model (vs Groq 16)        + Fallback chain antar model
  + DeepSeek, Claude, Gemini, GPT  + Satu API key untuk semua provider
  - Kecepatan tidak sefast Groq    - Free: 50 req/hari (Groq: 1.000)
  - Model :free bisa dihapus       - Rate limit lebih ketat (20 RPM)
""")

# --- Auto-run saat model berubah ---
def on_change(change):
    if change["name"] == "value":
        run_all_tests(modeldd.value, visiondd.value)

modeldd.observe(on_change, names="value")
visiondd.observe(on_change, names="value")
```
