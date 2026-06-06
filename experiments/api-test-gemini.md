# 🧪 Gemini API — Experiments

Kumpulan eksperimen dengan Gemini API (Google GenAI).

---

## 1. List Available Models

```python
from google.colab import userdata
from google import genai

api_key = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

print("List of models that support generateContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "generateContent":
            print(m.name)

print("\nList of models that support embedContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "embedContent":
            print(m.name)
```

---

## 2. Image Analysis Test (Gemma 4)

```python
from google.colab import userdata
from google.colab import files
from google import genai
from google.genai import types
from PIL import Image
import io

# Load API key from Colab secrets
api_key = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

model = "models/gemma-4-26b-a4b-it"

# Upload & resize image to 512px
uploaded = files.upload()
filename = list(uploaded.keys())[0]

img = Image.open(io.BytesIO(uploaded[filename]))
img = img.resize((512, 512))
print(f"Image resized to 512x512 | Mode: {img.mode}")

# Convert to bytes
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

# Send to Gemma
response = client.models.generate_content(
    model=model,
    contents=[
        "Hello! Please describe this image in detail.",
        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    ]
)

print("\nModel response:")
print(response.text)
```

---

## 3. Photo Color Grading Pipeline

### Architecture

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant D as Drive
    participant G as Gemma4
    participant C as OpenCV

    rect rgb(240, 248, 255)
    Note over U,C: INIT
    U->>D: authenticate_user()
    U->>G: genai.Client(api_key)
    U->>D: find_folder(TARGET)
    alt Folder belum ada
        U->>D: create_folder(TARGET)
    end
    U->>D: list_files_in_folder()
    end

    rect rgb(255, 250, 240)
    Note over U,C: LOOP PER FOTO
    loop Setiap foto
        U->>D: list_images(SOURCE)
        alt File sudah ada
            U-->>U: SKIP
        else File baru
            U->>D: download(fname)
            U->>D: read EXIF
            rect rgb(255, 245, 238)
            Note over G: ANALISA
            U->>G: thumbnail(768px)
            U->>G: generate_content()
            G-->>U: JSON params
            end
            rect rgb(240, 255, 240)
            Note over C: OPENCV PIPELINE
            U->>C: load image
            U->>C: Gray World WB
            U->>C: WB fine-tune
            U->>C: Brightness
            U->>C: Contrast LUT
            U->>C: Highlights/Shadows
            U->>C: Blacks/Whites
            U->>C: Saturation/Vibrance
            U->>C: Clarity/Sharpness
            Note over C: QUALITY GUARD
            alt Green cast +5
                C-->>U: pakai original
            else OK
                C-->>U: simpan edited
            end
            end
            U->>D: upload(result)
            U-->>U: log
        end
        U->>U: sleep(3s)
    end
    end

    rect rgb(245, 245, 245)
    Note over U,C: CLEANUP
    U->>D: upload(log.json)
    U->>U: cleanup temp
    U-->>U: summary
    end
```

### Colab Notebook

```python
#@title 🎨 Gemini Lightroom — AI Photo Editor (Gemma 4) [FIXED]
# ─────────────────────────────────────────────
# ⚙️  KONFIGURASI
# ─────────────────────────────────────────────
DRIVE_FOLDER_URL    = ""  #@param {type:"string"}
TARGET_DIR_NAME     = "Edited by Gemma 4"  #@param {type:"string"}
GEMMA_MODEL         = "models/gemma-4-26b-a4b-it"  #@param {type:"string"}
JPEG_QUALITY        = 95  #@param {type:"integer"}
SLEEP_BETWEEN       = 3   # detik antar foto

# ─────────────────────────────────────────────
# 📦  INSTALL
# ─────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "google-genai",
                "opencv-python-headless",
                "google-api-python-client", "google-auth-httplib2",
                "google-auth-oauthlib", "Pillow", "tqdm"], check=True)

# ─────────────────────────────────────────────
# 📚  IMPORT
# ─────────────────────────────────────────────
import os, io, re, json, time, base64, shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm.notebook import tqdm

from google.colab import userdata, auth
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# 🔐  AUTENTIKASI
# ─────────────────────────────────────────────

# Google Drive
print("🔐 Login ke akun Google...")
auth.authenticate_user()
creds, _ = default()
drive_svc = build("drive", "v3", credentials=creds)
print("✅ Drive siap!")

# Gemma
api_key = userdata.get("GEMINI_API_KEY")
client  = genai.Client(api_key=api_key)
print(f"✅ Gemma siap! Model: {GEMMA_MODEL}")

# ─────────────────────────────────────────────
# 🛠️  UTILITAS DRIVE
# ─────────────────────────────────────────────

def extract_folder_id(url: str) -> str:
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"URL tidak valid: {url}")
    return m.group(1)


def list_images(folder_id: str) -> list:
    q = (
        f"'{folder_id}' in parents and trashed=false and "
        "(mimeType='image/jpeg' or mimeType='image/png' or "
        "mimeType='image/webp' or mimeType='image/tiff')"
    )
    return drive_svc.files().list(
        q=q, fields="files(id,name,mimeType,size)", pageSize=500
    ).execute().get("files", [])


def download(file_id: str, dest: str) -> str:
    req = drive_svc.files().get_media(fileId=file_id)
    with open(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return dest


def find_folder(name: str, parent_id: str = None) -> tuple:
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    results = drive_svc.files().list(q=q, fields="files(id,webViewLink)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"], files[0].get("webViewLink", "")
    return None, None


def create_folder(name: str, parent_id: str = None) -> tuple:
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    f = drive_svc.files().create(body=meta, fields="id,webViewLink").execute()
    return f["id"], f.get("webViewLink", "")


def list_files_in_folder(folder_id: str) -> set:
    q = f"'{folder_id}' in parents and trashed=false"
    results = drive_svc.files().list(q=q, fields="files(name)", pageSize=1000).execute()
    return {f["name"] for f in results.get("files", [])}


def upload(local_path: str, folder_id: str) -> None:
    name = os.path.basename(local_path)
    ext  = Path(local_path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".webp": "image/webp"}.get(ext, "image/jpeg")
    drive_svc.files().create(
        body={"name": name, "parents": [folder_id]},
        media_body=MediaFileUpload(local_path, mimetype=mime),
        fields="id",
    ).execute()


# ─────────────────────────────────────────────
# 🤖  SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional photo colorist. Analyze the image and return correction parameters as JSON.\n\n"

    "WORKFLOW (follow this order — each step depends on the previous):\n"
    "1) WHITE BALANCE — Is there a color cast? Check skin tones as reference (should be warm-neutral). "
    "   Also check for COLOR SPILL: environmental colors (green walls, blue sky, neon) bouncing onto skin/subjects.\n"
    "2) EXPOSURE — Too dark/bright? Check histogram mentally. Are highlights clipped? Blacks crushed?\n"
    "3) TONE — Contrast flat or harsh? For BACKLIGHT: do NOT flatten — instead lift shadows moderately "
    "   while PRESERVING contrast to maintain depth and separation.\n"
    "4) COLOR — Dull or oversaturated? Skin tones natural? Remove color spill from environment.\n"
    "5) DETAIL — Soft? Noisy?\n\n"

    "OUTPUT FORMAT (return ONLY this JSON, no markdown):\n"
    '{"b":0,"c":1.0,"s":1.0,"v":1.0,"h":0,"d":0,"k":0,"n":0,"w":0,"t":0,"p":1.0,"l":0,"x":"diagnosis"}\n\n'

    "PARAMETERS:\n"
    "b = brightness  (-80..80)    overall exposure offset\n"
    "c = contrast    (0.6..2.0)   S-curve strength\n"
    "s = saturation  (0.5..2.0)   linear color intensity\n"
    "v = vibrance    (0.8..2.0)   boosts low-sat areas, skin-safe\n"
    "h = highlights  (-80..20)    recovery (-) or boost (+)\n"
    "d = shadows     (-20..80)    lift shadows\n"
    "k = blacks      (-60..15)    lower black point\n"
    "n = whites      (0..60)      raise white point\n"
    "w = warmth      (-40..40)    negative=cool/blue, positive=warm/yellow\n"
    "t = tint        (-30..30)    negative=green shift, positive=magenta shift\n"
    "p = sharpness   (0.5..2.0)   unsharp mask strength\n"
    "l = clarity     (-20..60)    midtone contrast; negative=skin soften\n"
    "x = description              one-line diagnosis, max 12 words\n\n"

    "SPECIAL CASES:\n"
    "- COLOR SPILL (green walls/ceiling reflecting onto skin): "
    "  t=+15..+25 (magenta to cancel green spill), w=+5..+10, s=0.9..0.95. "
    "  Do NOT over-correct — only cancel the spill, don't shift entire image.\n"
    "- Green cast (LED/fluorescent, no spill on skin): "
    "  t=-15..-25, s=0.9..0.95, w=+5..+10\n"
    "- Low-light indoor: b=+15..+30, d=+25..+40, c=1.0..1.1 "
    "  (maintain some contrast for depth)\n"
    "- Backlight/contre-jour (bright background, dark subject): "
    "  d=+30..+50, b=+10..+20, h=-30..-50, c=0.95..1.05 "
    "  (lift shadows but KEEP contrast — don't flatten). "
    "  If subject has color spill from background, add tint correction.\n"
    "- Skin tones: keep v<=1.3 and s<=1.2. If skin looks orange/unnatural, reduce s and v.\n\n"

    "RULES: Output ONLY the JSON. Never refuse. Never add explanation outside JSON."
)

# ─────────────────────────────────────────────
# 🤖  KEY MAP & DEFAULTS
# ─────────────────────────────────────────────

_KEY_MAP = {
    "b": "brightness", "c": "contrast",   "s": "saturation",
    "v": "vibrance",   "h": "highlights", "d": "shadows",
    "k": "blacks",     "n": "whites",     "w": "warmth",
    "t": "tint",       "p": "sharpness",  "l": "clarity",
    "x": "description",
}

DEFAULT_PARAMS = {
    "brightness": 8,    "contrast": 1.05,  "saturation": 1.0,
    "vibrance": 1.1,    "highlights": -8,  "shadows": 20,
    "blacks": -5,       "whites": 5,       "warmth": 0,
    "tint": 0,          "sharpness": 1.05, "clarity": 8,
    "description": "fallback: conservative correction",
}


def analyze_image(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > 768:
        img.thumbnail((768, 768), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    img_bytes = buf.getvalue()

    try:
        resp = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                "Analyze this image. Return ONLY the JSON.",
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )

        print(f"    🔍 RAW: {repr(resp.text)[:300]}")

        if resp.text is None:
            raise ValueError("Response kosong (text=None)")

        text = resp.text.strip()

        if any(kw in text.lower() for kw in ("rejected", "high risk", "cannot analyze", "i'm sorry")):
            raise ValueError(f"Safety rejection: {text[:120]}")

        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON in response: {text[:150]}")

        raw = json.loads(m.group(0))
        if isinstance(raw, list):
            raw = raw[0] if raw else {}

        result = {_KEY_MAP.get(k, k): v for k, v in raw.items()}

        NEUTRAL = {
            "brightness": 0, "contrast": 1.0, "saturation": 1.0,
            "vibrance": 1.0, "highlights": 0, "shadows": 0,
            "blacks": 0,     "whites": 0,     "warmth": 0,
            "tint": 0,       "sharpness": 1.0,"clarity": 0,
        }
        for k, neutral_val in NEUTRAL.items():
            if k not in result:
                result[k] = neutral_val

        return result

    except Exception as e:
        print(f"    ⚠️ Gemma error: {e}, pakai default")
        return DEFAULT_PARAMS.copy()


# ─────────────────────────────────────────────
# 🖼️  OPENCV PIPELINE
# ─────────────────────────────────────────────

def build_contrast_lut(contrast: float) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        x = i / 255.0
        if x < 0.5:
            y = 0.5 - (0.5 - x) ** (1.0 / max(contrast, 0.01))
        else:
            y = 0.5 + (x - 0.5) ** (1.0 / max(contrast, 0.01))
        lut[i] = np.clip(y * 255, 0, 255)
    return lut


def is_backlight(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h = gray.shape[0]
    mean_top    = np.mean(gray[:int(h * 0.4), :])
    mean_bottom = np.mean(gray[int(h * 0.4):, :])
    detected    = (mean_top - mean_bottom) > 50
    if detected:
        print(f"    🌅 Backlight terdeteksi (top={mean_top:.1f} bottom={mean_bottom:.1f}): skip Gray World WB")
    return detected


def gray_world_wb(img: np.ndarray, strength: float = 0.7) -> np.ndarray:
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg   = (avg_b + avg_g + avg_r) / 3.0

    scale_b = (avg / avg_b) ** strength
    scale_g = (avg / avg_g) ** strength
    scale_r = (avg / avg_r) ** strength

    img[:, :, 0] = np.clip(img[:, :, 0] * scale_b, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * scale_r, 0, 255)

    print(f"    🌡️  Gray World: B×{scale_b:.2f} G×{scale_g:.2f} R×{scale_r:.2f}")
    return img


def green_excess(img: np.ndarray) -> float:
    return float(np.mean(img[:, :, 1]) - (np.mean(img[:, :, 0]) + np.mean(img[:, :, 2])) / 2.0)


def apply_corrections(input_path: str, params: dict, output_path: str) -> str:
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Tidak bisa baca: {input_path}")

    original_bgr = img_bgr.copy()
    img = img_bgr.astype(np.float32)

    if not is_backlight(img):
        img = gray_world_wb(img, strength=0.7)

    warmth = params.get("warmth", 0)
    tint   = params.get("tint",   0)
    wb_r = 1.0 + warmth / 200.0
    wb_b = 1.0 - warmth / 200.0
    wb_g = 1.0 - tint   / 200.0

    img[:, :, 2] = np.clip(img[:, :, 2] * wb_r, 0, 255)
    img[:, :, 0] = np.clip(img[:, :, 0] * wb_b, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * wb_g, 0, 255)

    img = np.clip(img + params.get("brightness", 0) * 2.0, 0, 255)

    lut = build_contrast_lut(params.get("contrast", 1.0))
    img = cv2.LUT(img.astype(np.uint8), lut).astype(np.float32)

    norm = np.clip(img, 0, 255) / 255.0

    img += params.get("highlights", 0) / 100.0 * (norm ** 2)          * 100.0
    img += params.get("shadows",    0) / 100.0 * ((1.0 - norm) ** 2) * 100.0
    img  = np.clip(img, 0, 255)

    norm = np.clip(img, 0, 255) / 255.0

    img += params.get("blacks", 0) / 100.0 * ((1.0 - norm) ** 3) * 150.0
    img += params.get("whites", 0) / 100.0 * (norm ** 3)          * 150.0
    img  = np.clip(img, 0, 255)

    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    s_original = hsv[:, :, 1].copy()
    s = np.clip(s_original * params.get("saturation", 1.0), 0, 255)

    low_sat_mask  = (1.0 - s_original / 255.0) ** 2
    vib_factor    = params.get("vibrance", 1.0)
    effective_vib = 1.0 + (vib_factor - 1.0) * low_sat_mask
    s = np.clip(s * effective_vib, 0, 255)

    hsv[:, :, 1] = s
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    h_px, w_px = img.shape[:2]

    sigma_clarity = max(h_px, w_px) * 0.004
    clarity = params.get("clarity", 0)
    if clarity != 0:
        u8      = img.astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=sigma_clarity)
        lum     = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mid     = 4.0 * lum * (1.0 - lum)
        img     = np.clip(img + (clarity / 100.0) * (img - blurred.astype(np.float32))
                          * mid[:, :, np.newaxis], 0, 255)

    sigma_sharp = max(h_px, w_px) * 0.0006
    sharp = params.get("sharpness", 1.0)
    if sharp != 1.0:
        u8      = img.astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=sigma_sharp)
        img     = np.clip(img + (sharp - 1.0) * (img - blurred.astype(np.float32)) * 2, 0, 255)

    result = img.astype(np.uint8)

    ge_before = green_excess(original_bgr.astype(np.float32))
    ge_after  = green_excess(result.astype(np.float32))
    if ge_after > ge_before + 5:
        print(f"    ⚠️  Quality guard: edit memperburuk green cast "
              f"({ge_before:.1f} → {ge_after:.1f}), pakai original")
        shutil.copy(input_path, output_path)
        return output_path

    ext = Path(output_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        cv2.imwrite(output_path, result)
    return output_path


# ─────────────────────────────────────────────
# 🚀  MAIN
# ─────────────────────────────────────────────

if not DRIVE_FOLDER_URL:
    raise ValueError("❌ Isi DRIVE_FOLDER_URL dulu!")

INPUT_DIR  = Path("/tmp/gl_input");  INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("/tmp/gl_output"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

src_id = extract_folder_id(DRIVE_FOLDER_URL)
print(f"📁 Source folder: {src_id}")

images = list_images(src_id)
if not images:
    raise RuntimeError("❌ Tidak ada foto di folder!")
print(f"📸 Ditemukan {len(images)} foto")

dst_id, dst_link = find_folder(TARGET_DIR_NAME)
if dst_id:
    print(f"📂 Folder output sudah ada, gunakan: {dst_link}")
else:
    dst_id, dst_link = create_folder(TARGET_DIR_NAME)
    print(f"📂 Folder output baru: {dst_link}")

existing_files = list_files_in_folder(dst_id)
print(f"📋 Sudah ada {len(existing_files)} file di target")

ok, fail, skip, log = 0, 0, 0, []

for f in tqdm(images, desc="Memproses"):
    fname     = f["name"]
    local_in  = str(INPUT_DIR  / fname)
    local_out = str(OUTPUT_DIR / fname)

    if fname in existing_files:
        skip += 1
        print(f"  ⏭️ {fname}: sudah ada, skip")
        log.append({"file": fname, "status": "skip"})
        time.sleep(0.5)
        continue

    try:
        download(f["id"], local_in)
        params = analyze_image(local_in)
        apply_corrections(local_in, params, local_out)
        upload(local_out, dst_id)
        ok += 1
        desc = params.get("description", "")
        print(f"  ✅ {fname}: {desc}")
        log.append({"file": fname, "status": "ok", **{k: params[k] for k in
                    ("contrast", "saturation", "warmth", "tint", "description") if k in params}})
    except Exception as e:
        fail += 1
        print(f"  ❌ {fname}: {e}")
        log.append({"file": fname, "status": "error", "note": str(e)})

    time.sleep(SLEEP_BETWEEN)

log_path = str(OUTPUT_DIR / "log.json")
with open(log_path, "w") as fh:
    json.dump(log, fh, indent=2, ensure_ascii=False)
upload(log_path, dst_id)

shutil.rmtree(str(INPUT_DIR),  ignore_errors=True)
shutil.rmtree(str(OUTPUT_DIR), ignore_errors=True)

print(f"\n{'═'*50}")
print(f"🎉 Selesai! ✅ {ok} berhasil  ⏭️ {skip} skip  ❌ {fail} gagal")
print(f"📂 {dst_link}")
print(f"{'═'*50}")
```
