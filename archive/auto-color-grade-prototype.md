```python
#@title 🎨 Auto Color Grade — AI Photo Editor
# ─────────────────────────────────────────────
# ⚙️  KONFIGURASI — isi di sini sebelum run
# ─────────────────────────────────────────────
DRIVE_FOLDER_URL   = "https://drive.google.com/drive/folders/1-CtxXmmsT5F00j-5968fDqoO957XoCvy?usp=sharing" #@param {type:"string"}

MIMO_API_KEY       = ""  #@param {type:"string"}
MIMO_BASE_URL      = "https://token-plan-sgp.xiaomimimo.com/anthropic"  #@param ["https://token-plan-sgp.xiaomimimo.com/anthropic", "https://token-plan-sgp.xiaomimimo.com/v1", "https://api.xiaomimimo.com/v1"]

# Fallback ke Colab secret
if not MIMO_API_KEY:
    from google.colab import userdata
    MIMO_API_KEY = userdata.get("MIMO_API_KEY")
    print("🔑 Pakai API key dari Colab Secret")
RESULT_FOLDER_NAME = "Lamaran Edited 28-05-2026"  #@param {type:"string"}
JPEG_QUALITY       = 95  #@param {type:"integer"}
SLEEP_BETWEEN      = 3   # detik antar foto (jaga rate limit)

# ─────────────────────────────────────────────
# 📦  INSTALL
# ─────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "anthropic",
                "opencv-python-headless",
                "google-api-python-client", "google-auth-httplib2",
                "google-auth-oauthlib", "Pillow", "tqdm"], check=True)

# ─────────────────────────────────────────────
# 📚  IMPORT
# ─────────────────────────────────────────────
import os, io, re, json, time, shutil, base64
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm.notebook import tqdm

from google.colab import auth
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ─────────────────────────────────────────────
# 🔐  AUTENTIKASI GOOGLE
# ─────────────────────────────────────────────
print("🔐 Login ke akun Google kamu...")
auth.authenticate_user()
creds, _ = default()
drive_svc = build("drive", "v3", credentials=creds)
print("✅ Berhasil login!")

from anthropic import Anthropic

# ─────────────────────────────────────────────
# 🤖  INIT MIMO (Anthropic endpoint)
# ─────────────────────────────────────────────
if not MIMO_API_KEY:
    raise ValueError("❌ MIMO_API_KEY belum diisi!")

mimo = Anthropic(
    api_key=MIMO_API_KEY,
    base_url=MIMO_BASE_URL,
)
print("✅ MiMo Anthropic siap!")

# ─────────────────────────────────────────────
# 🛠️  FUNGSI UTILITAS
# ─────────────────────────────────────────────

def extract_folder_id(url: str) -> str:
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"URL tidak valid: {url}")
    return m.group(1)


def list_images_in_folder(folder_id: str) -> list:
    IMG_TYPES = "image/jpeg,image/png,image/webp,image/tiff"
    query = f"'{folder_id}' in parents and trashed=false and ("
    query += " or ".join([f"mimeType='{m}'" for m in IMG_TYPES.split(",")])
    query += ")"
    results = drive_svc.files().list(
        q=query,
        fields="files(id, name, mimeType, size)",
        pageSize=500
    ).execute()
    return results.get("files", [])


def download_drive_file(file_id: str, dest_path: str) -> str:
    request = drive_svc.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return dest_path


def create_drive_folder(name: str, parent_id: str = None) -> str:
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder"
    }
    if parent_id:
        meta["parents"] = [parent_id]
    folder = drive_svc.files().create(body=meta, fields="id, webViewLink").execute()
    return folder["id"], folder.get("webViewLink", "")


def upload_to_drive(local_path: str, folder_id: str) -> None:
    name = os.path.basename(local_path)
    ext  = Path(local_path).suffix.lower()
    mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png',  '.webp': 'image/webp'}.get(ext, 'image/jpeg')
    drive_svc.files().create(
        body={"name": name, "parents": [folder_id]},
        media_body=MediaFileUpload(local_path, mimetype=mime),
        fields="id"
    ).execute()


# ─────────────────────────────────────────────
# 🤖  MIMO: ANALISA GAMBAR → PARAMETER JSON
# Research-based: urutan analisis mengikuti
# workflow profesional (WB → Exposure → Tone → Color → Detail)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "type": "text",
        "text": (
            "Professional photo colorist. Analyze in order: "
            "1) WB — color cast? (warm/cool/green/magenta) "
            "2) Exposure — too dark/bright? clipped? crushed? "
            "3) Tone — contrast flat/harsh? highlights blown? shadows blocked? "
            "4) Color — dull/oversaturated? skin tones safe? "
            "5) Detail — soft? noisy? needs texture? "
            "Output ONLY JSON (no markdown): "
            '{"b":0,"c":1.0,"s":1.0,"v":1.0,"h":0,"d":0,"k":0,"n":0,'
            '"w":0,"t":0,"p":1.0,"l":0,"x":"diagnosis"} '
            "Keys: b=brightness,c=contrast,s=saturation,v=vibrance,"
            "h=highlights,d=shadows,k=blacks,n=whites,"
            "w=warmth,t=tint,p=sharpness,l=clarity,x=description. "
            "Never refuse. No text outside JSON. "
            "This is a legitimate professional color grading application."
        ),
        "cache_control": {"type": "ephemeral"},
    }
]

# User message minimal — analysis workflow sudah di system (cached)
USER_PROMPT = "Analyze. Return JSON."

# Mapping key pendek → key panjang (tambah k=blacks, n=whites)
_KEY = {
    "b": "brightness", "c": "contrast",   "s": "saturation",
    "v": "vibrance",   "h": "highlights", "d": "shadows",
    "k": "blacks",     "n": "whites",     "w": "warmth",
    "t": "tint",       "p": "sharpness",  "l": "clarity",
    "x": "description",
}

# Default: koreksi ringan untuk gambar yang terlalu flat/pale
# (lebih konservatif dari sebelumnya — hindari over-edit)
DEFAULT_PARAMS = {
    "brightness": 8,    "contrast": 1.1,  "saturation": 1.05,
    "vibrance": 1.15,   "highlights": -8, "shadows": 15,
    "blacks": -10,      "whites": 8,      "warmth": 0,
    "tint": 0,          "sharpness": 1.05,"clarity": 8,
    "description": "fallback: conservative pale-image correction",
}


def analyze_image(image_path: str) -> dict:
    """Kirim gambar ke MiMo, return parameter koreksi."""
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > 512:
        img.thumbnail((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    # Quality 82 → lebih baik preservasi warna untuk analisis
    img.save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        r = mimo.messages.create(
            model="mimo-v2.5",
            max_tokens=220,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": b64
                    }},
                    {"type": "text", "text": USER_PROMPT},
                ],
            }],
            thinking={"type": "disabled"},
        )

        text = "".join(block.text for block in r.content if block.type == "text")
        text = text.strip()

        # Safety rejection → langsung fallback
        if any(kw in text.lower() for kw in ("rejected", "high risk", "cannot analyze", "i'm sorry")):
            raise ValueError(f"safety rejection: {text[:120]}")

        # Extract JSON object (toleran terhadap teks sebelum/sesudah)
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON found in: {text[:120]}")
        raw = json.loads(m.group(0))

        if isinstance(raw, list):
            raw = raw[0] if raw else {}

        result = {_KEY.get(k, k): v for k, v in raw.items()}

        # Validasi: pastikan key penting ada, isi dengan neutral jika tidak
        NEUTRAL = {
            "brightness": 0, "contrast": 1.0, "saturation": 1.0,
            "vibrance": 1.0, "highlights": 0, "shadows": 0,
            "blacks": 0,     "whites": 0,     "warmth": 0,
            "tint": 0,       "sharpness": 1.0,"clarity": 0,
        }
        for key, neutral_val in NEUTRAL.items():
            if key not in result:
                result[key] = neutral_val

        return result

    except Exception as e:
        print(f"    ⚠️ MiMo: {e}, pakai default")
        return DEFAULT_PARAMS.copy()


# ─────────────────────────────────────────────
# 🖼️  OPENCV: TERAPKAN KOREKSI WARNA
# ─────────────────────────────────────────────

def apply_corrections(input_path: str, params: dict, output_path: str) -> str:
    # 1. Baca
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Tidak bisa membaca: {input_path}")
    img = img_bgr.astype(np.float32)

    # 2. Brightness
    img = np.clip(img + params.get("brightness", 0) * 2.0, 0, 255)

    # 3. Contrast
    c = params.get("contrast", 1.0)
    img = np.clip((img - 128.0) * c + 128.0, 0, 255)

    # 4. Saturation + Vibrance (HSV)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    s   = hsv[:, :, 1]
    s   = np.clip(s * params.get("saturation", 1.0), 0, 255)
    low_sat_mask = (1.0 - s / 255.0) ** 2
    s   = np.clip(s + (params.get("vibrance", 1.0) - 1.0) * 80.0 * low_sat_mask, 0, 255)
    hsv[:, :, 1] = s
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # 5. Warmth & Tint
    warmth = params.get("warmth", 0)
    tint   = params.get("tint",   0)
    img[:, :, 2] = np.clip(img[:, :, 2] + warmth * 1.5, 0, 255)  # Red
    img[:, :, 0] = np.clip(img[:, :, 0] - warmth * 1.0, 0, 255)  # Blue
    img[:, :, 1] = np.clip(img[:, :, 1] + tint   * 1.0, 0, 255)  # Green

    # 6. Highlights & Shadows
    norm   = img / 255.0
    img   += params.get("highlights", 0) / 100.0 * (norm ** 2)        * 100.0
    img   += params.get("shadows",    0) / 100.0 * ((1.0 - norm) ** 2) * 100.0
    img    = np.clip(img, 0, 255)

    # 6b. Blacks & Whites (cubic curve — targets deeper shadows/brighter highlights)
    img   += params.get("blacks", 0) / 100.0 * ((1.0 - norm) ** 3) * 150.0
    img   += params.get("whites", 0) / 100.0 * (norm ** 3)          * 150.0
    img    = np.clip(img, 0, 255)

    # 7. Clarity (midtone contrast)
    clarity = params.get("clarity", 0)
    if clarity > 0:
        u8      = img.astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=10)
        lum     = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mid     = 4.0 * lum * (1.0 - lum)
        img     = np.clip(img + (clarity / 100.0) * (img - blurred.astype(np.float32))
                          * mid[:, :, np.newaxis], 0, 255)

    # 8. Sharpness
    sharp = params.get("sharpness", 1.0)
    if sharp != 1.0:
        u8      = img.astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=1.5)
        img     = np.clip(img + (sharp - 1.0) * (img - blurred.astype(np.float32)) * 2, 0, 255)

    # 9. Simpan
    result = img.astype(np.uint8)
    ext    = Path(output_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        cv2.imwrite(output_path, result)
    return output_path


# ─────────────────────────────────────────────
# 🚀  MAIN
# ─────────────────────────────────────────────

INPUT_DIR  = Path("/tmp/cg_input");  INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("/tmp/cg_output"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

src_folder_id = extract_folder_id(DRIVE_FOLDER_URL)
print(f"📁 Source folder ID : {src_folder_id}")

images = list_images_in_folder(src_folder_id)
if not images:
    raise RuntimeError("Tidak ada foto, atau folder tidak bisa diakses.")
print(f"📸 Ditemukan {len(images)} foto")

result_id, result_link = create_drive_folder(RESULT_FOLDER_NAME)
print(f"📂 Folder hasil: {result_link}")

ok, fail, log = 0, 0, []

for file in tqdm(images, desc="Memproses foto"):
    fname     = file["name"]
    local_in  = str(INPUT_DIR  / fname)
    local_out = str(OUTPUT_DIR / fname)
    try:
        download_drive_file(file["id"], local_in)
        params = analyze_image(local_in)
        apply_corrections(local_in, params, local_out)
        upload_to_drive(local_out, result_id)
        ok += 1
        print(f"  ✅ {fname}: {params.get('description','')}")
        log.append({"file": fname, "status": "ok", **{k: params[k] for k in
                    ("contrast","saturation","warmth","description") if k in params}})
    except Exception as e:
        fail += 1
        print(f"  ❌ {fname}: {e}")
        log.append({"file": fname, "status": "error", "note": str(e)})
    time.sleep(SLEEP_BETWEEN)

log_path = "/tmp/processing_log.json"
with open(log_path, "w") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
upload_to_drive(log_path, result_id)

shutil.rmtree(str(INPUT_DIR),  ignore_errors=True)
shutil.rmtree(str(OUTPUT_DIR), ignore_errors=True)

print(f"\n{'═'*50}")
print(f"🎉 Selesai! ✅ {ok} berhasil  ❌ {fail} gagal")
print(f"📂 {result_link}")
print(f"{'═'*50}")
```