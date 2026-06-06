# Infisical: Research Mendalam untuk Kaggle & Google Colab

> **Konteks:** Research ini dibuat untuk menggantikan kebiasaan input manual secret di Kaggle/Colab, dengan fokus pada Infisical Cloud (hosted) sebagai solusi terpusat.

---

## 1. Apa itu Infisical?

Infisical adalah platform open-source, end-to-end encrypted untuk manajemen secrets (API keys, tokens, environment variables, database credentials). Didirikan sebagai YC W23, Infisical memposisikan diri sebagai alternatif open-source dari HashiCorp Vault dan Doppler, dengan 25.700+ GitHub stars (per Juni 2026).

**Tagline:** *"Single Source of Truth"* untuk semua secrets di seluruh tim, device, dan infrastruktur.

**Repositori GitHub:** https://github.com/Infisical/infisical  
**Website:** https://infisical.com  
**Dokumentasi:** https://infisical.com/docs  
**Status page:** https://status.infisical.com

---

## 2. Arsitektur Keamanan

Ini salah satu keunggulan terbesar Infisical yang membedakannya dari kompetitor.

### 2.1 Blind Backend / Client-Side Encryption

Infisical menggunakan model **"Blind Backend"**: secrets dienkripsi di sisi client (browser/CLI) *sebelum* dikirim ke server. Server hanya menyimpan ciphertext — tidak pernah plaintext.

> Bahkan di Infisical Cloud, secara teknis tidak mungkin karyawan Infisical melihat nilai secrets kamu, kecuali kamu secara eksplisit memberikan akses.

Ini berbeda dari kebanyakan secrets manager lain (termasuk AWS Secrets Manager) yang enkripsi dilakukan "at rest" di sisi server — artinya provider *bisa* membaca datamu secara teori.

### 2.2 Enkripsi yang Digunakan

- **Algoritma:** AES-256-GCM (Galois Counter Mode) dengan 96-bit nonces
- **Skema:** Authenticated encryption — memberikan jaminan **confidentiality** sekaligus **integrity**
- **Arsitektur:** Multilayer key hierarchy — setiap layer mengenkripsi layer di bawahnya

### 2.3 Compliance

Infisical Cloud telah mendapatkan sertifikasi:
- **SOC 2 Type II**
- **HIPAA**
- **FIPS 140-3**
- Continuous penetration testing

### 2.4 Audit Logging

Setiap event (akses secret, perubahan policy, mutasi data) di-log dengan timestamp, informasi aktor, IP address, dan user-agent. Ini penting untuk traceability.

---

## 3. Fitur Utama

### 3.1 Core Features (Tersedia di Free Tier)
- **Dashboard UI** — antarmuka web untuk manage secrets
- **CLI** (`infisical run`, `infisical secrets`) — inject secrets langsung ke proses
- **SDK resmi** — Python, Node.js, Go, Java, Ruby, dll.
- **REST API** — akses programmatic penuh
- **Kubernetes Operator** — sync secrets ke native K8s Secrets
- **Infisical Agent** — daemon untuk inject secrets ke aplikasi legacy
- **Webhooks** — trigger event saat secret berubah
- **Secret Referencing** — satu secret bisa mereferensikan nilai secret lain
- **Secret Overrides** — override nilai untuk environment tertentu
- **Secret Scanning** — scan codebase untuk detect leaked secrets
- **Secret Sharing** — share secret secara aman via one-time link
- **2FA** — autentikasi dua faktor untuk akun

### 3.2 Fitur Pro (Berbayar)
- Secret Versioning
- Point-in-time Recovery
- Role-based Access Controls (RBAC) lanjutan
- Secret Rotation otomatis
- SAML SSO
- IP Allowlisting
- 90-day Audit Log Retention
- Temporary Access Provisioning

### 3.3 Fitur Enterprise (Custom)
- Dynamic Secrets (generate credentials on-demand)
- LDAP Authentication
- SCIM (user provisioning otomatis)
- KMIP & HSM Support
- Approval Workflows
- 99.99% SLA
- Dedicated Support Engineer

---

## 4. Pricing (Per Juni 2026)

Infisical menggunakan **identity-based pricing**. Satu "identity" bisa berupa manusia (user) maupun mesin (Machine Identity: CI/CD pipeline, aplikasi, Kubernetes pod, dll.).

| Plan | Harga | Identity Limit | Project Limit | Environment Limit | Integration Limit |
|---|---|---|---|---|---|
| **Free** | $0/bulan | **5 identities** | 3 | 3 | 10 |
| **Pro** | $18/identity/bulan | Unlimited | Unlimited | 12 | 50 |
| **Enterprise** | Custom | Unlimited | Unlimited | Unlimited | Unlimited |

### Catatan Penting untuk Solo Developer (Free Tier):

- **5 identities total** — ini mencakup akun kamu (1) + Machine Identities (sisa 4)
- Untuk penggunaan personal Kaggle/Colab: **1 user + 1 Machine Identity = 2 dari 5** → masih aman
- **3 projects** — misalnya: `ml-projects`, `web-projects`, `personal` 
- **3 environments** — default: `dev`, `staging`, `prod`
- Self-hosting gratis sepenuhnya jika nanti ingin migrasi

> **Kesimpulan pricing untuk solo dev:** Free tier **cukup** untuk penggunaan personal Kaggle/Colab. Kamu punya headroom 3 Machine Identity lagi untuk project lain.

---

## 5. Konsep Kunci yang Perlu Dipahami

### 5.1 Machine Identity

Machine Identity adalah entitas non-manusia (aplikasi, script, notebook) yang autentikasi ke Infisical. Ada beberapa metode auth:

| Metode | Cocok Untuk |
|---|---|
| **Universal Auth** | Semua platform (termasuk Kaggle/Colab) |
| Kubernetes Auth | K8s pods |
| AWS IAM Auth | AWS Lambda, EC2 |
| GCP IAM Auth | GCP Cloud Run, Cloud Functions |
| Azure Auth | Azure resources |
| OIDC Auth | GitHub Actions, GitLab CI/CD |

Untuk Kaggle dan Colab, gunakan **Universal Auth** — paling fleksibel karena tidak terikat cloud provider tertentu.

### 5.2 Client ID & Client Secret

Universal Auth menghasilkan pasangan `CLIENT_ID` + `CLIENT_SECRET`. Analoginya seperti username/password untuk Machine Identity. Yang perlu disimpan di Kaggle Secrets hanya dua nilai ini.

### 5.3 Project & Environment

- **Project** = wadah untuk sekumpulan secrets (misalnya satu project ML)
- **Environment** = `dev`, `staging`, `prod` (bisa custom)
- Secrets bisa di-scope ke environment tertentu dan punya path hierarki folder

---

## 6. Implementasi: Kaggle & Google Colab

### 6.1 Setup Awal di Infisical Cloud

1. Buat akun di https://app.infisical.com/signup
2. Buat **Project** baru (misal: `kaggle-ml`)
3. Tambahkan secrets via dashboard (misal: `OPENAI_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`)
4. Buat **Machine Identity** via `Organization Settings > Machine Identities`
5. Pilih metode auth: **Universal Auth**
6. Simpan `CLIENT_ID` dan `CLIENT_SECRET` yang dihasilkan
7. Assign Machine Identity ke project dengan permission **read** saja

### 6.2 Setup Secrets di Kaggle

Simpan hanya dua nilai di Kaggle Secrets:
- `INFISICAL_CLIENT_ID` → nilai CLIENT_ID dari step di atas
- `INFISICAL_CLIENT_SECRET` → nilai CLIENT_SECRET

### 6.3 Kode Python: Helper Function untuk Kaggle

```python
# ============================================================
# infisical_loader.py — Paste di cell pertama notebook
# ============================================================

def load_infisical_secrets(
    project_id: str,
    environment: str = "dev",
    secret_path: str = "/",
    set_env_vars: bool = True
) -> dict:
    """
    Load secrets dari Infisical Cloud dan set sebagai environment variables.
    
    Params:
        project_id   : Project ID dari Infisical dashboard
        environment  : Nama environment ('dev', 'staging', 'prod', dll.)
        secret_path  : Path folder secrets (default: root '/')
        set_env_vars : Jika True, auto-set ke os.environ
    
    Returns:
        dict berisi {secret_name: secret_value}
    """
    import os
    import subprocess
    
    # --- Install SDK jika belum ada ---
    try:
        from infisical_sdk import InfisicalSDKClient
    except ImportError:
        subprocess.run(["pip", "install", "infisicalsdk", "-q"], check=True)
        from infisical_sdk import InfisicalSDKClient
    
    # --- Ambil credentials dari Kaggle Secrets ---
    try:
        from kaggle_secrets import UserSecretsClient
        secrets_client = UserSecretsClient()
        client_id     = secrets_client.get_secret("INFISICAL_CLIENT_ID")
        client_secret = secrets_client.get_secret("INFISICAL_CLIENT_SECRET")
        print("[Kaggle] Loaded credentials from Kaggle Secrets")
    except Exception:
        # Fallback: coba dari environment variable (untuk Colab atau lokal)
        client_id     = os.environ.get("INFISICAL_CLIENT_ID")
        client_secret = os.environ.get("INFISICAL_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError(
                "Tidak ditemukan INFISICAL_CLIENT_ID / INFISICAL_CLIENT_SECRET. "
                "Set di Kaggle Secrets atau Colab Secrets terlebih dahulu."
            )
        print("[Env] Loaded credentials from environment variables")
    
    # --- Autentikasi ke Infisical ---
    client = InfisicalSDKClient(host="https://app.infisical.com")
    client.auth.universal_auth.login(
        client_id=client_id,
        client_secret=client_secret
    )
    
    # --- Ambil semua secrets dari path yang ditentukan ---
    secrets_response = client.secrets.list_secrets(
        project_id=project_id,
        environment_slug=environment,
        secret_path=secret_path
    )
    
    result = {}
    for secret in secrets_response.secrets:
        result[secret.secret_key] = secret.secret_value
        if set_env_vars:
            os.environ[secret.secret_key] = secret.secret_value
    
    print(f"[Infisical] Loaded {len(result)} secrets dari '{environment}/{secret_path}'")
    print(f"[Infisical] Keys: {list(result.keys())}")
    return result


# ============================================================
# USAGE (ganti PROJECT_ID dengan ID dari Infisical dashboard)
# ============================================================
PROJECT_ID = "your-project-id-here"  # dari URL: app.infisical.com/project/XXX/secrets

secrets = load_infisical_secrets(
    project_id=PROJECT_ID,
    environment="dev",
    secret_path="/"
)

# Setelah ini, semua secret sudah di os.environ
# Contoh akses:
# import os
# openai_key = os.environ["OPENAI_API_KEY"]
# hf_token   = os.environ["HF_TOKEN"]
```

### 6.4 Kode Python: Versi Universal (Kaggle + Colab)

```python
# ============================================================
# Snippet yang works di BOTH Kaggle AND Colab
# ============================================================

import os
import subprocess

def get_platform():
    """Detect current notebook platform."""
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return "kaggle"
    except ImportError:
        pass
    return "local"

def get_infisical_credentials():
    """Ambil CLIENT_ID dan CLIENT_SECRET sesuai platform."""
    platform = get_platform()
    
    if platform == "kaggle":
        from kaggle_secrets import UserSecretsClient
        sc = UserSecretsClient()
        return sc.get_secret("INFISICAL_CLIENT_ID"), sc.get_secret("INFISICAL_CLIENT_SECRET")
    
    elif platform == "colab":
        from google.colab import userdata
        return userdata.get("INFISICAL_CLIENT_ID"), userdata.get("INFISICAL_CLIENT_SECRET")
    
    else:  # local / CI
        return os.environ["INFISICAL_CLIENT_ID"], os.environ["INFISICAL_CLIENT_SECRET"]


def load_secrets(project_id: str, environment: str = "dev", path: str = "/"):
    """Load dan inject secrets ke os.environ."""
    subprocess.run(["pip", "install", "infisicalsdk", "-q"], check=True)
    from infisical_sdk import InfisicalSDKClient
    
    client_id, client_secret = get_infisical_credentials()
    
    client = InfisicalSDKClient(host="https://app.infisical.com")
    client.auth.universal_auth.login(
        client_id=client_id,
        client_secret=client_secret
    )
    
    resp = client.secrets.list_secrets(
        project_id=project_id,
        environment_slug=environment,
        secret_path=path
    )
    
    for s in resp.secrets:
        os.environ[s.secret_key] = s.secret_value
    
    print(f"✓ {len(resp.secrets)} secrets loaded [{get_platform()}]")


# Jalankan:
load_secrets(project_id="YOUR_PROJECT_ID")
```

### 6.5 Catatan Instalasi Package

> **Penting:** Nama package di PyPI adalah `infisicalsdk` (tanpa hyphen/underscore), bukan `infisical-sdk`.

```bash
pip install infisicalsdk
```

---

## 7. Struktur Rekomendasi di Infisical

Untuk penggunaan Kaggle/Colab, berikut struktur yang disarankan:

```
Project: kaggle-ml
├── Environment: dev
│   ├── / (root)
│   │   ├── HF_TOKEN
│   │   ├── WANDB_API_KEY
│   │   ├── OPENAI_API_KEY
│   │   └── ANTHROPIC_API_KEY
│   ├── /kaggle
│   │   └── KAGGLE_USERNAME
│   └── /database
│       └── DB_URL
└── Environment: prod
    └── (jika perlu pisah secrets production)
```

**Keuntungan menggunakan folder (`/kaggle`, `/database`):**
- Load hanya subset secrets yang dibutuhkan per notebook
- Prinsip least-privilege: notebook A tidak perlu tahu secrets notebook B

---

## 8. Perbandingan Infisical vs Alternatif

| Aspek | Infisical Cloud | Private Git Repo + .env | Kaggle/Colab Native Secrets | Doppler | AWS Secrets Manager |
|---|---|---|---|---|---|
| **Keamanan** | ⭐⭐⭐⭐⭐ E2EE | ⭐⭐ Plaintext di repo | ⭐⭐⭐ Platform-managed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Kemudahan setup** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Centralized** | ✅ | ✅ | ❌ (per platform) | ✅ | ✅ |
| **Audit log** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Secret rotation** | ✅ Pro | ❌ Manual | ❌ | ✅ | ✅ |
| **Open source** | ✅ MIT core | N/A | ❌ | ❌ | ❌ |
| **Self-host option** | ✅ Gratis | N/A | ❌ | ❌ | ❌ |
| **Free tier** | ✅ 5 identities | Gratis (GitHub) | ✅ | ✅ Terbatas | ❌ |
| **Gratis untuk solo** | ✅ | ✅ | ✅ | ✅ | Mahal |
| **Risk jika token bocor** | Terbatas (scope) | Semua secrets terbuka | Terbatas | Terbatas | Terbatas |

---

## 9. Klarifikasi Penting: Apa yang Benar-Benar Dibatasi di Free Tier

### 9.1 Secrets Per Environment: UNLIMITED

**Jumlah secret di dalam setiap environment tidak dibatasi di semua plan, termasuk free.**

Yang dibatasi hanyalah wadah dan akses:

| Yang Dibatasi | Free Limit | Yang TIDAK Dibatasi |
|---|---|---|
| Identities (user + machine) | 5 | **Jumlah secrets per environment** ✅ |
| Projects | 3 | **Jumlah secrets per project** ✅ |
| Environments per project | 3 | **API call via SDK** ✅ |
| Integrations (sync otomatis) | 10 | **Jumlah folder/path** ✅ |

Jadi meski project Vercel kamu punya 50+ env vars, semuanya bisa masuk ke satu environment Infisical tanpa masalah.

### 9.2 SDK vs Integration: Tidak Sama

Ini sering salah kaprah:

- **Integration** = fitur sync otomatis dua arah ke platform lain (Vercel ↔ Infisical, GitHub Actions ↔ Infisical, AWS Secrets Manager ↔ Infisical). Ini dikonfigurasi di dashboard, memakan 1 slot per platform.
- **SDK / REST API** = pull secrets secara programmatic dari notebook/app. **Tidak memakan integration slot sama sekali.**

Artinya pakai SDK dari Kaggle, Colab, lokal, server sebanyak apapun — **bebas, tanpa batas, tanpa mengurangi slot apapun**.

### 9.3 Machine Identity: 1 Identity untuk Semua Platform

Machine Identity **bukan per-platform**, melainkan per-entitas yang autentikasi. Satu `CLIENT_ID` + `CLIENT_SECRET` bisa dipakai dari Kaggle dan Colab sekaligus — keduanya tetap dihitung sebagai **1 identity yang sama**.

Analogi: seperti 1 API key yang bisa dipakai dari banyak device/server — tetap 1 akun.

**Estimasi slot identity untuk solo dev:**

| Identity | Slot |
|---|---|
| Akun kamu sendiri (human) | 1 |
| Machine Identity untuk Kaggle + Colab (shared) | 1 |
| Sisa untuk project lain | 3 |
| **Total terpakai** | **2 dari 5** |

### 9.4 Free Tier Constraints (Hal yang Benar-Benar Perlu Diperhatikan)
- **5 identities total** — untuk solo dev ini cukup, tapi rencanakan dengan baik
- **3 projects** — rencanakan struktur project sejak awal
- **3 environments** — gunakan `dev`, `staging`, `prod`; tidak bisa buat environment custom di free tier
- **10 integration limit** — hanya berlaku untuk fitur sync otomatis, bukan SDK
- Tidak ada secret versioning dan point-in-time recovery di free tier

### 9.5 Catatan Teknis Python SDK
- Package name: `infisicalsdk` (bukan `infisical-sdk` atau `infisical_sdk`)
- Minimum Python 3.7+
- SDK punya built-in client-side caching (default TTL: 300 detik)
- Universal Auth token punya default TTL 7200 detik — untuk training job panjang, perlu re-auth atau konfigurasi TTL lebih panjang

### 9.6 Kelemahan vs Alternatif
- **Identity-based pricing** bisa mahal saat scaling ke tim: 10 dev + 20 machine identities = $540/bulan
- Doppler lebih baik untuk tim besar karena pricing per user, bukan per identity
- Untuk Kaggle/Colab personal: tidak ada kelemahan berarti di free tier

### 9.7 Dependency pada Internet
Setiap kali notebook start, butuh koneksi ke `app.infisical.com` untuk pull secrets. Kaggle dan Colab selalu online jadi ini bukan masalah praktis.

---

## 10. Workflow yang Direkomendasikan

```
[SEKALI SAJA - Setup Awal]
1. Buat akun Infisical Cloud (gratis)
2. Buat project "kaggle-ml"
3. Tambahkan semua secrets via dashboard
4. Buat Machine Identity dengan Universal Auth
5. Simpan CLIENT_ID & CLIENT_SECRET di Kaggle Secrets (2 nilai saja)

[SETIAP NOTEBOOK BARU]
1. Copy-paste helper function (lihat Section 6.3)
2. Set PROJECT_ID
3. Panggil load_secrets()
4. Semua secrets langsung di os.environ ✓

[SAAT ADA SECRET BARU/BERUBAH]
1. Update di Infisical dashboard saja
2. Tidak perlu ubah apapun di notebook
```

---

## 11. Quick Start Checklist

- [ ] Buat akun di https://app.infisical.com/signup
- [ ] Buat project baru, catat **Project ID** dari URL
- [ ] Tambahkan secrets yang dibutuhkan (HF_TOKEN, OPENAI_API_KEY, dll.)
- [ ] Buka **Organization Settings > Machine Identities**
- [ ] Buat Machine Identity baru → pilih **Universal Auth**
- [ ] Simpan `CLIENT_ID` dan `CLIENT_SECRET` yang muncul (tampil sekali saja!)
- [ ] Assign Machine Identity ke project, set permission: **Read Only**
- [ ] Di Kaggle: buka **Add-ons > Secrets**, tambahkan:
  - `INFISICAL_CLIENT_ID` = nilai CLIENT_ID
  - `INFISICAL_CLIENT_SECRET` = nilai CLIENT_SECRET
- [ ] Di Colab: buka ikon kunci 🔑 di sidebar, tambahkan dua secrets yang sama
- [ ] Test dengan notebook menggunakan kode di Section 6.4
- [ ] Verifikasi: `import os; print(os.environ.get("HF_TOKEN"))` → harus muncul nilainya

---

---

## 12. Loading Strategy: Kapan dan Bagaimana Load Secrets

Ini bagian yang sering diabaikan — cara load secrets yang benar sangat mempengaruhi performa dan keandalan notebook/aplikasi.

### 12.1 Prinsip Dasar: Load Sekali, Pakai Berkali-kali

**Jangan pernah call Infisical API setiap kali butuh secret.** Load semua secrets yang dibutuhkan sekali di awal, simpan di `os.environ` atau variabel Python, lalu akses dari sana.

```
❌ Anti-pattern:              ✅ Yang benar:
─────────────────             ──────────────────────────
def train():                  # Di cell pertama notebook:
  key = fetch_secret()        load_secrets(project_id=...)
  call_api(key)
                              # Di cell manapun sesudahnya:
def evaluate():               def train():
  key = fetch_secret()          key = os.environ["API_KEY"]
  call_api(key)                 call_api(key)
```

### 12.2 Strategi 1: Eager Loading (Rekomendasi untuk Notebook)

Load **semua** secrets sekaligus di cell pertama notebook, set ke `os.environ`. Strategi paling simpel dan paling cocok untuk Kaggle/Colab.

```python
# Cell 1 — jalankan sekali saat notebook start
load_secrets(project_id="YOUR_PROJECT_ID", environment="dev", path="/")

# Cell 2, 3, dst — akses langsung dari os.environ, tanpa network call
import os
openai_key = os.environ["OPENAI_API_KEY"]
hf_token   = os.environ["HF_TOKEN"]
```

**Kapan cocok:** Hampir semua use case Kaggle/Colab. Session notebook = satu lifetime, load sekali di awal sudah cukup.

**Trade-off:** Semua secrets masuk memori sekaligus — tidak masalah untuk notebook personal.

### 12.3 Strategi 2: SDK Built-in Caching (Rekomendasi untuk Script/Aplikasi)

Infisical SDK punya **client-side caching bawaan**. Saat `get_secret()` dipanggil, hasilnya di-cache di memori. Request berikutnya ke secret yang sama tidak akan hit network sampai TTL habis. Jika TTL habis dan fetch gagal (network error), SDK tetap return nilai cache sebelumnya — tidak crash.

```python
from infisical_sdk import InfisicalSDKClient

# Buat client SEKALI — ini yang memegang cache
client = InfisicalSDKClient(
    host="https://app.infisical.com",
    cache_ttl=3600  # cache 1 jam; None untuk disable; default 60 detik
)
client.auth.universal_auth.login(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

# Panggilan berikutnya ke secret yang sama → dari cache, bukan network
def get_openai_key():
    return client.secrets.get_secret(
        secret_name="OPENAI_API_KEY",
        project_id=PROJECT_ID,
        environment_slug="dev",
        secret_path="/"
    ).secret_value
```

**Kapan cocok:** Script Python yang berjalan lama, di mana secret mungkin berubah di tengah jalan dan perlu direfresh otomatis setelah TTL.

### 12.4 Strategi 3: Singleton Pattern (Codebase Multi-modul)

Untuk codebase dengan banyak file `.py` yang semua butuh akses secret, gunakan singleton agar inisialisasi client hanya terjadi sekali meski dipanggil dari berbagai tempat.

```python
# secrets_manager.py — satu file, di-import dari mana saja
import os
from functools import lru_cache

@lru_cache(maxsize=None)
def _get_client():
    """Client hanya dibuat sekali berkat lru_cache."""
    from infisical_sdk import InfisicalSDKClient
    client = InfisicalSDKClient(host="https://app.infisical.com", cache_ttl=3600)
    client.auth.universal_auth.login(
        client_id=os.environ["INFISICAL_CLIENT_ID"],
        client_secret=os.environ["INFISICAL_CLIENT_SECRET"]
    )
    return client

def get_secret(name: str, project_id: str, env: str = "dev") -> str:
    return _get_client().secrets.get_secret(
        secret_name=name,
        project_id=project_id,
        environment_slug=env,
        secret_path="/"
    ).secret_value

# Di modul/notebook manapun:
# from secrets_manager import get_secret
# key = get_secret("OPENAI_API_KEY", project_id="xxx")
```

### 12.5 Strategi 4: Lazy Loading dengan Guard (Load On-demand, Sekali)

Load hanya saat pertama kali dibutuhkan, dan skip jika sudah pernah diload di session yang sama.

```python
_loaded = False

def ensure_loaded(project_id: str, env: str = "dev"):
    """Load secrets hanya jika belum pernah diload di session ini."""
    global _loaded
    if _loaded:
        return  # No-op — tidak hit network
    load_secrets(project_id=project_id, environment=env)
    _loaded = True

# Aman dipanggil berkali-kali, hanya load sekali
ensure_loaded("YOUR_PROJECT_ID")
ensure_loaded("YOUR_PROJECT_ID")  # diabaikan
```

**Kapan cocok:** Utility script yang terkadang dijalankan tanpa butuh Infisical.

### 12.6 Handling Token Expiry untuk Long-Running Job

Universal Auth token punya default TTL **7200 detik (2 jam)**. Ada dua pendekatan:

**Pola A — Eager Loading ke `os.environ` (paling praktis untuk notebook):**

Setelah `load_secrets()` dipanggil, semua secret sudah di `os.environ` — tidak ada dependency ke Infisical token lagi sepanjang session. Token expired tidak berpengaruh apapun.

```python
# Load di awal → os.environ → tidak perlu Infisical lagi selama session
load_secrets(project_id="...", environment="dev")

# Training 6 jam? Aman — os.environ tidak expired
for epoch in range(100):
    api_key = os.environ["OPENAI_API_KEY"]  # baca dari memory, bukan Infisical
    train_step(api_key)
```

**Pola B — Lazy Re-auth (untuk script/server long-running):**

```python
def get_secret_safe(client, name, project_id, env="dev"):
    """Coba ambil secret, re-auth otomatis jika token expired."""
    try:
        return client.secrets.get_secret(
            secret_name=name,
            project_id=project_id,
            environment_slug=env,
            secret_path="/"
        ).secret_value
    except Exception as e:
        if "unauthorized" in str(e).lower():
            client.auth.universal_auth.login(
                client_id=os.environ["INFISICAL_CLIENT_ID"],
                client_secret=os.environ["INFISICAL_CLIENT_SECRET"]
            )
            return client.secrets.get_secret(
                secret_name=name,
                project_id=project_id,
                environment_slug=env,
                secret_path="/"
            ).secret_value
        raise
```

### 12.7 Ringkasan: Pilih Strategi yang Tepat

| Situasi | Strategi | Alasan |
|---|---|---|
| Kaggle/Colab notebook (semua kasus) | **Eager Loading** → `os.environ` | Paling simpel, load sekali di cell pertama |
| Training job panjang (>2 jam) | **Eager Loading** → `os.environ` | Token expiry tidak bermasalah |
| Script Python pendek | **Eager Loading** | Simpel dan andal |
| Aplikasi/server long-running | **SDK Caching** (`cache_ttl` tinggi) | Auto-refresh via TTL |
| Codebase multi-file/modul | **Singleton Pattern** (`lru_cache`) | Satu client instance, banyak caller |
| Load tidak selalu diperlukan | **Lazy + Guard** | Hindari load saat tidak perlu |

> **TL;DR untuk Kaggle/Colab:** Selalu **Eager Load ke `os.environ`** di cell pertama. Setelah itu akses semua secret dari `os.environ` — tidak ada network call lagi, tidak ada token expiry issue, tidak ada latency.

## 13. Resources

| Resource | URL |
|---|---|
| Infisical Cloud | https://app.infisical.com |
| Dokumentasi | https://infisical.com/docs |
| Python SDK docs | https://infisical.com/docs/sdks/languages/python |
| Machine Identity setup | https://infisical.com/docs/documentation/platform/identities/overview |
| Universal Auth | https://infisical.com/docs/documentation/platform/identities/universal-auth |
| API Reference | https://infisical.com/docs/api-reference/overview/introduction |
| Community Slack | https://infisical.com/slack |
| GitHub | https://github.com/Infisical/infisical |
| Status Page | https://status.infisical.com |

---

*Research dibuat: Juni 2026 | Versi pricing dan fitur dapat berubah — selalu cek https://infisical.com/pricing untuk informasi terbaru.*
