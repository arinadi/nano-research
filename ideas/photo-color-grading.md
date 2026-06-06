# Research: Professional Photo Color Grading Workflow

## Workflow Order (WB → Exposure → Tone → Color → Detail)

Urutan ini mengikuti pipeline colorist profesional. Setiap tahap bergantung pada output tahap sebelumnya — white balance salah akan menghasilkan koreksi exposure yang salah, dst.

### 1. White Balance — Color Cast Detection
- **Apa yang dinilai:** Apakah gambar memiliki color cast? (warm/cool/green/magenta)
- **Mengapa pertama:** WB adalah fondasi. Semua keputusan exposure dan color setelahnya bergantung pada netralitas white point.
- **Parameter:** `w` (warmth), `t` (tint)
- **Implementasi:** Koreksi channel BGR langsung — warmth menambah Red/mengurangi Blue, tint menambah/mengurangi Green.

### 2. Exposure — Overall Brightness
- **Apa yang dinilai:** Overall terlalu gelap/terang? Ada clipped highlights? Crushed blacks?
- **Mengapa kedua:** Setelah WB netral, baru bisa menilai apakah exposure perlu adjustment global.
- **Parameter:** `b` (brightness offset)

### 3. Tonal Range — Contrast & Tone Curve
- **Apa yang dinilai:** Contrast flat atau harsh? Highlights blown? Shadows blocked?
- **Mengapa ketiga:** Setelah exposure benar, tonal range menentukan "mood" gambar.
- **Parameter:**
  - `c` (contrast) — S-curve global
  - `h` (highlights) — recovery area terang, kurva `norm²`
  - `d` (shadows) — lift area gelap, kurva `(1-norm)²`
  - `k` (blacks) — black point, kurva `(1-norm)³` (lebih curam, hanya target deepest shadows)
  - `n` (whites) — white point, kurva `norm³` (lebih curam, hanya target brightest highlights)

### 4. Color — Saturation & Vibran
- **Apa yang dinilai:** Warna dull atau oversaturated? Skin tone aman?
- **Mengapa keempat:** Setelah tonal range dikunci, baru adjust saturasi agar tidak over-edit.
- **Parameter:**
  - `s` (saturation) — multiplier global channel S di HSV
  - `v` (vibran) — non-linear, lebih agresif di area low-saturation, skin-safe

### 5. Detail — Sharpness & Texture
- **Apa yang dinilai:** Gambar soft? Noisy? Perlu texture?
- **Mengapa terakhir:** Sharpening dan clarity sebaiknya diaplikasikan setelah semua tonal/color adjustment selesai.
- **Parameter:**
  - `p` (sharpness) — unsharp mask via GaussianBlur (sigma=1.5)
  - `l` (clarity) — midtone contrast via GaussianBlur (sigma=10) × midtone mask

---

## Diferensiasi Kurva: Highlights/Shadows vs Blacks/Whites

| Parameter | Kurva | Target Tonal Range |
|---|---|---|
| Highlights (`h`) | `norm²` (quadratic) | Bright areas broadly |
| Shadows (`d`) | `(1-norm)²` (quadratic) | Dark areas broadly |
| Whites (`n`) | `norm³` (cubic) | Very brightest pixels only |
| Blacks (`k`) | `(1-norm)³` (cubic) | Very darkest pixels only |

Kurva cubic (³) lebih curam di ujung — efek minimal di midtone, kuat di ekstrim. Ini mencegah overlap berlebihan antara pasangan highlights/whites dan shadows/blacks.

---

## Vibran vs Saturation

```
Saturation = linear multiplier ke semua piksel
Vibran     = non-linear, lebih kuat di area low-saturation
```

Implementasi vibran menggunakan mask `(1 - s/255)²` — semakin rendah saturasi piksel, semakin besar efek vibran. Area yang sudah saturated tidak terpengaruh. Ini menjadikan vibran "skin-safe" karena skin tone biasanya sudah memiliki saturasi moderat.

---

## Rationale Default Values (Konservatif)

Default sebelumnya (brightness 15, contrast 1.2, saturation 1.3) terlalu agresif untuk batch processing tanpa review manual. Default baru lebih konservatif:

| Parameter | Lama | Baru | Alasan |
|---|---|---|---|
| brightness | 15 | 8 | Hindari over-exposure di gambar yang sudah terang |
| contrast | 1.2 | 1.1 | Contrast berlebih menghilangkan detail midtone |
| saturation | 1.3 | 1.05 | Saturation berlebih tidak natural, sulit di-reverse |
| highlights | -15 | -8 | Recovery agresif membuat gambar flat |
| shadows | 25 | 15 | Lift berlebih membuat gambar washed-out |
| blacks | - | -10 | Tambah depth tanpa crushing |
| whites | - | 8 | Sedikit ceiling raise untuk pop |
| clarity | 15 | 8 | Clarity berlebih membuat tekstur kasar |

---

## Token Optimization Strategy

1. **1-letter JSON keys** — `b`, `c`, `s`, `v`, `h`, `d`, `k`, `n`, `w`, `t`, `p`, `l`, `x` menggantikan nama panjang. Di-map ke nama deskriptif via `_KEY` dict di Python.
2. **Thumbnail 512px** — Gambar di-resize ke max 512px sebelum base64 encoding untuk analisis. Resolusi lebih dari cukup untuk color analysis, signifikan mengurangi token input.
3. **JPEG quality 82** — Lebih rendah dari default 95, tapi cukup untuk AI color analysis. Mengurangi ukuran base64 tanpa degradasi warna yang berarti (quality 82 adalah sweet spot untuk color accuracy vs size).
4. **Thinking disabled** — `thinking={"type": "disabled"}` mencegah MiMo membuang token untuk reasoning yang tidak menghasilkan output.
5. **max_tokens 220** — Cukup untuk JSON 12 parameter + deskripsi singkat (~15 kata).

---

## Safety Filter Mitigation

Beberapa foto (terutama portrait) ditolak MiMo safety filter sebagai "high risk". Strategi:

1. **System prompt** menegaskan konteks legitimate: "This is a legitimate professional color grading application"
2. **Keyword detection** — cek kata kunci rejection (`rejected`, `high risk`, `cannot analyze`, `i'm sorry`) sebelum JSON parsing
3. **Clean fallback** — jika terdeteksi rejection atau parsing gagal, gunakan `DEFAULT_PARAMS` tanpa error
4. **Validation** — semua key yang hilang diisi dengan nilai neutral sebelum return

---

## Range Parameter Summary

| Key | Name | Range | Neutral | Keterangan |
|---|---|---|---|---|
| `b` | brightness | -80..80 | 0 | Offset kecerahan |
| `c` | contrast | 0.6..2.0 | 1.0 | Multiplier contrast |
| `s` | saturation | 0.5..2.0 | 1.0 | Multiplier saturasi (linear) |
| `v` | vibrance | 0.8..2.0 | 1.0 | Multiplier vibrance (non-linear, skin-safe) |
| `h` | highlights | -80..20 | 0 | Recovery blown highlights |
| `d` | shadows | -20..80 | 0 | Lift dark areas |
| `k` | blacks | -60..15 | 0 | Black point (negative = deepen) |
| `n` | whites | 0..60 | 0 | White point (positive = raise ceiling) |
| `w` | warmth | -40..40 | 0 | Negative = cooler/blue, positive = warmer/yellow |
| `t` | tint | -30..30 | 0 | Negative = green, positive = magenta |
| `p` | sharpness | 0.5..2.0 | 1.0 | Multiplier unsharp mask |
| `l` | clarity | -20..60 | 0 | Midtone contrast (negative = skin soften) |
| `x` | description | text | - | Diagnosis (max 15 kata) |
