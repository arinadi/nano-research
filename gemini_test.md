# Gemini API Test

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
