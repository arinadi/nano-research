# Gemini - List Available Models

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
