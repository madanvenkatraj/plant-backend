import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: No GEMINI_API_KEY found in .env")
    exit(1)

genai.configure(api_key=api_key)

print(f"Testing Gemini API with key: {api_key[:10]}...")

# 1. List available models
print("\n--- Available Models ---")
try:
    models = genai.list_models()
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name} ({m.display_name})")
except Exception as e:
    print(f"Error listing models: {e}")

# 2. Test specific models
test_models = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-pro"
]

print("\n--- Testing Content Generation ---")
for model_name in test_models:
    try:
        print(f"Testing {model_name}...", end=" ", flush=True)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello' in one word.")
        print(f"SUCCESS: {response.text.strip()}")
    except Exception as e:
        print(f"FAILED: {e}")

print("\nTest Complete.")
