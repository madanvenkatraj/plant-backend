import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

try:
    print("Testing gemini-2.0-flash with new SDK...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say 'Hello' in one word."
    )
    print(f"SUCCESS: {response.text}")
except Exception as e:
    print(f"FAILED: {e}")
