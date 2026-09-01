from google import genai
from api import GEMINI_API_KEY


print("Gemini API key loaded.")
print("Connecting to Gemini...")


client = genai.Client(
    api_key=GEMINI_API_KEY
)


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply exactly with: Gemini API connection successful"
)


print()
print(response.text)