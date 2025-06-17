import os
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

load_dotenv()


# Get API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")



# Ensure API keys are available before proceeding
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Configure the clients
openai_client = OpenAI(api_key=openai_api_key)
gemini_client=genai.Client()

template = "sage das abc auf"

def call_ai(template, model="openai"):
    if model == "openai":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": template}]
        )
        return response.choices[0].message.content
    
    if model == "google":
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=template
        )
        return response.text
    raise ValueError("Use 'openai' or 'google'")
