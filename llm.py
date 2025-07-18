import os
import logging
from openai import OpenAI
from google import genai
import ollama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Get API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")



# Ensure API keys are available before proceeding
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables.")
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
if not google_api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Configure the clients
openai_client = OpenAI(api_key=openai_api_key)
gemini_client = genai.Client()
ollama_client = ollama.Client()

template = "sage das abc auf"

def call_ai(template, model="openai"):
    logger.debug(f"Calling AI model '{model}' with template: {template[:100]}...")
    
    if model == "openai":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": template}]
        )
        result = response.choices[0].message.content
        logger.debug(f"AI response from '{model}': {result[:100]}...")
        return result
    
    if model == "google":
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=template
        )
        result = response.text
        logger.debug(f"AI response from '{model}': {result[:100]}...")
        return result
    

    if model == "google_json":
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=template,
            config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "Reflection": {
                            "type": "string"
                        },
                        "improved_persona": {
                            "type": "string"
                        }
                    },
                    "required": ["Reflection", "improved_persona"]
                }
            }
        )
        import json
        result = json.loads(response.text)
        logger.debug(f"AI response from '{model}': {str(result)[:100]}...")
        return result
    if model == "ollama":
        response = ollama_client.generate(
            model='phi4:latest',
            prompt=template,
            stream=False
        )
        result = response['response']
        logger.debug(f"AI response from '{model}': {result[:100]}...")
        return result
    
    logger.error(f"Invalid model specified: {model}")
    raise ValueError("Use 'openai', 'google', 'google_json' or 'ollama'")
