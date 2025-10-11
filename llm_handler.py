# llm_handler.py

import os
import time

from google.genai import Client
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv

load_dotenv(".env")  

class LlmHandler:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY nicht gesetzt.")
        self.client = Client(api_key=self.api_key)
        self.model_name = model_name
        print(f"LlmHandler initialisiert mit Modell: {model_name}")

    def generate_response(self, prompt: str) -> str:
        print(f"--- Anfrage an GenAI-Client: Modell={self.model_name} ---")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    # hier kannst du Default-Parameter setzen
                    temperature=0.7,
                    # max_output_tokens, top_p usw.
                )
            )
            return response.text.strip()
        except Exception as e:
            print("Fehler bei GenAI-API-Call:", e)
            return ""


# --- Beispiel für die direkte Verwendung ---
if __name__ == "__main__":
    try:
        handler = LlmHandler()
        test_prompt = "Erkläre kurz das Konzept eines Large Language Models."
        response = handler.generate_response(test_prompt)
        
        print("\n--- Ergebnis des Testlaufs ---")
        print(response)
        print("--------------------------")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")