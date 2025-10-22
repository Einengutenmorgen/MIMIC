# llm_handler.py
import os
import yaml
from google.generativeai import Client as GoogleClient
from google.generativeai.types import GenerateContentConfig
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  

# Load API keys securely from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LlmHandler:
    """
    Manages calls to different LLM providers (Google Gemini, OpenAI GPT)
    based on the specified model name.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = None
        self.provider = None

        if model_name.startswith("gemini-"):
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY environment variable not set for Gemini models.")
            self.client = GoogleClient(api_key=GOOGLE_API_KEY)
            self.provider = "google"
            print(f"LlmHandler initialized for Google Gemini model: {model_name}")
        elif model_name.startswith("gpt-"):
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable not set for GPT models.")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            print(f"LlmHandler initialized for OpenAI GPT model: {model_name}")
        else:
            raise ValueError(f"Unsupported model provider for model name: {model_name}")

    def generate_response(self, prompt: str) -> str:
        """Sends a prompt to the configured LLM and returns the response."""
        print(f"--- Sending request to {self.provider} model: {self.model_name} ---")
        # print("Prompt snippet:", prompt[:100].replace('\n', ' ')) # Optional: Log prompt snippet

        try:
            if self.provider == "google":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    # config=GenerateContentConfig(temperature=0.7) # Optional: Configure generation
                )
                # Add check for blocked content
                if not response.parts:
                     print("Warning: Google API returned no parts (potentially blocked).")
                     # Check for feedback if available (structure might vary slightly)
                     if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                         print(f"Block Reason: {response.prompt_feedback.block_reason}")
                     return ""
                return response.text.strip()

            elif self.provider == "openai":
                # Ensure prompt is passed correctly for chat completions
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # temperature=0.7 # Optional: Configure generation
                )
                if response.choices:
                    return response.choices[0].message.content.strip()
                else:
                    print("Warning: OpenAI API returned no choices.")
                    return "" # Return empty string if no choices available


        except Exception as e:
            print(f"Error during API call to {self.provider} ({self.model_name}): {e}")
            return "" # Return empty string on error

# --- Example Usage (Optional) ---
# if __name__ == "__main__":
#     try:
#         # Test Gemini (requires GOOGLE_API_KEY set)
#         gemini_handler = LlmHandler(model_name="gemini-1.5-flash")
#         gemini_response = gemini_handler.generate_response("Explain quantum physics simply.")
#         print("\nGemini Response:", gemini_response)

#         # Test OpenAI (requires OPENAI_API_KEY set)
#         gpt_handler = LlmHandler(model_name="gpt-4o-mini") # Example mini model
#         gpt_response = gpt_handler.generate_response("Explain quantum physics simply.")
#         print("\nGPT Response:", gpt_response)

#     except ValueError as e:
#         print(e)