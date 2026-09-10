"""
Isolated Gemini Interactions API Connection Test Script for CLAUSE Enterprise RAG.

Tests direct connection and authentication using client.interactions.create()
without any RAG, FAISS, fallback, or mock layers.
"""

import os
from dotenv import load_dotenv
import google.genai
from google import genai
from google.genai.errors import APIError

load_dotenv()


def main():
    sdk_version = getattr(google.genai, "__version__", "unknown")
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    print(f"SDK Version: {sdk_version}")
    print(f"Model Name: {model_name}")

    if not api_key:
        print("API Status: FAILED - GEMINI_API_KEY environment variable is absent or empty.")
        return

    try:
        client = genai.Client(api_key=api_key)
        
        # Test using the Interactions API endpoint: client.interactions.create()
        interaction = client.interactions.create(
            model=model_name,
            input="Reply with exactly: CLAUSE_CONNECTION_OK",
        )
        output_text = getattr(interaction, "output_text", None) or str(interaction)
        print("API Status: SUCCESS")
        print(f"Returned Text: {output_text.strip()}")
    except APIError as api_err:
        err_msg = str(api_err)
        if api_key and api_key in err_msg:
            err_msg = err_msg.replace(api_key, "[REDACTED_API_KEY]")
        print(f"API Status: FAILED (APIError)")
        print(f"Sanitized Error: {err_msg}")
    except Exception as err:
        err_msg = str(err)
        if api_key and api_key in err_msg:
            err_msg = err_msg.replace(api_key, "[REDACTED_API_KEY]")
        print(f"API Status: FAILED ({type(err).__name__})")
        print(f"Sanitized Error: {err_msg}")


if __name__ == "__main__":
    main()
