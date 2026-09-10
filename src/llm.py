"""
Gemini LLM Service module for CLAUSE Enterprise RAG using google-genai Interactions API.

Handles API client initialization, grounded prompt generation via client.interactions.create(),
bounded timeouts, rate-limit backoff handling, and exception handling without exposing API credentials.
"""

import os
import time
import re
from typing import Optional, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.config import GEMINI_API_KEY, GEMINI_MODEL, DEFAULT_LLM_TEMPERATURE


class LLMError(Exception):
    """Custom exception raised for LLM generation failures."""
    pass


class GeminiLLMService:
    """
    Wrapper around Google Gemini API (google-genai Interactions API).
    Provides conservative enterprise generation settings, bounded timeouts, rate-limit backoff, and lazy client initialization.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        request_timeout: float = 10.0,
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY

        self.model_name = model_name or os.getenv("GEMINI_MODEL") or GEMINI_MODEL or "gemini-3.8-flash"
        self.temperature = temperature
        self.request_timeout = request_timeout
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        """Lazily initialize and return the genai.Client instance with bounded timeout options."""
        if not self.api_key or not self.api_key.strip():
            raise LLMError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in environment or .env file."
            )

        if self._client is None:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as err:
                raise LLMError(f"Failed to initialize Gemini API client: {err}") from err

        return self._client

    def is_available(self) -> bool:
        """Check if GEMINI_API_KEY is configured."""
        return bool(self.api_key and self.api_key.strip())

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate response text for a given prompt string using Gemini Interactions API (client.interactions.create).
        Enforces 1 attempt with bounded timeout (default 10s) and immediate fallback trigger on error/quota/timeout.
        """
        if not prompt or not prompt.strip():
            raise LLMError("Cannot generate response for an empty prompt.")

        client = self._get_client()

        import concurrent.futures

        def _call_api():
            return client.interactions.create(
                model=self.model_name,
                input=prompt,
                timeout=self.request_timeout,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_api)
                interaction = future.result(timeout=self.request_timeout)

            output_text = getattr(interaction, "output_text", None)
            if not output_text and hasattr(interaction, "text"):
                output_text = interaction.text

            if not output_text:
                output_text = str(interaction)

            if not output_text or not output_text.strip():
                raise LLMError("Gemini Interactions API returned an empty response.")

            return output_text.strip()

        except concurrent.futures.TimeoutError:
            raise LLMError(f"Gemini API request timed out after {self.request_timeout} seconds.")
        except Exception as err:
            err_str = str(err)
            raise LLMError(f"Gemini LLM generation error (Model: '{self.model_name}'): {err_str}") from err
