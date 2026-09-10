"""
Unit tests for src/llm.py using google-genai Interactions API.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.llm import GeminiLLMService, LLMError


def test_llm_service_initialization_without_key():
    service = GeminiLLMService(api_key="")
    assert service.is_available() is False

    with pytest.raises(LLMError, match="GEMINI_API_KEY is not configured"):
        service.generate("Test prompt")


def test_llm_service_generate_empty_prompt():
    service = GeminiLLMService(api_key="mock_key")
    with pytest.raises(LLMError, match="empty prompt"):
        service.generate("   ")


def test_llm_service_mocked_interactions_generation():
    service = GeminiLLMService(api_key="mock_key", model_name="gemini-3.8-flash")

    mock_client = MagicMock()
    mock_interaction = MagicMock()
    mock_interaction.output_text = "This is a mocked Gemini answer. [S1]"
    mock_client.interactions.create.return_value = mock_interaction

    with patch.object(service, "_get_client", return_value=mock_client):
        res = service.generate("What is the remote work policy?")
        assert res == "This is a mocked Gemini answer. [S1]"
        mock_client.interactions.create.assert_called_once()
