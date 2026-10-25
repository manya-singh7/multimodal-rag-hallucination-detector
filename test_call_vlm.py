"""Tests for call_vlm.py model fallback and retry behavior."""

import pytest
import call_vlm


class MockResponse:
    """Mock Gemini response containing text."""

    def __init__(self, text: str):
        self.text = text


@pytest.fixture(autouse=True)
def patch_sleep_and_api_key(monkeypatch):
    """Ensure all tests run instantly without sleep and with a valid mock API key."""
    monkeypatch.setattr(call_vlm.time, "sleep", lambda s: None)
    monkeypatch.setattr(call_vlm, "API_KEY", "test-api-key")


def test_primary_succeeds_no_fallback_used(monkeypatch):
    """When the primary model succeeds on the first attempt, no fallback is used."""
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls.append(model)
            return MockResponse("Primary answer")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(call_vlm.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.8-flash")

    result = call_vlm.call_vlm("What is multimodal RAG?")
    assert result == "Primary answer"
    assert calls == ["gemini-3.1-flash-lite"]


def test_primary_503_first_fallback_succeeds(monkeypatch, capsys):
    """When primary returns 503 every time, first fallback succeeds after MAX_ATTEMPTS on primary."""
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls.append(model)
            if model == "gemini-3.1-flash-lite":
                raise RuntimeError("503 Service Unavailable")
            return MockResponse("First fallback answer")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(call_vlm.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.8-flash")

    result = call_vlm.call_vlm("What is multimodal RAG?")
    assert result == "First fallback answer"
    # Primary model must be tried exactly MAX_ATTEMPTS times (4)
    assert calls.count("gemini-3.1-flash-lite") == call_vlm.MAX_ATTEMPTS
    # First fallback model succeeds on its first attempt
    assert calls.count("gemini-3.6-flash") == 1
    # Second fallback must not be invoked
    assert "gemini-3.8-flash" not in calls

    output = capsys.readouterr().out
    assert "Gemini model gemini-3.1-flash-lite unavailable, trying gemini-3.6-flash" in output


def test_primary_and_first_fallback_fail_second_fallback_succeeds(monkeypatch, capsys):
    """When primary and first fallback fail with retryable errors, second fallback succeeds."""
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls.append(model)
            if model == "gemini-3.1-flash-lite":
                raise RuntimeError("503 Service Unavailable")
            elif model == "gemini-3.6-flash":
                raise RuntimeError("429 Resource Exhausted")
            elif model == "gemini-3.8-flash":
                return MockResponse("Second fallback answer")
            raise RuntimeError("Unexpected model")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(call_vlm.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.8-flash")

    result = call_vlm.call_vlm("What is multimodal RAG?")
    assert result == "Second fallback answer"
    # Primary tried MAX_ATTEMPTS (4)
    assert calls.count("gemini-3.1-flash-lite") == call_vlm.MAX_ATTEMPTS
    # Fallback 1 tried FALLBACK_ATTEMPTS (2)
    assert calls.count("gemini-3.6-flash") == call_vlm.FALLBACK_ATTEMPTS
    # Fallback 2 tried 1 time and succeeded
    assert calls.count("gemini-3.8-flash") == 1

    output = capsys.readouterr().out
    assert "Gemini model gemini-3.1-flash-lite unavailable, trying gemini-3.6-flash" in output
    assert "Gemini model gemini-3.6-flash unavailable, trying gemini-3.8-flash" in output


def test_all_models_fail_raises_runtime_error(monkeypatch):
    """When all models fail with retryable errors, raises RuntimeError naming every model."""
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls.append(model)
            raise RuntimeError(f"503 Service Unavailable on {model}")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(call_vlm.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.8-flash")

    with pytest.raises(RuntimeError) as exc_info:
        call_vlm.call_vlm("What is multimodal RAG?")

    message = str(exc_info.value)
    # The message must name every model tried
    assert "gemini-3.1-flash-lite" in message
    assert "gemini-3.6-flash" in message
    assert "gemini-3.8-flash" in message
    assert "503 Service Unavailable" in message

    assert calls.count("gemini-3.1-flash-lite") == call_vlm.MAX_ATTEMPTS
    assert calls.count("gemini-3.6-flash") == call_vlm.FALLBACK_ATTEMPTS
    assert calls.count("gemini-3.8-flash") == call_vlm.FALLBACK_ATTEMPTS


def test_non_retryable_error_raises_immediately_no_fallback(monkeypatch):
    """Non-retryable errors (e.g. 400 invalid request, bad API key) raise at once with no fallback."""
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls.append(model)
            raise ValueError("400 Invalid argument: prompt exceeds token limit")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(call_vlm.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.8-flash")

    with pytest.raises(ValueError) as exc_info:
        call_vlm.call_vlm("What is multimodal RAG?")

    assert "400 Invalid argument" in str(exc_info.value)
    # Only tried once on primary model; no retry and no fallback calls
    assert calls == ["gemini-3.1-flash-lite"]


def test_api_key_error_raises_immediately_no_fallback(monkeypatch):
    """An API key validation error raises immediately without walking fallbacks."""
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls.append(model)
            raise RuntimeError("API key not valid. Please pass a valid API key.")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(call_vlm.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.8-flash")

    with pytest.raises(RuntimeError) as exc_info:
        call_vlm.call_vlm("What is multimodal RAG?")

    assert "API key not valid" in str(exc_info.value)
    assert calls == ["gemini-3.1-flash-lite"]


def test_fallback_models_parsing_and_deduplication(monkeypatch):
    """Verify GEMINI_FALLBACK_MODELS parsing, stripping, empty dropping, and de-duplication."""
    # Default fallbacks when env var not set
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    assert call_vlm._parse_fallback_models("gemini-3.1-flash-lite") == [
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    ]

    # Strips spaces and drops empty entries
    monkeypatch.setenv(
        "GEMINI_FALLBACK_MODELS",
        "  gemini-3.6-flash  , ,  gemini-3.8-flash , ",
    )
    assert call_vlm._parse_fallback_models("gemini-3.1-flash-lite") == [
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    ]

    # Drops entry equal to the primary model
    monkeypatch.setenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.1-flash-lite, gemini-3.6-flash, gemini-3.1-flash-lite",
    )
    assert call_vlm._parse_fallback_models("gemini-3.1-flash-lite") == [
        "gemini-3.6-flash",
    ]

    # De-duplicates preserving the original order
    monkeypatch.setenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.8-flash, gemini-3.6-flash, gemini-3.8-flash, gemini-3.6-flash",
    )
    assert call_vlm._parse_fallback_models("gemini-3.1-flash-lite") == [
        "gemini-3.8-flash",
        "gemini-3.6-flash",
    ]

    # Empty string yields empty list
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", " , ")
    assert call_vlm._parse_fallback_models("gemini-3.1-flash-lite") == []
