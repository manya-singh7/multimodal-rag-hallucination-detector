"""Gemini VLM wrapper with exponential backoff and model fallback."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

MAX_ATTEMPTS = 4
FALLBACK_ATTEMPTS = 2
INITIAL_RETRY_DELAY = 2


# Network errors that are worth retrying even when their message is empty
# (for example httpx.ReadError). Subclasses are matched by name too.
RETRYABLE_ERROR_TYPES = {
    "readerror",
    "readtimeout",
    "writetimeout",
    "connecttimeout",
    "pooltimeout",
    "connecterror",
    "remoteprotocolerror",
    "timeoutexception",
    "timeouterror",
}


def _is_retryable_error(error: Exception) -> bool:
    """Identify temporary API or network errors worth retrying."""
    error_types = {cls.__name__.lower() for cls in type(error).__mro__}

    if error_types & RETRYABLE_ERROR_TYPES:
        return True

    message = str(error).lower()

    retryable_indicators = (
        "503",
        "unavailable",
        "high demand",
        "429",
        "resource_exhausted",
        "500",
        "502",
        "504",
        "internal server error",
        "deadline exceeded",
        "timed out",
        "timeout",
        "connection reset",
        "connection error",
        "temporarily unavailable",
    )

    return any(indicator in message for indicator in retryable_indicators)


def _parse_fallback_models(primary_model: str | None = None) -> list[str]:
    """Parse comma-separated GEMINI_FALLBACK_MODELS, dropping empty, duplicates, and primary."""
    if primary_model is None:
        primary_model = os.getenv("GEMINI_MODEL", MODEL_ID).strip()
    raw = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.8-flash")
    fallbacks = []
    seen = {primary_model}
    for item in raw.split(","):
        name = item.strip()
        if name and name not in seen:
            seen.add(name)
            fallbacks.append(name)
    return fallbacks


def call_vlm(prompt: str, images: list = None) -> str:
    """Generate a response using Gemini, retrying temporary failures and falling back to backup models."""

    if not API_KEY or API_KEY == "put_your_api_key_here":
        raise RuntimeError(
            "Set GEMINI_API_KEY in your project's .env file."
        )

    client = genai.Client(api_key=API_KEY)
    contents = [prompt]

    # Load available images for multimodal requests.
    for image_path in images or []:
        path = Path(image_path)

        if not path.is_file():
            continue

        try:
            with Image.open(path) as image:
                contents.append(image.convert("RGB").copy())
        except Exception as exc:
            raise RuntimeError(
                f"Could not load image '{path.name}': {exc}"
            ) from exc

    primary_model = os.getenv("GEMINI_MODEL", MODEL_ID).strip()
    fallback_models = _parse_fallback_models(primary_model)
    models_to_try = [primary_model] + fallback_models

    models_tried = []
    last_error = None

    for model_index, model in enumerate(models_to_try):
        is_primary = (model_index == 0)
        max_attempts = MAX_ATTEMPTS if is_primary else FALLBACK_ATTEMPTS
        models_tried.append(model)

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config={"temperature": 0},  # repeatable answers and verdicts
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty text response."
                    )

                return response.text.strip()

            except Exception as exc:
                should_retry = _is_retryable_error(exc)
                last_error = exc

                if not should_retry:
                    raise

                final_attempt = (attempt == max_attempts)

                if not final_attempt:
                    delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))

                    print(
                        f"[RETRY] Gemini request failed "
                        f"({attempt}/{max_attempts}): {exc}"
                    )
                    print(f"[RETRY] Trying again in {delay} seconds...")

                    time.sleep(delay)
                else:
                    if model_index + 1 < len(models_to_try):
                        next_model = models_to_try[model_index + 1]
                        print(
                            f"Gemini model {model} unavailable, trying {next_model}"
                        )
                    break

    tried_str = ", ".join(models_tried)
    raise RuntimeError(
        f"Gemini remained unavailable after trying models: {tried_str}. "
        f"Please try again later. "
        f"Last error: {last_error}"
    ) from last_error
