
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


def call_vlm(prompt: str, images: list = None) -> str:
    """Generate a response using Gemini, retrying temporary failures."""

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

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
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
            final_attempt = attempt == MAX_ATTEMPTS

            if not should_retry or final_attempt:
                if should_retry:
                    raise RuntimeError(
                        f"Gemini remained unavailable after "
                        f"{MAX_ATTEMPTS} attempts. "
                        f"Model: {MODEL_ID}. "
                        f"Please try again later. "
                        f"Last error: {exc}"
                    ) from exc

                raise

            delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))

            print(
                f"[RETRY] Gemini request failed "
                f"({attempt}/{MAX_ATTEMPTS}): {exc}"
            )
            print(f"[RETRY] Trying again in {delay} seconds...")

            time.sleep(delay)

    raise RuntimeError("Gemini request failed unexpectedly.")
