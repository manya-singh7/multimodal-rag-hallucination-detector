
import json
import re
from pathlib import Path

from call_vlm import call_vlm


ALLOWED_VERDICTS = {
    "Supported",
    "Contradicted",
    "Insufficient Evidence",
}

# Cap on images sent to the model per answer (keeps latency and cost bounded).
MAX_IMAGES = 3

# Abbreviations whose trailing period must not end a sentence. Matching is
# case-sensitive so units such as "ms." or the word "no." are not affected.
_ABBREVIATIONS = (
    "et al", "e.g", "i.e", "cf", "vs", "approx",
    "Fig", "Figs", "Eq", "Eqs", "Ref", "Refs",
    "Dr", "Mr", "Mrs", "Ms", "Prof",
)
_ABBREVIATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(item) for item in _ABBREVIATIONS) + r")\."
)
_PROTECTED_DOT = ""


def _protect_abbreviations(text: str) -> str:
    return _ABBREVIATION_RE.sub(lambda m: m.group(1) + _PROTECTED_DOT, text)


def _restore_abbreviations(text: str) -> str:
    return text.replace(_PROTECTED_DOT, ".")


def split_claims(answer: str) -> list[str]:
    """Split an answer into claims without corrupting numbers or currency."""
    if not answer or not answer.strip():
        return []

    # Remove the appended supporting-text note.
    answer = re.split(
        r"(?im)^\s*(?:possible\s+)?supporting\s+text\b",
        answer,
        maxsplit=1,
    )[0].strip()

    # Remove Markdown formatting while preserving the enclosed text.
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    answer = re.sub(r"__(.*?)__", r"\1", answer)
    answer = re.sub(r"`([^`]*)`", r"\1", answer)

    claims = []
    current = []

    def flush():
        nonlocal current

        if not current:
            return

        heading = current[0].strip()
        bullets = [
            item.strip().rstrip(";")
            for item in current[1:]
            if item.strip()
        ]

        if bullets:
            claim = heading.rstrip(":") + ": " + "; ".join(bullets)
        else:
            claim = heading.rstrip(":")

        claim = re.sub(r"\s+", " ", claim).strip()

        if re.search(r"[A-Za-z]{2,}", claim):
            claims.append(claim)

        current = []

    for raw in answer.splitlines():
        if not raw.strip():
            continue

        indent = len(raw) - len(raw.lstrip())
        text = raw.strip()

        if re.fullmatch(r"\d+[.)]", text):
            continue

        numbered = bool(re.match(r"^\d+[.)]\s+", text))
        bullet = bool(re.match(r"^[-*+]\s+", text))

        if numbered:
            flush()
            text = re.sub(r"^\d+[.)]\s+", "", text).strip()
            current = [text]

        elif bullet:
            text = re.sub(r"^[-*+]\s+", "", text).strip()

            if current:
                current.append(text)
            else:
                current = [text]

        elif current and indent > 0:
            current.append(text)

        else:
            flush()

            # Abbreviations such as "et al.", "Fig." and "e.g." are protected
            # so a citation does not break a sentence into fragments.
            parts = re.split(
                r"(?<=[.!?])\s+(?=[A-Z])",
                _protect_abbreviations(text),
            )
            claims.extend(
                _restore_abbreviations(part).strip()
                for part in parts
                if part.strip()
            )

    flush()

    # Deduplicate while preserving order.
    result = []
    seen = set()

    for claim in claims:
        key = re.sub(r"\s+", " ", claim).casefold()

        if key and key not in seen:
            seen.add(key)
            result.append(claim)

    return result


def _normalize_quote(text: str) -> str:
    """Normalize whitespace and quote characters only."""
    text = str(text or "").casefold()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def _quote_exists(quote: str, evidence_texts: list[str]) -> bool:
    """Check whether a quoted passage occurs in retrieved evidence."""
    quote = _normalize_quote(quote)

    if not quote:
        return False

    return any(
        quote in _normalize_quote(text)
        for text in evidence_texts
        if text
    )


def _build_result(verified: list[dict]) -> dict:
    """Assemble the final result in the agreed output schema."""
    total = len(verified)
    supported = sum(item["verdict"] == "Supported" for item in verified)
    contradicted = sum(item["verdict"] == "Contradicted" for item in verified)

    # Percentage of claims labeled Supported; not factual accuracy probability.
    trust_score = round(100 * supported / total) if total else 0

    return {
        "claims": verified,
        "trust_score": trust_score,
        "total_claims": total,
        "supported_claims": supported,
        "n_contradicted": contradicted,
    }


def _insufficient(claims: list[str], explanation: str) -> list[dict]:
    return [
        {
            "claim": claim,
            "verdict": "Insufficient Evidence",
            "evidence": [],
            "modality": "none",
            "explanation": explanation,
        }
        for claim in claims
    ]


def _parse_json_claims(response) -> list:
    """Pull the "claims" list out of a model reply that should be JSON."""
    if isinstance(response, dict):
        raw = response.get("text", "")
    else:
        raw = str(response or "")

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model response.")

    parsed = json.loads(match.group(0))
    results = parsed.get("claims", [])

    if not isinstance(results, list):
        raise ValueError("The claims field must be a list.")

    return results


def _verify_text(
    claims: list[str],
    text_chunks: list[dict],
) -> list[dict]:
    """Verify claims against retrieved text only (one model call)."""

    evidence_texts = []
    evidence_blocks = []

    for index, chunk in enumerate(text_chunks or [], start=1):
        if isinstance(chunk, dict):
            text = str(chunk.get("text", "") or "").strip()
            source = (
                chunk.get("source")
                or chunk.get("source_doc")
                or f"Evidence {index}"
            )
            page = chunk.get("page", "")
        else:
            text = str(chunk).strip()
            source = f"Evidence {index}"
            page = ""

        if not text:
            continue

        evidence_texts.append(text)
        evidence_blocks.append(
            f"[Evidence {len(evidence_blocks) + 1}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Text: {text}"
        )

    if not evidence_blocks:
        return _insufficient(claims, "No retrieved evidence was provided.")

    prompt = f"""
You are an evidence-grounded claim verification system.

Verify every claim using ONLY the retrieved evidence.

Rules:
1. Evaluate the complete claim, not just similar wording.
2. Preserve and check all numbers, currency values, dates, names,
   and qualifiers.
3. Supported means the evidence directly supports the entire claim.
4. Contradicted means the evidence directly conflicts with the claim.
5. Insufficient Evidence means neither conclusion is established.
6. Never infer support from partial text overlap alone.
7. Return exact quotations copied from retrieved evidence.
8. If no relevant quotation exists, return an empty evidence list.
9. Return one result per claim, in the same order.
10. Return valid JSON only, without Markdown fences.

Claims:
{json.dumps(claims, ensure_ascii=False, indent=2)}

Retrieved evidence:
{chr(10).join(evidence_blocks)}

Required JSON:
{{
  "claims": [
    {{
      "claim": "exact claim text",
      "verdict": "Supported",
      "evidence": ["exact quotation"],
      "explanation": "brief explanation"
    }}
  ]
}}
"""

    try:
        model_results = _parse_json_claims(call_vlm(prompt))

    except Exception as exc:
        return _insufficient(
            claims,
            "Verification failed: " + type(exc).__name__,
        )

    verified = []

    for index, claim in enumerate(claims):
        item = (
            model_results[index]
            if index < len(model_results)
            and isinstance(model_results[index], dict)
            else {}
        )

        verdict = item.get("verdict", "Insufficient Evidence")

        if verdict not in ALLOWED_VERDICTS:
            verdict = "Insufficient Evidence"

        raw_quotes = item.get("evidence", [])

        if isinstance(raw_quotes, str):
            raw_quotes = [raw_quotes]

        if not isinstance(raw_quotes, list):
            raw_quotes = []

        valid_quotes = [
            quote.strip()
            for quote in raw_quotes
            if isinstance(quote, str)
            and _quote_exists(quote, evidence_texts)
        ]

        if verdict == "Supported" and not valid_quotes:
            verdict = "Insufficient Evidence"

        verified.append(
            {
                "claim": claim,
                "verdict": verdict,
                "evidence": valid_quotes,
                "modality": (
                    "text"
                    if verdict in ("Supported", "Contradicted")
                    else "none"
                ),
                "explanation": str(
                    item.get("explanation", "")
                    or "No explanation provided."
                ),
            }
        )

    return verified


# ---------------------------------------------------------------------------
# Image (pixel) verification
# ---------------------------------------------------------------------------

def _prepare_images(images: list) -> list[dict]:
    """Keep up to MAX_IMAGES images whose files actually exist.

    Accepts the agreed {path, caption, source_doc} dicts (captions are never
    read) or plain path strings. Missing files are skipped so the model is
    never asked to judge an image it cannot see.
    """
    prepared = []

    for image in images or []:
        if isinstance(image, dict):
            path = image.get("path") or image.get("image_path")
            source = image.get("source_doc") or image.get("source") or ""
        else:
            path = image
            source = ""

        if not path or not Path(str(path)).is_file():
            continue

        prepared.append(
            {
                "path": str(path),
                "source": str(source) or Path(str(path)).name,
            }
        )

        if len(prepared) >= MAX_IMAGES:
            break

    return prepared


def _image_prompt(claims: list[str]) -> str:
    """Prompt that checks claims against what an image actually shows.

    The caption is deliberately NOT included: the image is judged on its
    pixels, so a wrong caption cannot leak into the verdict.
    """
    return f"""
You are checking claims against ONE image.

Look carefully at the image. Judge only what the image actually shows.
Ignore any caption or surrounding text you might expect to accompany it,
and do not use outside knowledge.

For every claim return one verdict:
- "Supported": the image clearly shows what the claim says.
- "Contradicted": the image clearly shows the OPPOSITE of what the claim
  says about this image (for example the claim says a value rises but the
  image shows it falling).
- "Not Shown": the image is unrelated to the claim, or does not show enough
  to decide. Use this whenever the claim is not about the visual content.

Rules:
1. Check trends, directions, counts, labels, and numbers against the image.
2. Return one result per claim, in the same order.
3. In "description", write one short sentence about what you actually see
   that is relevant to the claim.
4. Return valid JSON only, without Markdown fences.

Claims:
{json.dumps(claims, ensure_ascii=False, indent=2)}

Required JSON:
{{
  "claims": [
    {{
      "verdict": "Supported",
      "description": "what the image shows"
    }}
  ]
}}
"""


def _normalize_image_verdict(value) -> str:
    text = str(value or "").strip().casefold().replace("_", " ")

    if text.startswith("contradict"):
        return "Contradicted"

    if text == "supported" or text.startswith("support"):
        return "Supported"

    return "Not Shown"


def _check_image(claims: list[str], image: dict):
    """Check every claim against one image (one model call).

    Returns a list with one {"verdict", "description"} per claim, or None if
    the call or its reply failed, so a bad image never breaks the text result.
    """
    try:
        results = _parse_json_claims(
            call_vlm(_image_prompt(claims), images=[image["path"]])
        )
    except Exception:
        return None

    checked = []

    for index in range(len(claims)):
        item = (
            results[index]
            if index < len(results) and isinstance(results[index], dict)
            else {}
        )
        checked.append(
            {
                "verdict": _normalize_image_verdict(item.get("verdict")),
                "description": str(item.get("description", "") or "").strip(),
            }
        )

    return checked


def _merge_with_images(text_item: dict, findings: list[tuple]) -> dict:
    """Combine a text verdict with image findings for one claim.

    findings: (label, verdict, description) per checked image.
    Contradiction override: if any image shows the opposite of the claim, the
    verdict is Contradicted even when the retrieved text supports the claim.
    """
    text_verdict = text_item["verdict"]
    contradicting = [f for f in findings if f[1] == "Contradicted"]
    supporting = [f for f in findings if f[1] == "Supported"]

    def describe(finding):
        label, _, description = finding
        return f"{label}: {description}" if description else label

    if contradicting:
        explanation = (
            f"{contradicting[0][0]} contradicts this claim"
            + (f": {contradicting[0][2]}" if contradicting[0][2] else ".")
        )

        if text_verdict == "Supported":
            explanation += (
                " The retrieved text supports the claim, "
                "but the image shows the opposite."
            )

        return {
            "claim": text_item["claim"],
            "verdict": "Contradicted",
            "evidence": text_item["evidence"]
            + [describe(f) for f in contradicting],
            "modality": (
                "text+image" if text_verdict == "Contradicted" else "image"
            ),
            "explanation": explanation,
        }

    if text_verdict == "Contradicted":
        return text_item

    if supporting:
        evidence = text_item["evidence"] + [describe(f) for f in supporting]

        if text_verdict == "Supported":
            return {
                **text_item,
                "evidence": evidence,
                "modality": "text+image",
                "explanation": text_item["explanation"]
                + " The image also supports this claim.",
            }

        return {
            "claim": text_item["claim"],
            "verdict": "Supported",
            "evidence": evidence,
            "modality": "image",
            "explanation": (
                f"{supporting[0][0]} supports this claim"
                + (f": {supporting[0][2]}" if supporting[0][2] else ".")
            ),
        }

    return text_item


def verify_claims(
    answer: str,
    text_chunks: list[dict],
    images: list[dict] = None,
) -> dict:
    """Verify answer claims against retrieved text and, optionally, images.

    Text is checked first. If images are given, each (up to MAX_IMAGES) is then
    checked on its pixels only, and an image that shows the opposite of a claim
    overrides the text verdict with "Contradicted".
    """

    claims = split_claims(answer)

    if not claims:
        return _build_result([])

    text_items = _verify_text(claims, text_chunks)

    findings = [[] for _ in claims]

    for number, image in enumerate(_prepare_images(images), start=1):
        checked = _check_image(claims, image)

        if checked is None:
            continue

        label = f"Image {number} ({image['source']})"

        for index, result in enumerate(checked):
            findings[index].append(
                (label, result["verdict"], result["description"])
            )

    verified = [
        _merge_with_images(item, findings[index])
        for index, item in enumerate(text_items)
    ]

    return _build_result(verified)
