
import json
import re

from call_vlm import call_vlm


ALLOWED_VERDICTS = {
    "Supported",
    "Contradicted",
    "Insufficient Evidence",
}


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

            parts = re.split(
                r"(?<=[.!?])\s+(?=[A-Z])",
                text,
            )
            claims.extend(
                part.strip()
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


def verify_claims(answer: str, text_chunks: list[dict]) -> dict:
    """Verify answer claims against retrieved text using the configured model."""

    claims = split_claims(answer)

    if not claims:
        return {
            "claims": [],
            "trust_score": 0,
            "total_claims": 0,
            "supported_claims": 0,
        }

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
        return {
            "claims": [
                {
                    "claim": claim,
                    "verdict": "Insufficient Evidence",
                    "evidence": [],
                    "explanation": "No retrieved evidence was provided.",
                }
                for claim in claims
            ],
            "trust_score": 0,
            "total_claims": len(claims),
            "supported_claims": 0,
        }

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
        response = call_vlm(prompt)

        if isinstance(response, dict):
            raw = response.get("text", "")
        else:
            raw = str(response or "")

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)

        if not match:
            raise ValueError("No JSON object found in model response.")

        parsed = json.loads(match.group(0))
        model_results = parsed.get("claims", [])

        if not isinstance(model_results, list):
            raise ValueError("The claims field must be a list.")

    except Exception as exc:
        return {
            "claims": [
                {
                    "claim": claim,
                    "verdict": "Insufficient Evidence",
                    "evidence": [],
                    "explanation": (
                        "Verification failed: "
                        + type(exc).__name__
                    ),
                }
                for claim in claims
            ],
            "trust_score": 0,
            "total_claims": len(claims),
            "supported_claims": 0,
        }

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
                "explanation": str(
                    item.get("explanation", "")
                    or "No explanation provided."
                ),
            }
        )

    supported = sum(
        item["verdict"] == "Supported"
        for item in verified
    )

    # Percentage of claims labeled Supported; not factual accuracy probability.
    trust_score = round(100 * supported / len(claims))

    return {
        "claims": verified,
        "trust_score": trust_score,
        "total_claims": len(claims),
        "supported_claims": supported,
    }
