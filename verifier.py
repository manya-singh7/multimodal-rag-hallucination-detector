
import json
import re
from pathlib import Path

from call_vlm import call_vlm


ALLOWED_VERDICTS = {
    "Supported",
    "Contradicted",
    "Insufficient Evidence",
}

# Verdict for a claim whose check itself failed (API error, unreadable reply).
# It is NOT something the model may answer. Such claims are left out of the
# trust score and counted in "failed_claims".
FAILED_VERDICT = "Verification Failed"

# Cap on images sent to the model per answer (keeps latency and cost bounded).
# Images are checked in the order given. demo_ui.py passes them in retrieval
# rank order, so the best-ranked images are the ones that are kept.
MAX_IMAGES = 2

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

# Source labels from the generator's prompt that models often echo back,
# for example "[Text 1 | source: paper1.pdf]", "(Image 2)" or "(Source: x)".
_CITATION_MARKER_RES = (
    re.compile(
        r"\s*[\[(]\s*(?:text|evidence|image|source)s?\s*\d+[^\])]*[\])]",
        re.IGNORECASE,
    ),
    re.compile(r"\s*\(\s*sources?\s*:[^)]*\)", re.IGNORECASE),
    re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]"),
)

# "Answer:", "Summary:" and similar label prefixes are not part of a claim.
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+)?(?:answer|summary|response|conclusion)\s*:\s*",
    re.IGNORECASE,
)

# Sentence boundary: ., ! or ? (optionally followed by a closing quote or
# bracket), whitespace, then a capital letter (optionally after an opening
# quote or bracket).
_SENTENCE_BOUNDARY_RE = re.compile(
    "(?:(?<=[.!?])|(?<=[.!?][\"”’)\\]]))\\s+(?=[\"“‘(\\[]?[A-Z])"
)

# ---------------------------------------------------------------------------
# Meta-sentences: statements about the evidence itself, not about the topic.
# ("The evidence is insufficient.", "I don't know.") They are not claims, so
# they are not verified or scored.
# ---------------------------------------------------------------------------
_EVIDENCE_NOUN = (
    r"(?:(?:retrieved|provided|given|available|supplied)\s+)?"
    r"(?:evidence|context|sources?|documents?|passages?|chunks?|excerpts?)"
    r"(?:\s+(?:provided|given|retrieved|available|supplied))?"
    r"|(?:retrieved|provided|given|available|supplied)\s+"
    r"(?:text|information|material|images?|figures?|charts?|diagrams?)"
    r"|(?:text|information)\s+(?:provided|given|retrieved|available)"
    r"|(?:text|texts)\s+and\s+(?:the\s+)?(?:images?|figures?)"
)
_META_SUBJECT = (
    r"^(?:based\s+on\s+[^,]{1,60},\s*)?(?:the\s+|these\s+|this\s+)?"
    r"(?:" + _EVIDENCE_NOUN + r")"
)
_META_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        # "The evidence is insufficient (to answer ...)."
        _META_SUBJECT
        + r"\s+(?:is|are|was|were|seems?|appears?)\s+(?:\w+\s+)?"
        r"(?:insufficient|inadequate|incomplete|contradictory|conflicting|"
        r"inconsistent|unclear|ambiguous|irrelevant|vague|"
        r"not\s+(?:sufficient|enough|adequate|clear|relevant))\b",
        # "The context does not mention/contain/specify ..."
        _META_SUBJECT
        + r"\s+(?:does|do|did)\s+not\s+(?:contain|include|mention|provide|"
        r"specify|state|say|discuss|address|cover|offer|give|describe|"
        r"list|name|identify|define|explain|report|indicate|clarify|"
        r"detail|reveal|show)\b",
        # "There is no information about ..." / "Not enough evidence ..."
        r"^(?:there\s+(?:is|are)\s+)?(?:no|not\s+enough|insufficient)\s+"
        r"(?:relevant\s+|sufficient\s+)?(?:information|evidence|context|data)\b",
        # "I don't know." / "I cannot answer ..." / "I am unable to ..."
        r"^i\s+(?:do\s+not|don't|cannot|can't|could\s+not|couldn't|"
        r"am\s+unable\s+to|am\s+not\s+able\s+to)\b",
        # "... cannot be determined from the provided context."
        r"\b(?:cannot|can't|could\s+not|not)\s+be\s+"
        r"(?:found|determined|answered|concluded|inferred)\s+"
        r"(?:in|from|using|based\s+on)\s+(?:the\s+)?(?:" + _EVIDENCE_NOUN + r")",
    )
)
_META_LEAD_RE = re.compile(
    r"^(?:unfortunately|sorry|however|note\s+that|please\s+note\s+that|"
    r"overall|in\s+summary),?\s+"
)
# A sentence that goes on to say something real ("... but the method improves
# recall") is kept, so the real part still gets verified.
_CONTRAST_RE = re.compile(
    r"\b(?:but|however|although|though|yet|whereas|while)\b"
)


def _is_meta_claim(claim: str) -> bool:
    """True for a sentence that only comments on the evidence or refuses."""
    text = claim.strip().strip("\"'“”‘’").lower()
    text = _META_LEAD_RE.sub("", text)

    if _CONTRAST_RE.search(text):
        return False

    return any(pattern.search(text) for pattern in _META_PATTERNS)


def _protect_abbreviations(text: str) -> str:
    return _ABBREVIATION_RE.sub(lambda m: m.group(1) + _PROTECTED_DOT, text)


def _restore_abbreviations(text: str) -> str:
    return text.replace(_PROTECTED_DOT, ".")


def split_claims(answer: str) -> list[str]:
    """Split an answer into claims without corrupting numbers or currency.

    Not claims, and so dropped: source labels the model echoed back, "Answer:"
    style prefixes, intro lines that end with a colon and have nothing under
    them, one-word fragments, and meta-sentences about the evidence itself.
    """
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

    # Remove echoed source labels such as "[Text 1 | source: paper1.pdf]".
    for marker_re in _CITATION_MARKER_RES:
        answer = marker_re.sub("", answer)

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

        current = []

        # An intro such as "The key points are:" with nothing under it.
        if not bullets and heading.endswith(":"):
            return

        if bullets:
            claim = heading.rstrip(":") + ": " + "; ".join(bullets)
        else:
            claim = heading.rstrip(":")

        claim = re.sub(r"\s+", " ", claim).strip()

        if re.search(r"[A-Za-z]{2,}", claim):
            claims.append(claim)

    for raw in answer.splitlines():
        if not raw.strip():
            continue

        indent = len(raw) - len(raw.lstrip())
        text = _LABEL_PREFIX_RE.sub("", raw.strip()).strip()

        if not text:
            continue

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

        elif text.endswith(":"):
            # A heading such as "Key results:" opens a list. Bullets under it
            # are joined to it; with nothing under it, flush() drops it.
            flush()
            current = [text]

        else:
            flush()

            # Abbreviations such as "et al.", "Fig." and "e.g." are protected
            # so a citation does not break a sentence into fragments.
            parts = _SENTENCE_BOUNDARY_RE.split(_protect_abbreviations(text))
            claims.extend(
                _restore_abbreviations(part).strip()
                for part in parts
                if part.strip()
            )

    flush()

    # Drop fragments and meta-sentences, then deduplicate in order.
    result = []
    seen = set()

    for claim in claims:
        claim = claim.strip()

        if len(claim.split()) < 2 or _is_meta_claim(claim):
            continue

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
    """Assemble the final result in the agreed output schema.

    Claims whose check failed are not scored: the trust score is the share of
    Supported claims among the claims that were actually checked.
    """
    total = len(verified)
    supported = sum(item["verdict"] == "Supported" for item in verified)
    contradicted = sum(item["verdict"] == "Contradicted" for item in verified)
    failed = sum(item["verdict"] == FAILED_VERDICT for item in verified)
    scored = total - failed

    # Percentage of checked claims labeled Supported; not factual accuracy.
    trust_score = round(100 * supported / scored) if scored else 0

    return {
        "claims": verified,
        "trust_score": trust_score,
        "total_claims": total,
        "supported_claims": supported,
        "n_contradicted": contradicted,
        "failed_claims": failed,
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


def _failed(claim: str, reason: str) -> dict:
    """A claim whose check could not be completed (not a judgment on it)."""
    return {
        "claim": claim,
        "verdict": FAILED_VERDICT,
        "evidence": [],
        "modality": "none",
        "explanation": (
            f"Verification failed: {reason}. This claim was not scored."
        ),
    }


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
        # The check itself failed (API error, timeout, unreadable reply).
        return [_failed(claim, type(exc).__name__) for claim in claims]

    verified = []

    for index, claim in enumerate(claims):
        item = (
            model_results[index]
            if index < len(model_results)
            and isinstance(model_results[index], dict)
            else None
        )

        # The model skipped this claim or gave a verdict that is not allowed.
        if item is None or item.get("verdict") not in ALLOWED_VERDICTS:
            verified.append(
                _failed(claim, "the model returned no valid verdict")
            )
            continue

        verdict = item["verdict"]

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
    """Keep up to MAX_IMAGES images whose files actually exist, in order.

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


# A claim that is explicitly about what an image shows ("the image shows a
# consistent decline") is judged by the image. The retrieved text cannot
# refute a statement about how a picture looks.
_IMAGE_SUBJECT_RE = re.compile(
    r"\b(?:the\s+|this\s+|these\s+)?(?:(?:provided|given|retrieved|attached)\s+)?"
    r"(?:images?|figures?|charts?|graphs?|diagrams?|plots?|pictures?|photos?)\b"
    r"[^.]{0,40}?\b(?:shows?|depicts?|displays?|illustrates?|indicates?|reveals?)\b"
)


def _is_about_image(claim: str) -> bool:
    return bool(_IMAGE_SUBJECT_RE.search(claim.lower()))


def _merge_with_images(text_item: dict, findings: list[tuple]) -> dict:
    """Combine a text verdict with image findings for one claim.

    findings: (label, verdict, description) per checked image.
    Contradiction override: if any image shows the opposite of the claim, the
    verdict is Contradicted even when the retrieved text supports the claim.
    A claim whose text check failed can still be settled by an image.
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

    if text_verdict == "Contradicted" and not (
        supporting and _is_about_image(text_item["claim"])
    ):
        return text_item

    if supporting:
        # When the text contradicted a claim about the image, the image alone
        # settles it, so the contradicting text quote is not shown as support.
        kept = [] if text_verdict == "Contradicted" else text_item["evidence"]
        evidence = kept + [describe(f) for f in supporting]

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

    Text is checked first. If images are given, each (up to MAX_IMAGES, in the
    order given) is then checked on its pixels only, and an image that shows
    the opposite of a claim overrides the text verdict with "Contradicted".

    A claim whose check fails gets the verdict "Verification Failed" and is
    left out of trust_score (see "failed_claims" in the result).
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
