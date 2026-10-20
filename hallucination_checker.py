"""Person C: cross-modal hallucination checker.

check_hallucinations(answer, retrieved) splits an answer into claims and checks
each claim against (a) the retrieved text (including image captions) and
(b) the retrieved IMAGES THEMSELVES, with the caption deliberately hidden.

Verdicts per claim:
    supported     - backed by text and/or an image
    contradicted  - a source says the OPPOSITE (e.g. the image shows a decline
                    while the claim/caption says growth)
    unsupported   - nothing in the sources backs it up (no contradiction found)
    unclear       - the model's reply could not be parsed

Return format is a superset of the stub in demo_ui.py, so the UI keeps working:
    {"claims": [{"claim", "verdict", "evidence", "modality"}],
     "overall_score": float,          # supported / total claims
     "caption_conflicts": [...],      # images whose caption doesn't match pixels
     "n_contradicted": int}
"""
import re

MAX_IMAGES = 3

_REFUSAL_HINTS = (
    "don't know", "do not know", "cannot answer", "can't answer",
    "not mentioned in the context", "context does not", "not enough information",
)


def _vlm(prompt, images=None):
    # Imported lazily: call_vlm.py loads a 7B model at import time (Colab/GPU only).
    # Tests replace this function, so they run anywhere.
    from call_vlm import call_vlm
    return call_vlm(prompt, images=images)


# --------------------------------------------------------------------------
# Claim extraction
# --------------------------------------------------------------------------
def extract_claims(answer):
    """Split an answer into sentence-level claims, dropping refusals and fragments."""
    if not answer:
        return []
    claims = []
    for line in re.split(r"\n+", answer):
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = sent.strip()
            if len(sent.split()) < 4:
                continue
            if any(h in sent.lower() for h in _REFUSAL_HINTS):
                continue
            claims.append(sent)
    return claims


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
def _text_sources(retrieved):
    parts = []
    for i, c in enumerate(retrieved.get("text_chunks") or [], 1):
        parts.append(f"[Text {i} | {c.get('source_doc', 'unknown')}] {c['text']}")
    for i, img in enumerate(retrieved.get("images") or [], 1):
        if img.get("caption"):
            parts.append(f"[Caption of image {i} | {img.get('source_doc', 'unknown')}] {img['caption']}")
    return "\n".join(parts)


def _text_prompt(claim, sources):
    return (
        "TASK: TEXT CHECK\n"
        "You are a strict fact-checker. Using ONLY the source text below, decide "
        "whether the claim is supported, contradicted, or not found.\n\n"
        f"SOURCE TEXT:\n{sources}\n\n"
        f"CLAIM: {claim}\n\n"
        "Reply in exactly this format:\n"
        "VERDICT: SUPPORTED or CONTRADICTED or NOT_FOUND\n"
        "EVIDENCE: one short sentence, quoting the source if possible"
    )


def _image_prompt(claim):
    # The caption is intentionally NOT included: we want to test the claim against
    # what the pixels show, not against what someone wrote about them.
    return (
        "TASK: IMAGE CHECK\n"
        "Look carefully at the image. Ignore any text you might expect to "
        "accompany it. Decide whether the image supports the claim, contradicts "
        "it (shows the opposite), or does not show anything about it.\n\n"
        f"CLAIM: {claim}\n\n"
        "Reply in exactly this format:\n"
        "VERDICT: SUPPORTED or CONTRADICTED or NOT_SHOWN\n"
        "EVIDENCE: one short sentence describing what you actually see"
    )


def _caption_prompt(caption):
    return (
        "TASK: CAPTION CHECK\n"
        f"CAPTION: {caption}\n\n"
        "Does this caption accurately describe the image? Pay attention to "
        "direction of trends, counts, objects and labels.\n"
        "Reply in exactly this format:\n"
        "VERDICT: MATCHES or PARTIAL or MISMATCH\n"
        "EVIDENCE: one short sentence describing what the image actually shows"
    )


# --------------------------------------------------------------------------
# Parsing the model's reply (7B 4-bit models don't always follow formats)
# --------------------------------------------------------------------------
_VERDICT_RE = re.compile(r"VERDICT\s*:\s*([^\n]+)", re.I)
_EVIDENCE_RE = re.compile(r"EVIDENCE\s*:\s*(.+)", re.I | re.S)


def _split_reply(raw):
    text = (raw or "").strip()
    m = _VERDICT_RE.search(text)
    head = (m.group(1) if m else text[:80]).upper().replace("_", " ")
    e = _EVIDENCE_RE.search(text)
    evidence = e.group(1).strip().split("\n")[0].strip() if e else ""
    return head, evidence


def parse_claim_reply(raw):
    """-> (verdict in {supported, contradicted, not_found, None}, evidence)"""
    head, evidence = _split_reply(raw)
    if "CONTRADICT" in head or "OPPOSITE" in head:
        return "contradicted", evidence
    if any(k in head for k in ("NOT FOUND", "NOT SHOWN", "NOT SUPPORTED",
                               "UNSUPPORTED", "NOT MENTIONED", "NO EVIDENCE")):
        return "not_found", evidence
    if "SUPPORT" in head:
        return "supported", evidence
    return None, evidence


def parse_caption_reply(raw):
    """-> (verdict in {match, partial, mismatch, None}, evidence)"""
    head, evidence = _split_reply(raw)
    if "MISMATCH" in head or "DOES NOT MATCH" in head or "CONTRADICT" in head:
        return "mismatch", evidence
    if "PARTIAL" in head:
        return "partial", evidence
    if "MATCH" in head:
        return "match", evidence
    return None, evidence


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def verify_claim(claim, retrieved, check_images=True):
    text_v, text_ev = (None, "")
    sources = _text_sources(retrieved)
    if sources:
        text_v, text_ev = parse_claim_reply(_vlm(_text_prompt(claim, sources)))
    else:
        text_v, text_ev = "not_found", "No text sources were retrieved."

    image_results = []  # (label, verdict, evidence)
    if check_images:
        for i, img in enumerate((retrieved.get("images") or [])[:MAX_IMAGES], 1):
            v, ev = parse_claim_reply(_vlm(_image_prompt(claim), images=[img["path"]]))
            image_results.append((f"Image {i} ({img.get('source_doc', 'unknown')})", v, ev))

    # Conservative resolution: any contradiction wins, in either modality.
    for label, v, ev in image_results:
        if v == "contradicted":
            note = f"{label} contradicts this claim: {ev}".strip()
            if text_v == "supported":
                note += " (the text/caption supports it, but the image does not)"
            return {"claim": claim, "verdict": "contradicted", "evidence": note, "modality": "image"}
    if text_v == "contradicted":
        return {"claim": claim, "verdict": "contradicted",
                "evidence": f"Source text contradicts this claim: {text_ev}".strip(),
                "modality": "text"}

    for label, v, ev in image_results:
        if v == "supported":
            modality = "text+image" if text_v == "supported" else "image"
            return {"claim": claim, "verdict": "supported",
                    "evidence": f"{label}: {ev}".strip(), "modality": modality}
    if text_v == "supported":
        return {"claim": claim, "verdict": "supported", "evidence": text_ev, "modality": "text"}

    verdicts = [text_v] + [v for _, v, _ in image_results]
    if any(v is None for v in verdicts):
        return {"claim": claim, "verdict": "unclear",
                "evidence": "The checker's reply could not be parsed.", "modality": "none"}
    return {"claim": claim, "verdict": "unsupported",
            "evidence": "No retrieved text or image backs this claim.", "modality": "none"}


def check_captions(retrieved):
    """Flag retrieved images whose caption disagrees with what the image shows."""
    conflicts = []
    for i, img in enumerate((retrieved.get("images") or [])[:MAX_IMAGES], 1):
        caption = img.get("caption")
        if not caption:
            continue
        v, ev = parse_caption_reply(_vlm(_caption_prompt(caption), images=[img["path"]]))
        if v == "mismatch":
            conflicts.append({"image": i, "path": img["path"], "caption": caption,
                              "evidence": ev, "source_doc": img.get("source_doc")})
    return conflicts


def check_hallucinations(answer, retrieved, check_images=True, check_caption_consistency=True):
    claims = [verify_claim(c, retrieved, check_images) for c in extract_claims(answer)]
    supported = sum(1 for c in claims if c["verdict"] == "supported")
    return {
        "claims": claims,
        "overall_score": round(supported / len(claims), 2) if claims else 1.0,
        "caption_conflicts": check_captions(retrieved) if check_caption_consistency else [],
        "n_contradicted": sum(1 for c in claims if c["verdict"] == "contradicted"),
    }
