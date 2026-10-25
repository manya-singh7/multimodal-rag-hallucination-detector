import re
from pathlib import Path

from call_vlm import call_vlm

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
}


def _extract_words(text: str) -> set[str]:
    """Extract lowercased non-stopword tokens from text."""
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {w for w in tokens if w not in STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    """Split text into individual sentences, stripping whitespace and trailing periods."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for s in parts:
        cleaned = s.strip().rstrip(".")
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _build_prompt(query, text_chunks, images):
    lines = [
        "Answer the question using only the supplied evidence.",
        "If the evidence is insufficient, say so clearly.",
        "Do not invent facts or claim that an image shows something "
        "unless you can actually verify it.",
        "Answer in 2 or 3 short sentences, one fact per sentence, "
        "and do not list every item from a figure or table; mention only what the question asks.",
        "",
        "TEXT EVIDENCE:",
    ]

    for i, chunk in enumerate(text_chunks, 1):
        if isinstance(chunk, dict):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "Unknown")
            text = chunk.get("text", "")
        else:
            source = "Unknown"
            page = "Unknown"
            text = str(chunk)
        lines.append(
            f"[Evidence {i} | Source: {source} | Page: {page}]\n"
            f"{text}"
        )

    lines.append("")
    if images:
        lines.append("Relevant images are attached separately.")
    lines.extend([f"Question: {query}", "Answer:"])

    return "\n\n".join(lines)


def _find_supporting_sentence(answer, text_chunks):
    """Find a possible supporting sentence; not a verified claim check."""
    if not answer or not text_chunks:
        return "none"

    answer_sentences = _split_sentences(answer)
    if not answer_sentences:
        return "none"

    ans_word_sets = [_extract_words(s) for s in answer_sentences]
    ans_word_sets = [ws for ws in ans_word_sets if ws]
    if not ans_word_sets:
        return "none"

    best_sentence = "none"
    best_overlap = 0

    for chunk in text_chunks:
        chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        for sentence in _split_sentences(chunk_text):
            src_words = _extract_words(sentence)
            if not src_words:
                continue

            for ans_words in ans_word_sets:
                overlap = len(ans_words & src_words)
                if overlap >= 3 and overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sentence

    return best_sentence if best_overlap >= 3 else "none"


def generate_answer(query, retrieved):
    # retriever.py returns result dictionaries and image_paths.
    text_chunks = retrieved.get("results") or retrieved.get("text_chunks") or []
    image_paths = retrieved.get("image_paths") or [
        img["path"] for img in retrieved.get("images", []) if isinstance(img, dict) and "path" in img
    ] or []

    if not text_chunks and not image_paths:
        return {
            "answer": "I couldn't find relevant evidence in the documents.",
            "supporting_sentence": "none",
            "used_text": [],
            "used_images": [],
        }

    prompt = _build_prompt(query, text_chunks, image_paths)
    answer = call_vlm(prompt, images=image_paths or None)

    used_images = []
    for path in image_paths:
        p = path.get("path") if isinstance(path, dict) else str(path)
        used_images.append({
            "path": p,
            "source_doc": Path(p).name,
        })

    return {
        "answer": answer.strip(),
        "supporting_sentence": _find_supporting_sentence(
            answer, text_chunks
        ),
        "used_text": text_chunks,
        "used_images": used_images,
    }
