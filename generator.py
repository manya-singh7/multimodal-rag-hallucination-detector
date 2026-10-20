
from pathlib import Path

from call_vlm import call_vlm


def _build_prompt(query, text_chunks, images):
    lines = [
        "Answer the question using only the supplied evidence.",
        "If the evidence is insufficient, say so clearly.",
        "Do not invent facts or claim that an image shows something "
        "unless you can actually verify it.",
        "",
        "TEXT EVIDENCE:",
    ]

    for i, chunk in enumerate(text_chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "Unknown")
        lines.append(
            f"[Evidence {i} | Source: {source} | Page: {page}]\n"
            f"{chunk.get('text', '')}"
        )

    lines.extend(["", "Relevant images are attached separately.",
                  f"Question: {query}", "Answer:"])

    return "\n\n".join(lines)


def _find_supporting_sentence(answer, text_chunks):
    """Find a possible supporting sentence; not a verified claim check."""
    if not answer or not text_chunks:
        return "none"

    answer_words = set(answer.lower().split())
    best_sentence = "none"
    best_overlap = 0

    for chunk in text_chunks:
        for sentence in chunk.get("text", "").split(". "):
            sentence = sentence.strip().rstrip(".")
            if not sentence:
                continue

            overlap = len(answer_words & set(sentence.lower().split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

    return best_sentence if best_overlap else "none"


def generate_answer(query, retrieved):
    # retriever.py returns result dictionaries and image_paths.
    text_chunks = retrieved.get("results") or []
    image_paths = retrieved.get("image_paths") or []

    if not text_chunks and not image_paths:
        return {
            "answer": "I couldn't find relevant evidence in the documents.",
            "supporting_sentence": "none",
            "used_text": [],
            "used_images": [],
        }

    prompt = _build_prompt(query, text_chunks, image_paths)
    answer = call_vlm(prompt, images=image_paths or None)

    return {
        "answer": answer.strip(),
        "supporting_sentence": _find_supporting_sentence(
            answer, text_chunks
        ),
        "used_text": text_chunks,
        "used_images": [
            {"path": path, "source_doc": Path(path).name}
            for path in image_paths
        ],
    }
