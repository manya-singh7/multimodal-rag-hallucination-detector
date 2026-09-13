from call_vlm import call_vlm


def _build_prompt(query, text_chunks, images):
    lines = [
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you don't know.",
        "",
        "Context:",
    ]
    for i, chunk in enumerate(text_chunks, 1):
        lines.append(f"[Text {i} | source: {chunk['source_doc']}]\n{chunk['text']}")

    for i, img in enumerate(images, 1):
        caption = img.get("caption") or "no caption"
        lines.append(f"[Image {i} | source: {img['source_doc']}] {caption}")

    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("Answer:")
    return "\n\n".join(lines)


def _find_supporting_sentence(answer, text_chunks):
    """Pick the sentence from the retrieved text that overlaps most with the answer."""
    if not text_chunks or not answer:
        return "none"

    answer_words = set(answer.lower().split())
    best_sentence = "none"
    best_overlap = 0

    for chunk in text_chunks:
        for sentence in chunk["text"].split(". "):
            sentence = sentence.strip().rstrip(".")
            if not sentence:
                continue
            sentence_words = set(sentence.lower().split())
            overlap = len(answer_words & sentence_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

    return best_sentence if best_overlap > 0 else "none"


def generate_answer(query, retrieved):
    text_chunks = retrieved.get("text_chunks") or []
    images = retrieved.get("images") or []

    prompt = _build_prompt(query, text_chunks, images)
    image_paths = [img["path"] for img in images]

    answer = call_vlm(prompt, images=image_paths or None)
    supporting_sentence = _find_supporting_sentence(answer, text_chunks)

    return {
        "answer": answer.strip(),
        "supporting_sentence": supporting_sentence,
        "used_text": text_chunks,
        "used_images": images,
    }