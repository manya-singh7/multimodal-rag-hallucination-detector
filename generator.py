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


def generate_answer(query, retrieved):
    text_chunks = retrieved.get("text_chunks") or []
    images = retrieved.get("images") or []

    prompt = _build_prompt(query, text_chunks, images)
    image_paths = [img["path"] for img in images]

    answer = call_vlm(prompt, images=image_paths or None)

    return {
        "answer": answer,
        "used_text": text_chunks,
        "used_images": images,
    }
