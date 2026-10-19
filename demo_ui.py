
import html
import time
import gradio as gr

try:
    from generator import generate_answer as _generate_answer_real
    HAS_REAL_GENERATOR = True
except Exception:
    _generate_answer_real = None
    HAS_REAL_GENERATOR = False


def _generate_answer_stub(query: str, retrieved: dict) -> dict:
    """Offline fallback when the real model is unavailable."""
    time.sleep(0.2)
    chunks = retrieved.get("text_chunks") or []

    if chunks:
        answer = (
            "The retrieved sample context says that RAG uses relevant "
            "documents as context for generating answers. This is a "
            "demonstration response, not a real model-generated answer."
        )
        supporting_sentence = chunks[0]["text"]
    else:
        answer = (
            "I could not find relevant evidence. Add research PDFs and "
            "process them before relying on this pipeline."
        )
        supporting_sentence = "none"

    return {
        "answer": answer,
        "supporting_sentence": supporting_sentence,
        "used_text": chunks,
        "used_images": retrieved.get("images") or [],
    }


def generate_answer(query: str, retrieved: dict) -> dict:
    """Use the real generator when available; otherwise use a labelled fallback."""
    if HAS_REAL_GENERATOR:
        try:
            return _generate_answer_real(query, retrieved)
        except Exception as exc:
            return {
                "answer": (
                    "The real generator could not run. Check the model "
                    f"environment and call_vlm.py. Error: {exc}"
                ),
                "supporting_sentence": "none",
                "used_text": retrieved.get("text_chunks") or [],
                "used_images": retrieved.get("images") or [],
            }

    return _generate_answer_stub(query, retrieved)


def retrieve(query: str) -> dict:
    """Connect the existing UI to retriever.py."""
    from retriever import retrieve as retrieve_documents

    output = retrieve_documents(query, top_k=3)
    text_chunks = []
    images = []

    for item in output.get("results", []):
        text_chunks.append({
            "text": item.get("text", ""),
            "source_doc": item.get("source", "unknown"),
        })

        image_path = item.get("image_path")
        if image_path:
            from pathlib import Path

            if Path(image_path).exists():
                images.append({
                    "path": image_path,
                    "caption": item.get("caption", ""),
                    "source_doc": item.get("source", "unknown"),
                })

    return {
        "text_chunks": text_chunks,
        "images": images,
        "is_sample_data": output.get("is_sample_data", False),
    }


def check_hallucinations(answer: str, retrieved: dict) -> dict:
    """
    Placeholder only. This does not genuinely verify claims.
    Replace with the project's real claim-verification implementation.
    """
    return {
        "claims": [{
            "claim": "Automated claim verification is not implemented yet.",
            "verdict": "unclear",
            "evidence": (
                "This is a placeholder. No real claim-level verification "
                "has been performed."
            ),
        }],
        "overall_score": None,
        "status": "NOT VERIFIED",
    }


TEST_QUERIES = [
    "How does retrieval augmented generation use evidence?",
    "What is multimodal hallucination?",
    "How can retrieved evidence help verify an answer?",
    "What are the limitations of image-based question answering?",
    "Why should AI answers be checked against sources?",
]


def sources_html(text_chunks):
    if not text_chunks:
        return "<p>No relevant text sources were found.</p>"

    cards = []
    for index, chunk in enumerate(text_chunks, start=1):
        source = html.escape(str(chunk.get("source_doc", "unknown")))
        content = html.escape(str(chunk.get("text", "")))
        cards.append(
            f"<div style='border:1px solid #ccc;padding:12px;"
            f"border-radius:8px;margin-bottom:10px'>"
            f"<b>Source {index}: {source}</b>"
            f"<p>{content}</p></div>"
        )

    return "".join(cards)


def run_pipeline(query: str):
    if not query or not query.strip():
        return (
            "<i>Enter a query above.</i>",
            "<i>—</i>",
            [],
            "<i>—</i>",
            "",
        )

    try:
        retrieved = retrieve(query)
        gen = generate_answer(query, retrieved)
        check = check_hallucinations(gen["answer"], retrieved)

        gallery_items = [
            (img["path"], img.get("caption", ""))
            for img in retrieved.get("images", [])
        ]

        answer_display = gen.get("answer", "No answer returned.")

        if gen.get("supporting_sentence") not in (None, "", "none"):
            answer_display += (
                "\n\nSupporting text (not independent verification): "
                + gen["supporting_sentence"]
            )

        if retrieved.get("is_sample_data"):
            answer_display = (
                "[SAMPLE DATA — NOT REAL RESEARCH EVIDENCE]\n\n"
                + answer_display
            )

        claims = check.get("claims", [])
        claims_display = "<h3>Claim-check status: NOT VERIFIED</h3>"
        claims_display += (
            "<p>The claim checker is currently a placeholder. "
            "Do not interpret this as a real hallucination score.</p>"
        )

        for claim in claims:
            claims_display += (
                "<div style='border:1px solid #ccc;padding:10px;"
                "border-radius:8px;margin-bottom:8px'>"
                f"<b>{html.escape(str(claim.get('verdict', 'unclear')).upper())}"
                "</b><p>"
                f"{html.escape(str(claim.get('claim', '')))}"
                "</p><p>"
                f"{html.escape(str(claim.get('evidence', '')))}"
                "</p></div>"
            )

        badge = (
            "<div style='padding:10px;border-radius:8px;"
            "border:1px solid #cc9900'>"
            "<b>NOT VERIFIED</b> — claim verification is not implemented."
            "</div>"
        )

        return (
            sources_html(retrieved.get("text_chunks", [])),
            answer_display,
            gallery_items,
            claims_display,
            badge,
        )

    except Exception as exc:
        error = html.escape(str(exc))
        message = f"<p><b>Pipeline error:</b> {error}</p>"
        return message, f"Pipeline error: {exc}", [], message, ""


with gr.Blocks(title="Multimodal RAG Hallucination Detector") as demo:
    gr.Markdown("# Multimodal RAG + Hallucination Detection")
    gr.Markdown(
        "Enter a question to retrieve context and generate a response. "
        "Claim verification is currently a placeholder."
    )

    if not HAS_REAL_GENERATOR:
        gr.Markdown(
            "**Demo mode:** the real generator could not be imported. "
            "The interface will use a labelled fallback response."
        )

    with gr.Row():
        query_box = gr.Textbox(
            label="Question",
            placeholder="Ask a question about the research corpus...",
            scale=4,
        )
        submit_btn = gr.Button("Run", variant="primary", scale=1)

    example_dropdown = gr.Dropdown(
        choices=TEST_QUERIES,
        label="Example questions",
        value=None,
    )

    with gr.Row():
        with gr.Column(scale=1):
            sources_box = gr.HTML(label="Retrieved sources")
            image_gallery = gr.Gallery(
                label="Retrieved images",
                columns=2,
                height=220,
            )

        with gr.Column(scale=1):
            answer_box = gr.Textbox(
                label="Generated answer",
                lines=8,
                interactive=False,
            )
            claims_box = gr.HTML(label="Claim verification")
            badge_box = gr.HTML(label="Verification status")

    submit_btn.click(
        fn=run_pipeline,
        inputs=query_box,
        outputs=[
            sources_box,
            answer_box,
            image_gallery,
            claims_box,
            badge_box,
        ],
    )

    example_dropdown.change(
        fn=lambda value: value or "",
        inputs=example_dropdown,
        outputs=query_box,
    )


if __name__ == "__main__":
    demo.launch()
