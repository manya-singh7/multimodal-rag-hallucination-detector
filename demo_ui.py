
import html
import time
import gradio as gr


try:
    from generator import generate_answer as _generate_answer_real
    HAS_REAL_GENERATOR = True
except Exception as exc:
    _generate_answer_real = None
    HAS_REAL_GENERATOR = False
    GENERATOR_IMPORT_ERROR = str(exc)
else:
    GENERATOR_IMPORT_ERROR = ""


try:
    from verifier import verify_claims
    HAS_VERIFIER = True
except Exception as exc:
    verify_claims = None
    HAS_VERIFIER = False
    VERIFIER_IMPORT_ERROR = str(exc)
else:
    VERIFIER_IMPORT_ERROR = ""


TEST_QUERIES = [
    "How does retrieval augmented generation use evidence?",
    "What is multimodal hallucination?",
    "How can retrieved evidence help verify an answer?",
    "What are the limitations of image-based question answering?",
    "Why should AI answers be checked against sources?",
]


def retrieve(query: str) -> dict:
    """Retrieve relevant text and images from the existing retriever."""
    from retriever import retrieve as retrieve_documents

    output = retrieve_documents(query, top_k=3)

    text_chunks = []
    images = []

    for item in output.get("results", []):
        text_chunks.append({
            "text": item.get("text", ""),
            "source": item.get("source", "unknown"),
            "source_doc": item.get("source", "unknown"),
            "page": item.get("page", "Unknown"),
        })

        image_path = item.get("image_path")

        if image_path:
            from pathlib import Path

            if Path(image_path).is_file():
                images.append({
                    "path": image_path,
                    "caption": item.get("caption", ""),
                    "source_doc": item.get("source", "unknown"),
                })

    return {
        "text_chunks": text_chunks,
        "results": text_chunks,
        "images": images,
        "image_paths": [img["path"] for img in images],
        "is_sample_data": output.get("is_sample_data", False),
    }


def generate_answer(query: str, retrieved: dict) -> dict:
    """Generate an answer using the real model."""
    if not HAS_REAL_GENERATOR:
        raise RuntimeError(
            "The real generator could not be imported. "
            f"Details: {GENERATOR_IMPORT_ERROR}"
        )

    return _generate_answer_real(query, retrieved)


def check_hallucinations(answer: str, retrieved: dict) -> dict:
    """Run the real claim verifier."""
    if not HAS_VERIFIER:
        raise RuntimeError(
            "The claim verifier could not be imported. "
            f"Details: {VERIFIER_IMPORT_ERROR}"
        )

    return verify_claims(
        answer=answer,
        text_chunks=retrieved.get("text_chunks", []),
    )


def sources_html(text_chunks):
    if not text_chunks:
        return "<p>No relevant text sources were found.</p>"

    cards = []

    for index, chunk in enumerate(text_chunks, start=1):
        source = html.escape(
            str(chunk.get("source_doc", chunk.get("source", "unknown")))
        )
        page = html.escape(str(chunk.get("page", "Unknown")))
        content = html.escape(str(chunk.get("text", "")))

        cards.append(
            "<div style='border:1px solid #ccc;padding:12px;"
            "border-radius:8px;margin-bottom:10px'>"
            f"<b>Source {index}: {source}</b>"
            f"<p>Page: {page}</p>"
            f"<p>{content}</p></div>"
        )

    return "".join(cards)


def claims_html(check):
    claims = check.get("claims", [])

    if not claims:
        return "<p>No answer claims were available to verify.</p>"

    cards = []

    for item in claims:
        verdict = html.escape(
            str(item.get("verdict", "Insufficient Evidence"))
        )
        claim = html.escape(str(item.get("claim", "")))
        explanation = html.escape(str(item.get("explanation", "")))

        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        evidence_items = "".join(
            f"<li>{html.escape(str(quote))}</li>"
            for quote in evidence
        )

        if not evidence_items:
            evidence_items = "<li>No validated evidence quote returned.</li>"

        cards.append(
            "<div style='border:1px solid #ccc;padding:10px;"
            "border-radius:8px;margin-bottom:8px'>"
            f"<b>{verdict.upper()}</b>"
            f"<p><b>Claim:</b> {claim}</p>"
            f"<p><b>Explanation:</b> {explanation}</p>"
            f"<p><b>Evidence:</b></p><ul>{evidence_items}</ul>"
            "</div>"
        )

    return "".join(cards)


def run_pipeline(query: str):
    if not query or not query.strip():
        return (
            "<i>Enter a query above.</i>",
            "",
            [],
            "<i>—</i>",
            "",
        )

    try:
        # Stage 1: Retrieval
        start = time.perf_counter()
        retrieved = retrieve(query)
        retrieval_time = time.perf_counter() - start

        print(f"[TIMING] Retrieval: {retrieval_time:.2f}s", flush=True)
        print(
            f"[INFO] Retrieved {len(retrieved.get('text_chunks', []))} "
            f"text chunks and {len(retrieved.get('images', []))} images.",
            flush=True,
        )

        # Stage 2: Answer generation
        start = time.perf_counter()
        gen = generate_answer(query, retrieved)
        generation_time = time.perf_counter() - start

        print(
            f"[TIMING] Answer generation: {generation_time:.2f}s",
            flush=True,
        )

        answer = gen.get("answer", "No answer returned.")
        if not isinstance(answer, str):
            answer = str(answer)

        supporting_sentence = gen.get("supporting_sentence")

        if supporting_sentence not in (None, "", "none"):
            answer += (
                "\n\nPossible supporting text "
                "(not independent verification): "
                + str(supporting_sentence)
            )

        gallery_items = [
            (img["path"], img.get("caption", ""))
            for img in retrieved.get("images", [])
        ]

        # Stage 3: Claim verification
        if retrieved.get("is_sample_data"):
            claims_display = (
                "<p>Verification skipped because the retriever "
                "reported sample data.</p>"
            )
            badge = (
                "<div style='padding:10px;border:1px solid #cc9900;"
                "border-radius:8px'>"
                "<b>SAMPLE DATA</b> — not verified against a real "
                "research corpus.</div>"
            )
            verification_time = 0.0

        else:
            start = time.perf_counter()
            check = check_hallucinations(answer, retrieved)
            verification_time = time.perf_counter() - start

            print(
                f"[TIMING] Claim verification: "
                f"{verification_time:.2f}s",
                flush=True,
            )

            claims_display = claims_html(check)

            score = check.get("trust_score")
            if isinstance(score, (int, float)):
                score_text = f"{score}% evidence-support rate"
            else:
                score_text = "Score unavailable"

            badge = (
                "<div style='padding:10px;border:1px solid #999;"
                "border-radius:8px'>"
                f"<b>{html.escape(score_text)}</b><br>"
                "This is the percentage of claims classified as "
                "Supported, not a calibrated probability of correctness."
                "</div>"
            )

        print(
            f"[TIMING] Total pipeline: "
            f"{retrieval_time + generation_time + verification_time:.2f}s",
            flush=True,
        )

        if retrieved.get("is_sample_data"):
            answer = (
                "[SAMPLE DATA — NOT REAL RESEARCH EVIDENCE]\n\n"
                + answer
            )

        return (
            sources_html(retrieved.get("text_chunks", [])),
            answer,
            gallery_items,
            claims_display,
            badge,
        )

    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)

        error = html.escape(str(exc))
        message = f"<p><b>Pipeline error:</b> {error}</p>"

        return (
            message,
            f"Pipeline error: {exc}",
            [],
            message,
            "",
        )


with gr.Blocks(title="Multimodal RAG Hallucination Detector") as demo:
    gr.Markdown("# Multimodal RAG + Hallucination Detection")

    gr.Markdown(
        "Retrieve research evidence, generate an answer, and evaluate "
        "individual claims against retrieved text. The evidence-support "
        "score is not a calibrated probability of correctness."
    )

    if not HAS_REAL_GENERATOR:
        gr.Markdown(
            "**Generator unavailable:** check generator.py and "
            f"call_vlm.py. Details: {GENERATOR_IMPORT_ERROR}"
        )

    if not HAS_VERIFIER:
        gr.Markdown(
            "**Verifier unavailable:** check verifier.py. "
            f"Details: {VERIFIER_IMPORT_ERROR}"
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
            badge_box = gr.HTML(label="Evidence-support score")

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
    demo.queue(default_concurrency_limit=1).launch()
