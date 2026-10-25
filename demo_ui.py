import html
import os
import re
import time
from pathlib import Path

import gradio as gr

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
    # Normal text questions
    "What are the two primary stages from which hallucinations arise in RAG models?",
    "What is the middle curse in long contexts?",
    "What is the difference between intrinsic and extrinsic hallucination?",
    "Why do the authors say prompt engineering is useful for hallucination mitigation?",
    "How does the RAPTOR method organize retrieved text?",
    "How many videos and question-answer pairs does the ANCHOR benchmark contain?",
    "How is the difficulty of each question assigned in ANCHOR?",
    "What happens to accuracy when video is removed in the grounding-cue ablation?",
    "How well do the LLM judges agree with human raters in ANCHOR?",
    # Figure questions
    "What does Figure 1 of the review show about the causes of hallucinations and their mitigation methods?",
    "What does Figure 2 of the review show about the basic workflow of retrieval-augmented generation?",
    "What does the ANCHOR dataset composition figure show about video duration and resolution?",
    "What does Figure 4 show about reasoning similarity between models?",
    # Trick queries (false premises)
    "The review says hallucinations in RAG come only from the generation phase. Explain that claim.",
    "ANCHOR contains over 50,000 videos. Which domains do they cover?",
    "In the ablation, removing audio caused a much larger accuracy drop than removing video. Why?",
    "What did the review conclude about quantum retrieval methods?",
    "Figure 4 shows that Claude and GPT-5.2 have the lowest agreement of any model pair. Explain this.",
    # Contradiction demo (needs data/processed/text/demo_contradiction.json
    # and data/processed/images/demo_contradiction_p1_img1.png)
    "How did annual revenue change from 2019 to 2023?",
]

SUPPORTED_STYLE = (
    "background-color:#d4edda;color:#155724;padding:2px 6px;"
    "border-radius:4px;margin:0 2px;"
)
UNSUPPORTED_STYLE = (
    "background-color:#f8d7da;color:#721c24;padding:2px 6px;"
    "border-radius:4px;margin:0 2px;"
)
# Claims whose check itself failed: neutral grey with a dashed border, so they
# are never mistaken for a real "Contradicted / Unsupported" verdict.
FAILED_STYLE = (
    "background-color:#e2e3e5;color:#383d41;padding:2px 6px;"
    "border-radius:4px;margin:0 2px;border:1px dashed #6c757d;"
)

_IMAGE_NAME_RE = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)_img")

MODALITY_LABELS = {
    "text": "retrieved text",
    "image": "image",
    "text+image": "retrieved text + image",
    "none": "no evidence found",
}


def _count(value) -> int:
    """failed_claims may be a number or a list; return a plain count."""
    if isinstance(value, (list, tuple)):
        return len(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_failed(verdict) -> bool:
    return str(verdict).strip().lower() == "verification failed"


def api_key_missing() -> bool:
    key = os.getenv("GEMINI_API_KEY")
    return not key or key == "put_your_api_key_here"


API_KEY_MESSAGE = (
    "<p><b>Gemini API key missing.</b> Create a file named <code>.env</code> "
    "in the project folder containing <code>GEMINI_API_KEY=your_key_here</code>, "
    "then restart the app.</p>"
)


def _image_label(path: str) -> str:
    """Gallery label parsed from the retriever's <doc>_p<page>_img* filenames."""
    match = _IMAGE_NAME_RE.match(Path(path).stem)
    if match:
        return f"{match.group('doc')}, page {match.group('page')}"
    return Path(path).name


def retrieve(query: str) -> dict:
    """Retrieve relevant text and images and normalize them for the UI."""
    from retriever import retrieve as retrieve_documents

    output = retrieve_documents(query, top_k=3)

    text_chunks = []
    for item in output.get("results", []):
        text_chunks.append({
            "text": item.get("text", ""),
            "source": item.get("source", "unknown"),
            "source_doc": item.get("source", "unknown"),
            "page": item.get("page", "Unknown"),
        })

    # The retriever returns page images in "image_paths"
    # (the per-result "image_path" field is always None).
    images = []
    for path in output.get("image_paths", []):
        if Path(path).is_file():
            images.append({
                "path": path,
                "caption": "",
                "source_doc": Path(path).name,
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
    """Run the multimodal claim verifier (retrieved text + images)."""
    if not HAS_VERIFIER:
        raise RuntimeError(
            "The claim verifier could not be imported. "
            f"Details: {VERIFIER_IMPORT_ERROR}"
        )

    return verify_claims(
        answer=answer,
        text_chunks=retrieved["text_chunks"],
        images=retrieved.get("images", []),
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
        raw_verdict = str(item.get("verdict", "Insufficient Evidence"))
        verdict = html.escape(raw_verdict)
        claim = html.escape(str(item.get("claim", "")))
        explanation = html.escape(str(item.get("explanation", "")))

        modality = str(item.get("modality", "none"))
        modality_label = html.escape(MODALITY_LABELS.get(modality, modality))

        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        evidence_items = "".join(
            f"<li>{html.escape(str(quote))}</li>"
            for quote in evidence
        )

        if not evidence_items:
            evidence_items = "<li>No validated evidence returned.</li>"

        # Failed checks get a neutral dashed card instead of the normal one.
        if _is_failed(raw_verdict):
            card_style = (
                "border:1px dashed #6c757d;background:#f1f2f3;"
                "padding:10px;border-radius:8px;margin-bottom:8px"
            )
        else:
            card_style = (
                "border:1px solid #ccc;padding:10px;"
                "border-radius:8px;margin-bottom:8px"
            )

        cards.append(
            f"<div style='{card_style}'>"
            f"<b>{verdict.upper()}</b>"
            f"<p><b>Claim:</b> {claim}</p>"
            f"<p><b>Checked against:</b> {modality_label}</p>"
            f"<p><b>Explanation:</b> {explanation}</p>"
            f"<p><b>Evidence:</b></p><ul>{evidence_items}</ul>"
            "</div>"
        )

    return "".join(cards)


def status_badge(check: dict) -> str:
    """Feature #2: categorical status badge.

    Part 5 thresholds, plus the "Verification Failed" handling agreed in
    Part 9: claims whose check failed are not scored, and the badge never says
    Verified or Partially Supported while something could not be checked.
    Contradicted still wins (red), but the failed count is shown.
    """
    total = _count(check.get("total_claims", 0))
    failed = _count(check.get("failed_claims", 0))
    checked = max(total - failed, 0)
    score = check.get("trust_score", 0)
    n_contradicted = _count(check.get("n_contradicted", 0))
    supported = _count(check.get("supported_claims", 0))

    if total == 0:
        icon, label = "⚪", "No verifiable claims"
        bg, fg = "#e2e3e5", "#383d41"
        detail = "The answer contained no claims to check."
    elif failed == total:
        icon, label = "🟠", "Verification failed, try again"
        bg, fg = "#ffe5d0", "#8a4b08"
        detail = (
            f"None of the {total} claims could be checked (API error or "
            "unreadable reply). Run the question again."
        )
    else:
        if n_contradicted > 0:
            icon, label = "🔴", "Unsupported / Contradicted"
            bg, fg = "#f8d7da", "#721c24"
        elif failed > 0:
            icon, label = "🟠", "Verification incomplete, try again"
            bg, fg = "#ffe5d0", "#8a4b08"
        elif score == 100:
            icon, label = "🟢", "Verified"
            bg, fg = "#d4edda", "#155724"
        elif score > 0:
            icon, label = "🟡", "Partially Supported"
            bg, fg = "#fff3cd", "#856404"
        else:
            icon, label = "🔴", "Unsupported / Contradicted"
            bg, fg = "#f8d7da", "#721c24"

        detail = (
            f"{score}% evidence-support rate "
            f"({supported}/{checked} checked claims supported, "
            f"{n_contradicted} contradicted)"
        )
        if failed:
            detail += f". {failed} of {total} claims could not be checked."

    return (
        f"<div style='padding:14px;border-radius:10px;background:{bg};"
        f"color:{fg};margin-bottom:10px'>"
        f"<div style='font-size:1.2em'><b>{icon} {label}</b></div>"
        f"<div>{html.escape(detail)}</div>"
        "<div style='font-size:0.85em;margin-top:4px'>"
        "The evidence-support rate is the percentage of checked claims "
        "classified as Supported, not accuracy or a calibrated "
        "probability.</div></div>"
    )


def highlighted_answer(answer: str, claims: list[dict]) -> str:
    """Feature #1: each verified claim wrapped in a green, red or grey span.

    Built from the verifier's claims (not by string-matching the raw answer),
    because the verifier normalizes text (strips Markdown, merges list items).
    Claims that could not be checked are neutral grey, not red.
    If there are no claims (for example a refusal), the raw answer is shown.
    """
    if not claims:
        return f"<p>{html.escape(answer)}</p>"

    spans = []
    for item in claims:
        verdict = str(item.get("verdict", "Insufficient Evidence"))

        if verdict == "Supported":
            style = SUPPORTED_STYLE
        elif _is_failed(verdict):
            style = FAILED_STYLE
        else:
            style = UNSUPPORTED_STYLE

        spans.append(
            f'<span style="{style}" title="{html.escape(verdict)}">'
            f'{html.escape(str(item.get("claim", "")))}</span>'
        )

    return (
        "<div style='line-height:2.2'>"
        + " ".join(spans)
        + "</div>"
        "<div style='font-size:0.85em;margin-top:8px'>"
        f"<span style='{SUPPORTED_STYLE}'>Supported</span> "
        f"<span style='{UNSUPPORTED_STYLE}'>Contradicted / Unsupported</span> "
        f"<span style='{FAILED_STYLE}'>Not checked</span>"
        "</div>"
    )


def supporting_note(gen: dict) -> str:
    sentence = gen.get("supporting_sentence")
    if sentence in (None, "", "none"):
        return ""
    return (
        "<p style='font-size:0.85em'><i>Possible supporting text "
        "(not independent verification): "
        f"{html.escape(str(sentence))}</i></p>"
    )


def run_pipeline(query: str):
    if not query or not query.strip():
        return (
            "<i>Enter a query above.</i>",
            "",
            [],
            "<i>—</i>",
            "",
        )

    if api_key_missing():
        return (
            API_KEY_MESSAGE,
            API_KEY_MESSAGE,
            [],
            API_KEY_MESSAGE,
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

        gallery_items = [
            (img["path"], _image_label(img["path"]))
            for img in retrieved.get("images", [])
        ]

        # Stage 3: Claim verification
        if retrieved.get("is_sample_data"):
            claims_display = (
                "<p>Verification skipped because the retriever "
                "reported sample data.</p>"
            )
            badge = (
                "<div style='padding:14px;border-radius:10px;"
                "background:#fff3cd;color:#856404'>"
                "<b>SAMPLE DATA</b> — not verified against a real "
                "research corpus.</div>"
            )
            answer_display = (
                "<p><b>[SAMPLE DATA — NOT REAL RESEARCH EVIDENCE]</b></p>"
                f"<p>{html.escape(answer)}</p>" + supporting_note(gen)
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
            badge = status_badge(check)
            answer_display = (
                highlighted_answer(answer, check.get("claims", []))
                + supporting_note(gen)
            )

        print(
            f"[TIMING] Total pipeline: "
            f"{retrieval_time + generation_time + verification_time:.2f}s",
            flush=True,
        )

        return (
            sources_html(retrieved.get("text_chunks", [])),
            answer_display,
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
            message,
            [],
            message,
            "",
        )


with gr.Blocks(title="Multimodal RAG Hallucination Detector") as demo:
    gr.Markdown("# Multimodal RAG + Hallucination Detection")

    gr.Markdown(
        "Retrieve research evidence, generate an answer, and evaluate "
        "individual claims against retrieved text and images. The "
        "evidence-support rate is not accuracy or a calibrated probability."
    )

    if api_key_missing():
        gr.HTML(API_KEY_MESSAGE)

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
            lines=1,
            max_lines=1,
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
                object_fit="contain",
            )

        with gr.Column(scale=1):
            badge_box = gr.HTML(label="Evidence-support rate")
            answer_box = gr.HTML(label="Generated answer")
            claims_box = gr.HTML(label="Claim verification")

    pipeline_outputs = [
        sources_box,
        answer_box,
        image_gallery,
        claims_box,
        badge_box,
    ]

    submit_btn.click(
        fn=run_pipeline,
        inputs=query_box,
        outputs=pipeline_outputs,
    )

    # Pressing Enter in the question box runs the same pipeline.
    query_box.submit(
        fn=run_pipeline,
        inputs=query_box,
        outputs=pipeline_outputs,
    )

    example_dropdown.change(
        fn=lambda value: value or "",
        inputs=example_dropdown,
        outputs=query_box,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()