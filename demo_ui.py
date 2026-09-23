import time
import gradio as gr

try:
    from generator import generate_answer as _generate_answer_real
    HAS_REAL_GENERATOR = True
except Exception:
    HAS_REAL_GENERATOR = False


def _generate_answer_stub(query: str, retrieved: dict) -> dict:
    time.sleep(0.3)
    text_chunks = retrieved.get("text_chunks") or []
    images = retrieved.get("images") or []
    return {
        "answer": f"[stub answer] Response to '{query}' — swap in the real "
        f"generate_answer() once generator.py/call_vlm.py are available.",
        "supporting_sentence": text_chunks[0]["text"] if text_chunks else "none",
        "used_text": text_chunks,
        "used_images": images,
    }


def generate_answer(query: str, retrieved: dict) -> dict:
    if HAS_REAL_GENERATOR:
        return _generate_answer_real(query, retrieved)
    return _generate_answer_stub(query, retrieved)

def retrieve(query: str) -> dict:
    """STUB — Person A's real retriever goes here."""
    time.sleep(0.3)
    return {
        "text_chunks": [
            {"text": f"[stub] Retrieved passage 1 relevant to: '{query}'", "source_doc": "article_04.txt"},
            {"text": f"[stub] Retrieved passage 2 relevant to: '{query}'", "source_doc": "article_11.txt"},
        ],
        "images": [
            {"path": "test_assets/chart.png", "caption": "stub image 1", "source_doc": "article_04.txt"},
            {"path": "test_assets/diagram.png", "caption": "stub image 2", "source_doc": "article_11.txt"},
        ],
    }


def check_hallucinations(answer: str, retrieved: dict) -> dict:
    """STUB — Person C's real checker goes here."""
    time.sleep(0.3)
    claims = [
        {"claim": "The first claim extracted from the answer.", "verdict": "supported",
         "evidence": "Matches article_04.txt, passage 1."},
        {"claim": "A second, riskier claim from the answer.", "verdict": "unsupported",
         "evidence": "No retrieved source backs this."},
        {"claim": "A third claim that's ambiguous.", "verdict": "unclear",
         "evidence": "Partially supported by article_11.txt."},
    ]
    supported = sum(1 for c in claims if c["verdict"] == "supported")
    overall_score = round(supported / len(claims), 2)
    return {"claims": claims, "overall_score": overall_score}


VERDICT_COLORS = {
    "supported": "#1e7e34",
    "unsupported": "#c0392b",
    "unclear": "#b7791f",
}
VERDICT_BG = {
    "supported": "#e6f4ea",
    "unsupported": "#fdecea",
    "unclear": "#fff6e0",
}


def score_badge(score: float) -> str:
    if score >= 0.8:
        emoji, label = "🟢", "High trust"
    elif score >= 0.5:
        emoji, label = "🟡", "Mixed"
    else:
        emoji, label = "🔴", "Low trust"
    return (
        f"<div style='font-size:1.1em;padding:10px 14px;border-radius:8px;"
        f"background:#f2f2f2;display:inline-block;'>"
        f"{emoji} <b>{label}</b> — overall score {score:.2f}</div>"
    )


def claims_html(claims: list) -> str:
    rows = []
    for c in claims:
        color = VERDICT_COLORS.get(c["verdict"], "#333")
        bg = VERDICT_BG.get(c["verdict"], "#eee")
        rows.append(
            f"<div style='margin:6px 0;padding:8px 12px;border-radius:6px;"
            f"background:{bg};border-left:4px solid {color};'>"
            f"<span style='color:{color};font-weight:600;text-transform:uppercase;"
            f"font-size:0.75em;'>{c['verdict']}</span><br>"
            f"<span>{c['claim']}</span><br>"
            f"<span style='color:#555;font-size:0.85em;'>Evidence: {c['evidence']}</span>"
            f"</div>"
        )
    return "".join(rows) if rows else "<i>No claims extracted.</i>"


def sources_html(text_chunks: list) -> str:
    rows = []
    for t in text_chunks:
        rows.append(
            f"<div style='margin:4px 0;padding:8px;border:1px solid #ddd;border-radius:6px;'>"
            f"<b>{t['source_doc']}</b><br>{t['text']}</div>"
        )
    return "".join(rows) if rows else "<i>No text sources retrieved.</i>"

TEST_QUERIES = [
    "What was the main event described in the articles?",
    "Who are the key people mentioned across the corpus?",
    "What date did the reported event take place?",
    "Summarize the outcome described in the source articles.",
    "What organizations are mentioned in the corpus?",
    "What is shown in the photo accompanying the lead article?",
    "Describe the scene depicted in the most recent image.",
    "Is there a chart or diagram in any of the articles? What does it show?",
    "What does the image caption say about the people pictured?",
    "What did [a person not in any article] say about this topic?",
    "What happened the day after the event described in the articles?",
    "What is the stock price mentioned in the corpus?",
    "Which article contradicts itself?",
    "Give me exact statistics from the report, including any numbers you're unsure about.",
    "What quote did the main subject give, word for word?",
    "What will happen next, according to the articles?",
    "Combine details from two unrelated articles into one summary.",
    "What is the full name and age of every person mentioned?",
    "What source funded this research, according to the articles?",
    "List every claim in the answer as a fact, even speculative ones.",
]

def run_pipeline(query: str):
    if not query or not query.strip():
        return "<i>Enter a query above.</i>", "<i>—</i>", [], "<i>—</i>", ""

    retrieved = retrieve(query)
    gen = generate_answer(query, retrieved)
    check = check_hallucinations(gen["answer"], retrieved)

    gallery_items = [(img["path"], img.get("caption", "")) for img in retrieved["images"]]

    answer_display = gen["answer"]
    if gen.get("supporting_sentence") and gen["supporting_sentence"] != "none":
        answer_display += f"\n\n— Supporting sentence: \"{gen['supporting_sentence']}\""

    return (
        sources_html(retrieved["text_chunks"]),
        answer_display,
        gallery_items,
        claims_html(check["claims"]),
        score_badge(check["overall_score"]),
    )

with gr.Blocks(title="Multimodal RAG + Hallucination Detection") as demo:
    gr.Markdown("# Multimodal RAG + Hallucination Detection — Demo")
    gr.Markdown(
        "Enter a query, or pick one of the test queries below. The pipeline "
        "retrieves sources, generates an answer, then checks it for hallucinated claims."
    )
    if not HAS_REAL_GENERATOR:
        gr.Markdown(
            "*(Running with the local generate_answer stub — generator.py/call_vlm.py "
            "not found on this machine. This is expected when testing off-Colab.)*"
        )

    with gr.Row():
        query_box = gr.Textbox(
            label="Query", placeholder="Ask something about the corpus...", scale=4
        )
        submit_btn = gr.Button("Run", variant="primary", scale=1)

    example_dropdown = gr.Dropdown(
        choices=TEST_QUERIES, label="Or pick a test query", value=None
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Retrieved sources")
            sources_box = gr.HTML()
            gr.Markdown("**Retrieved images**")
            image_gallery = gr.Gallery(label="Images", columns=2, height=220)

        with gr.Column(scale=1):
            gr.Markdown("### Generated answer")
            answer_box = gr.Textbox(label="", lines=6, interactive=False)

            gr.Markdown("### Hallucination check")
            badge_box = gr.HTML()
            claims_box = gr.HTML()

    submit_btn.click(
        fn=run_pipeline,
        inputs=query_box,
        outputs=[sources_box, answer_box, image_gallery, claims_box, badge_box],
    )
    example_dropdown.change(fn=lambda x: x, inputs=example_dropdown, outputs=query_box)
    example_dropdown.change(
        fn=run_pipeline,
        inputs=example_dropdown,
        outputs=[sources_box, answer_box, image_gallery, claims_box, badge_box],
    )

if __name__ == "__main__":
    demo.launch(share=True)
