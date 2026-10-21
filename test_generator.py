import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import generator
import verifier
from generator import _find_supporting_sentence, generate_answer

STUB_IMAGE_DIR = "test_assets"


def make_stub_image(path, text, color):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (400, 300), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill="black")
    img.save(path)


def print_result(label, result):
    print(f"\n=== {label} ===")
    print(result["answer"])
    print("\nPossible supporting sentence:")
    print(result.get("supporting_sentence", "N/A"))
    print("\nText evidence:")
    for chunk in result.get("used_text", []):
        print(f"- ({chunk.get('source', 'Unknown')}) "
              f"{chunk.get('text', '')[:100]}...")
    print("\nImages:")
    for img in result.get("used_images", []):
        print(f"- {img.get('source_doc', 'Unknown')}: "
              f"{img.get('path', '')}")


def sample_text_chunks():
    return [
        {
            "text": (
                "The company reported revenue of $5.2 million in 2023, "
                "up 18% year over year."
            ),
            "source": "annual_report_2023.pdf",
            "page": 1,
        },
        {
            "text": (
                "Growth was driven primarily by the new enterprise product line."
            ),
            "source": "annual_report_2023.pdf",
            "page": 2,
        },
    ]


def sample_images():
    paths = [
        (os.path.join(STUB_IMAGE_DIR, "chart.png"),
         "Revenue chart 2023", "lightblue"),
        (os.path.join(STUB_IMAGE_DIR, "diagram.png"),
         "System diagram", "lightgreen"),
    ]

    result = []
    for path, label, color in paths:
        make_stub_image(path, label, color)
        result.append(path)
    return result


# ---------------- Offline Mocked Tests ----------------


def test_text_and_images(monkeypatch):
    monkeypatch.setattr(
        generator,
        "call_vlm",
        lambda prompt, images=None: (
            "In 2023, the company reported revenue of $5.2 million, "
            "driven by enterprise sales."
        ),
    )

    retrieved = {
        "results": sample_text_chunks(),
        "image_paths": sample_images(),
    }
    query = "What was the company's revenue in 2023 and what drove its growth?"
    result = generate_answer(query, retrieved)

    assert "revenue of $5.2 million" in result["answer"]
    assert len(result["used_text"]) == 2
    assert len(result["used_images"]) == 2
    assert result["used_images"][0]["path"] == retrieved["image_paths"][0]
    assert result["used_images"][0]["source_doc"] == Path(retrieved["image_paths"][0]).name
    assert result["supporting_sentence"] == (
        "The company reported revenue of $5.2 million in 2023, "
        "up 18% year over year"
    )


def test_text_only(monkeypatch):
    monkeypatch.setattr(
        generator,
        "call_vlm",
        lambda prompt, images=None: (
            "Growth was driven by the new enterprise product line."
        ),
    )

    retrieved = {
        "results": sample_text_chunks(),
        "image_paths": [],
    }
    query = "What drove company growth?"
    result = generate_answer(query, retrieved)

    assert "enterprise product line" in result["answer"]
    assert len(result["used_text"]) == 2
    assert result["used_images"] == []
    assert result["supporting_sentence"] == (
        "Growth was driven primarily by the new enterprise product line"
    )


def test_images_only(monkeypatch):
    monkeypatch.setattr(
        generator,
        "call_vlm",
        lambda prompt, images=None: "The chart shows quarterly performance.",
    )

    retrieved = {
        "results": [],
        "image_paths": sample_images(),
    }
    query = "Describe the supplied images."
    result = generate_answer(query, retrieved)

    assert result["answer"] == "The chart shows quarterly performance."
    assert result["used_text"] == []
    assert len(result["used_images"]) == 2
    assert result["supporting_sentence"] == "none"


def test_no_evidence_at_all():
    retrieved = {
        "results": [],
        "image_paths": [],
    }
    result = generate_answer("What is the revenue?", retrieved)

    assert result["answer"] == "I couldn't find relevant evidence in the documents."
    assert result["supporting_sentence"] == "none"
    assert result["used_text"] == []
    assert result["used_images"] == []


# ---------------- Supporting Sentence Behavior Tests ----------------


def test_supporting_sentence_stopword_only_returns_none():
    answer = "The was in on at."
    chunks = [{"text": "It was by the and on this."}]
    sentence = _find_supporting_sentence(answer, chunks)
    assert sentence == "none"

    # Common words between answer and chunk are only stopwords
    answer2 = "The cat is in the garden."
    chunks2 = [{"text": "The dog was in the car."}]
    sentence2 = _find_supporting_sentence(answer2, chunks2)
    assert sentence2 == "none"


def test_supporting_sentence_best_sentence_wins():
    answer = "Revenue reached five million dollars in 2023."
    chunks = [
        {
            "text": (
                "The market experienced volatility in 2023. "
                "Revenue reached five million dollars during the fiscal year. "
                "Operational expenses were stable."
            )
        }
    ]
    sentence = _find_supporting_sentence(answer, chunks)
    assert sentence == "Revenue reached five million dollars during the fiscal year"


def test_supporting_sentence_per_sentence_matching():
    # Answer has two sentences:
    # Sentence 1: "Alpha beta gamma." (words: alpha, beta, gamma)
    # Sentence 2: "Delta epsilon zeta." (words: delta, epsilon, zeta)
    answer = "Alpha beta gamma. Delta epsilon zeta."

    # Candidate A matches Sentence 1 on 3 words (alpha, beta, gamma).
    # Candidate B matches Sentence 1 on 1 word (alpha) and Sentence 2 on 2 words (delta, epsilon).
    # In whole-answer overlap, Candidate B (3 words: alpha, delta, epsilon) might tie or beat Candidate A.
    # But in per-sentence matching, Candidate A matches Sentence 1 with overlap 3,
    # whereas Candidate B has max per-sentence overlap of 2 (Sentence 2).
    # Therefore, Candidate A must win!
    chunks = [
        {
            "text": (
                "Alpha delta epsilon theta. "
                "Alpha beta gamma omega."
            )
        }
    ]
    sentence = _find_supporting_sentence(answer, chunks)
    assert sentence == "Alpha beta gamma omega"


# ---------------- Offline Multimodal Integration Test ----------------


def test_offline_multimodal_integration(monkeypatch):
    image_path = "test_assets/decline_chart.png"
    chunk_text = "The company reported revenue growth throughout 2023."

    retrieved = {
        "text_chunks": [
            {
                "text": chunk_text,
                "source": "annual_report.pdf",
                "source_doc": "annual_report.pdf",
                "page": 1,
            }
        ],
        "results": [
            {
                "text": chunk_text,
                "source": "annual_report.pdf",
                "source_doc": "annual_report.pdf",
                "page": 1,
            }
        ],
        "images": [
            {
                "path": image_path,
                "caption": "Revenue chart",
                "source_doc": Path(image_path).name,
            }
        ],
        "image_paths": [image_path],
        "is_sample_data": False,
    }

    generator_calls = []

    def fake_generator_vlm(prompt, images=None):
        generator_calls.append((prompt, images))
        return "The company experienced steady revenue growth across the entire period."

    verifier_calls = []

    def fake_verifier_vlm(prompt, images=None):
        verifier_calls.append((prompt, images))
        if "checking claims against ONE image" in prompt:
            return json.dumps({
                "claims": [
                    {
                        "verdict": "Contradicted",
                        "description": "The chart shows revenue falling from 120 to 52.",
                    }
                ]
            })
        elif "evidence-grounded claim verification" in prompt:
            return json.dumps({
                "claims": [
                    {
                        "claim": "The company experienced steady revenue growth across the entire period.",
                        "verdict": "Supported",
                        "evidence": [chunk_text],
                        "explanation": "Revenue growth is stated in the retrieved text.",
                    }
                ]
            })
        raise ValueError(f"Unexpected verifier prompt: {prompt[:100]}")

    monkeypatch.setattr(generator, "call_vlm", fake_generator_vlm)
    monkeypatch.setattr(verifier, "call_vlm", fake_verifier_vlm)

    query = "Did the company experience revenue growth?"
    gen_result = generator.generate_answer(query, retrieved)
    answer = gen_result["answer"]

    # First call: verify with images passed
    ver_result_with_images = verifier.verify_claims(
        answer,
        retrieved["text_chunks"],
        images=retrieved["images"],
    )

    # 1. Assert: the image path reached the generator's model call
    assert len(generator_calls) == 1
    assert generator_calls[0][1] == [image_path]

    # 2. Assert: the image path reached the verifier's image call
    image_vlm_calls = [c for c in verifier_calls if c[1] is not None and len(c[1]) > 0]
    assert len(image_vlm_calls) == 1
    assert image_vlm_calls[0][1] == [image_path]

    # 3. Assert: the final claim verdict is "Contradicted"
    assert ver_result_with_images["claims"][0]["verdict"] == "Contradicted"

    # 4. Assert: n_contradicted is 1
    assert ver_result_with_images["n_contradicted"] == 1

    # 5. Assert: the result has all contract keys
    expected_schema_keys = {
        "claims",
        "trust_score",
        "total_claims",
        "supported_claims",
        "n_contradicted",
    }
    assert set(ver_result_with_images.keys()) == expected_schema_keys
    expected_claim_keys = {
        "claim",
        "verdict",
        "evidence",
        "modality",
        "explanation",
    }
    assert set(ver_result_with_images["claims"][0].keys()) == expected_claim_keys

    # Second assertion test: if images is NOT passed to verify_claims,
    # the verdict stays "Supported" (documents the bug where the UI forgets to pass images)
    ver_result_without_images = verifier.verify_claims(
        answer,
        retrieved["text_chunks"],
    )
    assert ver_result_without_images["claims"][0]["verdict"] == "Supported"
    assert ver_result_without_images["n_contradicted"] == 0
    assert ver_result_without_images["supported_claims"] == 1
    assert ver_result_without_images["trust_score"] == 100


# ---------------- Live Test ----------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1",
    reason="calls real API; set RUN_LIVE=1",
)
def test_live_generation():
    retrieved = {
        "results": sample_text_chunks(),
        "image_paths": sample_images(),
    }
    query = "What was the company's revenue in 2023?"
    result = generate_answer(query, retrieved)
    assert result["answer"]
    assert result["supporting_sentence"] != "none"
