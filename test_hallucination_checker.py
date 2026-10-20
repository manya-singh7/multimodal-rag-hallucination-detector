"""Tests for hallucination_checker.py.

Most tests replace the VLM with a scripted fake, so they run anywhere (no GPU).
The last test runs the REAL model on the deliberate mismatch image; it only runs
on Colab with:  RUN_LIVE=1 pytest -k live -s
"""
import os
import pytest

import hallucination_checker as hc

DECLINE = "test_assets/decline_chart.png"
GROWTH = "test_assets/growth_chart.png"


class FakeVLM:
    """Routes on the TASK header in the prompt and records every call."""

    def __init__(self, text="VERDICT: NOT_FOUND\nEVIDENCE: none",
                 image="VERDICT: NOT_SHOWN\nEVIDENCE: nothing relevant",
                 caption="VERDICT: MATCHES\nEVIDENCE: ok"):
        self.replies = {"TEXT CHECK": text, "IMAGE CHECK": image, "CAPTION CHECK": caption}
        self.calls = []

    def __call__(self, prompt, images=None):
        self.calls.append((prompt, images))
        for task, reply in self.replies.items():
            if f"TASK: {task}" in prompt:
                return reply
        raise AssertionError("unexpected prompt")


def _retrieved(caption="Chart showing revenue growth", path=DECLINE):
    return {
        "text_chunks": [{"text": "The company reported its annual results.", "source_doc": "article_04.txt"}],
        "images": [{"path": path, "caption": caption, "source_doc": "article_04.txt"}],
    }


# ---------------- parsing & extraction ----------------
def test_parse_not_supported_is_not_misread_as_supported():
    assert hc.parse_claim_reply("VERDICT: NOT SUPPORTED\nEVIDENCE: x")[0] == "not_found"
    assert hc.parse_claim_reply("VERDICT: **CONTRADICTED**\nEVIDENCE: x")[0] == "contradicted"
    assert hc.parse_claim_reply("VERDICT: SUPPORTED\nEVIDENCE: yes")[0] == "supported"
    assert hc.parse_claim_reply("Hmm, hard to say.")[0] is None


def test_parse_caption_mismatch_is_not_misread_as_match():
    assert hc.parse_caption_reply("VERDICT: MISMATCH\nEVIDENCE: x")[0] == "mismatch"
    assert hc.parse_caption_reply("VERDICT: MATCHES\nEVIDENCE: x")[0] == "match"
    assert hc.parse_caption_reply("VERDICT: PARTIAL\nEVIDENCE: x")[0] == "partial"


def test_extract_claims_drops_refusals_and_fragments():
    ans = "Revenue grew every year. Profit doubled in 2023.\n- I don't know the exact figure.\nOk."
    assert hc.extract_claims(ans) == ["Revenue grew every year.", "Profit doubled in 2023."]


# ---------------- basic verdicts ----------------
def test_supported_by_text(monkeypatch):
    monkeypatch.setattr(hc, "_vlm", FakeVLM(text="VERDICT: SUPPORTED\nEVIDENCE: matches Text 1"))
    out = hc.check_hallucinations("The company reported annual results.", {"text_chunks": _retrieved()["text_chunks"]})
    assert out["claims"][0]["verdict"] == "supported"
    assert out["overall_score"] == 1.0


def test_unsupported_is_distinct_from_contradicted(monkeypatch):
    monkeypatch.setattr(hc, "_vlm", FakeVLM())
    out = hc.check_hallucinations("The CEO resigned in March.", _retrieved(caption="A chart"))
    assert out["claims"][0]["verdict"] == "unsupported"
    assert out["n_contradicted"] == 0
    assert out["overall_score"] == 0.0


def test_unparseable_reply_is_unclear(monkeypatch):
    monkeypatch.setattr(hc, "_vlm", FakeVLM(text="no idea", image="shrug"))
    out = hc.check_hallucinations("The CEO resigned in March.", _retrieved())
    assert out["claims"][0]["verdict"] == "unclear"


# ---------------- THE contradiction test case ----------------
def test_caption_says_growth_image_shows_decline_is_flagged_as_contradicted(monkeypatch):
    """Text/caption supports the claim, the image shows the opposite."""
    fake = FakeVLM(
        text="VERDICT: SUPPORTED\nEVIDENCE: Caption of image 1 says revenue growth.",
        image="VERDICT: CONTRADICTED\nEVIDENCE: The line falls from 120 to 52 over 2019-2023.",
        caption="VERDICT: MISMATCH\nEVIDENCE: The chart shows revenue declining, not growing.",
    )
    monkeypatch.setattr(hc, "_vlm", fake)

    out = hc.check_hallucinations("The chart shows revenue growth.", _retrieved())
    claim = out["claims"][0]

    # flagged specifically as a contradiction, NOT generic "unsupported"
    assert claim["verdict"] == "contradicted"
    assert claim["verdict"] != "unsupported"
    assert claim["modality"] == "image"
    assert "Image 1" in claim["evidence"] and "contradicts" in claim["evidence"]
    assert out["n_contradicted"] == 1
    assert out["overall_score"] == 0.0

    # the bad caption itself is surfaced
    assert len(out["caption_conflicts"]) == 1
    assert out["caption_conflicts"][0]["caption"] == "Chart showing revenue growth"


def test_image_check_never_sees_the_caption(monkeypatch):
    """Design invariant: the image is judged on pixels only, so a wrong caption can't leak in."""
    fake = FakeVLM()
    monkeypatch.setattr(hc, "_vlm", fake)
    hc.check_hallucinations("The chart shows revenue growth.", _retrieved(caption="SECRET-CAPTION-TEXT"),
                            check_caption_consistency=False)
    image_calls = [(p, imgs) for p, imgs in fake.calls if "TASK: IMAGE CHECK" in p]
    assert image_calls, "image check should have run"
    for prompt, imgs in image_calls:
        assert "SECRET-CAPTION-TEXT" not in prompt
        assert imgs == [DECLINE]


def test_control_matching_caption_is_supported_with_no_conflict(monkeypatch):
    fake = FakeVLM(
        text="VERDICT: SUPPORTED\nEVIDENCE: caption says growth",
        image="VERDICT: SUPPORTED\nEVIDENCE: The line rises from 52 to 120.",
        caption="VERDICT: MATCHES\nEVIDENCE: rising line",
    )
    monkeypatch.setattr(hc, "_vlm", fake)
    out = hc.check_hallucinations("The chart shows revenue growth.", _retrieved(path=GROWTH))
    assert out["claims"][0]["verdict"] == "supported"
    assert out["claims"][0]["modality"] == "text+image"
    assert out["caption_conflicts"] == []
    assert out["overall_score"] == 1.0


# ---------------- live run on the real model (Colab only) ----------------
@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="needs GPU/Colab; set RUN_LIVE=1")
def test_live_real_model_flags_decline_chart():
    assert os.path.exists(DECLINE), "run make_test_assets.py first"
    out = hc.check_hallucinations("The chart shows revenue growth.", _retrieved())
    print(out)
    assert out["claims"][0]["verdict"] == "contradicted"
