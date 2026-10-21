"""Tests for the multimodal checks in verifier.py.

(hallucination_checker.py was merged into verifier.py and removed; this file
keeps its name and now covers verify_claims() with images.)

Most tests replace call_vlm with a scripted fake, so they run without an API
key or network. The last test calls the REAL model on the deliberate mismatch
image; it only runs when GEMINI_API_KEY is set and you ask for it:
    RUN_LIVE=1 pytest -k live -s
"""
import json
import os

import pytest

import verifier

DECLINE = "test_assets/decline_chart.png"
GROWTH = "test_assets/growth_chart.png"

SCHEMA_KEYS = {"claims", "trust_score", "total_claims", "supported_claims", "n_contradicted"}
CLAIM_KEYS = {"claim", "verdict", "evidence", "modality", "explanation"}

GROWTH_TEXT = "Revenue growth was reported across the whole period."
GROWTH_CLAIM = "Revenue grew every year from 2019 to 2023."


def text_reply(verdict="Supported", quote=GROWTH_TEXT, explanation="Matches the text."):
    return json.dumps({"claims": [{
        "claim": "x", "verdict": verdict,
        "evidence": [quote] if quote else [], "explanation": explanation,
    }]})


def image_reply(verdict="Contradicted", description="The line falls from 120 to 52."):
    return json.dumps({"claims": [{"verdict": verdict, "description": description}]})


class FakeVLM:
    """Routes on the prompt header and records every call."""

    def __init__(self, text=None, image=None):
        self.text = text if text is not None else text_reply()
        self.image = image if image is not None else image_reply("Not Shown", "")
        self.calls = []

    def __call__(self, prompt, images=None):
        self.calls.append((prompt, images))
        if "evidence-grounded claim verification" in prompt:
            reply = self.text
        elif "checking claims against ONE image" in prompt:
            reply = self.image
        else:
            raise AssertionError("unexpected prompt")
        if isinstance(reply, Exception):
            raise reply
        return reply

    def image_calls(self):
        return [c for c in self.calls if "checking claims against ONE image" in c[0]]

    def text_calls(self):
        return [c for c in self.calls if "evidence-grounded claim verification" in c[0]]


def chunks(text=GROWTH_TEXT):
    return [{"text": text, "source_doc": "article_04.txt", "page": 1}]


def images(path=DECLINE, caption="Chart showing revenue growth"):
    return [{"path": path, "caption": caption, "source_doc": "article_04.txt"}]


def run(monkeypatch, fake, claim=GROWTH_CLAIM, text_chunks=None, imgs=None):
    monkeypatch.setattr(verifier, "call_vlm", fake)
    return verifier.verify_claims(
        claim, chunks() if text_chunks is None else text_chunks, imgs
    )


# ---------------- schema & backwards compatibility ----------------
def test_text_only_call_still_works_and_makes_no_image_calls(monkeypatch):
    fake = FakeVLM()
    out = run(monkeypatch, fake)  # images omitted: same call shape demo_ui.py uses today
    assert set(out) == SCHEMA_KEYS
    assert set(out["claims"][0]) == CLAIM_KEYS
    assert out["claims"][0]["verdict"] == "Supported"
    assert out["claims"][0]["modality"] == "text"
    assert out["trust_score"] == 100 and out["n_contradicted"] == 0
    assert fake.image_calls() == []


def test_no_claims_returns_schema(monkeypatch):
    out = run(monkeypatch, FakeVLM(), claim="")
    assert set(out) == SCHEMA_KEYS
    assert out["total_claims"] == 0 and out["n_contradicted"] == 0 and out["trust_score"] == 0


def test_fabricated_quote_is_still_downgraded(monkeypatch):
    fake = FakeVLM(text=text_reply("Supported", quote="A sentence that is not in the sources."))
    out = run(monkeypatch, fake)
    assert out["claims"][0]["verdict"] == "Insufficient Evidence"


# ---------------- THE contradiction override ----------------
def test_text_supports_but_image_shows_decline_is_contradicted(monkeypatch):
    fake = FakeVLM(image=image_reply("Contradicted", "The line falls from 120 to 52 over 2019-2023."))
    out = run(monkeypatch, fake, imgs=images(DECLINE))
    claim = out["claims"][0]

    assert claim["verdict"] == "Contradicted"
    assert claim["modality"] == "image"
    assert any("falls from 120 to 52" in e for e in claim["evidence"])
    assert "text supports the claim" in claim["explanation"]
    assert out["n_contradicted"] == 1
    assert out["supported_claims"] == 0 and out["trust_score"] == 0


def test_control_matching_chart_is_supported_by_text_and_image(monkeypatch):
    fake = FakeVLM(image=image_reply("Supported", "The line rises from 52 to 120."))
    out = run(monkeypatch, fake, imgs=images(GROWTH))
    claim = out["claims"][0]
    assert claim["verdict"] == "Supported"
    assert claim["modality"] == "text+image"
    assert out["trust_score"] == 100 and out["n_contradicted"] == 0


def test_image_alone_can_support_a_claim(monkeypatch):
    fake = FakeVLM(text=text_reply("Insufficient Evidence", quote=None),
                   image=image_reply("Supported", "The line rises from 52 to 120."))
    out = run(monkeypatch, fake, imgs=images(GROWTH))
    assert out["claims"][0]["verdict"] == "Supported"
    assert out["claims"][0]["modality"] == "image"


def test_unrelated_claim_stays_insufficient_not_contradicted(monkeypatch):
    fake = FakeVLM(text=text_reply("Insufficient Evidence", quote=None),
                   image=image_reply("Not Shown", "A chart of revenue."))
    out = run(monkeypatch, fake, claim="The CEO resigned in March.", imgs=images(GROWTH))
    claim = out["claims"][0]
    assert claim["verdict"] == "Insufficient Evidence"
    assert claim["modality"] == "none"
    assert out["n_contradicted"] == 0


def test_images_work_even_without_any_text_chunks(monkeypatch):
    fake = FakeVLM(image=image_reply("Contradicted", "The line falls."))
    out = run(monkeypatch, fake, text_chunks=[], imgs=images(DECLINE))
    assert out["claims"][0]["verdict"] == "Contradicted"
    assert fake.text_calls() == []  # nothing to check text against, so no text call


# ---------------- image handling rules ----------------
def test_image_check_never_sees_the_caption(monkeypatch):
    fake = FakeVLM()
    run(monkeypatch, fake, imgs=images(DECLINE, caption="SECRET-CAPTION-TEXT"))
    assert fake.image_calls(), "image check should have run"
    for prompt, sent in fake.image_calls():
        assert "SECRET-CAPTION-TEXT" not in prompt
        assert sent == [DECLINE]


def test_at_most_max_images_are_sent(monkeypatch):
    fake = FakeVLM()
    run(monkeypatch, fake, imgs=images(DECLINE) * 5)
    assert len(fake.image_calls()) == verifier.MAX_IMAGES


def test_missing_image_file_is_skipped(monkeypatch):
    fake = FakeVLM()
    run(monkeypatch, fake, imgs=images("test_assets/does_not_exist.png"))
    assert fake.image_calls() == []


def test_plain_path_strings_are_accepted(monkeypatch):
    fake = FakeVLM(image=image_reply("Contradicted", "falls"))
    out = run(monkeypatch, fake, imgs=[DECLINE])
    assert out["claims"][0]["verdict"] == "Contradicted"


def test_failed_image_call_keeps_the_text_verdict(monkeypatch):
    fake = FakeVLM(image=RuntimeError("API unavailable"))
    out = run(monkeypatch, fake, imgs=images(DECLINE))
    assert out["claims"][0]["verdict"] == "Supported"
    assert out["claims"][0]["modality"] == "text"


def test_unparseable_image_reply_keeps_the_text_verdict(monkeypatch):
    fake = FakeVLM(image="I think the chart looks fine.")
    out = run(monkeypatch, fake, imgs=images(DECLINE))
    assert out["claims"][0]["verdict"] == "Supported"


# ---------------- claim splitting ----------------
def test_split_claims_keeps_abbreviations_together():
    answer = ("Smith et al. This method improves accuracy (see Fig. 2). "
              "It also reduces cost, e.g. by 10%.")
    claims = verifier.split_claims(answer)
    assert len(claims) == 2
    assert "et al. This method" in claims[0] and "Fig. 2" in claims[0]
    assert "e.g. by 10%" in claims[1]


def test_split_claims_still_splits_normal_sentences_and_keeps_numbers():
    assert verifier.split_claims("Revenue grew. Profit fell.") == ["Revenue grew.", "Profit fell."]
    assert verifier.split_claims("The price was $3.50 in 2023. It rose later.") == [
        "The price was $3.50 in 2023.", "It rose later."]


# ---------------- live run on the real model ----------------
@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="calls the real API; set RUN_LIVE=1")
def test_live_real_model_flags_decline_chart():
    assert os.path.exists(DECLINE), "run make_test_assets.py first"
    out = verifier.verify_claims(GROWTH_CLAIM, chunks(), images(DECLINE))
    print(json.dumps(out, indent=2))
    assert out["claims"][0]["verdict"] == "Contradicted"
