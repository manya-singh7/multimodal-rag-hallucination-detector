"""Tests for the multimodal checks in verifier.py (and the two small
call_vlm.py changes that support them).

(hallucination_checker.py was merged into verifier.py and removed; this file
keeps its name and covers verify_claims() with images.)

Most tests replace call_vlm with a scripted fake, so they run without an API
key or network. The last test calls the REAL model on the deliberate mismatch
image; it only runs when GEMINI_API_KEY is set and you ask for it:
    RUN_LIVE=1 pytest -k live -s
"""
import json
import os

import pytest

import call_vlm as call_vlm_module
import verifier

DECLINE = "test_assets/decline_chart.png"
GROWTH = "test_assets/growth_chart.png"

SCHEMA_KEYS = {
    "claims", "trust_score", "total_claims", "supported_claims",
    "n_contradicted", "failed_claims",
}
CLAIM_KEYS = {"claim", "verdict", "evidence", "modality", "explanation"}

GROWTH_TEXT = "Revenue growth was reported across the whole period."
GROWTH_CLAIM = "Revenue grew every year from 2019 to 2023."


def text_reply(verdict="Supported", quote=GROWTH_TEXT, explanation="Matches the text."):
    return json.dumps({"claims": [{
        "claim": "x", "verdict": verdict,
        "evidence": [quote] if quote else [], "explanation": explanation,
    }]})


def multi_text_reply(*items):
    """One (verdict, quote) pair per claim the model answers."""
    return json.dumps({"claims": [{
        "claim": "x", "verdict": verdict,
        "evidence": [quote] if quote else [], "explanation": "e",
    } for verdict, quote in items]})


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
    out = run(monkeypatch, fake)  # images omitted: the call shape older code uses
    assert set(out) == SCHEMA_KEYS
    assert set(out["claims"][0]) == CLAIM_KEYS
    assert out["claims"][0]["verdict"] == "Supported"
    assert out["claims"][0]["modality"] == "text"
    assert out["trust_score"] == 100 and out["n_contradicted"] == 0
    assert out["failed_claims"] == 0
    assert fake.image_calls() == []


def test_no_claims_returns_schema(monkeypatch):
    out = run(monkeypatch, FakeVLM(), claim="")
    assert set(out) == SCHEMA_KEYS
    assert out["total_claims"] == 0 and out["n_contradicted"] == 0 and out["trust_score"] == 0
    assert out["failed_claims"] == 0


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


def test_only_the_two_best_ranked_images_are_checked_in_order(monkeypatch):
    ranked = [
        {"path": DECLINE, "caption": "", "source_doc": "rank1"},
        {"path": GROWTH, "caption": "", "source_doc": "rank2"},
        {"path": DECLINE, "caption": "", "source_doc": "rank3"},
    ]
    fake = FakeVLM()
    run(monkeypatch, fake, imgs=ranked)
    assert verifier.MAX_IMAGES == 2
    assert [sent for _, sent in fake.image_calls()] == [[DECLINE], [GROWTH]]


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


# ---------------- failed checks are not verdicts ----------------
def test_failed_call_is_reported_as_verification_failed(monkeypatch):
    fake = FakeVLM(text=RuntimeError("boom"))
    out = run(monkeypatch, fake, claim="Revenue grew. Profit doubled.")
    assert [c["verdict"] for c in out["claims"]] == ["Verification Failed"] * 2
    assert out["failed_claims"] == 2 and out["total_claims"] == 2
    assert out["supported_claims"] == 0 and out["trust_score"] == 0
    assert "not scored" in out["claims"][0]["explanation"]


def test_failed_claims_are_left_out_of_the_score(monkeypatch):
    # The model answers claim 1 and skips claim 2.
    fake = FakeVLM(text=multi_text_reply(("Supported", GROWTH_TEXT)))
    out = run(monkeypatch, fake, claim="Revenue grew every year. Profit doubled in 2022.")
    assert [c["verdict"] for c in out["claims"]] == ["Supported", "Verification Failed"]
    assert out["total_claims"] == 2 and out["failed_claims"] == 1
    assert out["supported_claims"] == 1
    assert out["trust_score"] == 100  # 1 of the 1 claim that could be checked


def test_an_invalid_verdict_counts_as_failed_not_as_insufficient(monkeypatch):
    fake = FakeVLM(text=multi_text_reply(("Maybe", None)))
    out = run(monkeypatch, fake)
    assert out["claims"][0]["verdict"] == "Verification Failed"
    assert out["failed_claims"] == 1


def test_an_unreadable_reply_counts_as_failed(monkeypatch):
    out = run(monkeypatch, FakeVLM(text="Sorry, I could not do that."))
    assert out["claims"][0]["verdict"] == "Verification Failed"


def test_an_image_can_still_settle_a_claim_whose_text_check_failed(monkeypatch):
    fake = FakeVLM(text=RuntimeError("ReadError"),
                   image=image_reply("Contradicted", "The line falls."))
    out = run(monkeypatch, fake, imgs=images(DECLINE))
    assert out["claims"][0]["verdict"] == "Contradicted"
    assert out["failed_claims"] == 0 and out["n_contradicted"] == 1


def test_score_math_with_one_failed_claim():
    out = verifier._build_result([
        {"verdict": "Supported"},
        {"verdict": "Insufficient Evidence"},
        {"verdict": "Verification Failed"},
    ])
    assert out["trust_score"] == 50  # 1 supported of the 2 that were checked
    assert out["total_claims"] == 3 and out["failed_claims"] == 1


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


def test_echoed_source_labels_and_answer_prefix_are_removed():
    answer = ("Answer: Retrieval improves accuracy [Text 1 | source: paper1.pdf]. "
              "It also lowers cost (Image 2) [3].")
    assert verifier.split_claims(answer) == [
        "Retrieval improves accuracy.", "It also lowers cost."]


def test_intro_lines_and_one_word_fragments_are_not_claims():
    assert verifier.split_claims("The key points are:\nRetrieval helps accuracy.") == [
        "Retrieval helps accuracy."]
    assert verifier.split_claims("Yes. Retrieval improves accuracy.") == [
        "Retrieval improves accuracy."]


def test_a_list_still_becomes_one_claim_with_its_heading():
    answer = "Key results:\n- Recall rises\n- Cost drops"
    assert verifier.split_claims(answer) == ["Key results: Recall rises; Cost drops"]


def test_a_sentence_after_a_closing_quote_or_bracket_starts_a_new_claim():
    answer = 'The authors call it "robust." The method is fast (see Fig. 2). It scales well.'
    assert len(verifier.split_claims(answer)) == 3


# ---------------- meta-sentences are not claims ----------------
def test_meta_sentences_about_the_evidence_are_not_claims():
    answer = ("The evidence is insufficient to answer this question. "
              "The context does not mention the dataset size. "
              "I don't know. "
              "There is no information about training cost in the provided context. "
              "The answer cannot be determined from the retrieved evidence.")
    assert verifier.split_claims(answer) == []


def test_real_and_mixed_sentences_are_kept():
    answer = ("The paper does not report accuracy. "
              "The evidence does not specify the dataset size, but the method improves recall by 5%. "
              "The sources are limited to two papers.")
    assert len(verifier.split_claims(answer)) == 3


def test_an_answer_of_only_meta_sentences_makes_no_model_calls(monkeypatch):
    fake = FakeVLM()
    out = run(monkeypatch, fake, claim="The evidence is insufficient to answer this. I don't know.",
              imgs=images(DECLINE))
    assert out["total_claims"] == 0 and out["trust_score"] == 0
    assert out["failed_claims"] == 0
    assert fake.calls == []


def test_more_meta_sentences_are_not_claims():
    for sentence in (
        "The text and the image do not mention quantum computing.",
        "The evidence provided is insufficient to answer the question.",
        "The provided text does not list the specific domains.",
        "The provided evidence is contradictory.",
    ):
        assert verifier.split_claims(sentence) == [], sentence


def test_real_sentences_that_look_similar_are_kept():
    for sentence in (
        "The provided image shows a consistent decline.",
        "The sources are limited to two papers.",
        "The paper does not report accuracy.",
    ):
        assert verifier.split_claims(sentence) == [sentence], sentence


# ---------------- a claim about the image is judged by the image ----------------
def test_claim_about_the_image_is_supported_when_only_the_text_disagrees(monkeypatch):
    claim = "The provided image shows a consistent decline in revenue."
    fake = FakeVLM(text=text_reply("Contradicted", quote=GROWTH_TEXT),
                   image=image_reply("Supported", "The line falls from 120 to 52."))
    out = run(monkeypatch, fake, claim=claim, imgs=images(DECLINE))
    item = out["claims"][0]
    assert item["verdict"] == "Supported"
    assert item["modality"] == "image"
    assert GROWTH_TEXT not in item["evidence"]
    assert out["trust_score"] == 100 and out["n_contradicted"] == 0


def test_text_contradiction_still_wins_for_a_claim_not_about_the_image(monkeypatch):
    fake = FakeVLM(text=text_reply("Contradicted", quote=GROWTH_TEXT),
                   image=image_reply("Supported", "The line falls."))
    out = run(monkeypatch, fake, claim="Revenue fell every year.", imgs=images(DECLINE))
    assert out["claims"][0]["verdict"] == "Contradicted"


# ---------------- call_vlm.py: retries and temperature ----------------
class TimeoutException(Exception):
    pass


class ReadTimeout(TimeoutException):  # like httpx.ReadTimeout
    pass


class ReadError(Exception):  # like httpx.ReadError, whose message is often empty
    pass


def test_network_errors_with_empty_messages_are_retryable():
    retryable = call_vlm_module._is_retryable_error
    assert retryable(ReadError())
    assert retryable(ReadTimeout())
    assert retryable(RuntimeError("503 UNAVAILABLE"))
    assert not retryable(ValueError("400 invalid argument"))
    assert not retryable(RuntimeError("Set GEMINI_API_KEY in your project's .env file."))


def test_call_vlm_retries_a_read_error_and_uses_temperature_zero(monkeypatch):
    configs = []

    class FakeResponse:
        text = " ok "

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            configs.append(config)
            if len(configs) == 1:
                raise ReadError()
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(call_vlm_module, "API_KEY", "test-key")
    monkeypatch.setattr(call_vlm_module.genai, "Client", FakeClient)
    monkeypatch.setattr(call_vlm_module.time, "sleep", lambda seconds: None)

    assert call_vlm_module.call_vlm("hello") == "ok"
    assert len(configs) == 2
    assert all(config == {"temperature": 0} for config in configs)


# ---------------- live run on the real model ----------------
@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="calls the real API; set RUN_LIVE=1")
def test_live_real_model_flags_decline_chart():
    assert os.path.exists(DECLINE), "run make_test_assets.py first"
    out = verifier.verify_claims(GROWTH_CLAIM, chunks(), images(DECLINE))
    print(json.dumps(out, indent=2))
    assert out["claims"][0]["verdict"] == "Contradicted"
