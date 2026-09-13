# Multimodal RAG + Hallucination Detection

A retrieval-augmented generation (RAG) system that answers questions using both text and images from a document corpus, with a built-in verification layer that flags when a generated answer isn't actually supported by its sources.

## Live Demo

Once the notebook is running and Gradio is launched, it produces a public link. That link goes here:

**Demo link:** _add the link here once the demo is live_

## Overview

Standard AI chatbots answer purely from memory, which means they can confidently state things that aren't true (they "hallucinate"). This project reduces and detects that problem in two ways.

1. **Retrieval-augmented generation.** Instead of answering from memory alone, the system retrieves relevant text passages and images from a document corpus before generating an answer, so the response is grounded in real source material.
2. **Multimodal hallucination detection.** Most hallucination-checking approaches only verify text against text. This project also checks whether generated claims are consistent with retrieved images, and can catch cases where a claim contradicts what an image actually shows.

## Features

- **Grounded generation.** Answers are produced only from retrieved text and images, and the model is instructed to say it doesn't know rather than guess when the context isn't enough to answer.
- **Source snippet highlighting.** Each answer is paired with the exact source sentence that supports it, found by matching overlapping words against the retrieved text, rather than relying on the model to report its own citation.
- **Cross-modal hallucination detection.** Generated claims are checked against both retrieved text and retrieved images, so the system can catch a claim that contradicts what an image actually shows, not just a claim with no textual backing.
- **Color-coded claim verdicts.** In the demo UI, each claim in an answer is highlighted to show whether it's supported or unsupported, instead of showing one undifferentiated block of text.
- **Hallucination score badge.** Each answer carries a simple summary of how well-supported it is overall, based on the underlying claim-level verdicts.
- **Trust score tracking.** Verification scores across a set of test questions are logged and charted, so there's a picture of where the system performs well and where it struggles, rather than relying on a handful of hand-picked examples.

## Architecture

The pipeline has four stages.

1. **Retrieve.** Given a question, search the document corpus and return the most relevant text chunks and images.
2. **Augment.** Build a labeled prompt from the retrieved text and images, with source attribution, instructing the model to answer only from the given context.
3. **Generate.** A vision-language model produces an answer grounded in the retrieved context, along with the specific source sentence that supports it.
4. **Verify.** The generated answer is checked claim by claim against the retrieved sources, flagging anything unsupported or contradicted.

```
User question
     |
     v
retrieve()  ->  text_chunks[] + images[]
     |
     v
generate_answer()  ->  answer + supporting_sentence
     |
     v
check_hallucinations()  ->  per-claim verdicts + overall_score
     |
     v
Gradio UI  ->  color-coded, annotated answer
```

## Tech Stack

- **Corpus:** a small set of real articles paired with images, used as the system's searchable knowledge base
- **Text embeddings:** `sentence-transformers` (all-MiniLM-L6-v2)
- **Image embeddings:** CLIP / `open_clip` (ViT-B-32)
- **Retrieval:** cosine similarity over embedded text and image vectors, run as two parallel searches (one per modality), since MiniLM and CLIP embeddings aren't compatible with each other
- **Generation and verification model:** Qwen2-VL-7B-Instruct, 4-bit quantized, run on a free Google Colab T4 GPU
- **Demo UI:** Gradio (`gr.Blocks()`)
- **Language:** Python

## Repository Structure

Files are organized by pipeline stage rather than listed individually, since new files will be added as the project grows.

- **Root-level `.py` files** hold the core pipeline code. For example, `call_vlm.py` loads the VLM and exposes `call_vlm(prompt, images)`, and `generator.py` builds prompts from retrieved content and generates grounded answers. Retrieval and hallucination-checking modules follow the same pattern.
- **`test_*.py` files** hold test cases for the matching module, covering text-only, image-only, and combined scenarios where relevant.
- **`notebooks/`** holds the Colab notebooks used to develop and test GPU-dependent code (model loading, generation, verification) before it's finalized into the `.py` files above.
- **`data/`** holds the document corpus (articles, images, metadata) used as the system's searchable knowledge base, where applicable.

## How to Run

The generation and verification models need a GPU, so development, testing, and the live demo all run on Google Colab's free tier rather than locally.

1. Open the relevant notebook in `notebooks/` on Google Colab.
2. Set the runtime to a T4 GPU under Runtime > Change runtime type > T4 GPU.
3. Run all cells from top to bottom. The first run downloads and loads the model, which can take several minutes.
4. Once the model is loaded and the pipeline code (retrieval, generation, verification) is in place, run the Gradio launch cell at the end of the notebook. This starts the demo and prints a live public link that opens the interface in any browser. From that point on there's no need to interact with the notebook directly.
5. Open the printed link to use the demo: ask a question, view the retrieved sources, and see the generated answer along with its hallucination verdicts. Paste that link into the Live Demo section at the top of this README so anyone opening the repo can find it right away.

Colab's free tier disconnects after periods of inactivity, which clears the loaded model and closes the Gradio link. When that happens, the setup cells and the Gradio launch cell need to be run again. This is a known limitation of using free infrastructure rather than a paid, always-on server.

## Known Limitations

- The generation model and the verification step currently use the same underlying model, which means the hallucination checker could be unreliable in some of the same ways the generator is. This is a known weakness of self-verification approaches in general, not something unique to this project.
- Source-sentence attribution is implemented as a simple word-overlap heuristic rather than having the model report its own citation, after testing showed the model was inconsistent at following that instruction reliably.
- The system runs on free-tier hardware (a Colab T4 GPU, with a 4-bit quantized 7B model), which trades some accuracy and instruction-following reliability for being fully free to run.
- The corpus is small and hand-curated for demo purposes, not a large-scale or production-representative dataset.
