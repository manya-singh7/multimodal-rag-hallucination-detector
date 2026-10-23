# Multimodal RAG and Hallucination Detector

A multimodal retrieval-augmented generation (RAG) system with an evidence-grounded claim verification layer. The system answers user questions using retrieved text passages and figures from research papers, then verifies the generated answer against both text and image evidence to detect hallucinations.

## Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your Gemini API key in a `.env` file:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. Launch the Gradio demo:
   ```bash
   python demo_ui.py
   ```

The UI highlights each verified claim inline in green for Supported claims and red for Contradicted or Unsupported claims. It also displays an overall categorical trust badge: green (Verified, 100% supported with 0 contradictions), yellow (Partially Supported, between 0% and 100% supported with 0 contradictions), red (Unsupported / Contradicted, if any claim is contradicted or unsupported), or grey (No verifiable claims, when there are no claims to check).

## Overview

Large language and vision models often sound confident even when they make factual mistakes. When answering purely from memory, they can easily hallucinate details. Retrieval-augmented generation reduces this risk by giving the model relevant documents to read before formulating an answer.

However, standard RAG systems can still produce hallucinations, especially when synthesizing complex information across multiple documents or modalities. Most hallucination checkers only compare text against text. This project adds a verification stage that checks generated claims against both retrieved text passages and the raw pixels of retrieved figures. If a figure shows the opposite of what a claim states, the system flags the contradiction immediately.

## Features

* Grounded generation: Answers are produced directly from retrieved text chunks and figures, with explicit instructions not to extrapolate beyond the provided evidence.
* Heuristic source attribution: The system highlights the single best supporting sentence from the sources using code-based word-overlap matching with stopwords removed, rather than asking the model to cite itself.
* Multimodal claim verification: Each answer is split into distinct claims. Claims are verified against retrieved text excerpts and against retrieved images.
* Visual contradiction override: Figures are evaluated purely on image pixels with captions hidden. If an image contradicts a claim, that claim is marked Contradicted even if retrieved text appeared to support it.
* Trust score tracking: Each verification calculates a trust score representing the percent of claims supported by the evidence, not accuracy.
* Interactive Gradio demo: A browser interface that displays retrieved text sources, figures, claim cards with evidence and explanations, and an overall trust score summary.

## Architecture

The system operates as a four-stage pipeline:

1. Retrieve: Given a user query, the retriever runs hybrid search combining BM25 keyword matching with dense MiniLM sentence embeddings. It selects the top text chunks and their associated figures from the processed corpus.
2. Generate: The generator constructs a structured evidence prompt containing the retrieved text and passes any accompanying images to Gemini via call_vlm.py. It produces a grounded answer and identifies the strongest supporting source sentence through code-based word-overlap matching.
3. Verify: The verifier breaks the answer into individual claims. It checks every claim against the retrieved text, then inspects image pixels independently with captions hidden. An image contradiction overrides text support, producing a verdict of Supported, Contradicted, or Insufficient Evidence for every claim, alongside an overall trust score.
4. Demo: The Gradio web interface coordinates the full workflow, displaying retrieved sources, generated answers with candidate supporting text, individual claim cards with evidence, and overall trust score metrics.

## Tech Stack

* Vision-Language Model: Google Gemini API accessed via the google-genai SDK in call_vlm.py
* Text Retrieval: Hybrid retrieval combining BM25 keyword scoring with sentence-transformers (all-MiniLM-L6-v2) dense embeddings
* Document Corpus: 2 arXiv research papers with extracted figures and paragraph chunks
* Image Processing: Pillow and PyMuPDF for handling figures extracted from paper PDFs
* Verification Engine: Multi-stage prompt verification against text passages and raw image pixels
* Evaluation: Matplotlib for trust score charts and standard CSV logging
* Web Application: Gradio web dashboard

## Repository Structure

The codebase is organized by pipeline category:

* Core pipeline modules: Files at the repository root implement each stage of the pipeline, including model access, document retrieval, answer generation, and claim verification.
* Test suite: Test files verify generator behavior and cross-modal claim verification offline using mocks, with optional live API tests.
* Evaluation tools: Standalone evaluation scripts run test queries end to end, export verification metrics to CSV, and plot trust score charts.
* Document processing: Extraction utilities process raw PDF research papers into structured paragraph chunks and extracted figures.
* Demo interface: A Gradio application provides an interactive web dashboard for user queries, source inspection, and claim verification.
* Corpus storage: Processed text chunks from the paper corpus and associated figures are stored under the data directory.

## How to Run

1. Clone the repository and enter the directory:
   cd multimodal-rag-hallucination-detector

2. Install dependencies:
   pip install -r requirements.txt

3. Configure your Gemini API key:
   Create a .env file in the repository root (never committed to git):
   GEMINI_API_KEY=your_gemini_api_key_here
   Optionally specify a model (defaults to gemini-3.1-flash-lite):
   GEMINI_MODEL=gemini-3.1-flash-lite

4. Run tests offline without an API key:
   pytest

5. Evaluate test queries:
   python trust_chart.py --dry-run
   To run with the live Gemini API, omit the dry-run flag:
   python trust_chart.py

6. Launch the Gradio demo:
   python demo_ui.py

## Known Limitations

* Self-verification model bias: The same underlying Gemini model is used for both generating answers and verifying claims. While structured prompts and independent pixel checks mitigate this, model biases can still carry across stages.
* Heuristic sentence attribution: The supporting sentence is selected via a word-overlap heuristic rather than deep syntactic or semantic entailment.
* The supporting sentence only points to text found by word overlap, and claims that come from images are checked by the verifier, not by this line.
* The demo needs a Gemini API key and is subject to rate limits, and real model behavior has been tested only partly.
* Small corpus: The current knowledge base contains only 2 arXiv research papers, limiting the breadth of questions that have retrieved evidence.
* Extracted PDF figures have no captions: Figures extracted from the PDFs do not retain their original captions in the image files, so image verification relies entirely on what is visible within the image pixels.
* Trust score represents support: The trust score is the percent of claims supported, not accuracy.
