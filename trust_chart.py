"""Evaluate trust score across test queries and generate a summary chart and CSV.

Pipeline per query:
1. retrieve() from retriever.py (hybrid BM25 + semantic retrieval)
2. generate_answer() from generator.py (grounded answer generation)
3. verify_claims() from verifier.py (cross-modal claim-level verification)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generator
import retriever
import verifier

# Try importing standard test queries from demo_ui; fallback to 5 built-in questions.
try:
    from demo_ui import TEST_QUERIES
except Exception:
    TEST_QUERIES = [
        "How does retrieval augmented generation use evidence?",
        "What is multimodal hallucination?",
        "How can retrieved evidence help verify an answer?",
        "What are the limitations of image-based question answering?",
        "Why should AI answers be checked against sources?",
    ]


class DryRunVLM:
    """Mock VLM for --dry-run execution without an API key or GPU."""

    def __init__(self):
        self._query_index = 0

    def __call__(self, prompt: str, images: list = None) -> str:
        # Verifier: text verification prompt
        if "evidence-grounded claim verification" in prompt:
            claims = self._extract_claims(prompt)
            evidence_quotes = self._extract_evidence(prompt)
            quote = evidence_quotes[0] if evidence_quotes else ""

            results = []
            for idx, claim in enumerate(claims):
                # Introduce slight variation for demonstration in charts
                # (e.g. 100%, 50%, 100%, 100%, 50%)
                is_supported = (self._query_index + idx) % 3 != 1
                if is_supported and quote:
                    verdict = "Supported"
                    evidence = [quote]
                    explanation = "Directly supported by retrieved source text."
                else:
                    verdict = "Insufficient Evidence"
                    evidence = []
                    explanation = "Context does not provide sufficient evidence."

                results.append({
                    "claim": claim,
                    "verdict": verdict,
                    "evidence": evidence,
                    "explanation": explanation,
                })

            return json.dumps({"claims": results})

        # Verifier: image verification prompt
        elif "checking claims against ONE image" in prompt:
            claims = self._extract_claims(prompt)
            return json.dumps({
                "claims": [
                    {
                        "verdict": "Not Shown",
                        "description": "Image does not depict this specific claim.",
                    }
                    for _ in claims
                ]
            })

        # Generator: grounded answer generation prompt
        else:
            self._query_index += 1
            # Produce 2 distinct claims that verifier.split_claims can separate
            return (
                "Retrieval augmented generation incorporates retrieved documents to ground answers. "
                "Document evidence is evaluated to verify claim accuracy."
            )

    @staticmethod
    def _extract_claims(prompt: str) -> list[str]:
        match = re.search(r"Claims:\s*(\[\s*.*?\s*\])", prompt, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    return [str(c) for c in parsed]
            except Exception:
                pass
        return ["Claim 1: Context is grounded", "Claim 2: Evidence is evaluated"]

    @staticmethod
    def _extract_evidence(prompt: str) -> list[str]:
        # Evidence text lines formatted as 'Text: <chunk text>' in verifier prompt
        quotes = []
        for match in re.finditer(r"Text:\s*(.+?)(?=\n\n|\n\[Evidence|\nRequired|\Z)", prompt, re.DOTALL):
            text = match.group(1).strip()
            # Use first clean sentence or snippet
            first_sentence = text.split(". ")[0].strip()
            if first_sentence:
                quotes.append(first_sentence)
        return quotes


def load_queries(query_file: str | None = None) -> list[str]:
    """Load queries from a file or fall back to TEST_QUERIES."""
    if not query_file:
        return list(TEST_QUERIES)

    path = Path(query_file)
    if not path.is_file():
        print(f"Warning: Query file '{query_file}' not found. Using default queries.")
        return list(TEST_QUERIES)

    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                queries.append(cleaned)

    if not queries:
        print(f"Warning: Query file '{query_file}' was empty. Using default queries.")
        return list(TEST_QUERIES)

    return queries


def build_retrieved_dict(raw: dict) -> tuple[dict, list[dict], list[dict]]:
    """Build the standard retrieved dict contract expected by generator and verifier."""
    text_chunks = [
        {
            "text": item.get("text", ""),
            "source": item.get("source", "unknown"),
            "source_doc": item.get("source", "unknown"),
            "page": item.get("page", "Unknown"),
        }
        for item in raw.get("results", [])
    ]

    images = [
        {
            "path": p,
            "caption": "",
            "source_doc": Path(p).name,
        }
        for p in raw.get("image_paths", [])
    ]

    image_paths = [img["path"] for img in images]

    retrieved = {
        "text_chunks": text_chunks,
        "results": text_chunks,
        "images": images,
        "image_paths": image_paths,
        "is_sample_data": raw.get("is_sample_data", False),
    }

    return retrieved, text_chunks, images


def evaluate_query(query: str, query_id: str) -> dict:
    """Run retrieval, generation, and claim verification for one query."""
    raw = retriever.retrieve(query, top_k=3)
    retrieved, text_chunks, images = build_retrieved_dict(raw)

    gen_result = generator.generate_answer(query, retrieved)
    answer = gen_result.get("answer", "")

    ver_result = verifier.verify_claims(answer, text_chunks, images=images)

    trust_score = ver_result.get("trust_score", 0)
    total_claims = ver_result.get("total_claims", 0)
    supported_claims = ver_result.get("supported_claims", 0)
    n_contradicted = ver_result.get("n_contradicted", 0)

    return {
        "query_id": query_id,
        "query": query,
        "answer": answer,
        "trust_score": trust_score,
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "n_contradicted": n_contradicted,
        "status": "Success",
    }


def make_short_label(query_id: str, query: str) -> str:
    """Generate concise x-axis label for bar chart."""
    words = query.split()
    snippet = " ".join(words[:3]) + "..." if len(words) > 3 else query
    return f"{query_id}\n{snippet}"


def save_chart(results: list[dict], output_path: Path):
    """Save a bar chart of trust score per query."""
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.6), 5.5), dpi=130)

    labels = [make_short_label(r["query_id"], r["query"]) for r in results]
    scores = [r["trust_score"] for r in results]

    bars = ax.bar(labels, scores, color="#3b82f6", width=0.55, edgecolor="#1d4ed8", linewidth=1.2)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Trust Score (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Claim Verification Trust Score per Query\n(Percent of Claims Supported - Not Accuracy)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bar values
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f"{score}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


def save_csv_results(results: list[dict], output_path: Path):
    """Save evaluation metrics to CSV."""
    fieldnames = [
        "query_id",
        "query",
        "trust_score",
        "total_claims",
        "supported_claims",
        "n_contradicted",
        "status",
        "answer",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"CSV saved to: {output_path}")


def print_summary(results: list[dict]):
    """Print formatted summary table to console."""
    print("\n" + "=" * 90)
    print("EVALUATION SUMMARY: TRUST SCORE ACROSS TEST QUERIES")
    print("=" * 90)
    header = f"{'ID':<4} {'Trust (%)':<11} {'Supported':<11} {'Contradicted':<14} {'Total Claims':<14} {'Query':<32}"
    print(header)
    print("-" * 90)
    for r in results:
        q_text = r["query"] if len(r["query"]) <= 32 else r["query"][:29] + "..."
        score_text = f"{r['trust_score']}%"
        print(
            f"{r['query_id']:<4} {score_text:<11} {r['supported_claims']:<11} "
            f"{r['n_contradicted']:<14} {r['total_claims']:<14} {q_text:<32}"
        )
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run trust score evaluation across test queries and plot results."
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Optional path to a text file with one query per line.",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Run offline using a fake VLM without requiring an API key or GPU.",
    )
    parser.add_argument(
        "--output-chart",
        type=str,
        default="trust_chart.png",
        help="Output PNG path for the trust score bar chart (default: trust_chart.png).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="trust_results.csv",
        help="Output CSV path for query results (default: trust_results.csv).",
    )

    args = parser.parse_args()

    # Dry-run monkeypatching
    if args.dry_run:
        print("[Dry Run] Activating mock VLM for offline execution.")
        dry_vlm = DryRunVLM()
        generator.call_vlm = dry_vlm
        verifier.call_vlm = dry_vlm

    queries = load_queries(args.queries)
    print(f"Running evaluation on {len(queries)} queries...")

    results = []
    base_dir = Path(__file__).resolve().parent

    for idx, query in enumerate(queries, 1):
        query_id = f"Q{idx}"
        try:
            print(f"[{query_id}/{len(queries)}] Processing: '{query}'")
            res = evaluate_query(query, query_id)
            results.append(res)
        except Exception as exc:
            print(f"  [Error] Query '{query}' failed: {exc}", file=sys.stderr)
            results.append({
                "query_id": query_id,
                "query": query,
                "answer": "N/A",
                "trust_score": 0,
                "total_claims": 0,
                "supported_claims": 0,
                "n_contradicted": 0,
                "status": f"Error: {exc}",
            })

    output_chart_path = base_dir / args.output_chart
    output_csv_path = base_dir / args.output_csv

    save_chart(results, output_chart_path)
    save_csv_results(results, output_csv_path)
    print_summary(results)


if __name__ == "__main__":
    main()
