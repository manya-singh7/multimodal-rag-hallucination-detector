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
import time
from pathlib import Path

import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
            if "quantum" in prompt.lower() or self._query_index == 17:
                return (
                    "The provided text does not mention quantum retrieval methods. "
                    "Therefore, the evidence is insufficient to answer this question."
                )
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
    failed_claims = ver_result.get("failed_claims", 0)

    if total_claims == 0:
        status = "refused"
    elif n_contradicted > 0 or trust_score < 100:
        status = "contradicted"
    else:
        status = "supported"

    return {
        "query_id": query_id,
        "query": query,
        "answer": answer,
        "trust_score": trust_score,
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "n_contradicted": n_contradicted,
        "failed_claims": failed_claims,
        "status": status,
    }


def make_short_label(query_id: str, query: str) -> str:
    """Generate single-line y-axis label: 'Q1  Query...' truncated with '...' at ~70 chars."""
    full = f"{query_id}  {query.strip()}"
    if len(full) > 70:
        return full[:67].rstrip() + "..."
    return full


def save_chart(results: list[dict], output_path: Path):
    """Save a horizontal bar chart of trust score per query sized 14x9 inches at 200 dpi."""
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200)

    labels = [make_short_label(r["query_id"], r["query"]) for r in results]
    y_pos = list(range(len(results)))

    bar_widths = []
    bar_colors = []
    bar_edges = []
    bar_hatches = []

    for r in results:
        if r.get("total_claims", 0) == 0:
            # Refusal (e.g. Q17): draw as short grey hatched bar
            bar_widths.append(8.0)
            bar_colors.append("#e5e7eb")
            bar_edges.append("#6b7280")
            bar_hatches.append("///")
        elif r.get("trust_score", 0) == 100 and r.get("n_contradicted", 0) == 0:
            # 100% supported: green
            bar_widths.append(100.0)
            bar_colors.append("#22c55e")
            bar_edges.append("#15803d")
            bar_hatches.append(None)
        else:
            # Contradicted / under 100%: red
            bar_widths.append(float(r.get("trust_score", 0)))
            bar_colors.append("#ef4444")
            bar_edges.append("#b91c1c")
            bar_hatches.append(None)

    bars = ax.barh(
        y_pos,
        bar_widths,
        color=bar_colors,
        edgecolor=bar_edges,
        hatch=bar_hatches,
        height=0.6,
        linewidth=1.2,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlim(0, 118)
    ax.set_xlabel("Trust Score (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Claim Verification Trust Score per Query\n(Percent of Claims Supported - Not Accuracy)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate bar values: N/A (refused) for refusals, percentage for scored queries
    for bar, r in zip(bars, results):
        y_center = bar.get_y() + bar.get_height() / 2
        width = bar.get_width()
        if r.get("total_claims", 0) == 0:
            ax.annotate(
                "N/A (refused)",
                xy=(width, y_center),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#4b5563",
            )
        else:
            score = r.get("trust_score", 0)
            ax.annotate(
                f"{score}%",
                xy=(width, y_center),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#15803d" if score == 100 else "#b91c1c",
            )

    # Legend elements as requested
    legend_elements = [
        Patch(facecolor="#22c55e", edgecolor="#15803d", label="All claims supported"),
        Patch(facecolor="#ef4444", edgecolor="#b91c1c", label="Contradicted claim found"),
        Patch(facecolor="#e5e7eb", edgecolor="#6b7280", hatch="///", label="No claims to verify (refusal)"),
    ]

    # Calculate and draw vertical average reference line excluding refusals
    scored = [r for r in results if r.get("total_claims", 0) > 0]
    if scored:
        avg_trust = sum(r["trust_score"] for r in scored) / len(scored)
        ax.axvline(
            avg_trust,
            color="#4b5563",
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="#4b5563",
                linestyle="--",
                linewidth=1.4,
                label=f"Average Trust ({avg_trust:.1f}%) [excl. refusals]",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        framealpha=0.95,
        fontsize=9,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


def _parse_num(val: any, default: int | float = 0) -> int | float:
    """Safely parse a numeric string into int or float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip().rstrip("%")
    if not s:
        return default
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return default


def load_results_from_csv(csv_path: Path | str) -> list[dict]:
    """Load evaluation results from a CSV file into list of dicts matching evaluate_query output."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get("query_id", "")
            query = row.get("query", "")
            answer = row.get("answer", "")

            trust_score = _parse_num(row.get("trust_score"), 0)
            total_claims = _parse_num(row.get("total_claims"), 0)
            supported_claims = _parse_num(row.get("supported_claims"), 0)
            n_contradicted = _parse_num(row.get("n_contradicted"), 0)
            failed_claims = _parse_num(row.get("failed_claims"), 0)

            status = (row.get("status") or "").strip().lower()
            if status not in ("supported", "contradicted", "refused"):
                if total_claims == 0:
                    status = "refused"
                elif n_contradicted > 0 or trust_score < 100:
                    status = "contradicted"
                else:
                    status = "supported"

            results.append({
                "query_id": query_id,
                "query": query,
                "answer": answer,
                "trust_score": trust_score,
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "n_contradicted": n_contradicted,
                "failed_claims": failed_claims,
                "status": status,
            })

    return results


def save_csv_results(results: list[dict], output_path: Path):
    """Save evaluation metrics to CSV."""
    fieldnames = [
        "query_id",
        "query",
        "trust_score",
        "total_claims",
        "supported_claims",
        "n_contradicted",
        "failed_claims",
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
    print("\n" + "=" * 105)
    print("EVALUATION SUMMARY: TRUST SCORE ACROSS TEST QUERIES")
    print("=" * 105)
    header = f"{'ID':<4} {'Trust (%)':<11} {'Supported':<11} {'Contradicted':<14} {'Failed':<9} {'Total':<8} {'Status':<14} {'Query':<30}"
    print(header)
    print("-" * 105)
    for r in results:
        q_text = r["query"] if len(r["query"]) <= 30 else r["query"][:27] + "..."
        if r.get("total_claims", 0) == 0:
            score_text = "N/A"
        else:
            score_text = f"{r['trust_score']}%"
        status_text = r.get("status", "")
        print(
            f"{r['query_id']:<4} {score_text:<11} {r['supported_claims']:<11} "
            f"{r['n_contradicted']:<14} {r.get('failed_claims', 0):<9} {r['total_claims']:<8} "
            f"{status_text:<14} {q_text:<30}"
        )
    print("=" * 105)

    scored_results = [r for r in results if r.get("total_claims", 0) > 0]
    refused_results = [r for r in results if r.get("total_claims", 0) == 0]
    if scored_results:
        avg_score = sum(r["trust_score"] for r in scored_results) / len(scored_results)
        print(f"Average Trust Score (excluding {len(refused_results)} refusal(s)): {avg_score:.1f}%\n")
    else:
        print()


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
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds between queries to avoid rate limits (default: 5.0).",
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
    parser.add_argument(
        "--from-csv",
        nargs="?",
        const="trust_results.csv",
        default=None,
        metavar="PATH",
        help=(
            "Skip evaluation loop and generate chart and summary directly from "
            "a CSV file (default: trust_results.csv)."
        ),
    )

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent

    if args.from_csv:
        csv_path = Path(args.from_csv)
        if not csv_path.is_absolute():
            if not csv_path.is_file() and (base_dir / csv_path).is_file():
                csv_path = base_dir / csv_path

        if not csv_path.is_file():
            print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading results from CSV: {csv_path} (skipping evaluation loop)...")
        results = load_results_from_csv(csv_path)

        output_chart_path = Path(args.output_chart)
        if not output_chart_path.is_absolute():
            output_chart_path = base_dir / output_chart_path

        save_chart(results, output_chart_path)
        print_summary(results)
        return

    # Dry-run monkeypatching
    if args.dry_run:
        print("[Dry Run] Activating mock VLM for offline execution.")
        dry_vlm = DryRunVLM()
        generator.call_vlm = dry_vlm
        verifier.call_vlm = dry_vlm

    queries = load_queries(args.queries)
    print(f"Running evaluation on {len(queries)} queries...", flush=True)

    results = []

    for idx, query in enumerate(queries, 1):
        query_id = f"Q{idx}"
        try:
            print(f"[{query_id}/{len(queries)}] Processing: '{query}'", flush=True)
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
                "failed_claims": 0,
                "status": f"Error: {exc}",
            })

        if not args.dry_run and idx < len(queries) and args.delay > 0:
            time.sleep(args.delay)

    output_chart_path = Path(args.output_chart)
    if not output_chart_path.is_absolute():
        output_chart_path = base_dir / output_chart_path

    output_csv_path = Path(args.output_csv)
    if not output_csv_path.is_absolute():
        output_csv_path = base_dir / output_csv_path

    save_chart(results, output_chart_path)
    save_csv_results(results, output_csv_path)
    print_summary(results)


if __name__ == "__main__":
    main()
