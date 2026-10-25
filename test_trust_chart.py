import csv
import subprocess
import sys
from pathlib import Path

import pytest
import trust_chart
from trust_chart import load_results_from_csv


def test_from_csv_cli_generates_chart_and_exits_zero(tmp_path):
    """Test --from-csv with a 100% row, a 50% row, and a total_claims=0 row."""
    csv_file = tmp_path / "test_trust_results.csv"
    chart_file = tmp_path / "test_trust_chart.png"

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
    rows = [
        {
            "query_id": "Q1",
            "query": "Fully supported query",
            "trust_score": "100",
            "total_claims": "2",
            "supported_claims": "2",
            "n_contradicted": "0",
            "failed_claims": "0",
            "status": "supported",
            "answer": "Answer 1 grounded in documents.",
        },
        {
            "query_id": "Q2",
            "query": "Partially contradicted query",
            "trust_score": "50",
            "total_claims": "2",
            "supported_claims": "1",
            "n_contradicted": "1",
            "failed_claims": "0",
            "status": "contradicted",
            "answer": "Answer 2 with one hallucinated claim.",
        },
        {
            "query_id": "Q3",
            "query": "Refused query with no verifiable claims",
            "trust_score": "0",
            "total_claims": "0",
            "supported_claims": "0",
            "n_contradicted": "0",
            "failed_claims": "0",
            "status": "refused",
            "answer": "The provided text does not mention quantum retrieval methods.",
        },
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mtime_before = csv_file.stat().st_mtime_ns

    cmd = [
        sys.executable,
        str(Path(trust_chart.__file__).resolve()),
        "--from-csv",
        str(csv_file),
        "--output-chart",
        str(chart_file),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, f"Process failed: {proc.stderr}"
    assert chart_file.exists(), "Chart file was not created"
    assert chart_file.stat().st_size > 0, "Chart file is empty"

    # Verify CSV was not overwritten
    mtime_after = csv_file.stat().st_mtime_ns
    assert mtime_before == mtime_after, "CSV file should not be modified in --from-csv mode"

    # Verify output summary table in stdout
    assert "Q1" in proc.stdout
    assert "Q2" in proc.stdout
    assert "Q3" in proc.stdout
    assert "N/A" in proc.stdout
    assert "refused" in proc.stdout


def test_load_results_from_csv_recomputes_missing_status(tmp_path):
    """Verify that load_results_from_csv recomputes status if column is missing."""
    csv_file = tmp_path / "no_status.csv"

    # CSV without 'status' column
    fieldnames = [
        "query_id",
        "query",
        "trust_score",
        "total_claims",
        "supported_claims",
        "n_contradicted",
        "failed_claims",
        "answer",
    ]
    rows = [
        {
            "query_id": "Q1",
            "query": "100 percent query",
            "trust_score": "100",
            "total_claims": "2",
            "supported_claims": "2",
            "n_contradicted": "0",
            "failed_claims": "0",
            "answer": "Supported answer.",
        },
        {
            "query_id": "Q2",
            "query": "Contradicted query",
            "trust_score": "50",
            "total_claims": "2",
            "supported_claims": "1",
            "n_contradicted": "1",
            "failed_claims": "0",
            "answer": "Contradicted answer.",
        },
        {
            "query_id": "Q3",
            "query": "Refusal query",
            "trust_score": "0",
            "total_claims": "0",
            "supported_claims": "0",
            "n_contradicted": "0",
            "failed_claims": "0",
            "answer": "Refusal answer.",
        },
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    results = load_results_from_csv(csv_file)
    assert len(results) == 3
    assert results[0]["status"] == "supported"
    assert results[0]["total_claims"] == 2
    assert results[1]["status"] == "contradicted"
    assert results[2]["status"] == "refused"
    assert results[2]["total_claims"] == 0


def test_make_short_label_single_line_and_truncation():
    """Verify make_short_label produces single-line labels truncated at ~70 chars."""
    long_query = "What are the two primary stages from which hallucinations arise in RAG models?"
    label = trust_chart.make_short_label("Q1", long_query)
    assert "\n" not in label
    assert label.startswith("Q1  What are")
    assert label.endswith("...")
    assert len(label) == 70

    short_query = "Short query?"
    label_short = trust_chart.make_short_label("Q2", short_query)
    assert label_short == "Q2  Short query?"

