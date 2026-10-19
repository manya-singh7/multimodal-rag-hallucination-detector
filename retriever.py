
"""Research retrieval using semantic embeddings and source-aware results."""

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "data" / "processed" / "text"
IMAGE_DIR = BASE_DIR / "data" / "processed" / "images"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def load_documents() -> list[dict]:
    """Load text chunks from processed PDF JSON files."""
    documents = []

    if not TEXT_DIR.exists():
        return documents

    for json_path in sorted(TEXT_DIR.glob("*.json")):
        try:
            chunks = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read {json_path.name}: {exc}")
            continue

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            documents.append({
                "id": chunk.get("id", f"{json_path.stem}_{len(documents)}"),
                "text": text,
                "source": chunk.get("source", json_path.stem + ".pdf"),
                "page": chunk.get("page"),
                "image_path": None,
            })

    return documents


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once; CPU mode works without CUDA."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def get_index():
    """Encode all document chunks once and cache their embeddings."""
    documents = load_documents()

    if not documents:
        return documents, np.empty((0, 0), dtype=np.float32)

    model = get_model()
    embeddings = model.encode(
        [doc["text"] for doc in documents],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return documents, embeddings


def retrieve(query: str, top_k: int = 5, documents=None) -> dict:
    """Return semantically relevant chunks and related extracted images."""
    empty_result = {
        "text_chunks": [],
        "image_paths": [],
        "results": [],
        "is_sample_data": False,
    }

    if not query or not query.strip():
        return empty_result

    if documents is None:
        docs, doc_embeddings = get_index()
    else:
        # Supports custom documents for tests without using the cached index.
        docs = documents
        if not docs:
            return empty_result

        model = get_model()
        doc_embeddings = model.encode(
            [doc.get("text", "") for doc in docs],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    if not docs:
        return empty_result

    model = get_model()
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # Embeddings are normalized, so dot product gives cosine similarity.
    scores = doc_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:max(1, top_k)]

    results = []
    image_paths = set()

    for index in top_indices:
        item = dict(docs[int(index)])
        item["score"] = float(scores[int(index)])
        item["retrieval_method"] = "semantic"
        results.append(item)

        page = item.get("page")
        source_stem = Path(item["source"]).stem

        if page is not None and IMAGE_DIR.exists():
            pattern = f"{source_stem}_p{page}_img*"
            for image_file in IMAGE_DIR.glob(pattern):
                image_paths.add(str(image_file.resolve()))

    return {
        "text_chunks": [item["text"] for item in results],
        "image_paths": sorted(image_paths),
        "results": results,
        "is_sample_data": False,
    }


if __name__ == "__main__":
    print(f"Loaded documents: {len(load_documents())}")
    question = input("Enter a research question: ").strip()
    output = retrieve(question)

    print(f"\nRetrieved chunks: {len(output['results'])}")
    for rank, item in enumerate(output["results"], start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Cosine similarity: {item['score']:.3f}")
        print(f"Source: {item['source']}")
        print(f"Page: {item['page']}")
        print(item["text"][:700])

    print(f"\nAssociated images: {len(output['image_paths'])}")
