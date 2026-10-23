"""Hybrid semantic + keyword retrieval for processed research papers.

Handles two kinds of questions differently:

* Specific questions ("How does ANCHOR evaluate grounding?")
    -> hybrid ranking: embeddings (semantic) + BM25 (keyword).
* Broad questions ("What are the main objectives and contributions?")
    -> same hybrid score PLUS a "role" score that favours abstracts,
       introductions, conclusions and contribution-style sentences, and
       selects evidence from EACH paper instead of letting one paper win.

Output format of retrieve() is unchanged:
    text_chunks, image_paths, results, is_sample_data
(plus an extra "query_type" key, which is harmless to existing code).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "data" / "processed" / "text"
IMAGE_DIR = BASE_DIR / "data" / "processed" / "images"
CACHE_DIR = BASE_DIR / "data" / "processed" / ".retriever_cache"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Specific-question weights.
SEMANTIC_WEIGHT = 0.70
KEYWORD_WEIGHT = 0.30

# Broad-question weights (role = abstract/intro/conclusion/contribution cues).
BROAD_SEMANTIC_WEIGHT = 0.45
BROAD_KEYWORD_WEIGHT = 0.15
BROAD_ROLE_WEIGHT = 0.40

# Minimum score cutoff for retrieved results
MIN_SCORE = 0.20

# Reference / bibliography page ranges to skip per document source
BIBLIOGRAPHY_PAGES: dict[str, range] = {
    "paper1.pdf": range(27, 34),
    "paper2.pdf": range(10, 13),
}

MIN_CHUNK_WORDS = 12      # chunks shorter than this are dropped (page numbers etc.)
SHORT_CHUNK_WORDS = 25    # chunks shorter than this are penalised
MAX_PER_PAGE = 2          # max chunks from one (paper, page) in the results
MAX_IMAGES = 6            # cap on images returned per query

BROAD_EXPANSION = (
    "Abstract. In this paper we propose and introduce the main objectives, "
    "contributions and key findings of this work. Conclusion."
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "for", "and", "or", "but", "with", "by", "from", "at",
    "as", "it", "its", "this", "that", "these", "those", "what", "which",
    "who", "how", "why", "when", "where", "do", "does", "did", "can", "could",
    "would", "should", "their", "they", "them", "we", "our", "you", "your",
    "me", "my", "about", "into", "than", "then", "there", "such", "also",
    "tell", "give", "explain", "describe", "main", "following",
}

# Words that only signal "this is an overview question"; they say nothing
# about WHICH paper or topic is meant.
GENERIC_QUERY_WORDS = {
    "paper", "papers", "research", "study", "work", "review", "article",
    "objective", "objectives", "contribution", "contributions", "aim", "aims",
    "purpose", "goal", "goals", "overview", "summary", "summarize", "summarise",
    "idea", "finding", "findings", "takeaway", "takeaways", "key", "novelty",
}

BROAD_RE = re.compile(
    r"\b(objectives?|contributions?|aims?|purpose|goals?|overview|"
    r"summar(?:y|ize|ise)|main (?:idea|point|findings?|takeaways?)|"
    r"key (?:findings?|takeaways?|ideas?)|novelty|"
    r"about this (?:paper|work|study)|"
    r"what does this (?:paper|work|study|review)|"
    r"this (?:paper|work|study|review|research) "
    r"(?:is about|does|proposes|presents|covers))\b",
    re.I,
)

FIGURE_QUERY_RE = re.compile(
    r"\b(figure|fig|table|plot|chart|diagram|graph|image|illustration)s?\b", re.I
)

# ---------------------------------------------------------------- noise ----

NOISE_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\{question\}",
        r"\{output\}",
        r"\{answer\}",
        r"\{context\}",
        r"first line of your response must contain",
        r"evaluation guidelines",
        r"author contributions\s*:",
        r"conflicts? of interest\s*:",
        r"data availability statement",
        r"institutional review board statement",
        r"informed consent statement",
        r"\bfunding\s*:",
    )
]

CAPTION_RE = re.compile(r"^(?:figure|fig\.|table)\s*[a-z]?\d+[\s.:]", re.I)
REFERENCE_HINT_RE = re.compile(
    r"\b(?:arxiv|doi|proceedings|preprint|in:|pp\.|vol\.|conference on)\b", re.I
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def is_reference_list(text: str) -> bool:
    """Heuristic: dense years + bibliographic keywords = reference list."""
    words = text.split()
    if len(words) < 30:
        return False
    years = len(YEAR_RE.findall(text))
    hints = len(REFERENCE_HINT_RE.findall(text))
    return hints >= 3 and years / len(words) > 0.04


def is_noise(text: str) -> bool:
    """True for prompt templates, boilerplate sections and reference lists."""
    if any(p.search(text) for p in NOISE_PATTERNS):
        return True
    return is_reference_list(text)


def is_bibliography_chunk(source: str | None, page: int | None) -> bool:
    """True if chunk falls into known bibliography/reference page ranges."""
    if not source or page is None:
        return False
    source_name = Path(source).name
    stem = Path(source).stem
    pages = BIBLIOGRAPHY_PAGES.get(source_name) or BIBLIOGRAPHY_PAGES.get(stem)
    return pages is not None and page in pages


# ------------------------------------------------------------ tokenizing ----

_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s")


def stem(word: str) -> str:
    """Very light stemmer so 'contributions' matches 'contribution'."""
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, stem."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def is_broad_query(query: str) -> bool:
    return bool(BROAD_RE.search(query))


# ------------------------------------------------------------- loading ----

def load_documents() -> list[dict]:
    """Load usable text chunks from data/processed/text/*.json."""
    documents: list[dict] = []
    seen_texts: set[str] = set()

    if not TEXT_DIR.exists():
        return documents

    for json_path in sorted(TEXT_DIR.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read {json_path.name}: {exc}")
            continue

        if isinstance(payload, dict):
            payload = payload.get("chunks", [])
        if not isinstance(payload, list):
            continue

        for position, chunk in enumerate(payload):
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if not isinstance(text, str):
                continue
            text = " ".join(text.split())
            if len(text.split()) < MIN_CHUNK_WORDS:
                continue
            if is_noise(text) or text in seen_texts:
                continue

            page = chunk.get("page")
            try:
                page = int(page) if page is not None else None
            except (TypeError, ValueError):
                page = None

            source = chunk.get("source", json_path.stem + ".pdf")
            if is_bibliography_chunk(source, page):
                continue

            seen_texts.add(text)

            documents.append({
                "id": chunk.get("id", f"{json_path.stem}_{position}"),
                "text": text,
                "source": source,
                "page": page,
                "image_path": None,
            })

    return documents


# --------------------------------------------------------- embeddings ----

@lru_cache(maxsize=1)
def _get_model():
    """Load the embedding model once. Returns None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # missing package, offline, etc.
        print(f"Semantic model unavailable ({exc}); using keyword scoring only.")
        return None


def _encode(texts: list[str]) -> np.ndarray | None:
    model = _get_model()
    if model is None:
        return None
    return np.asarray(
        model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def _embed_corpus(texts: list[str]) -> np.ndarray | None:
    """Embed all chunks, caching on disk (keyed by model + corpus content)."""
    digest = hashlib.sha1(
        (MODEL_NAME + "\n" + "\n".join(texts)).encode("utf-8")
    ).hexdigest()
    cache_file = CACHE_DIR / f"emb_{digest}.npy"

    if cache_file.exists():
        try:
            cached = np.load(cache_file)
            if cached.shape[0] == len(texts):
                return cached
        except (OSError, ValueError):
            pass

    embeddings = _encode(texts)
    if embeddings is not None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)
        except OSError:
            pass
    return embeddings


# ----------------------------------------------------------- the index ----

class _Index:
    """BM25 statistics + embeddings for a list of chunks."""

    K1 = 1.5
    B = 0.75

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.stems = [tokenize(d["text"]) for d in docs]
        self.tf = [Counter(s) for s in self.stems]
        lengths = np.array([len(s) for s in self.stems], dtype=float)
        self.lengths = lengths
        self.avg_len = float(lengths.mean()) if len(lengths) else 1.0

        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {
            t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

        # Per-paper document frequency (to detect which paper a query names).
        self.source_df: dict[str, Counter] = defaultdict(Counter)
        for doc, counts in zip(docs, self.tf):
            self.source_df[doc["source"]].update(counts.keys())

        self.embeddings = _embed_corpus([d["text"] for d in docs]) if docs else None

    @property
    def sources(self) -> list[str]:
        return sorted({d["source"] for d in self.docs})

    def bm25(self, query_stems: list[str]) -> np.ndarray:
        """BM25 scores for every chunk, normalised to 0..1 by the max."""
        scores = np.zeros(len(self.docs), dtype=float)
        for term in set(query_stems):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                freq = counts.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.K1 * (
                    1 - self.B + self.B * self.lengths[i] / self.avg_len
                )
                scores[i] += idf * freq * (self.K1 + 1) / denom
        top = scores.max() if len(scores) else 0.0
        return scores / top if top > 0 else scores

    def semantic(self, query: str, broad: bool) -> np.ndarray | None:
        """Cosine similarity of every chunk to the query (0..1)."""
        if self.embeddings is None:
            return None
        texts = [query, f"{query} {BROAD_EXPANSION}"] if broad else [query]
        vectors = _encode(texts)
        if vectors is None:
            return None
        if broad:
            # Keep the user's own wording dominant; the expansion only nudges.
            q_vec = 0.7 * vectors[0] + 0.3 * vectors[1]
        else:
            q_vec = vectors[0]
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        return np.clip(self.embeddings @ q_vec, 0.0, 1.0)

    def focus_sources(self, query: str) -> list[str] | None:
        """If a broad query names one paper's topic, return just that paper."""
        focus = [
            s for s in tokenize(query)
            if s not in {stem(w) for w in GENERIC_QUERY_WORDS}
        ]
        if not focus or len(self.sources) < 2:
            return None
        coverage = {
            src: sum(1 for s in focus if self.source_df[src].get(s, 0) >= 2)
            for src in self.sources
        }
        best = max(coverage.values())
        winners = [s for s, c in coverage.items() if c == best]
        if best > 0 and len(winners) == 1:
            return winners
        return None


_DEFAULT_INDEX: _Index | None = None


def _get_index(documents: list[dict] | None) -> _Index | None:
    """Build (and cache) the index for the processed corpus."""
    global _DEFAULT_INDEX
    if documents is not None:
        docs = [
            d for d in documents
            if isinstance(d, dict)
            and d.get("text")
            and not is_bibliography_chunk(d.get("source"), d.get("page"))
        ]
        return _Index(docs) if docs else None
    if _DEFAULT_INDEX is None:
        docs = load_documents()
        if not docs:
            return None
        _DEFAULT_INDEX = _Index(docs)
    return _DEFAULT_INDEX


# --------------------------------------------------- role / section score ----

_HEAD_ABSTRACT = re.compile(r"(?:^|\s)(?:Abstract|ABSTRACT)(?=[\s:.\-\u2014])")
_HEAD_INTRO = re.compile(
    r"(?:^|\s)(?:\d+\.?\s+)?(?:Introduction|INTRODUCTION)\s+[A-Z]"
)
_HEAD_CONCLUSION = re.compile(
    r"(?:^|\s)(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:Conclusions?|CONCLUSIONS?|Concluding Remarks|Discussion and Conclusions?)"
    r"\s+[A-Z]"
)

CONTRIBUTION_CUES = (
    "we propose", "we introduce", "we present", "we develop", "we construct",
    "we conduct", "we benchmark", "this paper", "in this paper", "in this work",
    "in this review", "this review", "this work", "this study", "this survey",
    "this benchmark", "our contribution", "main contribution", "key contribution",
    "contributions are", "to address this", "to this end", "aims to",
    "we find", "we show", "we demonstrate", "experiments reveal", "results show",
    "our findings", "these findings", "in conclusion", "we conclude",
    "future work", "future research", "comprehensive investigation",
)


def role_score(text: str, page: int | None) -> float:
    """0..1: how much this chunk looks like an abstract/intro/conclusion."""
    heading = (
        1.0
        if (
            _HEAD_ABSTRACT.search(text)
            or _HEAD_INTRO.search(text)
            or _HEAD_CONCLUSION.search(text)
        )
        else 0.0
    )
    lowered = text.lower()
    cue_hits = sum(1 for cue in CONTRIBUTION_CUES if cue in lowered)
    cue_norm = min(1.0, cue_hits / 4)

    early = 0.0
    if page is not None:
        early = 1.0 if page <= 2 else 0.5 if page == 3 else 0.0

    return min(1.0, 0.45 * heading + 0.35 * cue_norm + 0.20 * early)


FRONT_MATTER_RE = re.compile(
    r"academic editor|licensee|creative commons|copyright\s*:|"
    r"received\s*:|accepted\s*:|published\s*:|citation\s*:|doi\.org",
    re.I,
)


def quality_penalty(text: str, wants_figures: bool) -> float:
    """Negative adjustment for captions, tiny fragments and journal front matter."""
    penalty = 0.0
    if len({m.group(0).lower() for m in FRONT_MATTER_RE.finditer(text)}) >= 3:
        penalty -= 0.10
    if CAPTION_RE.match(text.strip()) and not wants_figures:
        penalty -= 0.15
    if len(text.split()) < SHORT_CHUNK_WORDS:
        penalty -= 0.10
    return penalty


# -------------------------------------------------------------- selection ----

def _select(scored: list[dict], top_k: int, broad: bool, sources: list[str]) -> list[dict]:
    """Pick top_k results; for broad queries, give every paper a fair share."""
    ordered = sorted(scored, key=lambda item: item["score"], reverse=True)
    chosen: list[dict] = []
    chosen_ids: set[int] = set()
    per_page: Counter = Counter()
    per_source: Counter = Counter()

    def take(item: dict, respect_page_cap: bool = True) -> bool:
        key = (item["source"], item.get("page"))
        if respect_page_cap and per_page[key] >= MAX_PER_PAGE:
            return False
        if id(item) in chosen_ids:
            return False
        chosen.append(item)
        chosen_ids.add(id(item))
        per_page[key] += 1
        per_source[item["source"]] += 1
        return True

    if broad and len(sources) > 1:
        quota = max(1, math.ceil(top_k / len(sources)))
        # First pass: at most one chunk per page, so each paper contributes
        # its abstract, introduction AND conclusion rather than 3 chunks of page 1.
        for item in ordered:
            if len(chosen) >= top_k:
                break
            key = (item["source"], item.get("page"))
            if per_source[item["source"]] < quota and per_page[key] < 1:
                take(item)

    for item in ordered:
        if len(chosen) >= top_k:
            break
        take(item)

    for item in ordered:  # relax the page cap only if we still lack results
        if len(chosen) >= top_k:
            break
        take(item, respect_page_cap=False)

    return sorted(chosen, key=lambda item: item["score"], reverse=True)


def _associate_images(results: list[dict]) -> list[str]:
    """Images from the source PDF + page of the best results (capped)."""
    if not IMAGE_DIR.exists():
        return []

    image_paths: list[str] = []
    seen_pages: set[tuple[str, int]] = set()

    for item in results:
        page = item.get("page")
        if page is None:
            continue
        stem_name = Path(item["source"]).stem
        key = (stem_name, page)
        if key in seen_pages:
            continue
        seen_pages.add(key)

        for image_file in sorted(IMAGE_DIR.glob(f"{stem_name}_p{page}_img*")):
            path = str(image_file.resolve())
            if path not in image_paths:
                image_paths.append(path)
            if len(image_paths) >= MAX_IMAGES:
                return image_paths

    return image_paths


# ---------------------------------------------------------------- retrieve ----

def _empty() -> dict:
    return {
        "text_chunks": [],
        "image_paths": [],
        "results": [],
        "is_sample_data": False,
        "query_type": "none",
    }


def retrieve(
    query: str,
    top_k: int = 5,
    documents=None,
    sources: list[str] | None = None,
) -> dict:
    """Return the most relevant chunks (and page images) for a question.

    sources: optionally restrict to certain papers, e.g. ["paper2.pdf"].
    """
    if not isinstance(query, str) or not query.strip():
        return _empty()

    try:
        top_k = max(1, int(top_k))
    except (TypeError, ValueError):
        top_k = 5

    index = _get_index(documents)
    if index is None:
        return _empty()

    broad = is_broad_query(query)
    wants_figures = bool(FIGURE_QUERY_RE.search(query))

    # Which papers are eligible?
    if sources:
        wanted = {Path(s).stem for s in sources}
        allowed = [s for s in index.sources if Path(s).stem in wanted]
    elif broad:
        allowed = index.focus_sources(query) or index.sources
    else:
        allowed = index.sources
    if not allowed:
        allowed = index.sources

    keyword = index.bm25(tokenize(query))
    semantic = index.semantic(query, broad)
    use_semantic = semantic is not None
    if semantic is None:
        semantic = np.zeros(len(index.docs))

    if broad:
        w_sem, w_kw, w_role = (
            (BROAD_SEMANTIC_WEIGHT, BROAD_KEYWORD_WEIGHT, BROAD_ROLE_WEIGHT)
            if use_semantic
            else (0.0, 0.45, 0.55)
        )
    else:
        w_sem, w_kw, w_role = (
            (SEMANTIC_WEIGHT, KEYWORD_WEIGHT, 0.0) if use_semantic else (0.0, 1.0, 0.0)
        )

    scored: list[dict] = []
    for i, doc in enumerate(index.docs):
        if doc["source"] not in allowed:
            continue

        penalty = quality_penalty(doc["text"], wants_figures)
        bonus = penalty
        if broad:
            bonus += w_role * role_score(doc["text"], doc.get("page"))

        score = w_sem * float(semantic[i]) + w_kw * float(keyword[i]) + bonus

        scored.append({
            **doc,
            "score": score,
            "semantic_score": float(semantic[i]),
            "keyword_score": float(keyword[i]),
            "section_bonus": bonus,
        })

    if scored:
        best_doc = max(scored, key=lambda item: item["score"])
        above_min = [d for d in scored if d["score"] >= MIN_SCORE]
        if above_min:
            results = _select(above_min, top_k, broad, allowed)
            results = [r for r in results if r["score"] >= MIN_SCORE]
            if not results:
                results = [best_doc]
        else:
            results = [best_doc]
    else:
        results = []

    return {
        "text_chunks": [item["text"] for item in results],
        "image_paths": _associate_images(results),
        "results": results,
        "is_sample_data": False,
        "query_type": "broad" if broad else "specific",
    }


# --------------------------------------------------------------------- CLI ----

if __name__ == "__main__":
    print(f"Loaded usable documents: {len(load_documents())}")
    question = input("Enter a research question: ")
    output = retrieve(question)

    print(f"\nQuery type: {output['query_type']}")
    print(f"Retrieved chunks: {len(output['results'])}")

    for number, item in enumerate(output["results"], start=1):
        print(f"\n--- Result {number} ---")
        print(f"Combined score: {item['score']:.3f}")
        print(f"Semantic score: {item['semantic_score']:.3f}")
        print(f"Keyword score:  {item['keyword_score']:.3f}")
        print(f"Section bonus:  {item['section_bonus']:.3f}")
        print(f"Source: {item['source']}")
        print(f"Page: {item['page']}")
        print(f"Text: {item['text'][:700]}")

    print(f"\nAssociated images: {len(output['image_paths'])}")
