#!/usr/bin/env python3
"""
Create topic clusters for review texts using BERTopic.

This script reads from ./neurips/2021/raw_qas.json (an array of JSON objects
with a 'review' field), extracts the review texts, runs BERTopic to find
topics, and writes out:

- <output_dir>/document_topics.csv : document index, topic, probability, text (truncated)
- <output_dir>/topics.json : mapping topic_id -> keywords, count, example indices
- <output_dir>/topic_model : saved BERTopic model directory (if BERTopic supports .save)

Usage:
    python clusters.py --input ./neurips/2021/raw_qas.json --output_dir ./neurips/2021/topics

Notes:
 - Requires the `bertopic` package and an embedding model (sentence-transformers).
   If missing, the script will print an installation hint.
 - This script prefers loading the whole JSON array (the file in this repo is ~50MB,
   which should fit in memory on typical development machines). If you need a
   streaming/low-memory approach, we can adjust it to use ijson.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Dict, Any


def load_reviews(path: Path) -> List[str]:
    """Load the JSON array and extract the 'review' field from each object.

    Args:
        path: Path to the JSON file (array of objects).

    Returns:
        List of review strings (empty ones filtered out).
    """
    logging.info("Loading reviews from %s", path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    reviews: List[str] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue
        r = obj.get("review")
        if r and isinstance(r, str):
            # Basic cleanup
            text = " ".join(r.split())
            if text:
                reviews.append(text)
        else:
            # skip entries without a review
            continue

    logging.info("Loaded %d reviews (filtered)", len(reviews))
    return reviews


def run_bertopic(docs: List[str], embedding_model: str = "all-MiniLM-L6-v2", extra_stopwords: str | None = None, **kwargs) -> Any:
    """Instantiate and fit BERTopic on the provided documents.

    Returns the fitted topic model and the topics/probabilities for each doc.
    """
    try:
        from bertopic import BERTopic
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "BERTopic is not installed or failed to import.\n"
            "Install it with: pip install bertopic[all] or pip install bertopic sentence-transformers\n"
            f"Original error: {e}"
        )

    # Create a CountVectorizer with stop words so topic keywords exclude filler words.
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception:
        ENGLISH_STOP_WORDS = set()
        CountVectorizer = None

    stop_words = set(map(str.lower, ENGLISH_STOP_WORDS)) if CountVectorizer is not None else set()
    if extra_stopwords:
        # allow comma-separated additional stop words
        extra = [w.strip().lower() for w in extra_stopwords.split(",") if w.strip()]
        stop_words.update(extra)

    logging.info("Creating BERTopic model with embedding_model=%s (stopwords=%d extra)", embedding_model, len(stop_words))

    vectorizer = None
    if CountVectorizer is not None:
        # sklearn expects stop_words to be a str ('english'), list, or None — convert set to list
        stop_words_list = list(stop_words) if stop_words else None
        # reasonable defaults: unigrams and bigrams, ignore very frequent terms
        vectorizer = CountVectorizer(stop_words=stop_words_list, ngram_range=(1, 2), max_df=0.95, min_df=2)

    topic_model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer, **kwargs)

    logging.info("Fitting BERTopic to %d documents...", len(docs))
    topics, probs = topic_model.fit_transform(docs)

    return topic_model, topics, probs


def save_outputs(topic_model: Any, topics: List[int], probs: List[float], docs: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Categorize documents (fast keyword-based default) ---
    def categorize_keyword(text: str) -> Dict[str, float]:
        # Map of category -> list of trigger words
        KEYWORDS = {
            "clarity": ["clarity", "clear", "unclear", "well written", "readable", "confusing", "organized", "presentation", "writing"],
            "experiments": ["experiment", "experiments", "results", "accuracy", "evaluation", "datasets", "fig", "figure", "table", "ablation", "training"],
            "methods": ["method", "approach", "algorithm", "technique", "model", "proposed"],
            "theory": ["theorem", "proof", "analysis", "bound", "lemma", "corollary", "theoretical"],
            "reproducibility": ["code", "reproducible", "implementation", "reproduce", "released", "hyperparameter", "seed"],
            "novelty": ["novel", "originality", "novelty", "contribution", "new"],
            "baselines": ["baseline", "compare", "comparison", "competitor", "state-of-the-art", "soa", "previous work"],
            "organization": ["organization", "structure", "section", "appendix", "flow", "order"],
            "writing": ["typo", "typos", "grammar", "spelling", "language"],
            "limitations": ["limitation", "weakness", "future work", "concern", "issue", "shortcoming"],
        }

        text_l = text.lower()
        scores: Dict[str, float] = {}
        for cat, triggers in KEYWORDS.items():
            score = 0
            for t in triggers:
                if t in text_l:
                    score += 1
            if score > 0:
                # normalize by number of triggers to make categories comparable
                scores[cat] = score / max(1, len(triggers))
        # if nothing matched, mark as 'other'
        if not scores:
            scores["other"] = 1.0
        return scores

    # Save per-document assignments
    import csv

    doc_csv = out_dir / "document_topics.csv"
    logging.info("Writing document-topic assignments to %s", doc_csv)
    with doc_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["doc_index", "topic", "probability", "categories", "text_sample"])
        # compute categories for each doc and write them
        doc_categories: List[Dict[str, float]] = []
        for i, (t, p) in enumerate(zip(topics, probs)):
            sample = docs[i][:200].replace("\n", " ")
            prob_value = ""
            try:
                # p may be None, a float, or an array-like with a single value
                if p is None:
                    prob_value = ""
                elif isinstance(p, (list, tuple)) and len(p) > 0:
                    prob_value = float(p[0])
                else:
                    prob_value = float(p)
            except Exception:
                prob_value = ""

            cats = categorize_keyword(docs[i])
            doc_categories.append(cats)
            # flatten category dict to a semi-colon list of category:score
            cat_str = ";".join([f"{k}:{v:.2f}" for k, v in sorted(cats.items(), key=lambda x: -x[1])])
            writer.writerow([i, t, prob_value, cat_str, sample])

    # Collect topic metadata from the model
    topics_info: Dict[str, Any] = {}
    # topic_model.get_topic_info() may be available (often a pandas.DataFrame)
    try:
        info = topic_model.get_topic_info()
    except Exception:
        info = None

    # Build a mapping topic_id -> keywords, count, example indices
    topic_to_indices: Dict[int, List[int]] = {}
    for idx, t in enumerate(topics):
        topic_to_indices.setdefault(t, []).append(idx)

    for topic_id, indices in topic_to_indices.items():
        if topic_id == -1:
            keywords = ["_outlier"]
        else:
            try:
                kws = topic_model.get_topic(topic_id)
                keywords = [k for k, _ in kws]
            except Exception:
                keywords = []

        # aggregate category distribution for this topic
        agg: Dict[str, float] = {}
        for idx in indices:
            for c, s in doc_categories[idx].items():
                agg[c] = agg.get(c, 0.0) + float(s)
        # normalize by topic count
        if agg:
            for c in list(agg.keys()):
                agg[c] = agg[c] / len(indices)

        topics_info[str(topic_id)] = {
            "topic_id": topic_id,
            "count": len(indices),
            "keywords": keywords,
            "category_distribution": agg,
            "example_indices": indices[:5],
            "example_texts": [docs[i][:500] for i in indices[:5]],
        }

    # Make sure topic info is JSON serializable (DataFrame -> list of dicts)
    info_serializable = None
    if info is not None:
        try:
            # pandas DataFrame
            info_serializable = info.to_dict(orient="records")
        except Exception:
            try:
                # Fallback: try list conversion
                info_serializable = info.tolist() if hasattr(info, "tolist") else str(info)
            except Exception:
                info_serializable = str(info)

    topics_json = out_dir / "topics.json"
    logging.info("Writing topics summary to %s", topics_json)
    with topics_json.open("w", encoding="utf-8") as fh:
        json.dump({"topics": topics_info, "topic_info_table": info_serializable}, fh, indent=2)

    # Try saving the model (BERTopic supports .save(path))
    try:
        model_dir = out_dir / "topic_model"
        logging.info("Saving BERTopic model to %s", model_dir)
        topic_model.save(str(model_dir))
    except Exception as e:
        logging.warning("Failed to save BERTopic model: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster review texts using BERTopic")
    parser.add_argument("--input", type=str, default="./neurips/2021/raw_qas.json", help="Path to raw_qas.json")
    parser.add_argument("--output_dir", type=str, default="./neurips/2021/topics", help="Directory to write topic outputs")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="sentence-transformers embedding model name")
    parser.add_argument("--min_topic_size", type=int, default=10, help="minimum topic size for BERTopic")
    parser.add_argument("--nr_topics", type=int, default=None, help="Reduce number of topics (BERTopic parameter)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents (for quick testing)")
    parser.add_argument("--extra_stopwords", type=str, default=None, help="Comma-separated extra stop words to remove from keywords")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    infile = Path(args.input)
    out_dir = Path(args.output_dir)

    if not infile.exists():
        raise SystemExit(f"Input file not found: {infile}")

    docs = load_reviews(infile)
    if args.limit:
        docs = docs[: args.limit]

    if not docs:
        raise SystemExit("No review texts found to cluster.")

    # BERTopic kwargs
    bm_kwargs = {"min_topic_size": args.min_topic_size}
    if args.nr_topics is not None:
        bm_kwargs["nr_topics"] = args.nr_topics

    topic_model, topics, probs = run_bertopic(docs, embedding_model=args.embedding_model, extra_stopwords=args.extra_stopwords, **bm_kwargs)

    save_outputs(topic_model, topics, probs, docs, out_dir)

    logging.info("Done. Topics and assignments were written to %s", out_dir)


if __name__ == "__main__":
    main()
