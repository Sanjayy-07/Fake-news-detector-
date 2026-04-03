import os
import pickle
import random
import re
import csv
import json
import time
import xml.etree.ElementTree as ET
from html import unescape
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

MAX_LEN = 200
MIN_CHARS = 10
# If your model's sigmoid output corresponds to "Real" probability, keep this True.
# If it corresponds to "Fake" probability, set it to False.
OUTPUT_PROB_FOR_REAL = True

# Optional: configure a news API key (e.g., from newsapi.org) via environment variable.
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

app = Flask(__name__)

# Demo sampling config (for "Try Real/Fake" buttons)
TRUE_CSV_PATH = os.path.join(BASE_DIR, "True.csv")
FAKE_CSV_PATH = os.path.join(BASE_DIR, "Fake.csv")
DEMO_POOL_SIZE = 250
DEMO_MAX_ATTEMPTS = 60

# Cached metrics for demo performance.
METRICS_CACHE_PATH = os.path.join(BASE_DIR, "metrics.json")
METRICS_TTL_SECONDS = 24 * 60 * 60
METRICS_POSITIVE_LABEL = 0  # 0 = Fake (matches Fake.csv label)


def _load_artifacts():
    # Loading at startup keeps predictions fast and avoids reloading on every request.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = _load_artifacts()

_demo_pool: Dict[str, List[Dict[str, str]]] = {"real": [], "fake": []}
_demo_pool_loaded = False


_non_alnum_re = re.compile(r"[^a-z0-9\s]+")
_multi_space_re = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Basic preprocessing to match typical training cleaning."""
    text = (text or "").lower()
    text = _non_alnum_re.sub(" ", text)
    text = _multi_space_re.sub(" ", text).strip()
    return text


def _build_rss_sources(category: str) -> List[str]:
    """Return a small set of open RSS feeds for a category.

    This avoids hard-coding a single provider and keeps things API-key free.
    You can customize/extend these sources as needed.
    """
    category = (category or "").lower()
    if category in {"sports", "cricket"}:
        return [
            # Broad + cricket-specific
            "https://www.espn.com/espn/rss/news",
            "https://feeds.bbci.co.uk/sport/rss.xml",
            "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        ]
    if category in {"business", "markets", "shares", "finance"}:
        return [
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://www.investing.com/rss/news_25.rss",
            "https://www.moneycontrol.com/rss/latestnews.xml",
            "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        ]
    if category in {"tech", "technology"}:
        return [
            "https://feeds.arstechnica.com/arstechnica/technology-lab",
            "https://feeds.feedburner.com/TechCrunch/",
            "https://www.theverge.com/rss/index.xml",
        ]
    if category in {"india"}:
        return [
            "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
            "https://www.thehindu.com/news/national/feeder/default.rss",
            "https://www.indiatoday.in/rss/home",
        ]
    if category in {"science"}:
        return [
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "https://www.sciencedaily.com/rss/top/science.xml",
        ]
    if category in {"health"}:
        return [
            "https://feeds.bbci.co.uk/news/health/rss.xml",
            "https://www.sciencedaily.com/rss/top/health.xml",
        ]
    if category in {"entertainment", "movies"}:
        return [
            "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "https://www.theguardian.com/film/rss",
        ]
    # Default: general / politics / world
    return [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.reutersagency.com/feed/?best-topics=world&post_type=best",
    ]


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = unescape(s)
    return _multi_space_re.sub(" ", s).strip()


def _parse_rss_items(xml_text: str, limit: int) -> List[dict]:
    items: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    # RSS: channel/item; Atom: entry. Handle both lightly.
    # Ignore namespaces by checking tag suffix.
    def _tag_name(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    candidates = []
    for node in root.iter():
        name = _tag_name(node.tag)
        if name in {"item", "entry"}:
            candidates.append(node)

    for it in candidates[: max(20, limit * 5)]:
        title = ""
        desc = ""
        link = ""
        for child in list(it):
            name = _tag_name(child.tag)
            if name == "title" and child.text:
                title = _strip_tags(child.text)
            elif name in {"description", "summary"}:
                desc = _strip_tags(child.text or "")
            elif name == "link":
                # Atom: <link href="..."/>
                href = child.attrib.get("href")
                if href:
                    link = href
                elif child.text:
                    link = child.text.strip()

        if not title:
            continue

        combined = f"{title}. {desc}".strip() if desc else title
        items.append({"title": title, "text": combined, "link": link})
        if len(items) >= limit:
            break
    return items


def _fetch_rss_items(url: str, limit: int = 10) -> List[dict]:
    """Fetch and parse up to `limit` RSS/Atom items."""
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "fake-news-demo/1.0"})
        if resp.status_code != 200 or not resp.text:
            return []
        return _parse_rss_items(resp.text, limit=limit)
    except Exception:
        return []


def fetch_live_news_samples(category: str, limit: int = 12) -> List[dict]:
    """Fetch multiple items, shuffle them, and return a deduped list."""
    sources = _build_rss_sources(category)
    all_items: List[dict] = []
    per_source = max(4, min(12, (limit // max(1, len(sources))) + 6))

    for url in sources:
        items = _fetch_rss_items(url, limit=per_source)
        for it in items:
            it["source"] = url
            it["category"] = category
        all_items.extend(items)

    # Deduplicate by title.
    seen = set()
    deduped = []
    for it in all_items:
        key = (it.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    random.shuffle(deduped)
    return deduped[:limit]


def _reservoir_sample_csv_texts(path: str, label: str, k: int) -> List[Dict[str, str]]:
    """Reservoir-sample up to k rows from a CSV file's 'text' column."""
    sample: List[Dict[str, str]] = []
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            return []

        for row in reader:
            txt = (row.get("text") or "").strip()
            if not txt:
                continue
            title = (row.get("title") or "").strip()
            n += 1
            item = {"title": title, "text": txt, "source": os.path.basename(path), "label": label}

            if len(sample) < k:
                sample.append(item)
            else:
                j = random.randint(1, n)
                if j <= k:
                    sample[j - 1] = item
    return sample


def _ensure_demo_pool_loaded():
    global _demo_pool_loaded
    if _demo_pool_loaded:
        return
    # Keep memory bounded by sampling a small pool.
    if os.path.exists(TRUE_CSV_PATH):
        _demo_pool["real"] = _reservoir_sample_csv_texts(TRUE_CSV_PATH, "real", DEMO_POOL_SIZE)
    if os.path.exists(FAKE_CSV_PATH):
        _demo_pool["fake"] = _reservoir_sample_csv_texts(FAKE_CSV_PATH, "fake", DEMO_POOL_SIZE)
    _demo_pool_loaded = True


def _pick_demo_item(label: str) -> Optional[Dict[str, str]]:
    _ensure_demo_pool_loaded()
    items = _demo_pool.get(label.lower(), [])
    if not items:
        return None
    return random.choice(items)


def _confidence_of_label(out: dict, desired_label: str) -> float:
    desired_label = desired_label.lower()
    if desired_label == "real":
        return out["real_prob"] * 100
    return out["fake_prob"] * 100


def _try_find_demo_item(
    label: str, min_conf: float, max_conf: float, attempts: int
) -> Tuple[Optional[Dict[str, str]], Optional[dict]]:
    """Try multiple random samples and return one within the confidence band."""
    best = None
    best_out = None
    best_gap = float("inf")

    for _ in range(max(1, attempts)):
        item = _pick_demo_item(label)
        if not item:
            break
        out = predict_proba(item["text"])
        conf = _confidence_of_label(out, label)

        # Prefer items whose model agrees with the desired label.
        agrees = (out["prediction"].lower() == label.lower())
        if agrees and (min_conf <= conf <= max_conf):
            return item, out

        # Track closest "agreeing" sample (or closest overall if none agree).
        gap = 0.0
        if conf < min_conf:
            gap = min_conf - conf
        elif conf > max_conf:
            gap = conf - max_conf
        else:
            gap = 0.0

        score_gap = gap + (50.0 if not agrees else 0.0)
        if score_gap < best_gap:
            best_gap = score_gap
            best = item
            best_out = out

    return best, best_out


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _compute_metrics() -> dict:
    """Compute accuracy/precision/recall/F1 for the current saved model.

    Precision/recall/F1 are computed with respect to the positive class:
    `METRICS_POSITIVE_LABEL` (default: Fake = 0).
    """
    def _load_labeled_texts(path: str, label: int) -> Tuple[List[str], List[int]]:
        texts: List[str] = []
        labels: List[int] = []
        if not os.path.exists(path):
            return texts, labels
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                txt = (row.get("text") or row.get("title") or "").strip()
                if not txt:
                    continue
                texts.append(txt)
                labels.append(label)
        return texts, labels

    true_texts, true_labels = _load_labeled_texts(TRUE_CSV_PATH, 1)
    fake_texts, fake_labels = _load_labeled_texts(FAKE_CSV_PATH, 0)

    texts = true_texts + fake_texts
    y_true = np.array(true_labels + fake_labels, dtype=np.int64)

    if len(texts) < 50:
        raise RuntimeError("Not enough data found to compute metrics.")

    cleaned_texts = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(seqs, maxlen=MAX_LEN)

    n = len(padded)
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.8 * n)
    test_idx = idx[split:]

    X_test = padded[test_idx]
    y_test = y_true[test_idx]

    preds = model.predict(X_test, verbose=0).reshape(-1)
    preds = np.clip(preds.astype(float), 0.0, 1.0)

    # Our predict_proba uses OUTPUT_PROB_FOR_REAL mapping; replicate it here for metrics.
    real_prob = preds if OUTPUT_PROB_FOR_REAL else (1.0 - preds)
    pred_label_real = (real_prob >= 0.5).astype(np.int64)  # 1=Real, 0=Fake

    y_pred = pred_label_real

    accuracy = float((y_pred == y_test).mean() * 100.0)

    pos = METRICS_POSITIVE_LABEL
    tp = float(((y_pred == pos) & (y_test == pos)).sum())
    fp = float(((y_pred == pos) & (y_test != pos)).sum())
    fn = float(((y_pred != pos) & (y_test == pos)).sum())

    precision = _safe_div(tp, tp + fp) * 100.0
    recall = _safe_div(tp, tp + fn) * 100.0
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "positive_label": pos,
    }


def _get_cached_metrics() -> dict:
    # Cache on disk so Flask startup remains fast.
    if os.path.exists(METRICS_CACHE_PATH):
        try:
            mtime = os.path.getmtime(METRICS_CACHE_PATH)
            if time.time() - mtime <= METRICS_TTL_SECONDS:
                with open(METRICS_CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

    metrics = _compute_metrics()
    try:
        with open(METRICS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception:
        # Metrics will still be returned even if we can't write cache.
        pass
    return metrics


def predict_proba(news_text: str) -> dict:
    cleaned = clean_text(news_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(padded, verbose=0)
    # Binary sigmoid model: output is probability for class "1".
    prob = float(pred[0][0])
    prob = max(0.0, min(1.0, prob))

    real_prob = prob if OUTPUT_PROB_FOR_REAL else (1.0 - prob)
    fake_prob = 1.0 - real_prob

    is_real = real_prob >= 0.5
    prediction = "Real" if is_real else "Fake"
    confidence = (real_prob if is_real else fake_prob) * 100

    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
    }


@app.get("/")
def home():
    return render_template("index.html", min_chars=MIN_CHARS, max_len=MAX_LEN)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("news", "") or request.form.get("news", "")
    text = text or ""

    stripped = text.strip()
    word_count = len(stripped.split()) if stripped else 0

    if not stripped:
        return (
            jsonify(
                {
                    "error": "Please enter some news text.",
                    "word_count": word_count,
                }
            ),
            400,
        )

    if len(stripped) < MIN_CHARS:
        return (
            jsonify(
                {
                    "error": f"Please enter at least {MIN_CHARS} characters.",
                    "word_count": word_count,
                }
            ),
            400,
        )

    try:
        out = predict_proba(stripped)
        return jsonify(
            {
                "prediction": out["prediction"],
                "confidence": round(out["confidence"], 2),
                "real_prob": round(out["real_prob"] * 100, 2),
                "fake_prob": round(out["fake_prob"] * 100, 2),
                "word_count": word_count,
                # Demo placeholders; replace with real metrics from your training run if desired.
                "metrics": {
                    "accuracy": "—",
                    "precision": "—",
                    "recall": "—",
                    "f1": "—",
                },
            }
        )
    except Exception:
        # Avoid leaking stack traces to the client.
        return (
            jsonify(
                {
                    "error": "Prediction failed. Please try again with different text.",
                    "word_count": word_count,
                }
            ),
            500,
        )


@app.get("/demo-sample")
def demo_sample():
    """Return a labeled demo sample from True.csv/Fake.csv (randomized).

    This is for reliable demos: the label comes from your dataset, and we optionally
    try to find a sample whose model confidence falls within a desired band.
    """
    label = (request.args.get("label", "real") or "real").lower()
    if label not in {"real", "fake"}:
        label = "real"

    try:
        min_conf = float(request.args.get("min_conf", "85" if label == "real" else "80"))
    except Exception:
        min_conf = 85.0 if label == "real" else 80.0
    try:
        max_conf = float(request.args.get("max_conf", "90"))
    except Exception:
        max_conf = 90.0

    item, out = _try_find_demo_item(
        label=label,
        min_conf=min_conf,
        max_conf=max_conf,
        attempts=DEMO_MAX_ATTEMPTS,
    )

    if not item or not out:
        return (
            jsonify(
                {
                    "error": "Demo samples unavailable. Ensure True.csv and Fake.csv exist with a 'text' column.",
                    "label": label,
                }
            ),
            500,
        )

    # Provide a single sample (as requested) and also the model output.
    wc = len((item.get("text") or "").split())
    return jsonify(
        {
            "requested_label": label,
            "item": {
                "title": item.get("title") or "",
                "text": item.get("text") or "",
                "source": item.get("source") or "",
            },
            "model": {
                "prediction": out["prediction"],
                "confidence": round(
                    (out["real_prob"] if out["prediction"] == "Real" else out["fake_prob"]) * 100, 2
                ),
                "real_prob": round(out["real_prob"] * 100, 2),
                "fake_prob": round(out["fake_prob"] * 100, 2),
                "word_count": wc,
            },
            "band": {"min_conf": min_conf, "max_conf": max_conf},
        }
    )


@app.get("/news")
def live_news():
    """Return multiple live headlines + snippets for a requested category."""
    category = request.args.get("category", "general")
    try:
        limit = int(request.args.get("limit", "12"))
    except Exception:
        limit = 12
    limit = max(3, min(30, limit))

    items = fetch_live_news_samples(category, limit=limit)
    if not items:
        return (
            jsonify(
                {
                    "error": "Unable to fetch live news at the moment. Please try again or use your own text.",
                    "category": category,
                }
            ),
            502,
        )

    return jsonify({"category": category, "count": len(items), "items": items})


def fetch_viral_news_samples(category: str, limit: int = 20) -> List[dict]:
    """Return viral/trending headlines by aggregating multiple RSS categories."""
    category = (category or "top").lower()

    feeds: List[str] = []
    if category in {"top", "world"}:
        feeds += [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "https://www.reutersagency.com/feed/?best-topics=world&post_type=best",
        ]
    if category in {"top", "politics"}:
        feeds += [
            "https://feeds.bbci.co.uk/news/politics/rss.xml",
        ]
    if category in {"top", "war", "conflict"}:
        feeds += [
            "https://www.reutersagency.com/feed/?best-topics=world&post_type=best",
            "https://feeds.bbci.co.uk/news/world/rss.xml",
        ]
    if category in {"top", "sports"}:
        feeds += [
            "https://feeds.bbci.co.uk/sport/rss.xml",
            "https://www.espn.com/espn/rss/news",
        ]
    if category in {"top", "health"}:
        feeds += [
            "https://feeds.bbci.co.uk/news/health/rss.xml",
            "https://www.sciencedaily.com/rss/top/health.xml",
        ]
    if category in {"top", "science"}:
        feeds += [
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "https://www.sciencedaily.com/rss/top/science.xml",
        ]
    if category in {"top", "tech", "technology"}:
        feeds += [
            "https://feeds.arstechnica.com/arstechnica/technology-lab",
            "https://www.theverge.com/rss/index.xml",
            "https://feeds.feedburner.com/TechCrunch/",
        ]
    if category in {"top", "business"}:
        feeds += [
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://www.moneycontrol.com/rss/latestnews.xml",
            "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        ]

    # Always include a couple of general feeds as fallback.
    feeds += [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    ]

    all_items: List[dict] = []
    for url in feeds:
        items = _fetch_rss_items(url, limit=8)
        for it in items:
            it["source"] = url
            it["category"] = category
        all_items.extend(items)

    # Dedup + shuffle.
    seen = set()
    deduped: List[dict] = []
    for it in all_items:
        key = (it.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    random.shuffle(deduped)
    return deduped[:limit]


@app.get("/metrics")
def metrics():
    """Return cached demo metrics computed from the current saved model."""
    try:
        data = _get_cached_metrics()
        return jsonify(
            {
                "accuracy": data["accuracy"],
                "precision": data["precision"],
                "recall": data["recall"],
                "f1": data["f1"],
                "positive_label": data.get("positive_label", METRICS_POSITIVE_LABEL),
                "cached": os.path.exists(METRICS_CACHE_PATH),
            }
        )
    except Exception:
        return (
            jsonify(
                {
                    "error": "Unable to compute metrics right now.",
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                }
            ),
            500,
        )


@app.get("/viral")
def viral():
    """Return viral/trending worldwide headlines across categories."""
    category = request.args.get("category", "top")
    try:
        limit = int(request.args.get("limit", "20"))
    except Exception:
        limit = 20
    limit = max(5, min(40, limit))

    items = fetch_viral_news_samples(category, limit=limit)
    if not items:
        return (
            jsonify({"error": "Unable to fetch viral news right now.", "category": category}),
            502,
        )
    return jsonify({"category": category, "count": len(items), "items": items})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)