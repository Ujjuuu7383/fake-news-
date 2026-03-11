import os
import json
import requests
import ollama
import torch
from flask import Flask, request, jsonify, render_template

# =========================
# CONFIG
# =========================

SERPER_URL = "https://google.serper.dev/news"

SERPER_API_KEY = os.getenv("SERPER_API_KEY") or "3cd9effd78d88fd5ac0b20a280f2a58e2c2a10c6"
if not SERPER_API_KEY:
    raise RuntimeError("SERPER_API_KEY not set")

OLLAMA_MODEL = "qwen2.5:14b"

app = Flask(__name__)

# =========================
# FEATURE TOOLS
# =========================

import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, GPT2LMHeadModel, GPT2TokenizerFast
from scipy.special import expit

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

toxicity_pipeline = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    tokenizer="unitary/toxic-bert",
    return_all_scores=True
)

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()


def readability_score(text: str) -> float:
    return textstat.flesch_reading_ease(text)


def sentiment_score(text: str) -> float:
    return sia.polarity_scores(text)["compound"]


def toxicity_score(text: str) -> float:
    scores = toxicity_pipeline(text)[0]
    for s in scores:
        if s["label"].lower() == "toxic":
            return float(s["score"])
    return 0.0


def detected_synthetic_score(text: str) -> float:
    enc = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = gpt2_model(**enc, labels=enc["input_ids"]).loss.item()
    perplexity = torch.exp(torch.tensor(loss)).item()
    return float(expit((50 - perplexity) / 10))


def extract_text_features(text: str) -> dict:
    return {
        "readability_score": readability_score(text),
        "sentiment_score": sentiment_score(text),
        "toxicity_score": toxicity_score(text),
        "detected_synthetic_score": detected_synthetic_score(text),
        "text_length": len(text)
    }

# =========================
# NEWS FETCHING
# =========================

def fetch_news_text(topic, num_results=5):
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {"q": topic, "num": num_results}

    response = requests.post(
        SERPER_URL,
        headers=headers,
        json=payload,
        timeout=15
    )

    if response.status_code != 200:
        return []

    data = response.json()

    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", "")
        }
        for item in data.get("news", [])
    ]


def fetch_news_with_features(topic, num_results=5):
    articles = fetch_news_text(topic, num_results)
    enriched = []

    for a in articles:
        text = f"{a['title']}. {a['snippet']}"
        enriched.append({
            "title": a["title"],
            "snippet": a["snippet"],
            "url": a["url"],
            **extract_text_features(text)
        })

    return enriched

# =========================
# LLM CLAIM–ARTICLE ANALYSIS
# =========================

def analyze_news_llama3(news_item: dict, user_claim: str) -> dict:
    prompt = f"""
You are a professional fact-checker.

USER CLAIM:
"{user_claim}"

NEWS ARTICLE:
Title: {news_item['title']}
Snippet: {news_item['snippet']}

TASK:
Determine how the article relates to the USER CLAIM.

Rules:
- SUPPORTS → claim is REAL
- REFUTES → claim is FAKE
- DOES NOT ADDRESS → UNKNOWN

A debunking article means the claim is FAKE.

AUXILIARY SIGNALS (supporting only):
- Readability: {news_item['readability_score']}
- Sentiment: {news_item['sentiment_score']}
- Toxicity: {news_item['toxicity_score']}
- Synthetic probability: {news_item['detected_synthetic_score']}

OUTPUT JSON ONLY:
{{
  "relationship": "SUPPORTS" | "REFUTES" | "NOT_ADDRESSED",
  "verdict": "REAL" | "FAKE" | "UNKNOWN",
  "confidence": 0 to 1,
  "reasoning": "concise explanation"
}}
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )

    try:
        return json.loads(response["message"]["content"])
    except Exception:
        return {
            "relationship": "UNKNOWN",
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": "Invalid LLM output"
        }


def aggregate_claim_verdict(analyses: list) -> dict:
    supports = sum(a["relationship"] == "SUPPORTS" for a in analyses)
    refutes = sum(a["relationship"] == "REFUTES" for a in analyses)

    if refutes > supports:
        return {"final_verdict": "FAKE", "confidence": min(0.9, refutes / len(analyses))}

    if supports > refutes:
        return {"final_verdict": "REAL", "confidence": min(0.9, supports / len(analyses))}

    return {"final_verdict": "UNKNOWN", "confidence": 0.5}

# =========================
# FLASK ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/factcheck", methods=["POST"])
def factcheck():
    data = request.get_json()
    if not data or "claim" not in data:
        return jsonify({"error": "Missing claim"}), 400

    claim = data["claim"]

    news = fetch_news_with_features(claim, 5)

    analyses = []
    for item in news:
        result = analyze_news_llama3(item, claim)
        analyses.append({
            "title": item["title"],
            "url": item["url"],
            **result
        })

    final = aggregate_claim_verdict(analyses)

    return jsonify({
        "claim": claim,
        "final_verdict": final["final_verdict"],
        "confidence": final["confidence"],
        "analyses": analyses
    })

# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
