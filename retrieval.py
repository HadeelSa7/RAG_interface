"""
retrieval.py - the four retrieval methods of the thesis.

The functions in PART 1 are copied UNCHANGED from the notebook
"Retrival_Methods_updated1.ipynb" (the cell number is written above each one).
PART 2 loads the data and the models exactly like the notebook did.
PART 3 runs one retrieval method with the same settings used in the experiments.
"""
import json
import os
import re

import numpy as np

# The final dataset of 101 chunks (a copy of the file saved by the notebook)
CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "data", "midocean_chunks_final.json")

# These variables are filled by load_models() (PART 2).
# The notebook functions below use them, just like the notebook variables.
all_chunks = []
corpus = []
bm25 = None
model = None              # BGE-M3
chunk_embeddings = None
tokenizer = None          # reranker tokenizer
reranker_model = None     # BGE-Reranker-v2-m3
torch = None


# =====================================================================
# PART 1 - ORIGINAL NOTEBOOK FUNCTIONS (copied without any change)
# =====================================================================

# ---- From notebook cell 10 (Step 2: Arabic normalization) ----
def normalize_arabic_keep_lines(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # توحيد أشكال الألف
        line = re.sub(r'[إأآٱ]', 'ا', line)

        # توحيد الألف المقصورة
        line = line.replace('ى', 'ي')

        # إزالة التشكيل
        line = re.sub(
            r'[\u0617-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]',
            '',
            line
        )

        # إزالة التطويل
        line = line.replace('ـ', '')

        # توحيد المسافات داخل السطر فقط
        line = re.sub(r'[ \t]+', ' ', line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ---- From notebook cell 28 (Tokenization for BM25) ----
ARABIC_STOPWORDS = {
    "في", "من", "علي", "الي", "عن", "مع", "بين", "حتي", "عند", "بعد", "قبل",
    "ما", "ماذا", "هل", "كيف", "متي", "اين", "لماذا", "اي",
    "هو", "هي", "هم", "هذا", "هذه", "ذلك", "تلك", "هناك",
    "التي", "الذي", "الذين", "ان", "انه", "انها", "او", "ثم", "كل", "كما", "ايضا", "قد", "الا", "لكن",
    "كان", "كانت", "يكون", "تكون", "يتم", "تم", "يمكن", "اذا", "حيث",
    "به", "بها", "له", "لها", "منه", "منها", "فيه", "فيها",
}

ARABIC_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")

DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

LIST_NUMBER_PATTERN = re.compile(r"(?<!\S)\d{1,2}\s*[-.)](?!\d)")


def remove_prefix(word):
    """Remove the Arabic definite article (and similar prefixes) from a word."""
    for prefix in ARABIC_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            return word[len(prefix):]
    return word


def tokenize_arabic(text):
    """Turn a text into a clean list of tokens. Used for BOTH the chunks and the questions."""
    #text = normalize_arabic(text)                 # 1) same normalization as Step 2
    text = normalize_arabic_keep_lines(text)
    text = text.translate(DIGIT_TABLE)            #    Arabic digits -> normal digits
    text = text.lower()                           #    English letters -> lowercase

    text = LIST_NUMBER_PATTERN.sub(" ", text)     # 2) remove list numbers like "1-"
    text = re.sub(r"[^\u0621-\u064Aa-z0-9\s]", " ", text)   # 3) keep only Arabic letters, English letters, digits

    tokens = []
    for word in text.split():                     # 4) split into words
        if word in ARABIC_STOPWORDS:              # 6) stop-word
            continue
        word = remove_prefix(word)                # 7) remove "ال"
        if len(word) < 2 or word in ARABIC_STOPWORDS:   # 5) too short / stop-word
            continue
        tokens.append(word)

    return tokens


# ---- From notebook cell 31 (A - BM25, sparse retrieval) ----
def bm25_search(query, top_k=5):
    
    # تجهيز الاستعلام
    query_tokens = tokenize_arabic(query)
    
    # حساب الدرجات
    scores = bm25.get_scores(query_tokens)
    
    # الحصول على أعلى النتائج
    top_indices = scores.argsort()[::-1][:top_k]
    
    results = []
    
    for rank, idx in enumerate(top_indices, start=1):
        
        chunk = all_chunks[idx]
        ##نرجع هنا
        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "section": chunk["section"],
            "score": float(scores[idx]),
            "text": chunk["text"]
        })
    
    return results


# ---- From notebook cell 44 (B - BGE-M3, dense retrieval) ----
def dense_search(query_embedding, top_k=10):
    scores = np.dot(chunk_embeddings, query_embedding)
    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        chunk = all_chunks[idx]

        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "section": chunk["section"],
            "score": float(scores[idx]),
            "text": chunk["text"]
        })

    return results


# ---- From notebook cell 47 (C - Hybrid retrieval with RRF) ----
def hybrid_search(query_embedding, query_text, top_k=10, rrf_k=60):
    
    # -------------------------
    # 1. BM25 ranking
    # -------------------------
    query_tokens = tokenize_arabic(query_text)
    bm25_scores = bm25.get_scores(query_tokens)
    
    bm25_indices = bm25_scores.argsort()[::-1]
    
    # -------------------------
    # 2. Dense ranking
    # -------------------------
    dense_scores = np.dot(chunk_embeddings, query_embedding)
    
    dense_indices = dense_scores.argsort()[::-1]
    
    # -------------------------
    # 3. RRF scores
    # -------------------------
    rrf_scores = np.zeros(len(all_chunks))
    
    for rank, idx in enumerate(bm25_indices, start=1):
        rrf_scores[idx] += 1 / (rrf_k + rank)
    
    for rank, idx in enumerate(dense_indices, start=1):
        rrf_scores[idx] += 1 / (rrf_k + rank)
    
    # -------------------------
    # 4. Get top-k
    # -------------------------
    top_indices = rrf_scores.argsort()[::-1][:top_k]
    
    results = []
    
    for rank, idx in enumerate(top_indices, start=1):
        chunk = all_chunks[idx]
        
        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "section": chunk["section"],
            "score": float(rrf_scores[idx]),
            "text": chunk["text"]
        })
    
    return results


# ---- From notebook cell 57 (RQ2 - cross-encoder reranking) ----
def rerank_results(query, results):
    pairs = [
        [query, result["text"]]
        for result in results
    ]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = reranker_model(**inputs)

    scores = outputs.logits.view(-1).cpu().numpy()

    reranked = []

    for result, score in zip(results, scores):
        reranked.append({
            **result,
            "reranker_score": float(score)
        })

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    for rank, result in enumerate(reranked, start=1):
        result["rerank_rank"] = rank

    return reranked


# =====================================================================
# PART 2 - LOAD DATA AND MODELS (same steps as the notebook)
# =====================================================================

def load_models():
    """Load the chunks, build BM25, load BGE-M3 and the reranker.
    This is slow (a few minutes the first time), so app.py calls it only once."""
    global all_chunks, corpus, bm25, model, chunk_embeddings, tokenizer, reranker_model, torch

    import torch as torch_library
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch = torch_library

    # 1. Load the 101 final chunks (saved by the notebook)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # 2. BM25 index (notebook cells 29-30)
    corpus = [tokenize_arabic(chunk["text"]) for chunk in all_chunks]
    bm25 = BM25Okapi(corpus)

    # 3. BGE-M3 model and chunk embeddings (notebook cells 40 and 42)
    model = SentenceTransformer("BAAI/bge-m3")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    # 4. Reranker model (notebook cell 55)
    reranker_name = "BAAI/bge-reranker-v2-m3"
    tokenizer = AutoTokenizer.from_pretrained(reranker_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
    reranker_model.eval()


# =====================================================================
# PART 3 - RUN ONE METHOD (same settings as the experiments)
# =====================================================================

def encode_query(question):
    """Turn the question into a BGE-M3 vector (same settings as notebook cell 43)."""
    return model.encode(
        [question],
        batch_size=8,
        show_progress_bar=False,
        normalize_embeddings=True
    )[0]


def run_method(method_name, question):
    """Run one retrieval method and return its top-10 results."""

    if method_name == "BM25":
        # notebook cell 37: bm25_search(query, top_k=10)
        return bm25_search(question, top_k=10)

    query_embedding = encode_query(question)

    if method_name == "BGE-M3":
        # notebook cell 44: dense_search(query_embedding, top_k=10)
        return dense_search(query_embedding, top_k=10)

    if method_name == "Hybrid":
        # notebook cell 48: hybrid_search(..., top_k=10), RRF k = 60 (default)
        return hybrid_search(query_embedding=query_embedding, query_text=question, top_k=10)

    if method_name == "Hybrid + Reranking":
        # notebook cell 58: take the Hybrid top-20, rerank them, keep the top-10
        hybrid_results = hybrid_search(query_embedding=query_embedding, query_text=question, top_k=20)
        reranked_results = rerank_results(query=question, results=hybrid_results)
        return reranked_results[:10]

    raise ValueError("Unknown method: " + method_name)
