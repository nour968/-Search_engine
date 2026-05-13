import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ---------------------------
# 1. CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------
# 2. LOAD + CLEAN DOCUMENTS (REMOVE HEADINGS)
# ---------------------------
def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("\n", ". ")
    sentences = text.split(". ")

    docs = []

    for s in sentences:
        s = s.strip()

        # remove headings / fragments
        if len(s.split()) < 6:
            continue

        if len(s) < 40:
            continue

        # keep only explanation sentences
        if any(k in s.lower() for k in [
            "is", "are", "means", "refers", "defined", "used", "consists"
        ]):
            docs.append(s)

    return docs


# ---------------------------
# 3. BUILD MODELS
# ---------------------------
def build_models(documents):

    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, show_progress_bar=False)

    return tfidf_vectorizer, tfidf_matrix, model, embeddings


# ---------------------------
# 4. IMPROVED ANSWER GENERATION (2–3 SENTENCES)
# ---------------------------
def get_short_answer(text, query):
    sentences = text.split(". ")
    query_words = set(query.lower().split())

    scored = []

    for s in sentences:
        words = set(s.lower().split())
        score = len(words.intersection(query_words))

        # boost definition sentences
        if any(k in s.lower() for k in ["is", "means", "refers", "defined"]):
            score += 2

        scored.append((score, s))

    # sort best sentences first
    scored.sort(reverse=True, key=lambda x: x[0])

    # take top 2–3 sentences
    top_sentences = [s for _, s in scored[:3]]

    answer = ". ".join(top_sentences)

    # safety limit (avoid huge outputs)
    if len(answer.split()) > 120:
        answer = " ".join(answer.split()[:120]) + "..."

    return answer


# ---------------------------
# UTILITIES: NORMALIZE + MMR
# ---------------------------
def normalize_scores(scores):
    """Min-max normalize a 1D numpy array safely."""
    min_v = np.min(scores)
    max_v = np.max(scores)
    if max_v - min_v < 1e-12:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)


def mmr_selection(candidate_embeddings, query_embedding, top_k, diversity=0.7):
    """
    Maximal Marginal Relevance (MMR) to pick diverse yet relevant items.

    candidate_embeddings: np.array of shape (n_candidates, dim)
    query_embedding: np.array of shape (1, dim)
    top_k: number of items to select
    diversity: lambda parameter balancing relevance vs diversity (0..1)
    Returns indices (relative to candidate_embeddings) of selected items.
    """
    if candidate_embeddings.shape[0] == 0:
        return []

    # cosine similarity between query and candidates
    sim_to_query = cosine_similarity(query_embedding, candidate_embeddings).flatten()

    # pairwise similarity among candidates
    pairwise_sim = cosine_similarity(candidate_embeddings)

    selected = []
    # pick the most relevant as first
    first = int(np.argmax(sim_to_query))
    selected.append(first)

    # iteratively select items that maximize MMR objective
    candidate_indices = set(range(candidate_embeddings.shape[0]))
    candidate_indices.remove(first)

    while len(selected) < min(top_k, candidate_embeddings.shape[0]):
        mmr_scores = {}
        for idx in candidate_indices:
            relevance = sim_to_query[idx]
            diversity_penalty = max([pairwise_sim[idx][s] for s in selected]) if selected else 0
            mmr_score = (diversity * relevance) - ((1 - diversity) * diversity_penalty)
            mmr_scores[idx] = mmr_score

        # pick best remaining
        next_idx = max(mmr_scores, key=mmr_scores.get)
        selected.append(int(next_idx))
        candidate_indices.remove(next_idx)

    return selected


def compose_answer_with_mmr(query, documents, ranked, final_scores, model, embeddings, top_k=5, max_sentences=3, diversity=0.7):
    """Compose an answer by selecting diverse, relevant sentences from top_k documents using MMR.

    documents: list of sentence strings (each document is already a sentence in this project)
    ranked: array of document indices sorted by relevance desc
    embeddings: array of embeddings matching documents
    model: sentence transformer model (used to encode the query when needed)
    """
    # limit candidates to top_k ranked documents
    candidate_idxs = [int(i) for i in ranked[:max(top_k, 1)]]
    if not candidate_idxs:
        return get_short_answer(documents[int(ranked[0])], query)

    candidate_embeddings = embeddings[candidate_idxs]
    query_embedding = model.encode([query])

    # run MMR selection on the candidate set
    pick_rel_indices = mmr_selection(candidate_embeddings, query_embedding, top_k=max_sentences, diversity=diversity)

    # map back to document indices and collect text
    selected_texts = [documents[candidate_idxs[i]] for i in pick_rel_indices]

    # fallback: if nothing selected, pick the highest ranked doc(s)
    if not selected_texts:
        best_idx = int(ranked[0])
        return get_short_answer(documents[best_idx], query)

    answer = '. '.join(selected_texts)
    # trim answer length for safety
    if len(answer.split()) > 300:
        answer = ' '.join(answer.split()[:300]) + '...'
    return answer


# ---------------------------
# 5. SEARCH FUNCTION (HYBRID MODEL)
# ---------------------------
def search(query, tfidf_vectorizer, tfidf_matrix, model, embeddings, documents):
    query_clean = clean_text(query)

    # TF-IDF similarity
    query_tfidf = tfidf_vectorizer.transform([query_clean])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Semantic similarity
    query_embedding = model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, embeddings).flatten()

    # Normalize scores before combining to make weights meaningful
    tfidf_norm = normalize_scores(tfidf_scores)
    semantic_norm = normalize_scores(semantic_scores)

    # Hybrid score (configurable mix)
    alpha = 0.3  # TF-IDF weight
    final_scores = (alpha * tfidf_norm) + ((1 - alpha) * semantic_norm)

    # 🔥 SORT BY SIMILARITY (DESCENDING)
    ranked = np.argsort(final_scores)[::-1]

    print("\n📌 TOP MATCHED RESULTS (RANKED):\n")

    for rank, idx in enumerate(ranked[:5], start=1):
        print(f"#{rank}")
        print(f"Score: {final_scores[idx]:.4f}")
        print(documents[idx])
        print("-" * 80)

    # Compose a more informative & diverse answer using MMR
    answer = compose_answer_with_mmr(query, documents, ranked, final_scores, model, embeddings, top_k=8, max_sentences=3, diversity=0.7)

    print("\n📌 FINAL ANSWER:\n")
    print(answer)

    best_idx = int(ranked[0])
    print("\n📊 Best Confidence:", round(final_scores[best_idx], 4))
# ---------------------------
# 6. MAIN LOOP
# ---------------------------
def main():

    print("📂 Loading documents...")
    documents = []
    documents += load_documents("doc1.txt")
    documents += load_documents("doc2.txt")
    documents += load_documents("doc3.txt")

    if not documents:
        print("❌ No valid QA sentences found!")
        return

    print(f"✅ Loaded {len(documents)} clean QA sentences")

    print("⚙️ Building QA system...")

    tfidf_vectorizer, tfidf_matrix, model, embeddings = build_models(documents)

    print("🚀 Ready for questions!")

    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        search(query, tfidf_vectorizer, tfidf_matrix, model, embeddings, documents)


if __name__ == "__main__":
    main()