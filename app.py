import streamlit as st
import numpy as np

from main import (
    clean_text,
    load_documents,
    build_models,
    get_short_answer,
    compose_answer_with_mmr,
    normalize_scores
)

from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Search Engine", page_icon="🔍")

st.title("🔍 AI Document Search Engine")
st.write("Ask questions from your documents using TF-IDF + AI Ranking")


# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top K results", 1, 10, 5)
    tfidf_weight = st.slider("TF-IDF weight (hybrid)", 0.0, 1.0, 0.3)
    diversity = st.slider("MMR diversity", 0.0, 1.0, 0.7)
    max_sentences = st.slider("Max answer sentences", 1, 8, 3)

    show_docs = st.checkbox("Show documents", False)
    detailed_answer = st.checkbox("Detailed answer", True)


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
documents = []
documents += load_documents("doc1.txt")
documents += load_documents("doc2.txt")
documents += load_documents("doc3.txt")

if not documents:
    st.error("❌ No valid documents found!")
    st.stop()

st.success(f"✅ Loaded {len(documents)} documents")


# -----------------------------
# BUILD MODEL (TF-IDF + EMBEDDINGS)
# -----------------------------
st.info("⚙️ Building model... please wait")

tfidf_vectorizer, tfidf_matrix, model, embeddings = build_models(documents)

st.success("🚀 System Ready!")


# -----------------------------
# QUERY INPUT
# -----------------------------
query = st.text_input("🔎 Enter your question:")


# -----------------------------
# SEARCH FUNCTION
# -----------------------------
if query:

    query_clean = clean_text(query)

    # TF-IDF similarity
    query_tfidf = tfidf_vectorizer.transform([query_clean])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Semantic similarity
    query_embedding = model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, embeddings).flatten()

    # Normalize
    tfidf_norm = normalize_scores(tfidf_scores)
    semantic_norm = normalize_scores(semantic_scores)

    # Hybrid score
    final_scores = (tfidf_weight * tfidf_norm) + ((1 - tfidf_weight) * semantic_norm)

    ranked = np.argsort(final_scores)[::-1]


    # -------------------------
    # TOP RESULTS
    # -------------------------
    st.subheader("📌 Top Results")

    for i, idx in enumerate(ranked[:top_k], 1):
        st.markdown(f"### #{i}")
        st.write(f"Score: {final_scores[idx]:.4f}")
        st.write(documents[int(idx)])
        st.markdown("---")


    # -------------------------
    # FINAL ANSWER
    # -------------------------
    if detailed_answer:
        answer = compose_answer_with_mmr(
            query,
            documents,
            ranked,
            final_scores,
            model,
            embeddings,
            top_k=max(top_k, 10),
            max_sentences=max_sentences,
            diversity=diversity
        )
    else:
        answer = get_short_answer(documents[int(ranked[0])], query)

    st.subheader("📌 Final Answer")
    st.write(answer)

    st.metric("Best Confidence", f"{final_scores[int(ranked[0])]:.4f}")