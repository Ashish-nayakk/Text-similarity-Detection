import streamlit as st
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import numpy as np

# ------------------------
# Load models (once)
# ------------------------
bi_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-large")

st.set_page_config(page_title="Academic Integrity Prediction", layout="wide")
st.title("📑 Academic Integrity Prediction using Text Similarity")

# ------------------------
# User Input
# ------------------------
suspect_text = st.text_area("Enter Suspect Text (student submission):", height=200)
source_text = st.text_area("Enter Source Text (reference / PAN corpus snippet):", height=200)

if st.button("Check Plagiarism"):
    if suspect_text and source_text:
        # Step 1: Split into sentences
        suspect_sentences = [s.strip() for s in suspect_text.split(".") if s.strip()]
        source_sentences = [s.strip() for s in source_text.split(".") if s.strip()]

        # Step 2: Encode (bi-encoder for retrieval)
        suspect_emb = bi_encoder.encode(suspect_sentences, convert_to_tensor=True)
        source_emb = bi_encoder.encode(source_sentences, convert_to_tensor=True)

        # Step 3: Candidate retrieval
        cosine_scores = util.cos_sim(suspect_emb, source_emb)
        top_pairs = []
        for i in range(len(suspect_sentences)):
            best_match_id = int(np.argmax(cosine_scores[i]))
            best_match_score = float(cosine_scores[i][best_match_id])
            top_pairs.append((suspect_sentences[i], source_sentences[best_match_id], best_match_score))

        # Step 4: Refine with cross-encoder
        cross_inputs = [(pair[0], pair[1]) for pair in top_pairs]
        cross_scores = cross_encoder.predict(cross_inputs)

        # Step 5: Aggregate plagiarism probability
        doc_score = float(np.mean(cross_scores))

        # ------------------------
        # Output
        # ------------------------
        st.subheader(f"📊 Document Plagiarism Probability: **{doc_score:.2f}** (0=Original, 1=Plagiarized)")

        st.write("---")
        st.subheader("🔍 Sentence-Level Matches")
        for (sus, src, _), score in zip(top_pairs, cross_scores):
            st.markdown(f"**Suspect:** {sus}")
            st.markdown(f"**Matched Source:** {src}")
            st.markdown(f"**Plagiarism Probability:** `{score:.2f}`")
            st.write("---")
    else:
        st.warning("Please enter both suspect and source texts.")
