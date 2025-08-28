import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx

# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to read different file types
def read_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return None

# Streamlit UI
st.title("📄 Text Similarity Detection")

st.write("Upload two documents (txt/pdf/docx) or type manually to compare similarity.")

# Upload or manual input
col1, col2 = st.columns(2)

with col1:
    file1 = st.file_uploader("Upload First Document", type=["txt", "pdf", "docx"])
    text1 = st.text_area("Or Enter First Document Text")

with col2:
    file2 = st.file_uploader("Upload Second Document", type=["txt", "pdf", "docx"])
    text2 = st.text_area("Or Enter Second Document Text")

if st.button("Check Similarity"):
    # Get text from file or input
    doc1 = read_file(file1) if file1 else text1
    doc2 = read_file(file2) if file2 else text2

    if not doc1 or not doc2:
        st.error("Please provide both documents (upload or type).")
    else:
        # Encode with SBERT
        emb1, emb2 = model.encode([doc1, doc2], convert_to_tensor=True)

        # Cosine similarity
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        st.success(f"Similarity Score: **{similarity:.4f}**")

        # Interpretation
        if similarity >= 0.7:
            st.write("🔹 The documents are **Highly Similar**")
        elif similarity >= 0.4:
            st.write("🔸 The documents are **Moderately Similar**")
        else:
            st.write("⚪ The documents are **Not Similar**")
