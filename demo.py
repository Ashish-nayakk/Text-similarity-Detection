# demo.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Input
text1 = input("Enter something: ")
text2 = input("Enter something: ")

# Convert the texts into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# Calculate cosine similarity
similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
percentage = similarity_score * 100
print(f"Similarity: {percentage:.2f}%")

# Tokenize into sets (case-insensitive, simple split)
words1 = set(text1.lower().split())
words2 = set(text2.lower().split())

# Matching words
matching_words = words1.intersection(words2)

if matching_words:
    print("Matching words:", ", ".join(matching_words))
else:
    print("No Exact word found which matches.")

# Precision, Recall, F1
TP = len(words1 & words2)
FP = len(words2 - words1)
FN = len(words1 - words2)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")