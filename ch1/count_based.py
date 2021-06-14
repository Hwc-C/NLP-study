from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "I like fruits. Fruits like bananas",
    "I love bananas but eat an apple",
    "An apple a day keeps the doctor away"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())

# X.toarray()

query = vectorizer.transform(["apple and bananas"])
print("\nCosine similarity: \n", cosine_similarity(X, query))