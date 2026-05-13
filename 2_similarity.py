from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Load the same model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Our Legal Knowledge
documents = [
    "The penalty for speeding is a fine of $150.",
    "Theft is defined as taking property without permission.",
    "A contract requires an offer, acceptance, and consideration.",
    "Speeding in a school zone results in a double fine."
]

# 3. User Query
query = "What is the cost for going over the speed limit?"

print(f"User Query: {query}\n")

# 4. Convert everything to math
doc_vectors = model.encode(documents)
query_vector = model.encode([query]) # Put in brackets because it expects a list

# 5. Calculate Similarity! 
# We compare the query_vector against ALL doc_vectors at once
scores = cosine_similarity(query_vector, doc_vectors)[0]

# 6. Show the results
for i in range(len(documents)):
    print(f"Score: {scores[i]:.4f} | Document: {documents[i]}")

# 7. Find the winner
best_idx = np.argmax(scores)
print(f"\n--- WINNING RESULT ---")
print(f"The most relevant law is: {documents[best_idx]}")