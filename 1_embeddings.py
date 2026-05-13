from sentence_transformers import SentenceTransformer

# 1. Load the 'Map Maker' (Embedding Model)
# This model is small, fast, and perfect for learning.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Our 'Knowledge Base' (Simple Law Examples)
documents = [
    "The penalty for speeding is a fine of $150.",
    "Theft is defined as taking property without permission.",
    "A contract requires an offer, acceptance, and consideration.",
    "Speeding in a school zone results in a double fine."
]

print("Converting text to math (Embeddings)...")

# 3. Convert our documents into Vectors
embeddings = model.encode(documents)

# 4. Let's look at one!
print(f"\nDocument: '{documents[0]}'")
print(f"Vector (First 5 numbers): {embeddings[0][:5]}")
print(f"Total numbers in this vector: {len(embeddings[0])}")

# 5. TEST: Let's see how the computer 'sees' a query
query = "How much do I pay for driving too fast?"
query_embedding = model.encode([query])

print(f"\nQuery: '{query}'")
print(f"Query Vector (First 5): {query_embedding[0][:5]}")