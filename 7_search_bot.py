import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the AI Model (The Translator)
# It MUST be the exact same model used to build the database!
print("Initializing AI Translator...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load the Database Files from your hard drive
print("Loading vector database indexes...")
index = faiss.read_index("my_vector_db.index")

with open("my_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 3. DEFINE YOUR QUESTION
# You can change this text to test different parts of your PDF!
user_query = "Who is this nondisclosure agreement made between?"

print(f"\nUser Question: '{user_query}'")
print("Searching database parallelly using Euclidean geometry...")

# 4. Convert your question into a math vector point
query_vector = model.encode([user_query])
query_vector_as_numpy = np.array(query_vector).astype('float32')

# 5. Search the FAISS geometric map
# k=2 means "retrieve the top 2 closest/most similar chunks"
distances, indices = index.search(query_vector_as_numpy, k=2)

print("\n--- 🤖 BOT RETRIEVED THE FOLLOWING EVIDENCE ---")

# Loop through the matching results and print them out
for i, chunk_index in enumerate(indices[0]):
    print(f"\n[Match #{i+1} - Chunk ID #{chunk_index}]")
    print(chunks[chunk_index])
    print("-" * 50)