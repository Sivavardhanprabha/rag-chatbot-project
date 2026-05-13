import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. INITIALIZE THE BRAIN (The Map Maker)
# ==========================================
# This model turns text into 384-dimensional vectors.
print("Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# 2. OUR KNOWLEDGE BASE (The Data)
# ==========================================
legal_docs = [
    "The penalty for speeding is a fine of $150.",
    "Theft is defined as taking property without permission.",
    "A contract requires an offer, acceptance, and consideration.",
    "Speeding in a school zone results in a double fine.",
    "Copyright protects original works of authorship.",
    "A tort is a civil wrong that causes a claimant to suffer loss.",
    "A person who commits theft may be sentenced to jail time.",
    "To drive a car, one must possess a valid driver's license."
]

# ==========================================
# 3. CONVERSION (Text -> Math)
# ==========================================
# FAISS is built in C++ and requires 'float32' numbers to be fast.
print("Creating embeddings for the database...")
doc_vectors = model.encode(legal_docs).astype('float32')

# ==========================================
# 4. THE WAREHOUSE (FAISS Index)
# ==========================================
# We tell FAISS: "Every vector we give you will have 384 dimensions."
dimension = doc_vectors.shape[1] 
index = faiss.IndexFlatL2(dimension) # 'FlatL2' calculates exact Euclidean distance

# Add our document vectors to the warehouse
index.add(doc_vectors)
print(f"Success! {index.ntotal} documents are now indexed and searchable.\n")

# ==========================================
# 5. THE USER QUERY (The Search)
# ==========================================
query = "What are the legal consequences of stealing?"
print(f"USER QUERY: {query}")

# Convert query to the same math format (384-D vector)
query_vector = model.encode([query]).astype('float32')

# ==========================================
# 6. RETRIEVAL (Top K Search)
# ==========================================
# k=3 means "Find the 3 closest matches in the whole library"
k = 3
distances, indices = index.search(query_vector, k)

# ==========================================
# 7. DISPLAY THE RESULTS
# ==========================================
print("\n--- TOP 3 SEARCH RESULTS ---")

for i in range(k):
    # 'indices' gives us the position in our 'legal_docs' list
    doc_id = indices[0][i]
    
    # 'distances' tells us how far away the point is (Lower = Closer/Better)
    dist = distances[0][i]
    
    content = legal_docs[doc_id]
    
    print(f"RANK {i+1}:")
    print(f"  Content : {content}")
    print(f"  Distance: {dist:.4f}")
    print("-" * 40)

# ==========================================
# 8. WHY THIS MATTERS FOR RAG (The 'So What?')
# ==========================================
# Tomorrow, we will take 'Rank 1' and 'Rank 2', and 
# feed them into Groq to generate a final answer.