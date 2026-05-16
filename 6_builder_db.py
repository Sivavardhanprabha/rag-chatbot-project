import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# 1. THE CHUNKER LOGIC (From your 5_chunker.py success)
def get_pdf_chunks(file_path, chunk_size=500, overlap=50):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    # Squish extra spaces just like you did!
    text = " ".join(text.split())

    chunks = []
    step = chunk_size - overlap 
    for i in range(0, len(text), step):
        chunks.append(text[i : i + chunk_size])
    return chunks

# --- EXECUTION ---
pdf_path = "D:/rag_chatbot_project/sample.pdf"

print("1. Slicing PDF into overlapping chunks...")
chunks = get_pdf_chunks(pdf_path)
print(f"   Done! Created {len(chunks)} chunks.")

print("\n2. Loading the AI Embedding Model (the word translator)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("\n3. Converting chunks into math vectors (31 points)...")
# We convert our list of chunks into a mathematical matrix (array)
vectors = model.encode(chunks)
vectors_as_numpy = np.array(vectors).astype('float32')

print("\n4. Initializing FAISS Vector Database...")
# 384 is the number of dimensional characteristics this specific AI model uses
dimension = vectors_as_numpy.shape[1]
index = faiss.IndexFlatL2(dimension)

# Drop the points into our map!
index.add(vectors_as_numpy)

print("\n5. Saving database to hard drive...")
# Save the math maps (the vector points)
faiss.write_index(index, "my_vector_db.index")

# Save the text strings so we can pull them out when a point matches
with open("my_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("\n✅ SUCCESS! Your PDF is now a permanent, local Vector Database.")
print("Look at your sidebar, you will see 'my_vector_db.index' and 'my_chunks.pkl'!")