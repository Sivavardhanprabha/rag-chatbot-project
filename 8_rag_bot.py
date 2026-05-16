import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. LOAD SECRET ENVIRONMENT VARIABLES
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found in your .env file!")
    exit()

# Initialize the Groq Client
client = Groq(api_key=GROQ_API_KEY)

# 2. LOAD OUR LOCAL EMBEDDING MODEL AND DATABASE FILES
print("🤖 Loading local AI models and database files...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("my_vector_db.index")

with open("my_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 3. DEFINE THE USER'S QUESTION
user_query = "Who is this nondisclosure agreement made between?"
print(f"\nUser Question: '{user_query}'")

# 4. RETRIEVAL PHASE (Find the matching evidence chunks)
query_vector = model.encode([user_query])
query_vector_as_numpy = np.array(query_vector).astype('float32')

# Grab the top 2 matching chunks
distances, indices = index.search(query_vector_as_numpy, k=2)

# Extract the actual text string from those matching index numbers
retrieved_context = ""
for chunk_index in indices[0]:
    retrieved_context += chunks[chunk_index] + "\n\n"

print("✅ Relevant evidence successfully retrieved from local database.")

# 5. AUGMENTATION & GENERATION PHASE (Feeding facts to Groq)
print("Sending evidence to Groq LLM for processing...")

# Construct a strict prompt template so the AI doesn't hallucinate
system_instruction = (
    "You are an expert legal assistant chatbot. Answer the user's question "
    "using ONLY the provided Context evidence below. If the answer cannot be found "
    "in the context, say 'I cannot find the answer in the provided document.' "
    "Do not make up facts."
)

user_prompt = f"""
Context Evidence:
---------------------
{retrieved_context}
---------------------

Question: {user_query}
Answer:"""

# Fire the request to Groq using a fast, smart model
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ],
    model="llama-3.1-8b-instant", # Utilizing Llama 3 on Groq infrastructure
    temperature=0.2 # Lower temperature makes the AI strict and stick to the facts
)

# 6. OUTPUT THE FINAL REFINED ANSWER
bot_response = chat_completion.choices[0].message.content

print("\n================== 🤖 CHATBOT RESPONSE ==================")
print(bot_response)
print("========================================================")