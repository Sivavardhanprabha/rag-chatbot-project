import PyPDF2

def get_pdf_chunks(file_path, chunk_size=500, overlap=50):
    """
    Reads a PDF, cleans the text, and splits it into 
    overlapping chunks for the AI.
    """
    text = ""
    try:
        # 1. Extract text from all pages
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "

        # 2. CLEANING: Remove extra spaces and newlines
        # This turns "N O N - D I S C L O S U R E" into "NON -DISCLOSURE"
        text = " ".join(text.split())

        # 3. CHUNKING: Create overlapping slices
        chunks = []
        # 'step' is how far we jump forward each time
        step = chunk_size - overlap 
        
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            chunks.append(chunk)
            
        return chunks

    except FileNotFoundError:
        print("Error: Could not find the PDF file.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# --- EXECUTION ---
path = "D:/rag_chatbot_project/sample.pdf"
all_chunks = get_pdf_chunks(path)

if all_chunks:
    print(f"✅ Success! Created {len(all_chunks)} overlapping chunks.")
    print("-" * 30)
    print("PREVIEW OF CHUNK #1:")
    print(all_chunks[0])
    print("-" * 30)
    print("PREVIEW OF CHUNK #2 (Notice the overlap at the start):")
    print(all_chunks[1])
else:
    print("No chunks were created. Check your file path!")