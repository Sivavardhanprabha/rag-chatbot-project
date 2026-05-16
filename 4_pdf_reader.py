import PyPDF2

def read_pdf(file_path):
    """
    This function opens a PDF, counts the pages, 
    and extracts text from the first page.
    """
    try:
        print(f"Opening file at: {file_path}")
        
        # 1. Open the PDF file in 'read binary' mode
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Count total pages
            num_pages = len(reader.pages)
            print(f"Success! This PDF has {num_pages} pages.")
            
            # 2. Extract text from the first page (Index 0)
            first_page = reader.pages[0]
            text = first_page.extract_text()
            
            return text
            
    except FileNotFoundError:
        return "Error: The file 'sample.pdf' was not found in the folder."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- EXECUTION AREA ---

# We use the forward slash (/) even on Windows to keep Python happy!
path_to_pdf = "D:/rag_chatbot_project/sample.pdf"

# Call the function and store the result
extracted_text = read_pdf(path_to_pdf)

print("\n--- HERE IS WHAT THE BOT READ (First 500 characters) ---")
print(extracted_text[:500])