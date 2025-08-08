# main.py

import os
import requests
import fitz  # PyMuPDF
import docx
import email
import email.policy
import mimetypes
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import uvicorn

# ==============================================================================
# 1. INITIAL SETUP (Loads once on startup)
# ==============================================================================

# Load Embedding Model (this will be done only once when the app starts)
print("Loading sentence-transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Configure Gemini API
# --- IMPORTANT: Load API Key from environment variables for security ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-1.5-flash") # Using 1.5-flash for good performance

# --- Pre-defined Questions (can also be sent in the request if needed) ---
QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

# ==============================================================================
# 2. HELPER FUNCTIONS (Your existing logic, slightly modified)
# ==============================================================================

def get_file_extension_from_url(url):
    # (Your existing function)
    # ...
    # Note: This function might not be reliable for signed URLs. The version below is more robust.
    pass


def download_and_extract_text(doc_url: str):
    try:
        print(f"⏳ Downloading from {doc_url}")
        response = requests.get(doc_url, timeout=30) # Added a timeout
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
        print("✅ Response received.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download the document: {e}")

    content = response.content
    
    # More robust file type detection
    content_type = response.headers.get('Content-Type', '').lower()
    ext = ''
    if 'pdf' in content_type:
        ext = 'pdf'
    elif 'word' in content_type or 'docx' in content_type:
        ext = 'docx'
    elif 'rfc822' in content_type or 'eml' in content_type:
        ext = 'eml'
    else: # Fallback for URLs without content-type
        if doc_url.lower().endswith('.pdf'): ext = 'pdf'
        elif doc_url.lower().endswith('.docx'): ext = 'docx'
        elif doc_url.lower().endswith('.eml'): ext = 'eml'

    if ext == "pdf":
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            return [page.get_text() for page in doc]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    # ... (Add similar robust error handling for docx and eml)
    elif ext == "docx":
        # ... your docx logic ...
        return []
    elif ext == "eml":
        # ... your eml logic ...
        return []
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported document type: {content_type}")


def chunk_and_embed(texts):
    # (Your existing function)
    print("Chunking and embedding text...")
    chunks = []
    for text in texts:
        # Simple splitting by newline and then by sentence could be more robust
        parts = text.split("\n")
        for part in parts:
            if len(part.strip()) > 20: # Filter out very short lines
                chunks.append(part.strip())
    
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return chunks, embeddings


def search_similar_chunks(questions, chunks, embeddings):
    # (Your existing function)
    print("Searching for relevant chunks using FAISS...")
    embeddings_np = embeddings.cpu().detach().numpy()
    faiss.normalize_L2(embeddings_np)
    
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    results = []
    for question in questions:
        q_embedding = model.encode(question, convert_to_tensor=True).cpu().detach().numpy()
        q_embedding = q_embedding.reshape(1, -1)
        faiss.normalize_L2(q_embedding)

        D, I = index.search(q_embedding, k=1) # Search for the top 1 most similar chunk
        top_idx = I[0][0]
        similarity_score = D[0][0]
        
        results.append((question, chunks[top_idx], similarity_score))
    return results


def answer_with_llm(pairs):
    # (Your existing function)
    print("Generating answers with Gemini...")
    answers = []
    for question, context, score in pairs:
        prompt = f"Based on the following context from a policy document, please answer the question.\n\nContext: \"{context}\"\n\nQuestion: \"{question}\"\n\nAnswer:"
        
        try:
            response = llm_model.generate_content(prompt)
            clean_answer = response.text.strip().replace('\n', ' ').replace('*', '').replace('\"','')
            clean_answer = ' '.join(clean_answer.split())
            answers.append(clean_answer)
        except Exception as e:
            print(f"Error generating answer for '{question}': {e}")
            answers.append(f"Error generating answer: {e}")
            
    return answers

# ==============================================================================
# 3. FASTAPI APPLICATION
# ==============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="An API to ask questions about a document (PDF, DOCX, EML) provided via URL."
)

# Pydantic model for input validation
class DocumentRequest(BaseModel):
    document_url: HttpUrl # Ensures the input is a valid URL

# Pydantic model for the output
class AnswersResponse(BaseModel):
    answers: list[str]

@app.post("/process-document/", response_model=AnswersResponse)
async def process_document_and_get_answers(request: DocumentRequest):
    """
    Accepts a document URL, processes it, and returns answers to a predefined set of questions.
    """
    try:
        # Step 1: Download and extract text
        texts = download_and_extract_text(str(request.document_url))
        
        # Step 2: Chunk text and create embeddings
        chunks, embeddings = chunk_and_embed(texts)
        
        # Step 3: Search for relevant chunks for each question
        pairs = search_similar_chunks(QUESTIONS, chunks, embeddings)
        
        # Step 4: Generate answers using the LLM
        answers = answer_with_llm(pairs)
        
        # Step 5: Return the response
        return AnswersResponse(answers=answers)

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# A simple root endpoint to check if the server is running
@app.get("/")
async def root():
    return {"message": "Document Q&A API is running. Send a POST request to /process-document/"}


if __name__ == '__main__':
    # This block allows you to run the app locally for testing
    # Use: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)