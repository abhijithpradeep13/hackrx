
import requests
import fitz  # PyMuPDF
import docx
import email
import email.policy
import io
import mimetypes
import os
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import json
import faiss                   # 🔁 New FAISS import
import numpy as np             # Required for FAISS operations
from dotenv import load_dotenv
load_dotenv()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDtuR06OSccodY-B6uyUlD1iUHiy8OCzZA"
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.5-flash")  # ✅ supported

# Constants
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
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

# Step 1: Download and extract text based on file type
def get_file_extension_from_url(url):
    print("file type extraction")
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type == 'application/pdf':
        return 'pdf'
    elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
        return 'docx'
    elif mime_type == 'message/rfc822':
        return 'eml'
    else:
        return None

def download_and_extract_text(doc_url):
    print("⏳ About to send request...", flush=True)
    response = requests.get(doc_url)
    print("✅ Response received.", flush=True)

    content = response.content
    ext = get_file_extension_from_url(doc_url)

    # If extension detection fails, try response headers
    if ext is None:
        content_type = response.headers.get('Content-Type', '').lower()
        if 'pdf' in content_type:
            ext = 'pdf'
        elif 'word' in content_type or 'docx' in content_type:
            ext = 'docx'
        elif 'rfc822' in content_type or 'eml' in content_type:
            ext = 'eml'

    if ext == "pdf":
        with open("temp.pdf", "wb") as f:
            f.write(content)
        doc = fitz.open("temp.pdf")
        return [page.get_text() for page in doc]

    elif ext == "docx":
        with open("temp.docx", "wb") as f:
            f.write(content)
        doc = docx.Document("temp.docx")
        return [para.text for para in doc.paragraphs if para.text.strip()]

    elif ext == "eml":
        msg = email.message_from_bytes(content, policy=email.policy.default)
        text_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    text_parts.append(part.get_content())
        else:
            text_parts.append(msg.get_content())
        return text_parts

    else:
        raise ValueError("Unsupported document type or unable to detect file type.")

# Step 2: Chunk text and embed
def chunk_and_embed(texts):
    print("converting to chuncks")
    chunks = []
    for text in texts:
        parts = text.split("\n")
        for part in parts:
            if len(part.strip()) > 20:
                chunks.append(part.strip())

    embeddings = model.encode(chunks, convert_to_tensor=True)
    return chunks, embeddings

# Step 3: Embed Questions and Search
# Step 3 (FAISS version): Embed Questions and Search
def search_similar_chunks(questions, chunks, embeddings):
    print("embedding....")
    embeddings_np = embeddings.cpu().detach().numpy()
    faiss.normalize_L2(embeddings_np)

    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    results = []
    for question in questions:
        q_embedding = model.encode(question, convert_to_tensor=True).cpu().detach().numpy()
        q_embedding = q_embedding.reshape(1, -1)          # reshape here
        faiss.normalize_L2(q_embedding)                    # then normalize

        D, I = index.search(q_embedding, k=1)
        top_idx = I[0][0]
        similarity_score = D[0][0]

        results.append((question, chunks[top_idx], similarity_score))
    return results



# Step 4: Generate Answers with Gemini API
def answer_with_llm(pairs):
    print("fetching answers from llm")
    answers = []
    for question, context, score in pairs:
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        response = llm_model.generate_content(prompt)
        print(response.text)
        # Clean answer: remove newlines and asterisks, and extra spaces
        clean_answer = response.text.strip().replace('\n', ' ').replace('*', '').replace('\"','')
        # Optional: collapse multiple spaces into one
        clean_answer = ' '.join(clean_answer.split())
        answers.append(clean_answer)
    return answers


# Step 5: JSON Output
def create_json_response(answers):
    return json.dumps({"answers": answers}, indent=4)

# Main flow
if __name__ == '__main__':
    texts = download_and_extract_text(DOCUMENT_URL)
    chunks, embeddings = chunk_and_embed(texts)
    pairs = search_similar_chunks(QUESTIONS, chunks, embeddings)
    answers = answer_with_llm(pairs)
    output_json = create_json_response(answers)
    print(output_json)
    with open("answers.json", "w") as f:
        f.write(output_json)

