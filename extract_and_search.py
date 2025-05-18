from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from openai import OpenAI
import os
import tempfile
import hashlib
from werkzeug.utils import secure_filename
from pymilvus import DataType, MilvusClient
from dotenv import load_dotenv
import logging

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Connect to Milvus
def connect_to_milvus_db():
    try:
        milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        collections = milvus_client.list_collections()
        app.logger.info("✅ Connected to Milvus. Collections: %s", collections)
        return milvus_client
    except Exception as e:
        app.logger.error("❌ Failed to connect to Milvus: %s", e)
        return None

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = "\n".join([page.get_text() for page in doc])
        return full_text
    except Exception as e:
        app.logger.error(f"❌ Error extracting text from PDF: {e}")
        raise

# Chunk text by paragraphs
def chunk_text(text, max_tokens=400):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len((current_chunk + para).split()) < max_tokens:
            current_chunk += "\n" + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Get OpenAI embedding
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Process chunks and get aggregate embedding
def process_text_chunks(text):
    chunks = chunk_text(text)
    
    # Get embeddings for all chunks
    chunk_embeddings = []
    for chunk in chunks:
        try:
            emb = get_embedding(chunk)
            if emb:
                chunk_embeddings.append(emb)
        except Exception as e:
            app.logger.error(f"⚠️ Embedding failed for chunk: {e}")
    
    # If we have embeddings, return the first one as representative
    # In a more advanced implementation, you might want to average all embeddings
    if chunk_embeddings:
        return chunk_embeddings[0]
    
    return None

# Search in Milvus with specified parameters
def search_similar_cases(milvus_client, query_embedding, court_level, collection_name="case_files", top_k=5):
    # Prepare search parameters
    search_params = {
        "vector": query_embedding,
        "filter": f"court_level == {court_level}", # Search for cases at the next court level
        "output_fields": ["file_id", "file_name", "case_decision"],
        "limit": 100  # Get more results initially to allow for grouping
    }
    
    # Execute search
    results = milvus_client.search(
        collection_name=collection_name,
        data=[query_embedding],
        filter=f"court_level == {court_level}",
        output_fields=["file_id", "file_name", "case_decision"],
        limit=100
    )
    
    # Group results by file_id
    grouped_results = {}
    for hit in results[0]:
        file_id = hit.get("entity").get("file_id")
        file_name = hit.get("entity").get("file_name")
        case_decision = hit.get("entity").get("case_decision")
        score = hit.get("distance")
        
        if file_id not in grouped_results:
            grouped_results[file_id] = {
                "file_id": file_id,
                "file_name": file_name,
                "case_decision": case_decision,
                "score": score  # Use the best score
            }
    
    # Sort grouped results by score and take top_k
    sorted_results = sorted(grouped_results.values(), key=lambda x: x["score"])[:top_k]
    
    return sorted_results

# Initialize Milvus client at app startup
milvus_client = None

# Using Flask 2.0+ compatible approach for initialization
with app.app_context():
    milvus_client = connect_to_milvus_db()

# API endpoint for case search
@app.route('/api/v1/search-similar-cases', methods=['POST'])
def search_similar_cases_endpoint():
    # Check if Milvus client is available
    global milvus_client
    if milvus_client is None:
        milvus_client = connect_to_milvus_db()
        if milvus_client is None:
            return jsonify({"error": "Failed to connect to vector database"}), 500
    
    # Validate required parameters
    if 'court_level' not in request.form:
        return jsonify({"error": "Missing court_level parameter"}), 400
    
    try:
        court_level = int(request.form.get('court_level'))
        # Calculate target court level (1 level higher than input)
        target_court_level = court_level + 1
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid court_level. It must be an integer."}), 400
    
    # Check if file is provided
    if 'case_file' not in request.files:
        return jsonify({"error": "No case_file in the request"}), 400
    
    file = request.files['case_file']
    # If user does not select file, browser may submit an empty file
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check if it's a PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    # Get optional collection name
    collection_name = request.form.get('collection', 'case_files')
    
    try:
        # Save the file temporarily
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(pdf_path)
        
        # Extract text from PDF
        full_text = extract_text_from_pdf(pdf_path)
        
        # Process text to get representative embedding
        query_embedding = process_text_chunks(full_text)
        
        if not query_embedding:
            return jsonify({"error": "Failed to generate embedding from the document"}), 500
        
        # Search for similar cases at higher court level
        similar_cases = search_similar_cases(
            milvus_client,
            query_embedding,
            target_court_level,
            collection_name
        )
        
        # Clean up temporary file
        os.remove(pdf_path)
        os.rmdir(temp_dir)
        
        # Return results
        return jsonify({
            "status": "success",
            "query": {
                "input_court_level": court_level,
                "target_court_level": target_court_level,
                "file_name": file.filename
            },
            "results": similar_cases,
            "result_count": len(similar_cases)
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error processing search request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "legal-case-search-api"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)