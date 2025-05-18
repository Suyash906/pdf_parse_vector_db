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
import re

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

# Create collection using the schema method
def create_milvus_collection(milvus_client, collection_name="case_files"):
    if milvus_client.has_collection(collection_name):
        app.logger.info(f"ℹ️ Collection '{collection_name}' already exists.")
        return

    # Define schema
    schema = milvus_client.create_schema()
    schema.add_field("chunk_id", DataType.INT64, is_primary=True, description="Unique chunk ID")
    schema.add_field("text", DataType.VARCHAR, max_length=65535, description="Document chunk text")
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536, description="OpenAI embedding")
    schema.add_field("file_name", DataType.VARCHAR, max_length=1000, description="Name of the uploaded file")
    schema.add_field("file_id", DataType.VARCHAR, max_length=1000, description="SHA256 hash of the file name")
    schema.add_field("court_level", DataType.INT8, description="level of the court")
    schema.add_field("case_decision", DataType.VARCHAR, max_length=20, description="Outcome of the case - appellant_won, appellant_lost, or invalid")
    
    # Define index
    index_params = milvus_client.prepare_index_params()
    index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})

    # Create and load collection
    milvus_client.create_collection(collection_name, schema=schema, index_params=index_params)
    milvus_client.load_collection(collection_name)
    app.logger.info(f"✅ Collection '{collection_name}' created and loaded.")

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
    
    # Calculate appellant winning percentage
    appellant_win_count = sum(1 for result in sorted_results if result["case_decision"] == "appellant_won")
    total_valid_decisions = sum(1 for result in sorted_results if result["case_decision"] in ["appellant_won", "appellant_lost"])
    
    # Calculate percentage (avoid division by zero)
    appellant_win_percentage = 0
    if total_valid_decisions > 0:
        appellant_win_percentage = (appellant_win_count / total_valid_decisions) * 100
    
    return sorted_results, appellant_win_percentage, appellant_win_count, total_valid_decisions

# Determine case decision - whether appellant won or lost
def determine_case_decision(text):
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # First check for customs appeal tribunal specific phrases
    # Example from AHMEDABAD_C-4-2012_29-03-2023.pdf
    if re.search(r'appeal\s+is\s+allowed', text_lower) or re.search(r'the\s+appeal\s+is\s+allowed\s+by\s+way\s+of', text_lower):
        return "appellant_won"
    
    if re.search(r'impugned\s+order\s+is\s+set-aside', text_lower) or re.search(r'impugned\s+order\s+is\s+set\s+aside', text_lower):
        return "appellant_won"
        
    # Check for remand outcomes - consider these as partial wins
    if re.search(r'remand\s+to\s+the\s+commissioner', text_lower) or re.search(r'matter\s+needs\s+to\s+be\s+remanded', text_lower):
        return "appellant_won"
    
    # Look for standard outcome statements
    if re.search(r'appeal\s+dismissed', text_lower) or re.search(r'dismiss\s+the\s+appeal', text_lower):
        return "appellant_lost"
    
    if re.search(r'appeal\s+allowed', text_lower) or re.search(r'allow\s+the\s+appeal', text_lower):
        return "appellant_won"
        
    if re.search(r'judgment\s+affirmed', text_lower) or re.search(r'affirm\s+the\s+judgment', text_lower):
        return "appellant_lost"
        
    if re.search(r'judgment\s+reversed', text_lower) or re.search(r'reverse\s+the\s+judgment', text_lower):
        return "appellant_won"
    
    # Check for order outcomes (common in customs/tax cases)
    if re.search(r'order\s+is\s+upheld', text_lower) or re.search(r'upheld\s+the\s+order', text_lower):
        return "appellant_lost"
        
    # Check for rejection/acceptance terms
    if re.search(r'refund\s+claim\s+is\s+rejected', text_lower) or re.search(r'petition\s+is\s+rejected', text_lower):
        return "appellant_lost"
        
    if re.search(r'refund\s+claim\s+is\s+accepted', text_lower) or re.search(r'petition\s+is\s+accepted', text_lower):
        return "appellant_won"
    
    # Check for success/win language
    win_indicators = [
        r'in\s+favor\s+of\s+appellant',
        r'appellant\s+prevails',
        r'grant\s+the\s+appeal',
        r'succeed\s+on\s+appeal',
        r'appeal\s+is\s+successful',
        r'judgment\s+set\s+aside',
        r'order\s+set\s+aside',
        r'appeal\s+is\s+partly\s+allowed',
        r'relief\s+granted',
        r'relief\s+is\s+granted'
    ]
    
    for pattern in win_indicators:
        if re.search(pattern, text_lower):
            return "appellant_won"
    
    # Default if no clear outcome can be determined
    return "invalid"

# Process PDF and insert into Milvus
def process_pdf_to_milvus(pdf_path, milvus_client, collection_name="case_files", original_filename=None, court_level=None):
    # Ensure collection exists
    create_milvus_collection(milvus_client, collection_name)

    # Extract text from PDF
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    
    # Determine case decision
    case_decision = determine_case_decision(full_text)
    app.logger.info(f"📊 Case decision determined: {case_decision}")
    
    chunks = chunk_text(full_text)

    file_name = original_filename or os.path.basename(pdf_path)
    file_id = hashlib.sha256(file_name.encode('utf-8')).hexdigest()
    
    # Process chunks and prepare rows for insertion
    rows = []
    for chunk in chunks:
        try:
            emb = get_embedding(chunk)
            if emb:
                # Create individual row as dictionary with proper field types
                row = {
                    "text": chunk,  # String value for varchar field
                    "embedding": emb,  # List of floats for vector field
                    "file_name": file_name,
                    "file_id": file_id,
                    "court_level": court_level,
                    "case_decision": case_decision
                }
                rows.append(row)
            else:
                app.logger.warning("⚠️ Skipping chunk due to empty embedding.")
        except Exception as e:
            app.logger.error(f"⚠️ Embedding failed for chunk: {e}")
    
    # Insert data if we have rows
    inserted_count = 0
    if rows:
        try:
            milvus_client.insert(collection_name, rows)
            milvus_client.flush(collection_name)
            inserted_count = len(rows)
            app.logger.info(f"✅ Inserted {inserted_count} chunks into Milvus collection '{collection_name}'")
        except Exception as e:
            app.logger.error(f"❌ Error inserting into Milvus: {e}")
            raise e
    else:
        app.logger.warning("⚠️ No valid data to insert.")
    
    return inserted_count, case_decision

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
        similar_cases, win_percentage, win_count, total_valid = search_similar_cases(
            milvus_client,
            query_embedding,
            target_court_level,
            collection_name
        )
        
        # Clean up temporary file
        os.remove(pdf_path)
        os.rmdir(temp_dir)
        
        # Return results with winning percentage
        return jsonify({
            "status": "success",
            "query": {
                "input_court_level": court_level,
                "target_court_level": target_court_level,
                "file_name": file.filename
            },
            "results": similar_cases,
            "result_count": len(similar_cases),
            "appellant_statistics": {
                "win_percentage": round(win_percentage, 2),
                "win_count": win_count,
                "total_valid_decisions": total_valid,
                "invalid_decisions": len(similar_cases) - total_valid
            }
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error processing search request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API endpoint to ingest PDF file
@app.route('/api/v1/ingest-legal-document', methods=['POST'])
def ingest_legal_document():
    # Check if Milvus client is available
    global milvus_client
    if milvus_client is None:
        milvus_client = connect_to_milvus_db()
        if milvus_client is None:
            return jsonify({"error": "Failed to connect to vector database"}), 500
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    # If user does not select file, browser may submit an empty file
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check if it's a PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    # Get optional collection name
    collection_name = request.form.get('collection', 'case_files')

    court_level = request.form.get("court_level")
    try:
        court_level = int(court_level)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing court_level. It must be an integer."}), 400
    
    try:
        # Save the file temporarily
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(pdf_path)
        
        # Process the PDF
        chunks_inserted, case_decision = process_pdf_to_milvus(
            pdf_path,
            milvus_client,
            collection_name,
            original_filename=file.filename,
            court_level=court_level
        )

        # Clean up
        os.remove(pdf_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            "status": "success",
            "message": f"Document processed successfully",
            "chunks_inserted": chunks_inserted,
            "case_decision": case_decision, 
            "collection": collection_name
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "legal-case-search-api"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)