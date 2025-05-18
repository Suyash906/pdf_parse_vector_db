import fitz  # PyMuPDF
from openai import OpenAI
import os
import uuid
import tempfile
from flask import Flask, request, jsonify
from pymilvus import DataType, MilvusClient
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB

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
def create_milvus_collection_v2(milvus_client, collection_name="case_files"):
    if milvus_client.has_collection(collection_name):
        app.logger.info(f"ℹ️ Collection '{collection_name}' already exists.")
        return

    # Define schema
    schema = milvus_client.create_schema()
    schema.add_field("chunk_id", DataType.INT64, is_primary=True, description="Unique chunk ID")
    schema.add_field("text", DataType.VARCHAR, max_length=65535, description="Document chunk text")
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536, description="OpenAI embedding")

    # Define index
    index_params = milvus_client.prepare_index_params()
    index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})

    # Create and load collection
    milvus_client.create_collection(collection_name, schema=schema, index_params=index_params)
    milvus_client.load_collection(collection_name)
    app.logger.info(f"✅ Collection '{collection_name}' created and loaded.")

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

# Process PDF and insert into Milvus
def process_pdf_to_milvus(pdf_path, milvus_client, collection_name="case_files"):
    # Ensure collection exists
    create_milvus_collection_v2(milvus_client, collection_name)

    # Extract text from PDF
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    chunks = chunk_text(full_text)
    
    # Process chunks and prepare rows for insertion
    rows = []
    for chunk in chunks:
        try:
            emb = get_embedding(chunk)
            if emb:
                # Create individual row as dictionary with proper field types
                row = {
                    "text": chunk,  # String value for varchar field
                    "embedding": emb  # List of floats for vector field
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
    
    return inserted_count

# Initialize Milvus client at app startup
milvus_client = None

# Using Flask 2.0+ compatible approach instead of before_first_request
with app.app_context():
    milvus_client = connect_to_milvus_db()

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
    
    try:
        # Save the file temporarily
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(pdf_path)
        
        # Process the PDF
        chunks_inserted = process_pdf_to_milvus(pdf_path, milvus_client, collection_name)
        
        # Clean up
        os.remove(pdf_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            "status": "success",
            "message": f"Document processed successfully",
            "chunks_inserted": chunks_inserted,
            "collection": collection_name
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "legal-document-ingestion-api"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)