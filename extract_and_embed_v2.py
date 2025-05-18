import fitz  # PyMuPDF
from openai import OpenAI
import os
import uuid
import tempfile
import re
from flask import Flask, request, jsonify
from pymilvus import DataType, MilvusClient
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import hashlib

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
    
    # Use document type detection to apply different strategies
    document_type = detect_document_type(text)
    
    # Use more advanced analysis with OpenAI if patterns don't match
    try:
        conclusion_section = extract_conclusion_section(text, document_type)
        if conclusion_section:
            decision = analyze_conclusion_with_ai(conclusion_section)
            if decision in ["appellant_won", "appellant_lost"]:
                return decision
    except Exception as e:
        app.logger.warning(f"⚠️ AI analysis of conclusion failed: {e}")
    
    # Default if no clear outcome can be determined
    return "invalid"

# Add new function to detect document type
def detect_document_type(text):
    """
    Detects the type of legal document to apply appropriate extraction strategies
    """
    text_lower = text.lower()
    
    # Check for appellate tribunal documents
    if re.search(r'appellate tribunal|cestat|final order no|coram|hon\'ble', text_lower):
        return "appellate_tribunal"
    
    # Check for order-in-original customs documents
    if re.search(r'order-in-original|commissioner of customs|central board of indirect taxes|office of the commissioner', text_lower):
        return "order_in_original"
    
    # Check for high court judgments
    if re.search(r'high court|writ petition|division bench|single bench', text_lower):
        return "high_court"
    
    # Check for supreme court judgments
    if re.search(r'supreme court|civil appeal no|criminal appeal no|constitution bench', text_lower):
        return "supreme_court"
    
    # Default document type
    return "generic_legal_document"

# Extract the conclusion section which is likely to contain the decision
def extract_conclusion_section(text, document_type="generic_legal_document"):
    """
    Extract the relevant conclusion section based on document type
    """
    # Different extraction strategies based on document type
    if document_type == "appellate_tribunal":
        # For appellate tribunal documents, look for specific sections
        conclusion_indicators = [
            r'accordingly', r'in\s+result', r'the\s+appeal\s+is', r'order', 
            r'final\s+order', r'for\s+these\s+reasons', r'therefore'
        ]
        # Also try to find paragraphs with numbers like "4.", "5." near the end
        numbered_paras = list(re.finditer(r'\n\s*\d+\.\s+', text))
        if numbered_paras and len(numbered_paras) > 2:
            # Get the last few numbered paragraphs
            last_paragraphs = text[numbered_paras[-3].start():]
            return last_paragraphs
    
    elif document_type == "order_in_original":
        # For order-in-original, look for decision sections
        conclusion_indicators = [
            r'i\s+order\s+that', r'i\s+hereby\s+order', r'hereby\s+ordered\s+that',
            r'decision', r'conclusion', r'adjudication', r'in\s+view\s+of\s+above'
        ]
    
    else:
        # Generic indicators for other document types
        conclusion_indicators = [
            r'CONCLUSION', r'DISPOSITION', r'ORDER', r'JUDGMENT',
            r'FOR THESE REASONS', r'THEREFORE', r'ACCORDINGLY',
            r'IT IS ORDERED THAT', r'IT IS SO ORDERED', r'In the result',
            r'In view of the above'
        ]
    
    # Join indicators with OR operator
    pattern = '|'.join(conclusion_indicators)
    
    # Find all matches
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    if matches:
        # Use the last match, which is typically the final conclusion
        last_match = matches[-1]
        # Get text from match position to end (or at most 2000 chars to keep context reasonable)
        conclusion = text[last_match.start():last_match.start() + 2000]
        return conclusion
    
    # Look for specific phrases often found in court orders
    order_phrases = [
        r'appeal\s+is\s+allowed', r'appeal\s+is\s+dismissed', 
        r'petition\s+is\s+allowed', r'petition\s+is\s+dismissed',
        r'affirm\s+the\s+judgment', r'set\s+aside\s+the\s+order',
        r'impugned\s+order\s+is'
    ]
    
    for phrase in order_phrases:
        match = re.search(phrase, text, re.IGNORECASE)
        if match:
            # Get some context around the match
            start = max(0, match.start() - 500)
            end = min(len(text), match.end() + 500)
            return text[start:end]
    
    # If no conclusion section found, try to identify by position
    # For appellate tribunal and similar documents, conclusion often in last 20% of the document
    text_length = len(text)
    if text_length > 5000:  # Reasonably long document
        return text[int(text_length * 0.8):]  # Return the last 20%
    
    # For shorter documents, return the last 1000 characters
    if text_length > 1000:
        return text[-1000:]
    
    return text  # Return whole text for very short documents

# Use OpenAI to analyze the conclusion if pattern matching fails
def analyze_conclusion_with_ai(conclusion_text):
    try:
        # Add context about document type for better analysis
        document_context = identify_document_context(conclusion_text)
        
        system_prompt = f"""You are a legal assistant that analyzes the conclusion of court cases.
Determine whether the appellant won or lost the case based on the text provided.
This appears to be {document_context}.
Respond with ONLY one of these exact terms:
- 'appellant_won' if the appellant/petitioner succeeded fully or partially, including remand orders
- 'appellant_lost' if the appellant/petitioner failed completely
- 'invalid' if you cannot determine with confidence

For customs/tax cases:
- If an appeal is allowed (even partially), that means appellant_won
- If a refund is ordered or upheld, that means appellant_won
- If a case is remanded for fresh consideration, that generally means appellant_won
- If an impugned order is set aside, that means appellant_won"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this conclusion from a legal case and tell me if the appellant won or lost:\n\n{conclusion_text}"}
            ],
            temperature=0.3,
            max_tokens=10
        )
        decision = response.choices[0].message.content.strip().lower()
        
        # Ensure response is one of our valid values
        if decision in ["appellant_won", "appellant_lost", "invalid"]:
            return decision
        return "invalid"
    except Exception as e:
        app.logger.error(f"❌ Error using AI to analyze conclusion: {e}")
        return "invalid"

def identify_document_context(text):
    """
    Identifies the context/type of legal document to guide AI analysis
    """
    text_lower = text.lower()
    
    if re.search(r'customs|cestat|excise|service tax|appellate tribunal', text_lower):
        return "a Customs/Tax Appellate Tribunal case"
    
    if re.search(r'writ petition|high court', text_lower):
        return "a High Court judgment"
    
    if re.search(r'supreme court|civil appeal|criminal appeal', text_lower):
        return "a Supreme Court judgment"
    
    if re.search(r'commissioner|order-in-original|central board', text_lower):
        return "an administrative order from a tax/customs authority"
    
    return "a legal document"

# Process PDF and insert into Milvus
def process_pdf_to_milvus(pdf_path, milvus_client, collection_name="case_files", original_filename=None, court_level=None):
    # Ensure collection exists
    create_milvus_collection(milvus_client, collection_name)

    # Extract text from PDF
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    
    # Extract metadata if possible (for better context)
    metadata = extract_pdf_metadata(doc)
    
    # Determine case decision
    document_type = detect_document_type(full_text)
    case_decision = determine_case_decision(full_text)
    app.logger.info(f"📊 Document type: {document_type}, Case decision determined: {case_decision}")
    
    # For challenging cases, use more context if decision is "invalid"
    if case_decision == "invalid" and metadata:
        app.logger.info(f"⚠️ Retrying decision detection with metadata context")
        enriched_text = f"{metadata}\n\n{full_text}"
        case_decision = determine_case_decision(enriched_text)
        
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

def extract_pdf_metadata(doc):
    """
    Extract metadata from PDF document for better context
    """
    metadata = {}
    try:
        # Try to get standard PDF metadata
        if doc.metadata:
            metadata.update(doc.metadata)
        
        # Try to extract case number, date, etc. from first page
        first_page_text = doc[0].get_text(sort=True)
        
        # Extract appeal/case numbers
        case_number_match = re.search(r'(?:appeal|application|petition|case|c\.a\.)\s+no\.?\s*([\w\d\.\-\/]+)', 
                                      first_page_text, re.IGNORECASE)
        if case_number_match:
            metadata['case_number'] = case_number_match.group(1).strip()
            
        # Extract dates
        date_matches = re.findall(r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4}', 
                                  first_page_text, re.IGNORECASE)
        if date_matches:
            metadata['dates_mentioned'] = date_matches
            
        # Extract parties (common formats like "X versus Y")
        parties_match = re.search(r'([A-Za-z\s\.,]+)\s+(?:versus|vs\.?|v\.)\s+([A-Za-z\s\.,]+)', 
                                 first_page_text, re.IGNORECASE)
        if parties_match:
            metadata['appellant'] = parties_match.group(1).strip()
            metadata['respondent'] = parties_match.group(2).strip()
        
        return metadata
    except Exception as e:
        app.logger.warning(f"⚠️ Error extracting PDF metadata: {e}")
        return {}

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
    return jsonify({"status": "healthy", "service": "legal-document-ingestion-api"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)