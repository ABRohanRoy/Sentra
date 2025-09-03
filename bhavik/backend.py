from flask import Flask, render_template, request, jsonify, flash
import os
import numpy as np
from dotenv import load_dotenv
import faiss
from langchain_openai import AzureOpenAIEmbeddings
from sentra.parser.s3_log_parser import parse_log_file
from sentra.agent.gpt_responder import ask_gpt
from werkzeug.utils import secure_filename
import json

# Load environment variables
load_dotenv()

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize embedding model using .env values (unchanged from your original code)
embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("Embedding_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("Embedding_AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("Embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("Embedding_AZURE_OPENAI_API_VERSION"),
    chunk_size=10,
)

# Your original backend functions (unchanged)
def load_log_chunks(file_path):
    log_dicts = parse_log_file(file_path)
    logs = [
        f"[{log['timestamp']}] IP: {log['ip']} | Action: {log['action']} | Endpoint: {log['endpoint']} | Status: {log['status']}"
        for log in log_dicts if log
    ]
    return logs

def store_vector(log_chunks):
    vectors = np.array(embedding_model.embed_documents(log_chunks)).astype("float32")
    dimension = vectors.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, "faiss_vector.index")
    # Save logs to file for lookup
    with open("logs_reference.txt", "w", encoding="utf-8") as f:
        for log in log_chunks:
            f.write(log + "\n")
    print("Vector stored successfully")
    return vectors

def search(query, top_k=3):
    index = faiss.read_index("faiss_vector.index")
    query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    # Retrieve matching logs
    with open("logs_reference.txt", "r", encoding="utf-8") as f:
        logs = f.readlines()
    results = [logs[i].strip() for i in indices[0]]
    return results

# Helper function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'log', 'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the log file using your existing backend
            log_chunks = load_log_chunks(file_path)
            
            if log_chunks:
                store_vector(log_chunks)
                return jsonify({
                    'success': True, 
                    'message': f'File uploaded and processed successfully! Found {len(log_chunks)} log entries.',
                    'log_count': len(log_chunks)
                })
            else:
                return jsonify({'success': False, 'message': 'Log file is empty or failed to parse.'}), 400
        else:
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload a .log or .txt file.'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search_logs():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Check if vector index exists
        if not os.path.exists("faiss_vector.index"):
            return jsonify({'error': 'No log data available. Please upload a log file first.'}), 400
        
        # Use your existing search function
        results = search(query)
        
        # Get GPT analysis using your existing function
        gpt_response = ask_gpt(query, results)
        
        return jsonify({
            'query': query,
            'results': results,
            'gpt_analysis': gpt_response
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check if the system is ready for queries"""
    has_index = os.path.exists("faiss_vector.index")
    has_reference = os.path.exists("logs_reference.txt")
    return jsonify({
        'ready': has_index and has_reference,
        'has_index': has_index,
        'has_reference': has_reference
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)