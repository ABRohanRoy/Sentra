from flask import Flask, render_template, request, jsonify, flash
import os
import numpy as np
from dotenv import load_dotenv
import faiss
from werkzeug.utils import secure_filename

# Import Sentra modules instead of redefining
from sentra.parser.s3_log_parser import parse_log_file
from sentra.agent.gpt_responder import ask_gpt

# Load environment variables
load_dotenv()

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "fallback-secret")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------ Reuse Sentra RAG functions ------------------
from langchain_openai import AzureOpenAIEmbeddings

embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("Embedding_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("Embedding_AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("Embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("Embedding_AZURE_OPENAI_API_VERSION"),
    chunk_size=10,
)

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

    with open("logs_reference.txt", "w", encoding="utf-8") as f:
        for log in log_chunks:
            f.write(log + "\n")

    print("✅ Vector stored successfully")
    return vectors

def search(query, top_k=3):
    index = faiss.read_index("faiss_vector.index")
    query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    with open("logs_reference.txt", "r", encoding="utf-8") as f:
        logs = f.readlines()

    results = [logs[i].strip() for i in indices[0]]
    return results

# ------------------ Flask Routes ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            flash("ERROR: No file selected")
            return render_template('index.html')

        file = request.files['file']
        if file.filename == '':
            flash("ERROR: No file selected")
            return render_template('index.html')

        if file and file.filename.rsplit('.', 1)[1].lower() in {'log', 'txt'}:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            log_chunks = load_log_chunks(file_path)

            if log_chunks:
                store_vector(log_chunks)
                flash(f"SUCCESS: File uploaded and processed successfully! Found {len(log_chunks)} log entries.")
            else:
                flash("ERROR: Log file is empty or failed to parse.")
        else:
            flash("ERROR: Invalid file type. Please upload a .log or .txt file.")

    except Exception as e:
        flash(f"ERROR: {str(e)}")

    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_logs():
    try:
        query = request.form.get('query', '').strip()

        if not query:
            flash("ERROR: Query cannot be empty")
            return render_template('index.html')

        if not os.path.exists("faiss_vector.index"):
            flash("ERROR: No log data available. Please upload a log file first.")
            return render_template('index.html')

        results = search(query)
        gpt_response = ask_gpt(query, results)

        return render_template('index.html', query=query, results=results, gpt_analysis=gpt_response)

    except Exception as e:
        flash(f"ERROR: Search failed: {str(e)}")
        return render_template('index.html')

@app.route('/status')
def status():
    has_index = os.path.exists("faiss_vector.index")
    has_reference = os.path.exists("logs_reference.txt")
    return jsonify({
        'ready': has_index and has_reference,
        'has_index': has_index,
        'has_reference': has_reference
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
