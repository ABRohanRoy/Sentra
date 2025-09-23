import streamlit as st
import os
import json
import re
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import pandas as pd

# Page config
st.set_page_config(
    page_title="Sentra - AWS Log Analysis RAG Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependencies with graceful fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.error("FAISS required. Install with: pip install faiss-cpu")

try:
    from langchain_openai import AzureOpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.error("LangChain not available. Install with: pip install langchain-openai")

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    st.error("Azure OpenAI not available. Install with: pip install openai")

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

VECTOR_INDEX_PATH = DATA_DIR / "faiss_vector.index"
METADATA_PATH = DATA_DIR / "logs_reference.txt"
CONVERSATION_PATH = DATA_DIR / "conversations.json"
FILE_HASHES_PATH = DATA_DIR / "file_hashes.json"

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = {}

@st.cache_resource
def initialize_embedding_model():
    """Initialize embedding model"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        embedding_model = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("Embedding_AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("Embedding_AZURE_OPENAI_ENDPOINT"),
            deployment=os.getenv("Embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("Embedding_AZURE_OPENAI_API_VERSION"),
            chunk_size=10,
        )
        logger.info("Embedding model initialized")
        return embedding_model
    except Exception as e:
        st.error(f"Failed to initialize embedding model: {e}")
        return None

@st.cache_resource
def initialize_azure_client():
    """Initialize Azure OpenAI client"""
    if not AZURE_AVAILABLE:
        return None
    
    try:
        azure_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        logger.info("Azure OpenAI client initialized")
        return azure_client
    except Exception as e:
        st.error(f"Failed to initialize Azure OpenAI client: {e}")
        return None

# Initialize models
embedding_model = initialize_embedding_model()
azure_client = initialize_azure_client()

# -------------------------
# File Deduplication Functions
# -------------------------

def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

def load_file_hashes() -> Dict[str, str]:
    """Load file hash records"""
    if FILE_HASHES_PATH.exists():
        try:
            with open(FILE_HASHES_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_file_hashes(hashes: Dict[str, str]):
    """Save file hash records"""
    try:
        with open(FILE_HASHES_PATH, "w") as f:
            json.dump(hashes, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving file hashes: {e}")

def is_file_duplicate(file_name: str, file_content: bytes) -> bool:
    """Check if file is duplicate based on content hash"""
    current_hash = compute_file_hash(file_content)
    existing_hashes = load_file_hashes()
    return existing_hashes.get(file_name) == current_hash

def mark_file_processed(file_name: str, file_content: bytes):
    """Mark file as processed"""
    file_hash = compute_file_hash(file_content)
    hashes = load_file_hashes()
    hashes[file_name] = file_hash
    save_file_hashes(hashes)

# -------------------------
# Enhanced Log Parsing Functions
# -------------------------

def parse_s3_log_line(line: str) -> Optional[Dict]:
    """Parse S3 access log line"""
    parts = []
    current = ""
    in_quotes = False
    
    for char in line:
        if char == '"' and (not current or current[-1] != '\\'):
            in_quotes = not in_quotes
            current += char
        elif char == ' ' and not in_quotes:
            if current:
                parts.append(current.strip('"'))
                current = ""
        else:
            current += char
    
    if current:
        parts.append(current.strip('"'))
    
    if len(parts) < 20:
        return None
    
    try:
        return {
            'bucket_owner': parts[0],
            'bucket': parts[1],
            'timestamp': parts[2] + " " + parts[3],
            'remote_ip': parts[4],
            'requester': parts[5],
            'request_id': parts[6],
            'operation': parts[7],
            'key': parts[8],
            'request_uri': parts[9],
            'http_status': parts[10],
            'error_code': parts[11],
            'bytes_sent': parts[12],
            'object_size': parts[13],
            'total_time': parts[14],
            'turn_around_time': parts[15],
            'referrer': parts[16],
            'user_agent': parts[17],
            'log_type': 'S3',
            'raw_line': line
        }
    except IndexError:
        return None

def parse_ec2_log_line(line: str) -> Optional[Dict]:
    """Parse EC2/generic log line"""
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})'
    ip_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
    
    timestamp_match = re.search(timestamp_pattern, line)
    ip_match = re.search(ip_pattern, line)
    
    entry = {
        'log_type': 'EC2',
        'raw_line': line
    }
    
    if timestamp_match:
        entry['timestamp'] = timestamp_match.group(1)
    
    if ip_match:
        entry['ip'] = ip_match.group(1)
    
    status_match = re.search(r'\b([1-5]\d{2})\b', line)
    if status_match:
        entry['status'] = status_match.group(1)
    
    method_match = re.search(r'\b(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\b', line)
    if method_match:
        entry['method'] = method_match.group(1)
    
    return entry

def parse_json_log(file_content: str, file_name: str) -> List[Dict]:
    """Parse JSON log files"""
    entries = []
    try:
        data = json.loads(file_content)
        
        if isinstance(data, dict):
            entry = {
                'log_type': 'JSON',
                'source_file': file_name,
                'timestamp': data.get('digestStartTime', data.get('eventTime', str(datetime.now()))),
                'aws_account': data.get('awsAccountId', ''),
                'service': 'CloudTrail' if 'digest' in file_name.lower() else 'AWS',
                'raw_data': json.dumps(data, separators=(',', ':')),
            }
            entries.append(entry)
            
        elif isinstance(data, list):
            for i, item in enumerate(data):
                entry = {
                    'log_type': 'JSON',
                    'source_file': file_name,
                    'line_number': i + 1,
                    'timestamp': item.get('eventTime', item.get('timestamp', str(datetime.now()))),
                    'raw_data': json.dumps(item, separators=(',', ':')),
                }
                entries.append(entry)
                
    except Exception as e:
        logger.error(f"Error parsing JSON file {file_name}: {e}")
    
    return entries

def parse_uploaded_file(uploaded_file) -> List[Dict]:
    """Parse uploaded file with format detection"""
    if is_file_duplicate(uploaded_file.name, uploaded_file.getvalue()):
        st.warning(f"Skipping duplicate file: {uploaded_file.name}")
        return []
    
    file_ext = Path(uploaded_file.name).suffix.lower()
    entries = []
    
    try:
        if file_ext == '.json':
            content = uploaded_file.getvalue().decode('utf-8')
            entries = parse_json_log(content, uploaded_file.name)
        else:
            content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
            lines = content.split('\n')
            
            sample_lines = [line.strip() for line in lines[:5] if line.strip()]
            is_s3_log = any('amazonaws.com' in line or 'WEBSITE.GET.OBJECT' in line for line in sample_lines)
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                if is_s3_log:
                    parsed = parse_s3_log_line(line)
                else:
                    parsed = parse_ec2_log_line(line)
                
                if parsed:
                    parsed['source_file'] = uploaded_file.name
                    parsed['line_number'] = line_num
                    entries.append(parsed)
        
        if entries:
            mark_file_processed(uploaded_file.name, uploaded_file.getvalue())
            
    except Exception as e:
        st.error(f"Error parsing file {uploaded_file.name}: {e}")
    
    return entries

def format_log_entries(entries: List[Dict]) -> List[str]:
    """Format entries as strings for vector storage"""
    formatted_chunks = []
    for entry in entries:
        if entry.get('log_type') == 'S3':
            chunk = f"[{entry.get('timestamp', 'N/A')}] IP: {entry.get('remote_ip', 'N/A')} | Operation: {entry.get('operation', 'N/A')} | Object: {entry.get('key', 'N/A')} | Status: {entry.get('http_status', 'N/A')} | Bytes: {entry.get('bytes_sent', 'N/A')}"
        elif entry.get('log_type') == 'JSON':
            chunk = f"[{entry.get('timestamp', 'N/A')}] Service: {entry.get('service', 'AWS')} | Account: {entry.get('aws_account', 'N/A')} | File: {entry.get('source_file', 'N/A')}"
        else:
            chunk = f"[{entry.get('timestamp', 'N/A')}] IP: {entry.get('ip', 'N/A')} | Method: {entry.get('method', 'N/A')} | Status: {entry.get('status', 'N/A')} | Raw: {entry.get('raw_line', '')[:100]}"
        
        formatted_chunks.append(chunk)
    
    return formatted_chunks

# -------------------------
# Vector Operations
# -------------------------

def store_vectors(log_chunks: List[str]):
    """Store vectors in FAISS index"""
    if not embedding_model:
        st.error("Embedding model not available")
        return False
    
    if not log_chunks:
        st.warning("No log chunks to process")
        return False
    
    try:
        with st.spinner("Generating embeddings..."):
            vectors = np.array(embedding_model.embed_documents(log_chunks)).astype("float32")
            dimension = vectors.shape[1]
            
            if VECTOR_INDEX_PATH.exists():
                index = faiss.read_index(str(VECTOR_INDEX_PATH))
                st.info(f"Loaded existing index with {index.ntotal} vectors")
            else:
                index = faiss.IndexFlatL2(dimension)
                st.info("Created new FAISS index")
            
            index.add(vectors)
            faiss.write_index(index, str(VECTOR_INDEX_PATH))
            
            with open(METADATA_PATH, "a", encoding="utf-8") as f:
                for log in log_chunks:
                    f.write(log + "\n")
            
            st.success(f"Stored {len(log_chunks)} vectors successfully")
            return True
            
    except Exception as e:
        st.error(f"Error storing vectors: {e}")
        return False

def search_logs(query: str, top_k: int = 5) -> List[str]:
    """Search for similar log entries"""
    if not embedding_model:
        return []
    
    if not VECTOR_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return []
    
    try:
        index = faiss.read_index(str(VECTOR_INDEX_PATH))
        query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")
        distances, indices = index.search(query_vector, min(top_k, index.ntotal))
        
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            logs = f.readlines()
        
        results = []
        for i in indices[0]:
            if 0 <= i < len(logs):
                results.append(logs[i].strip())
        
        return results
        
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# -------------------------
# Domain Restriction Functions
# -------------------------

def is_query_in_scope(query: str) -> bool:
    """Check if query is within allowed technical domains"""
    query_lower = query.lower()
    
    # Allowed technical keywords
    allowed_domains = [
        # IT/Development
        'log', 'logs', 'server', 'database', 'application', 'code', 'programming',
        'software', 'system', 'network', 'infrastructure', 'api', 'service',
        'deployment', 'configuration', 'performance', 'monitoring', 'debug',
        
        # Cybersecurity
        'security', 'attack', 'threat', 'vulnerability', 'breach', 'malware',
        'firewall', 'intrusion', 'authentication', 'authorization', 'encryption',
        'phishing', 'exploit', 'penetration', 'incident', 'forensic', 'brute force',
        'ddos', 'sql injection', 'xss', 'csrf', 'malicious', 'suspicious',
        
        # Cloud/AWS
        'aws', 'azure', 'gcp', 'cloud', 'ec2', 's3', 'lambda', 'cloudtrail',
        'cloudformation', 'kubernetes', 'docker', 'container', 'serverless',
        'vpc', 'iam', 'cloudwatch', 'load balancer', 'autoscaling',
        
        # DevOps
        'devops', 'ci/cd', 'pipeline', 'automation', 'jenkins', 'git', 'docker',
        'terraform', 'ansible', 'chef', 'puppet', 'monitoring', 'alerting',
        'deployment', 'orchestration', 'microservices',
        
        # AI/ML
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'model',
        'algorithm', 'neural network', 'deep learning', 'training', 'prediction',
        'classification', 'regression', 'clustering', 'nlp', 'computer vision'
    ]
    
    # Blocked non-technical topics
    blocked_topics = [
        'movie', 'film', 'actor', 'actress', 'director', 'cinema', 'hollywood',
        'sport', 'football', 'basketball', 'soccer', 'baseball', 'tennis',
        'music', 'song', 'singer', 'band', 'album', 'concert',
        'food', 'recipe', 'cooking', 'restaurant', 'cuisine',
        'travel', 'vacation', 'tourism', 'hotel', 'flight',
        'politics', 'government', 'election', 'politician',
        'weather', 'climate', 'temperature', 'rain', 'snow',
        'health', 'medicine', 'doctor', 'disease', 'treatment',
        'fashion', 'clothing', 'style', 'brand',
        'celebrity', 'gossip', 'entertainment', 'tv show',
        'book', 'novel', 'author', 'literature', 'poetry'
    ]
    
    # Check for blocked topics first
    for blocked in blocked_topics:
        if blocked in query_lower:
            return False
    
    # Check for allowed technical terms
    for allowed in allowed_domains:
        if allowed in query_lower:
            return True
    
    # Technical indicators
    technical_indicators = ['analyze', 'show', 'find', 'detect', 'monitor', 'trace', 'debug']
    if any(indicator in query_lower for indicator in technical_indicators):
        return True
    
    return False

def generate_response(query: str, context_results: List[str]) -> str:
    """Generate domain-restricted response using Azure OpenAI"""
    if not azure_client:
        return """Azure OpenAI not configured. Check your .env file:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY  
- AZURE_OPENAI_DEPLOYMENT_NAME"""
    
    # Check if query is within allowed domains
    if not is_query_in_scope(query):
        return """I'm Sentra, focused on IT, Cybersecurity, AI, DevOps, and Cloud technologies.

I can help with:
- Log analysis and security monitoring
- AWS services and cloud infrastructure  
- DevOps practices and automation
- Cybersecurity threats and analysis
- IT systems and development
- AI/ML applications in these domains

Please ask technical questions in these areas."""
    
    context = "\n".join(context_results)
    
    system_prompt = f"""You are Sentra, an AWS log analysis assistant specialized in IT, Cybersecurity, AI, DevOps, and Cloud technologies.

DOMAIN RESTRICTIONS: Only respond to queries about:
- IT Development & Systems
- Cybersecurity & Threat Analysis  
- AI/Machine Learning
- DevOps & Infrastructure
- Cloud Technologies (AWS, Azure, GCP)
- Log analysis and monitoring

Instructions:
- Answer the specific question asked
- Be direct and concise
- Focus on technical accuracy
- Only mention relevant findings from the logs"""
    
    user_prompt = f"""Query: {query}

Relevant log entries ({len(context_results)} found):
{context}

Provide a focused technical response."""
    
    try:
        with st.spinner("Generating AI response..."):
            response = azure_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            
            return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        
        if "DeploymentNotFound" in error_msg:
            return f"""Configuration Issue: Azure OpenAI deployment not found.

Check your .env file:
- AZURE_OPENAI_DEPLOYMENT_NAME should match your actual deployment
- Current setting: {os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "Not set")}
- Endpoint: {os.getenv("AZURE_OPENAI_ENDPOINT", "Not set")}"""
        
        elif "InvalidApiKey" in error_msg:
            return """API Key Issue: Invalid or expired Azure OpenAI API key.

Check your AZURE_OPENAI_API_KEY in the .env file."""
        
        else:
            return f"Analysis Error: {error_msg}\n\nFound {len(context_results)} relevant log entries."

# -------------------------
# Conversation Management
# -------------------------

def load_conversations():
    """Load conversation history"""
    if CONVERSATION_PATH.exists():
        try:
            with open(CONVERSATION_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_conversations(conversations):
    """Save conversation history"""
    try:
        with open(CONVERSATION_PATH, "w") as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving conversations: {e}")

def add_conversation(query: str, response: str, matching_logs: List[str]):
    """Add conversation to history"""
    conversations = load_conversations()
    conversations.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "matching_logs": matching_logs[:3]
    })
    
    if len(conversations) > 50:
        conversations = conversations[-50:]
    
    save_conversations(conversations)
    st.session_state.conversation_history = conversations

# -------------------------
# Streamlit UI
# -------------------------

def main():
    st.title("🚀 Sentra - AWS Log Analysis RAG Agent")
    st.markdown("**S3 Access Logs • EC2 Logs • CloudTrail JSON • Generic Text Logs**")
    
    # Sidebar for system info
    with st.sidebar:
        st.header("🔧 System Status")
        
        config_status = {
            "Azure OpenAI": azure_client is not None,
            "Embeddings": embedding_model is not None,
            "Vector Store": VECTOR_INDEX_PATH.exists()
        }
        
        for service, status in config_status.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} **{service}**")
        
        st.header("📊 Statistics")
        file_hashes = load_file_hashes()
        total_logs = 0
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r") as f:
                    total_logs = len(f.readlines())
            except Exception:
                pass
        
        st.metric("Files Processed", len(file_hashes))
        st.metric("Log Entries", total_logs)
        st.metric("Conversations", len(st.session_state.conversation_history))
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.sidebar.checkbox("Confirm deletion"):
                for path in [VECTOR_INDEX_PATH, METADATA_PATH, FILE_HASHES_PATH, CONVERSATION_PATH]:
                    if path.exists():
                        path.unlink()
                st.session_state.conversation_history = []
                st.success("All data cleared!")
                st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "🔍 Query Logs", "💬 Chat History"])
    
    with tab1:
        st.header("📂 Upload Log Files")
        st.markdown("**Supported formats:** .log, .txt, .json")
        
        uploaded_files = st.file_uploader(
            "Choose log files",
            accept_multiple_files=True,
            type=['log', 'txt', 'json'],
            help="Upload S3 access logs, EC2 logs, CloudTrail JSON files, or any text-based log files"
        )
        
        if uploaded_files:
            st.write(f"**Selected Files:** {len(uploaded_files)}")
            
            file_info = []
            for file in uploaded_files:
                file_info.append({
                    "Name": file.name,
                    "Size": f"{file.size:,} bytes",
                    "Type": file.type or "text/plain"
                })
            
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("⚡ Process Files", type="primary", use_container_width=True):
                    process_files(uploaded_files)
            
            with col2:
                if st.button("🔄 Reset Upload", use_container_width=True):
                    st.rerun()
    
    with tab2:
        st.header("🔍 Query Your Logs")
        
        if not METADATA_PATH.exists():
            st.warning("⚠️ No processed logs found. Please upload and process some log files first.")
            return
        
        query = st.text_area(
            "Ask questions about your logs:",
            placeholder="Examples:\n• Show me all failed requests\n• What are the top IP addresses?\n• Find any security issues\n• Analyze S3 bucket access patterns",
            height=100
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_button = st.button("🤖 Analyze Logs", type="primary", disabled=not query.strip())
        
        with col2:
            top_k = st.selectbox("Results to show:", [3, 5, 10, 15], index=1)
        
        if search_button and query.strip():
            query_logs_ui(query, top_k)
    
    with tab3:
        st.header("💬 Conversation History")
        
        conversations = load_conversations()
        st.session_state.conversation_history = conversations
        
        if not conversations:
            st.info("No conversations yet. Start by querying your logs!")
            return
        
        for i, conv in enumerate(reversed(conversations)):
            with st.expander(f"**Query #{len(conversations)-i}:** {conv['query'][:50]}...", expanded=i==0):
                st.write("**🕐 Time:**", datetime.fromisoformat(conv['timestamp']).strftime("%Y-%m-%d %H:%M:%S"))
                st.write("**❓ Query:**", conv['query'])
                st.write("**🤖 Response:**")
                st.markdown(conv['response'])
                
                if conv.get('matching_logs'):
                    st.write("**📝 Matching Logs:**")
                    for j, log in enumerate(conv['matching_logs'], 1):
                        st.code(f"{j}. {log}", language="text")
        
        if st.button("🗑️ Clear History"):
            if CONVERSATION_PATH.exists():
                CONVERSATION_PATH.unlink()
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
            st.rerun()

def process_files(uploaded_files):
    """Process uploaded files"""
    all_entries = []
    processed_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        entries = parse_uploaded_file(uploaded_file)
        if entries:
            all_entries.extend(entries)
            processed_count += 1
            st.success(f"✅ Processed {uploaded_file.name}: {len(entries)} entries")
        else:
            st.info(f"ℹ️ Skipped {uploaded_file.name} (duplicate or no valid entries)")
    
    if all_entries:
        status_text.text("Storing vectors...")
        log_chunks = format_log_entries(all_entries)
        
        if store_vectors(log_chunks):
            st.success(f"🎉 Successfully processed {processed_count} files with {len(all_entries)} total log entries!")
        else:
            st.error("❌ Failed to store vectors")
    else:
        st.warning("⚠️ No new entries to process")
    
    progress_bar.empty()
    status_text.empty()

def query_logs_ui(query: str, top_k: int):
    """Handle log querying in UI"""
    results = search_logs(query, top_k)
    
    if not results:
        st.warning("No matching log entries found.")
        return
    
    response = generate_response(query, results)
    
    st.subheader("🤖 AI Analysis")
    st.markdown(response)
    
    st.subheader(f"📝 Matching Log Entries ({len(results)} found)")
    
    for i, log_entry in enumerate(results, 1):
        with st.expander(f"Log Entry #{i}", expanded=i <= 3):
            st.code(log_entry, language="text")
    
    add_conversation(query, response, results)
    
    st.success("✅ Query completed and saved to history!")

if __name__ == "__main__":
    main()