import os
import json
import re
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
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
    raise RuntimeError("FAISS required. Install with: pip install faiss-cpu")

try:
    from langchain_openai import AzureOpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain-openai")

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure OpenAI not available. Install with: pip install openai")

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

VECTOR_INDEX_PATH = DATA_DIR / "faiss_vector.index"
METADATA_PATH = DATA_DIR / "logs_reference.txt"
CONVERSATION_PATH = DATA_DIR / "conversations.json"
FILE_HASHES_PATH = DATA_DIR / "file_hashes.json"

# Initialize embedding model using .env values (matching original format)
embedding_model = None
if LANGCHAIN_AVAILABLE:
    try:
        embedding_model = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("Embedding_AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("Embedding_AZURE_OPENAI_ENDPOINT"),
            deployment=os.getenv("Embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("Embedding_AZURE_OPENAI_API_VERSION"),
            chunk_size=10,
        )
        logger.info("Embedding model initialized")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")

# Azure OpenAI client for chat
azure_client = None
if AZURE_AVAILABLE:
    try:
        azure_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
        )
        logger.info("Azure OpenAI client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}")

# -------------------------
# File Deduplication
# -------------------------

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file content"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return ""

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

def is_file_duplicate(file_path: str) -> bool:
    """Check if file is duplicate based on content hash"""
    current_hash = compute_file_hash(file_path)
    if not current_hash:
        return False
    
    existing_hashes = load_file_hashes()
    file_name = os.path.basename(file_path)
    return existing_hashes.get(file_name) == current_hash

def mark_file_processed(file_path: str):
    """Mark file as processed"""
    file_hash = compute_file_hash(file_path)
    if file_hash:
        hashes = load_file_hashes()
        hashes[os.path.basename(file_path)] = file_hash
        save_file_hashes(hashes)

# -------------------------
# Enhanced Log Parsing
# -------------------------

def parse_s3_log_line(line: str) -> Optional[Dict]:
    """Parse S3 access log line"""
    # S3 log format is space-separated with some quoted fields
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
    
    if len(parts) < 20:  # S3 logs should have many fields
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
    # Try to extract timestamp, IP, and other common fields
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
    
    # Look for HTTP status codes
    status_match = re.search(r'\b([1-5]\d{2})\b', line)
    if status_match:
        entry['status'] = status_match.group(1)
    
    # Look for HTTP methods
    method_match = re.search(r'\b(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\b', line)
    if method_match:
        entry['method'] = method_match.group(1)
    
    return entry

def parse_json_log(file_path: str) -> List[Dict]:
    """Parse JSON log files (like CloudTrail)"""
    entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle different JSON log formats
            if isinstance(data, dict):
                # Single JSON object (like CloudTrail digest)
                entry = {
                    'log_type': 'JSON',
                    'source_file': os.path.basename(file_path),
                    'timestamp': data.get('digestStartTime', data.get('eventTime', str(datetime.now()))),
                    'aws_account': data.get('awsAccountId', ''),
                    'service': 'CloudTrail' if 'digest' in file_path.lower() else 'AWS',
                    'raw_data': json.dumps(data, separators=(',', ':')),
                }
                entries.append(entry)
                
            elif isinstance(data, list):
                # Array of JSON objects
                for i, item in enumerate(data):
                    entry = {
                        'log_type': 'JSON',
                        'source_file': os.path.basename(file_path),
                        'line_number': i + 1,
                        'timestamp': item.get('eventTime', item.get('timestamp', str(datetime.now()))),
                        'raw_data': json.dumps(item, separators=(',', ':')),
                    }
                    entries.append(entry)
                    
    except Exception as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
    
    return entries

def parse_log_file(file_path: str) -> List[Dict]:
    """Parse log file with format detection (matching original function name)"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    if is_file_duplicate(file_path):
        logger.info(f"Skipping duplicate file: {os.path.basename(file_path)}")
        return []
    
    file_ext = Path(file_path).suffix.lower()
    entries = []
    
    try:
        # Handle JSON files
        if file_ext == '.json':
            entries = parse_json_log(file_path)
        else:
            # Handle .log and .txt files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Detect log format from first few lines
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
                    parsed['source_file'] = os.path.basename(file_path)
                    parsed['line_number'] = line_num
                    entries.append(parsed)
        
        if entries:
            mark_file_processed(file_path)
            logger.info(f"Parsed {len(entries)} entries from {os.path.basename(file_path)}")
        
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {e}")
    
    return entries

def load_log_chunks(file_paths: List[str]) -> List[str]:
    """Load and parse multiple log files, return formatted chunks"""
    all_entries = []
    
    for file_path in file_paths:
        entries = parse_log_file(file_path)
        all_entries.extend(entries)
    
    # Format entries as strings (matching original format)
    formatted_chunks = []
    for entry in all_entries:
        if entry.get('log_type') == 'S3':
            chunk = f"[{entry.get('timestamp', 'N/A')}] IP: {entry.get('remote_ip', 'N/A')} | Operation: {entry.get('operation', 'N/A')} | Object: {entry.get('key', 'N/A')} | Status: {entry.get('http_status', 'N/A')} | Bytes: {entry.get('bytes_sent', 'N/A')}"
        elif entry.get('log_type') == 'JSON':
            chunk = f"[{entry.get('timestamp', 'N/A')}] Service: {entry.get('service', 'AWS')} | Account: {entry.get('aws_account', 'N/A')} | File: {entry.get('source_file', 'N/A')}"
        else:  # EC2 or generic
            chunk = f"[{entry.get('timestamp', 'N/A')}] IP: {entry.get('ip', 'N/A')} | Method: {entry.get('method', 'N/A')} | Status: {entry.get('status', 'N/A')} | Raw: {entry.get('raw_line', '')[:100]}"
        
        formatted_chunks.append(chunk)
    
    return formatted_chunks

def store_vector(log_chunks: List[str]):
    """Store vectors in FAISS index (matching original function)"""
    if not embedding_model:
        logger.error("Embedding model not available")
        return None
    
    if not log_chunks:
        logger.warning("No log chunks to process")
        return None
    
    try:
        vectors = np.array(embedding_model.embed_documents(log_chunks)).astype("float32")
        dimension = vectors.shape[1]
        
        # Load existing index or create new one
        if VECTOR_INDEX_PATH.exists():
            index = faiss.read_index(str(VECTOR_INDEX_PATH))
            logger.info(f"Loaded existing index with {index.ntotal} vectors")
        else:
            index = faiss.IndexFlatL2(dimension)
            logger.info("Created new FAISS index")
        
        # Add new vectors
        index.add(vectors)
        faiss.write_index(index, str(VECTOR_INDEX_PATH))
        
        # Append logs to reference file
        with open(METADATA_PATH, "a", encoding="utf-8") as f:
            for log in log_chunks:
                f.write(log + "\n")
        
        logger.info(f"Stored {len(log_chunks)} vectors successfully")
        return vectors
        
    except Exception as e:
        logger.error(f"Error storing vectors: {e}")
        return None

def search(query: str, top_k: int = 5) -> List[str]:
    """Search for similar log entries (matching original function)"""
    if not embedding_model:
        logger.error("Embedding model not available")
        return []
    
    if not VECTOR_INDEX_PATH.exists() or not METADATA_PATH.exists():
        logger.warning("No vector index or metadata found")
        return []
    
    try:
        index = faiss.read_index(str(VECTOR_INDEX_PATH))
        query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")
        distances, indices = index.search(query_vector, min(top_k, index.ntotal))
        
        # Retrieve matching logs
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            logs = f.readlines()
        
        results = []
        for i in indices[0]:
            if 0 <= i < len(logs):
                results.append(logs[i].strip())
        
        return results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def ask_gpt(query: str, context_results: List[str]) -> str:
    """Generate response using Azure OpenAI (matching original function name)"""
    if not azure_client:
        return "Azure OpenAI not configured. Please set up your Azure credentials."
    
    context = "\n".join(context_results)
    
    system_prompt = """You are Sentra, an expert AWS log analysis assistant. You help users understand their S3 access logs, EC2 logs, CloudTrail logs, and other AWS service logs.

Key capabilities:
- Analyze S3 access patterns, errors, and usage
- Identify security issues in access logs
- Explain CloudTrail events and API calls
- Detect anomalies in EC2 and application logs
- Provide actionable insights and recommendations

Always base your answers on the provided log data and be specific about what you find."""
    
    user_prompt = f"""Query: {query}

Relevant log entries:
{context}

Please analyze the above log data and provide a comprehensive answer."""
    
    try:
        response = azure_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"GPT response error: {e}")
        return f"I encountered an error analyzing your query: {str(e)}"

# -------------------------
# Chat Memory Management
# -------------------------

def load_conversation() -> List[Dict]:
    """Load conversation history"""
    if CONVERSATION_PATH.exists():
        try:
            with open(CONVERSATION_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_conversation(conversation: List[Dict]):
    """Save conversation history"""
    try:
        with open(CONVERSATION_PATH, "w") as f:
            json.dump(conversation, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")

def add_to_conversation(query: str, response: str, results: List[str]):
    """Add Q&A to conversation history"""
    conversation = load_conversation()
    conversation.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "matching_logs": results[:3]  # Store top 3 matches
    })
    
    # Keep only last 50 conversations
    if len(conversation) > 50:
        conversation = conversation[-50:]
    
    save_conversation(conversation)

def get_conversation_context() -> str:
    """Get recent conversation context for continuity"""
    conversation = load_conversation()
    if not conversation:
        return ""
    
    # Get last 3 conversations for context
    recent = conversation[-3:]
    context_parts = []
    
    for item in recent:
        context_parts.append(f"Previous Q: {item['query']}")
        context_parts.append(f"Previous A: {item['response'][:200]}...")
    
    return "\n".join(context_parts) if context_parts else ""

# -------------------------
# System Stats & Management
# -------------------------

def get_system_stats() -> Dict:
    """Get system statistics"""
    stats = {
        'has_data': METADATA_PATH.exists(),
        'azure_configured': azure_client is not None,
        'embeddings_configured': embedding_model is not None,
        'total_files': len(load_file_hashes()),
        'total_logs': 0
    }
    
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r") as f:
                stats['total_logs'] = len(f.readlines())
        except Exception:
            pass
    
    return stats

def clear_all_data():
    """Clear all processed data"""
    for path in [VECTOR_INDEX_PATH, METADATA_PATH, FILE_HASHES_PATH, CONVERSATION_PATH]:
        if path.exists():
            path.unlink()
    logger.info("All data cleared")

def process_files(file_paths: List[str]) -> Dict:
    """Process multiple files and return results"""
    if not file_paths:
        return {"success": False, "message": "No files provided"}
    
    try:
        # Load and process logs
        log_chunks = load_log_chunks(file_paths)
        
        if not log_chunks:
            return {"success": False, "message": "No valid log entries found"}
        
        # Store vectors
        vectors = store_vector(log_chunks)
        
        if vectors is not None:
            return {
                "success": True, 
                "message": f"Processed {len(log_chunks)} log entries from {len(file_paths)} files"
            }
        else:
            return {"success": False, "message": "Failed to store vectors"}
            
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

def query_logs(query: str, top_k: int = 5) -> Dict:
    """Query logs and return response with context"""
    try:
        # Get conversation context
        context = get_conversation_context()
        enhanced_query = f"{context}\n\nCurrent query: {query}" if context else query
        
        # Search for relevant logs
        results = search(enhanced_query, top_k)
        
        if not results:
            return {
                "success": False,
                "message": "No matching log entries found. Please make sure you have processed some log files first."
            }
        
        # Generate response
        response = ask_gpt(query, results)
        
        # Save to conversation
        add_to_conversation(query, response, results)
        
        return {
            "success": True,
            "query": query,
            "response": response,
            "matching_logs": results,
            "total_matches": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error querying logs: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

# -------------------------
# Main CLI Interface (matching original)
# -------------------------

if __name__ == "__main__":
    print("🚀 Sentra - AWS Log Analysis RAG Agent")
    print("Supports S3 Access Logs, EC2 Logs, CloudTrail JSON, and more!")
    print("-" * 50)
    
    # Check system status
    stats = get_system_stats()
    print(f"✅ Azure OpenAI: {'Configured' if stats['azure_configured'] else 'Not configured'}")
    print(f"✅ Embeddings: {'Configured' if stats['embeddings_configured'] else 'Not configured'}")
    print(f"📊 Processed Files: {stats['total_files']}")
    print(f"📝 Log Entries: {stats['total_logs']}")
    
    if not stats['embeddings_configured']:
        print("\n⚠️ Embedding model not configured. Please check your .env file.")
        exit(1)
    
    # Main loop
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("1. Process log files")
        print("2. Query logs") 
        print("3. View conversation history")
        print("4. Clear all data")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            print("\nEnter file paths (comma-separated):")
            print("Supported: .log, .txt, .json files")
            file_input = input("Files: ").strip()
            
            if file_input:
                file_paths = [f.strip() for f in file_input.split(",")]
                result = process_files(file_paths)
                print(f"\n{'✅' if result['success'] else '❌'} {result['message']}")
        
        elif choice == "2":
            if not stats['has_data']:
                print("\n⚠️ No data available. Please process some log files first.")
                continue
                
            query = input("\n🔍 Enter your query: ").strip()
            if query:
                print("\n🤖 Analyzing logs...")
                result = query_logs(query)
                
                if result['success']:
                    print(f"\n📄 Query: {result['query']}")
                    print("📄 Matching Log Entries:")
                    for i, log in enumerate(result['matching_logs'], 1):
                        print(f"  {i}. {log}")
                    
                    print(f"\n🤖 GPT Insight:")
                    print(result['response'])
                else:
                    print(f"\n❌ {result['message']}")
        
        elif choice == "3":
            conversation = load_conversation()
            if conversation:
                print(f"\n📜 Recent Conversations ({len(conversation)} total):")
                for i, item in enumerate(conversation[-5:], 1):  # Show last 5
                    print(f"\n{i}. [{item['timestamp']}]")
                    print(f"   Q: {item['query']}")
                    print(f"   A: {item['response'][:150]}...")
            else:
                print("\n📜 No conversation history found.")
        
        elif choice == "4":
            confirm = input("\n⚠️ Clear all data? (y/N): ").strip().lower()
            if confirm == 'y':
                clear_all_data()
                print("✅ All data cleared.")
        
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid option. Please try again.")