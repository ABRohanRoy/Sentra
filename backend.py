# backend.py
"""
Fast, optimized backend for Sentra log analysis
- Simplified imports and error handling
- Faster processing with batch operations
- Memory-efficient FAISS operations
- Robust parsing with fallbacks
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from itertools import islice
import numpy as np
from collections import defaultdict, Counter

# Simple embedding fallback using TF-IDF if transformers not available
_embedding_model = None
_use_tfidf = False

try:
    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    print("✓ Using sentence-transformers for embeddings")
except Exception:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _embedding_model = TfidfVectorizer(max_features=384, stop_words='english', ngram_range=(1,2))
        _use_tfidf = True
        print("✓ Using TF-IDF for embeddings (fallback)")
    except Exception:
        print("⚠ No embedding model available - install sentence-transformers or scikit-learn")

# FAISS import with error handling
try:
    import faiss
except Exception:
    print("❌ FAISS not found. Install with: pip install faiss-cpu")
    raise

# Paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
VECTOR_INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
MEMORY_PATH = os.path.join(DATA_DIR, "memory.json")

# Global cache
_INDEX_CACHE = None
_METADATA_CACHE = None

class FastLogParser:
    """Unified fast parser with pattern matching"""
    
    # Common patterns
    IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    TIMESTAMP_PATTERNS = [
        re.compile(r'\[([^\]]+)\]'),  # [timestamp]
        re.compile(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})'),  # ISO
        re.compile(r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'),  # syslog
    ]
    HTTP_PATTERN = re.compile(r'"(GET|POST|PUT|DELETE|HEAD|PATCH)\s+(\S+).*?"\s+(\d{3})')
    STATUS_PATTERN = re.compile(r'\b([2-5]\d{2})\b')
    
    def parse_file(self, file_path: str) -> List[Dict]:
        """Fast unified parsing"""
        entries = []
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    entry = {
                        'id': hashlib.md5(f"{filename}:{line_num}:{line}".encode()).hexdigest()[:12],
                        'source_file': filename,
                        'line_number': line_num,
                        'raw': line,
                        'text': line  # for embedding
                    }
                    
                    # Extract fields
                    self._extract_fields(entry, line)
                    entries.append(entry)
                    
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return entries
    
    def _extract_fields(self, entry: Dict, line: str):
        """Extract common fields fast"""
        # IP addresses
        ip_matches = self.IP_PATTERN.findall(line)
        if ip_matches:
            entry['ip'] = ip_matches[0]
        
        # Timestamps
        for pattern in self.TIMESTAMP_PATTERNS:
            match = pattern.search(line)
            if match:
                entry['timestamp'] = match.group(1)
                break
        
        # HTTP requests
        http_match = self.HTTP_PATTERN.search(line)
        if http_match:
            entry['method'] = http_match.group(1)
            entry['endpoint'] = http_match.group(2)
            entry['status'] = http_match.group(3)
        else:
            # Fallback status code search
            status_match = self.STATUS_PATTERN.search(line)
            if status_match:
                entry['status'] = status_match.group(1)

class FastEmbedder:
    """Fast embedding with caching"""
    
    def __init__(self):
        self.model = _embedding_model
        self.use_tfidf = _use_tfidf
        self._fitted = False
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Fast batch embedding"""
        if not self.model:
            # Fallback to simple word hashing
            return self._hash_embeddings(texts)
        
        if self.use_tfidf:
            if not self._fitted:
                self.model.fit(texts)
                self._fitted = True
            vectors = self.model.transform(texts).toarray()
        else:
            vectors = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        
        return vectors.astype('float32')
    
    def embed_query(self, query: str) -> np.ndarray:
        """Fast query embedding"""
        if not self.model:
            return self._hash_embeddings([query])
        
        if self.use_tfidf:
            if not self._fitted:
                # If not fitted, return zeros
                return np.zeros((1, 384), dtype='float32')
            return self.model.transform([query]).toarray().astype('float32')
        else:
            return self.model.encode([query]).astype('float32')
    
    def _hash_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback hash-based embeddings"""
        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            words = text.lower().split()
            vec = np.zeros(384)
            for i, word in enumerate(words[:50]):  # limit words
                hash_val = hash(word) % 384
                vec[hash_val] += 1.0 / (i + 1)  # position weighting
            embeddings.append(vec)
        return np.array(embeddings, dtype='float32')

class FastSearch:
    """Fast FAISS-based search"""
    
    def __init__(self):
        self.index = None
        self.embedder = FastEmbedder()
        self.metadata = []
        self.id_to_idx = {}
    
    def load_or_create_index(self) -> bool:
        """Load existing index or create new one"""
        global _INDEX_CACHE, _METADATA_CACHE
        
        if _INDEX_CACHE is not None:
            self.index = _INDEX_CACHE
            self.metadata = _METADATA_CACHE or []
            self._rebuild_id_mapping()
            return True
        
        if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(METADATA_PATH):
            try:
                self.index = faiss.read_index(VECTOR_INDEX_PATH)
                with open(METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
                self._rebuild_id_mapping()
                _INDEX_CACHE = self.index
                _METADATA_CACHE = self.metadata
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
        
        return False
    
    def _rebuild_id_mapping(self):
        """Rebuild ID to index mapping"""
        self.id_to_idx = {item['id']: i for i, item in enumerate(self.metadata)}
    
    def add_entries(self, entries: List[Dict]) -> int:
        """Add entries to index"""
        if not entries:
            return 0
        
        # Extract texts for embedding
        texts = [entry['text'] for entry in entries]
        vectors = self.embedder.embed_batch(texts)
        
        if self.index is None:
            # Create new index
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []
        
        # Add to index
        self.index.add(vectors)
        
        # Add metadata
        for entry in entries:
            self.metadata.append({
                'id': entry['id'],
                'source_file': entry['source_file'],
                'line_number': entry.get('line_number'),
                'ip': entry.get('ip'),
                'timestamp': entry.get('timestamp'),
                'method': entry.get('method'),
                'endpoint': entry.get('endpoint'),
                'status': entry.get('status'),
            })
        
        self._rebuild_id_mapping()
        self._save_index()
        
        # Update cache
        global _INDEX_CACHE, _METADATA_CACHE
        _INDEX_CACHE = self.index
        _METADATA_CACHE = self.metadata
        
        return len(entries)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Fast semantic search"""
        if not self.index or self.index.ntotal == 0:
            return []
        
        query_vector = self.embedder.embed_query(query)
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def _save_index(self):
        """Save index and metadata"""
        try:
            faiss.write_index(self.index, VECTOR_INDEX_PATH)
            with open(METADATA_PATH, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving index: {e}")

class Analytics:
    """Fast analytics and insights"""
    
    @staticmethod
    def analyze_results(results: List[Dict]) -> str:
        """Fast result analysis"""
        if not results:
            return "No results found."
        
        total = len(results)
        ips = Counter(r.get('ip') for r in results if r.get('ip'))
        statuses = Counter(r.get('status') for r in results if r.get('status'))
        methods = Counter(r.get('method') for r in results if r.get('method'))
        files = Counter(r.get('source_file') for r in results if r.get('source_file'))
        
        parts = [f"Found {total} matching entries."]
        
        if ips:
            top_ips = ips.most_common(3)
            parts.append(f"Top IPs: {', '.join(f'{ip}({c})' for ip, c in top_ips)}")
        
        if statuses:
            parts.append(f"Status codes: {', '.join(f'{s}({c})' for s, c in statuses.most_common(3))}")
        
        if methods:
            parts.append(f"Methods: {', '.join(f'{m}({c})' for m, c in methods.most_common(3))}")
        
        return " | ".join(parts)

# Global instances
_parser = FastLogParser()
_search = FastSearch()
_analytics = Analytics()

# Memory functions
def save_memory(data: Dict):
    """Save memory data"""
    try:
        with open(MEMORY_PATH, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass

def load_memory() -> Dict:
    """Load memory data"""
    try:
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

# Public API
def process_files(file_paths: List[str]) -> Tuple[int, int]:
    """Process log files and add to index"""
    _search.load_or_create_index()
    
    all_entries = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            entries = _parser.parse_file(file_path)
            all_entries.extend(entries)
    
    if all_entries:
        added = _search.add_entries(all_entries)
        return len(all_entries), added
    
    return 0, 0

def search_logs(query: str, top_k: int = 5) -> List[Dict]:
    """Search logs with semantic similarity"""
    _search.load_or_create_index()
    return _search.search(query, top_k)

def analyze_results(results: List[Dict]) -> str:
    """Analyze search results"""
    return _analytics.analyze_results(results)

def get_stats() -> Dict:
    """Get system statistics"""
    _search.load_or_create_index()
    metadata = _search.metadata or []
    
    return {
        'total_entries': len(metadata),
        'total_files': len(set(item.get('source_file') for item in metadata if item.get('source_file'))),
        'unique_ips': len(set(item.get('ip') for item in metadata if item.get('ip'))),
        'has_index': _search.index is not None and _search.index.ntotal > 0
    }

def remember_query(query: str):
    """Remember last query"""
    memory = load_memory()
    memory['last_query'] = query
    memory.setdefault('query_history', []).append(query)
    memory['query_history'] = memory['query_history'][-10:]  # keep last 10
    save_memory(memory)

def get_last_query() -> Optional[str]:
    """Get last query"""
    memory = load_memory()
    return memory.get('last_query')

def get_dashboard_data() -> Dict:
    """Get dashboard data"""
    _search.load_or_create_index()
    metadata = _search.metadata or []
    
    if not metadata:
        return {'error': 'No data available'}
    
    # Count by IP
    ip_counts = Counter(item.get('ip') for item in metadata if item.get('ip'))
    
    # Count by file
    file_counts = Counter(item.get('source_file') for item in metadata if item.get('source_file'))
    
    # Count by status
    status_counts = Counter(item.get('status') for item in metadata if item.get('status'))
    
    return {
        'top_ips': dict(ip_counts.most_common(10)),
        'file_stats': dict(file_counts),
        'status_codes': dict(status_counts.most_common(10)),
        'total_entries': len(metadata)
    }

# Initialize on import
if __name__ != "__main__":
    _search.load_or_create_index()