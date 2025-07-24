import os
import numpy as np
from dotenv import load_dotenv
import faiss
from langchain_openai import AzureOpenAIEmbeddings
from sentra.parser.s3_log_parser import parse_log_file

load_dotenv()

# Initialize embedding model using .env values
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

    # Save logs to file for lookup
    with open("logs_reference.txt", "w", encoding="utf-8") as f:
        for log in log_chunks:
            f.write(log + "\n")

    print("✅ Vector stored successfully")
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

# Main runner
if __name__ == "__main__":
    file_path = "data/sample_logs/s3_access_sample.log"  # Your log file

    log_chunks = load_log_chunks(file_path)
    if log_chunks:
        store_vector(log_chunks)

        query = "Were there any failed GET requests?"
        results = search(query)
        print(f"\n🔍 Query: {query}")
        print("📄 Matching Log Entries:")
        for r in results:
            print("➤", r)
        else:
            print("Log file is empty or failed to parse.")
