# sentra/agent/gpt_responder.py

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # e.g., "gpt-4o"

def ask_gpt(query, log_snippets):
    logs_text = "\n".join(log_snippets)
    prompt = f"""
You are a cybersecurity log expert.

The user asked: "{query}"

Here are some S3 access log entries:
{logs_text}

Please analyze and give a concise, human-friendly insight or summary based on the query and logs. Do not assume anything outside the logs.
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
