🚀 Simple-RAG: Privacy-First AI Data System


This is a Question & Answering system built on Retrieval-Augmented Generation (RAG) technology. This project focuses on Privacy-First AI, allowing users to process PDF documents and query them entirely on local infrastructure, effectively mitigating LLM hallucinations and data leakage risks.

🌟 Key Features
Privacy-First (Local LLM): Operates entirely on-premise using Ollama, ensuring sensitive PDF data never leaves your machine.

Document Parsing: Automatically reads and processes PDF text content using PyPDF.

Semantic Search: Utilizes Llama 3.2 embeddings to vectorise text for precise information retrieval.

Cross-Platform: Fully optimized for WSL2/Linux development environments.

Persistent Storage: Uses ChromaDB as a vector database for efficient data management.

🛠️ Tech Stack
Language: Python 3.9+

Environment: WSL2 / Ubuntu (Recommended)

Framework: LangChain / LangChain-Classic

LLM / Embeddings: Ollama (Llama 3.2)

Vector DB: ChromaDB

📦 Quick Start
1. Prerequisites
Ensure you have Ollama installed and the model downloaded:

Bash
# Install Ollama (Linux/WSL)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2
2. Setup Environment
Bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
3. Run the System
Place your PDF files in the data/ folder, then execute:

Bash
python3 main.py

## RESUME

My-RAG-Lab | Python, LangChain, OpenAI, ChromaDB

Developed a Retrieval-Augmented Generation (RAG) system to process and query private PDF documents using Ollama (Llama 3.2).

Implemented text chunking and semantic search using Ollama Embeddings and ChromaDB, reducing LLM hallucinations.

Designed a modular Python architecture for scalable document ingestion and retrieval.
