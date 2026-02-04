# 🚀 Simple-RAG: Intelligent PDF Q&A System

This is a Question & Answering system built on **Retrieval-Augmented Generation (RAG)** technology. Users can upload PDF documents, and the AI will provide accurate, grounded answers based on the content, effectively mitigating LLM hallucinations.

## 🌟 Key Features
- **Document Parsing**: Automatically reads and processes PDF text content.
- **Semantic Search**: Utilizes OpenAI Embeddings to vectorise text for precise information retrieval.
- **Intelligent Generation**: Combines GPT-4o with retrieved context to generate concise and accurate answers.
- **Persistent Storage**: Uses ChromaDB as a vector database for efficient data management.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Framework**: LangChain
- **LLM**: OpenAI GPT-4o
- **Vector DB**: ChromaDB
- **Environment**: python-dotenv

## 📦 Quick Start

# 1. Create a virtual environment named 'venv'
python3 -m venv venv

# 2. Activate it
# For Mac/Linux:
# source venv/bin/activate
# For Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

## RESUME

My-RAG-Lab | Python, LangChain, OpenAI, ChromaDB

Developed a Retrieval-Augmented Generation (RAG) system to process and query private PDF documents using GPT-4o.

Implemented text chunking and semantic search using OpenAI Embeddings and ChromaDB, reducing LLM hallucinations.

Designed a modular Python architecture for scalable document ingestion and retrieval.
