import os
import sys
from dotenv import load_dotenv

# Using the new Ollama integration
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Using your discovered langchain_classic workaround
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class LocalRAGLab:
    def __init__(self, data_folder="data", db_folder="vector_db"):
        """Fully localized RAG system powered by Ollama."""
        self.data_folder = data_folder
        self.db_folder = db_folder

        # Initialize Local LLM (Llama 3.2 is lightweight and fast)
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        # Initialize Local Embeddings (Same model for consistency)
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        self.vector_db = None

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def ingest_pdf(self, file_name):
        """Process PDF locally without sending data to the cloud."""
        file_path = os.path.join(self.data_folder, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        splits = text_splitter.split_documents(documents)

        # Store in local ChromaDB
        self.vector_db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_folder,
        )
        print(f"Local indexing complete: {len(splits)} chunks.")
        return True

    def query(self, question):
        """Handle local inference."""
        if not self.vector_db:
            return "Please ingest a document first."

        system_prompt = (
            "You are a helpful local assistant. Use the context provided to answer the question. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Using your classic chain logic
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(
            self.vector_db.as_retriever(), combine_docs_chain
        )

        response = rag_chain.invoke({"input": question})
        return response["answer"]


if __name__ == "__main__":
    lab = LocalRAGLab()

    # Auto-ingest first PDF found
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    if pdf_files:
        lab.ingest_pdf(pdf_files[0])

    print("\n--- Local RAG Lab Ready (Powered by Ollama) ---")
    while True:
        user_input = input("\n[Question]: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print(f"\n[AI]: {lab.query(user_input)}")
