import os
import sys
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


class MyRAGLab:
    def __init__(self, data_folder="data", db_folder="vector_db"):
        """Initialize the RAG system with English comments."""
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found in .env file.")
            sys.exit(1)

        self.data_folder = data_folder
        self.db_folder = db_folder
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = None

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def ingest_pdf(self, file_name):
        """Load and vectorise the PDF document."""
        file_path = os.path.join(self.data_folder, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)

        self.vector_db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_folder,
        )
        print(f"Indexed {len(splits)} chunks from {file_name}")
        return True

    def query(self, question):
        """Handle retrieval and response generation."""
        if not self.vector_db:
            return "Please ingest a document first."

        # Define the system prompt
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create the chains
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(
            self.vector_db.as_retriever(), question_answer_chain
        )

        response = rag_chain.invoke({"input": question})
        return response["answer"]


if __name__ == "__main__":
    lab = MyRAGLab()

    # Check for PDFs in the data folder
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    if pdf_files:
        lab.ingest_pdf(pdf_files[0])
    else:
        print("Note: No PDF found in /data folder. Please add one.")

    print("\n--- My RAG Lab Ready ---")
    while True:
        user_input = input("\n[Question]: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print(f"\n[AI]: {lab.query(user_input)}")
