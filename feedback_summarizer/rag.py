import pandas as pd
import json
import os
import tempfile
import ollama
import chromadb
from chromadb.utils import embedding_functions

class SurveyRAG:
    def __init__(self, model_name="llama3.2",collection_name="survey_collection"):
        self.model_name = model_name
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self._init_collection()

        # Use Ollama embedding model (local)
        # self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        #     api_key=None, model_name="nomic-embed-text"
        # )
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
        model_name="nomic-embed-text"  # or any Ollama-supported embedding model
            )

    def _init_collection(self):
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)

    def load_file(self, uploaded_file):
        """Reads uploaded CSV/Excel/JSON into a pandas DataFrame"""
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
        if file_ext == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_ext == ".json":
            df = pd.read_json(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Use CSV, Excel, or JSON.")
        return df

    def build_db(self, df):
        """Stores survey responses as documents in ChromaDB"""
        docs = df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist()
        ids = [f"doc_{i}" for i in range(len(docs))]
        embeddings = [self._embed_text(d) for d in docs]
        self.collection.add(ids=ids, embeddings=embeddings, documents=docs)

    def _embed_text(self, text):
        """Get embeddings using Ollama embedding model"""
        result = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return result["embedding"]

    def summarize(self):
        """Generate a summary using Ollama locally"""
        all_docs = [doc for doc in self.collection.get()["documents"]]
        combined_text = "\n".join(all_docs)
        prompt = f"Summarize the following survey responses:\n\n{combined_text}"
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    def query(self, question):
        """Retrieve relevant docs and answer with Ollama"""
        # Get all documents
        all_docs = self.collection.get()
        docs = all_docs["documents"]
        # Simple retrieval: top 5
        context = "\n".join(docs[:5])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
