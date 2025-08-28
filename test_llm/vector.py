import os
import uuid
import httpx
import tiktoken
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI  # if using Azure OpenAI embeddings

# --------------------------
# 1. Create the Search Index
# --------------------------
def create_search_index(search_index_name: str):
    load_dotenv()

    endpoint = os.getenv("SEARCH_ENDPOINT")
    admin_key = os.getenv("SEARCH_KEY")

    client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(admin_key))

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="KBA_ID", type=SearchFieldDataType.String),
        SimpleField(name="KBA_Title", type=SearchFieldDataType.String),
        SimpleField(name="KBA_Owner_Group", type=SearchFieldDataType.String),
        SimpleField(name="KBA_Author", type=SearchFieldDataType.String),
        SimpleField(name="KBA_Description", type=SearchFieldDataType.String),
        SimpleField(name="KBA_Template", type=SearchFieldDataType.String),
        SimpleField(name="Create_Date", type=SearchFieldDataType.String),
        SearchField(
            name="KBA_Vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="vector-profile-kba-itsd"
        ),
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vector-profile-kba-itsd",
                algorithm_configuration_name="my-hnsw-config"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-config",
                kind="hnsw",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ]
    )

    index = SearchIndex(name=search_index_name, fields=fields, vector_search=vector_search)
    client.create_index(index)
    print(f"Index '{search_index_name}' created successfully.")


# --------------------------
# 2. Embedding Function
# --------------------------
def embedding(text: str):
    """Generate embedding using Azure OpenAI (or replace with Ollama embeddings)."""
    print("Inside embedding function")
    load_dotenv()

    client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        http_client=httpx.Client(verify=False)
    )
    try:
        response = client.embeddings.create(input=text, model="doc_embedding")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


# --------------------------
# 3. Helper Functions
# --------------------------
def get_number_of_tokens(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def split_large_text(large_text: str, max_tokens: int):
    enc = tiktoken.get_encoding("cl100k_base")
    tokenized_text = enc.encode(large_text)

    chunks = []
    current_chunk = []
    for token in tokenized_text:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(enc.decode(current_chunk).rstrip('.,;'))
            current_chunk = []
    if current_chunk:
        chunks.append(enc.decode(current_chunk).rstrip('.,;'))
    return chunks


# --------------------------
# 4. Insert Documents
# --------------------------
def insert_documents_to_search_index(search_index_name: str):
    load_dotenv()

    endpoint = os.getenv("SEARCH_ENDPOINT")
    admin_key = os.getenv("SEARCH_KEY")

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=search_index_name,
        credential=AzureKeyCredential(admin_key)
    )

    # Example dummy documents
    documents = [
        {
            "id": str(uuid.uuid4()),
            "KBA_ID": "KBA001",
            "KBA_Title": "How to reset your password",
            "KBA_Owner_Group": "IT Support",
            "KBA_Author": "kundual",
            "KBA_Description": "Steps to reset your corporate password using the self-service portal.",
            "KBA_Template": "Standard",
            "Create_Date": "2025-08-01"
        },
        {
            "id": str(uuid.uuid4()),
            "KBA_ID": "KBA002",
            "KBA_Title": "VPN Troubleshooting Guide",
            "KBA_Owner_Group": "Network Team",
            "KBA_Author": "pillaid",
            "KBA_Description": "Common VPN issues and how to resolve them.",
            "KBA_Template": "Troubleshooting",
            "Create_Date": "2025-08-02"
        },
    ]

    # Process each document
    for doc in documents:
        combined_text = f"""
        ***KBA Title: {doc['KBA_Title']}***
        ***KBA Description: {doc['KBA_Description']}***
        """

        if get_number_of_tokens(combined_text) > 8000:
            combined_text = split_large_text(combined_text, 8000)[0]

        doc["KBA_Vector"] = embedding(combined_text)

        result = search_client.upload_documents([doc])
        print("Upload result:", result)


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    create_search_index("kba-itsd-index")
    insert_documents_to_search_index("kba-itsd-index")
