import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore
import openai
from pinecone import Pinecone

load_dotenv()

embed_model_name = os.getenv("EMBED_MODEL_NAME", "")
if not embed_model_name:
    raise Exception("EMBED_MODEL_NAME is not set in the environment variables")

index_name = os.getenv("PINECONE_INDEX_NAME", "")
if not index_name:
    raise Exception("PINECONE_INDEX_NAME is not set in the environment variables")

embed_client = openai.OpenAI(base_url = os.getenv("EMBEDDING_BASE_URL"))


class CustomEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        response = embed_client.embeddings.create(input=[text], model=embed_model_name)
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = embed_client.embeddings.create(input=texts, model=embed_model_name)
        return [data.embedding for data in response.data]


embeddings = CustomEmbeddings()
vdb_client = Pinecone()


# Create index if it doesn't exist
if index_name not in [idx.name for idx in vdb_client.list_indexes()]:
    # Generate a sample embedding and find size
    print("Generating sample embedding to determine dimension...")
    sample_embedding = embeddings.embed_query("Sample text for dimension check")
    dimension = len(sample_embedding)

    print("Creating Pinecone index:", index_name, "with dimension:", dimension)
    vdb_client.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        vector_type="dense",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    )

vectorstore = PineconeVectorStore(vdb_client.Index(index_name), embeddings)


# Custom retriever with reranking

class Retriever(BaseRetriever):
    k: int = 5

    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        # Get initial results
        results = search_documents(query, k=self.k)

        # Filter for paragraphs
        results = [r for r in results if r["metadata"].get("type") == "paragraph"]

        # Convert back to Document objects
        docs = []
        for result in results:
            doc = Document(page_content=result["content"], metadata=result["metadata"])
            docs.append(doc)

        return docs


def search_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search documents using semantic similarity."""
    if vectorstore is None:
        return []

    docs = vectorstore.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in docs:
        content = doc.metadata.get("paragraph_text", doc.page_content)
        results.append({"content": content, "metadata": doc.metadata, "score": score})
    return results


retriever = Retriever()
