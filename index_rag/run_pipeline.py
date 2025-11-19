from index_rag.utils.llm_utils import rag_chain
from index_rag.utils.ingestion import ingest_document, ingest_directory


if __name__ == "__main__":
	# Ingest documents
	print("Ingesting documents...")
	# ingest_directory("docs")
	print("Ingestion complete.")

	# Run RAG pipeline
	query = "What Is an Investment?"
	response = rag_chain.invoke(query)
	print("RAG Response:", response)
