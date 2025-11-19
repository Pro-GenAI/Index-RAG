# Copyright (c) Praneeth Vadlapati

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from index_rag.utils.retrieval_utils import retriever


# Initialize client
model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise Exception("OPENAI_MODEL is not set in the environment variables")
print(f"Model: {model}")
model_name_short = model.split("/")[-1].lower()

llm = ChatOpenAI(model=model, temperature=0.7)


# RAG prompt for investment tutoring
rag_prompt_template = """
You are an AI tutor specializing in investment education. Use the following pieces of context to answer the student's question.
If you don't know the answer based on the provided context, say "I don't have enough information from my knowledge base to answer this question accurately."

Context:
{context}

Question: {question}

Instructions:
- Provide clear, educational explanations
- Include relevant examples when possible
- ALWAYS cite sources for any factual claims, statistics, or specific investment advice
- If a claim cannot be supported by the provided context, do not make it
- Use phrases like "According to [source]" or "Based on [source]" when citing
- If you cannot find supporting evidence in the context, say "I don't have source-backed information for that"
- Encourage critical thinking about investments

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=rag_prompt_template, input_variables=["context", "question"]
)


def format_docs(docs):
    print("Retrieved", len(docs), "documents")
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    # Example usage
    question = "What are the key principles of value investing?"
    answer = rag_chain.invoke(question)
    print("Answer:", answer)
