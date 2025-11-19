from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader

from index_rag.utils.llm_utils import llm
from index_rag.utils.retrieval_utils import vectorstore


# Use LangChain's Pinecone wrapper


def load_paragraphs(file_path: str) -> List[Dict[str, Any]]:
    """Load paragraphs from PDF file with metadata."""
    reader = PdfReader(file_path)
    paragraphs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        pars = [p.strip() for p in text.split("\n\n") if p.strip()]
        line_num = 1
        for par in pars:
            line_end = line_num + par.count("\n")
            paragraphs.append(
                {
                    "text": par,
                    "page": page_num + 1,
                    "line_start": line_num,
                    "line_end": line_end,
                    "file": file_path,
                }
            )
            line_num = line_end + 1
    return paragraphs


def generate_questions(
    paragraph: str, num_questions: int = 5, attempt: int = 1
) -> List[str]:
    """Generate questions that can be answered by the paragraph."""
    prompt = (
        f"Generate {num_questions} diverse questions that can be "
        f"answered by this paragraph:\n\n{paragraph}\n\n"
        "Provide each question on a new line."
    )
    response = llm.invoke(input=prompt)
    content = response.text.strip()
    questions = [q.strip() for q in content.split("\n") if q.strip() and "?" in q]
    if not questions and attempt < 3:
        # Retry up to 3 attempts
        return generate_questions(paragraph, num_questions, attempt + 1)
    return questions[:num_questions]


def ingest_document(file_path: str) -> None:
    """Ingest a single document (PDF)."""
    if not file_path.endswith(".pdf"):
        raise ValueError("Unsupported file type. Only PDF files are supported.")

    paragraphs = load_paragraphs(file_path)

    for para in paragraphs:
        texts = generate_questions(para["text"]) + [para["text"]]
        metadatas = [
            {
                "file": para["file"],
                "page": para["page"],
                "line_start": para["line_start"],
                "line_end": para["line_end"],
                "type": "question" if text != para["text"] else "paragraph",
            }
            for text in texts
        ]
        vectorstore.add_texts(texts, metadatas)
    print(f"Ingested {len(paragraphs)} paragraphs from {file_path}")


def ingest_directory(directory_path: str) -> None:
    """Ingest all PDFs in a directory."""
    for pdf_file in Path(directory_path).glob("*.pdf"):
        ingest_document(str(pdf_file))


if __name__ == "__main__":
    # Example usage
    # ingest_document("path/to/document.pdf")
    # ingest_directory("path/to/pdf/folder")
    pass
