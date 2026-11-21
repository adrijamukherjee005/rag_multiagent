"""
Ingests PDF / PPTX / DOCX files from a directory into ChromaDB.

Usage:
    python -m ingestion.document_ingest --path data/documents
    python -m ingestion.document_ingest --path data/documents --file report.pdf
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings
from vectorstore.chroma_store import get_document_store

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx"}


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": i + 1, "type": "pdf"},
                )
            )
    return docs


def load_pptx(path: Path) -> list[Document]:
    from pptx import Presentation

    prs = Presentation(str(path))
    docs = []
    for i, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    texts.append(" | ".join(cell.text for cell in row.cells))
        slide_text = "\n".join(t for t in texts if t.strip())
        if slide_text.strip():
            docs.append(
                Document(
                    page_content=slide_text,
                    metadata={"source": path.name, "slide": i + 1, "type": "pptx"},
                )
            )
    return docs


def load_docx(path: Path) -> list[Document]:
    import docx

    d = docx.Document(str(path))
    paragraphs = [p.text for p in d.paragraphs if p.text.strip()]
    tables_text = []
    for table in d.tables:
        for row in table.rows:
            tables_text.append(" | ".join(cell.text for cell in row.cells))

    full_text = "\n".join(paragraphs + tables_text)
    if not full_text.strip():
        return []
    return [Document(page_content=full_text, metadata={"source": path.name, "type": "docx"})]


LOADERS = {".pdf": load_pdf, ".pptx": load_pptx, ".docx": load_docx}


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADERS[ext](path)


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_path(path: str, single_file: str | None = None) -> int:
    root = Path(path)
    if single_file:
        files = [root / single_file]
    elif root.is_file():
        files = [root]
    else:
        files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning("No supported documents found under %s", path)
        return 0

    store = get_document_store()
    total_chunks = 0
    for f in files:
        try:
            raw_docs = load_file(f)
            if not raw_docs:
                logger.warning("No extractable text in %s, skipping.", f.name)
                continue
            chunks = chunk_documents(raw_docs)
            for j, c in enumerate(chunks):
                c.metadata["chunk_id"] = f"{f.stem}-{j}"
            store.add_documents(chunks)
            total_chunks += len(chunks)
            logger.info("Ingested %s -> %d chunks", f.name, len(chunks))
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to ingest %s: %s", f, e)

    logger.info("Ingestion complete. Total chunks added: %d", total_chunks)
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--path", default=settings.documents_dir, help="Directory or file to ingest")
    parser.add_argument("--file", default=None, help="Optional single filename within --path directory")
    args = parser.parse_args()
    ingest_path(args.path, args.file)


if __name__ == "__main__":
    main()
