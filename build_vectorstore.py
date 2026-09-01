"""Build a Chroma vector store from the small NBC fire-rule knowledge base."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    NBC_RULES_PATH,
    ProjectConfigurationError,
    ensure_gemini_api_key,
    get_chat_model,
    get_embedding_model,
)


VECTORSTORE_METADATA_PATH = CHROMA_DIR / "metadata.json"


class VectorStoreBuildError(RuntimeError):
    """Raised when the Chroma vector store cannot be built."""


def get_langchain_dependencies() -> tuple[type[Any], type[Any], type[Any]]:
    """Import vector-store dependencies lazily for clear setup errors."""
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ModuleNotFoundError as error:
        raise VectorStoreBuildError(
            "Missing LangChain, Chroma, or Gemini dependency. "
            "Run: pip install -r requirements.txt"
        ) from error

    return Document, GoogleGenerativeAIEmbeddings, Chroma


def load_nbc_rules(path: Path = NBC_RULES_PATH) -> str:
    """Load the small NBC knowledge base."""
    if not path.exists():
        raise FileNotFoundError(f"NBC rules file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise VectorStoreBuildError(f"NBC rules file is empty: {path}")

    return text


def split_rule_chunks(text: str) -> list[str]:
    """Split nbc_rules.txt into retrievable rule chunks."""
    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*---\s*\n", text)
        if chunk.strip()
    ]
    if chunks:
        return chunks

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text)]
    return [paragraph for paragraph in paragraphs if paragraph]


def _extract_line(chunk: str, label: str, default: str = "") -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", chunk, re.MULTILINE)
    return match.group(1).strip() if match else default


def rule_metadata(chunk: str, index: int, source_name: str) -> dict[str, str | int]:
    """Create Chroma-compatible metadata for one NBC rule chunk."""
    return {
        "source": source_name,
        "chunk_index": index,
        "rule_id": _extract_line(chunk, "Rule ID", default=f"RULE-{index:03d}"),
        "category": _extract_line(chunk, "Category", default="GENERAL"),
        "reference": _extract_line(chunk, "Reference", default="Not specified"),
    }


def documents_from_rules(
    text: str,
    source_name: str,
    document_cls: type[Any] | None = None,
) -> list[Any]:
    """Create one LangChain Document per small NBC rule chunk."""
    if document_cls is None:
        document_cls = get_langchain_dependencies()[0]

    chunks = split_rule_chunks(text)
    if not chunks:
        raise VectorStoreBuildError("No rule chunks were found in nbc_rules.txt.")

    return [
        document_cls(
            page_content=chunk,
            metadata=rule_metadata(chunk, index, source_name),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def save_vectorstore_metadata(
    rule_count: int,
    vectorstore_dir: Path = CHROMA_DIR,
) -> None:
    """Save lightweight metadata for Streamlit status display."""
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "rule_count": rule_count,
        "collection_name": CHROMA_COLLECTION_NAME,
        "embedding_model": get_embedding_model(),
        "chat_model": get_chat_model(),
    }
    (vectorstore_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_vectorstore_metadata(vectorstore_dir: Path = CHROMA_DIR) -> dict[str, Any]:
    """Load saved Chroma build metadata when present."""
    metadata_path = vectorstore_dir / "metadata.json"
    if not metadata_path.exists():
        return {
            "rule_count": 0,
            "collection_name": CHROMA_COLLECTION_NAME,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "chat_model": DEFAULT_CHAT_MODEL,
        }

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "rule_count": 0,
            "collection_name": CHROMA_COLLECTION_NAME,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "chat_model": DEFAULT_CHAT_MODEL,
        }


def vectorstore_ready(vectorstore_dir: Path = CHROMA_DIR) -> bool:
    """Return whether a local Chroma store appears to have been built."""
    metadata_path = vectorstore_dir / "metadata.json"
    return vectorstore_dir.exists() and metadata_path.exists()


def build_vectorstore(
    rules_path: Path = NBC_RULES_PATH,
    vectorstore_dir: Path = CHROMA_DIR,
) -> Any:
    """Embed nbc_rules.txt into a persistent Chroma vector store."""
    ensure_gemini_api_key()
    Document, GoogleGenerativeAIEmbeddings, Chroma = get_langchain_dependencies()

    text = load_nbc_rules(rules_path)
    documents = documents_from_rules(
        text,
        source_name=rules_path.name,
        document_cls=Document,
    )

    if vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)

    embeddings = GoogleGenerativeAIEmbeddings(model=get_embedding_model())
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(vectorstore_dir),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    save_vectorstore_metadata(len(documents), vectorstore_dir=vectorstore_dir)

    print(f"Loaded {len(documents)} NBC rule chunks.")
    print(f"Saved Chroma vector store to {vectorstore_dir}")
    return vectorstore


def main() -> None:
    """Build the local Chroma vector store."""
    try:
        build_vectorstore()
    except (ProjectConfigurationError, VectorStoreBuildError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
