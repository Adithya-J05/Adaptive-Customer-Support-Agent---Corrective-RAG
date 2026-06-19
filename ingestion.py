import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config import get_embeddings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
PERSIST_DIR = "./chroma_db"

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


def _load_documents(source: str) -> list:
    if os.path.isdir(source):
        docs = []
        for fname in sorted(os.listdir(source)):
            fpath = os.path.join(source, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in LOADER_MAP:
                loader_cls = LOADER_MAP[ext]
                docs.extend(loader_cls(fpath).load())
                print(f"  Loaded {fname}")
        print(f"  Total: {len(docs)} page(s) from '{source}'")
        return docs

    ext = os.path.splitext(source)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(LOADER_MAP)}")
    loader = loader_cls(source)
    docs = loader.load()
    print(f"  Loaded {len(docs)} page(s) from '{source}'")
    return docs


def _split(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunk(s)")
    return chunks


def build_retriever(source: str, force_rebuild: bool = False) -> VectorStoreRetriever:
    embeddings = get_embeddings()

    if not force_rebuild:
        try:
            store = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
            )
            if store._collection.count() > 0:
                print(f"  Loaded existing vector store ({store._collection.count()} chunks).")
                return store.as_retriever(search_kwargs={"k": TOP_K})
        except Exception:
            pass

    print("  Building vector store...")
    docs = _load_documents(source)
    chunks = _split(docs)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"  Vector store built with {len(chunks)} chunks.")
    return store.as_retriever(search_kwargs={"k": TOP_K})
