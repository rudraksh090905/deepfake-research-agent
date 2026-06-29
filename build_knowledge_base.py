from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from embeddings import get_embeddings
import arxiv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", "./deepfake_db"))

SEARCH_QUERIES = [
    "deepfake detection",
    "media forensics",
    "face manipulation detection",
    "GAN image forgery detection",
    "transformer deepfake detection",
]

MAX_RESULTS_PER_QUERY = int(os.getenv("ARXIV_RESULTS_PER_QUERY", "8"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))


def normalize_title(title: str) -> str:
    title = title.lower().strip()
    return re.sub(r"\W+", " ", title).strip()


def fetch_arxiv_docs() -> list[Document]:
    client = arxiv.Client(
        page_size=MAX_RESULTS_PER_QUERY,
        delay_seconds=3,
        num_retries=3,
    )

    all_docs: list[Document] = []
    seen_titles: set[str] = set()

    for query in SEARCH_QUERIES:
        print(f"Searching arXiv for: {query}")
        search = arxiv.Search(query=query, max_results=MAX_RESULTS_PER_QUERY)

        try:
            results = client.results(search)
        except Exception as exc:
            print(f"  Warning: search failed for '{query}': {exc}")
            continue

        for paper in results:
            key = normalize_title(paper.title)
            if key in seen_titles:
                continue
            seen_titles.add(key)

            abstract = (paper.summary or "").strip()
            text = f"Title: {paper.title}\n\nAbstract: {abstract}"

            all_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "title": paper.title,
                        "year": paper.published.year if paper.published else None,
                        "source": "arxiv",
                        "arxiv_id": paper.entry_id,
                        "query": query,
                    },
                )
            )

    return all_docs


def main() -> None:
    print("Loading free embedding model (first run downloads the model)...")
    embeddings = get_embeddings()
    print("Embedding model loaded.")

    print("\nFetching arXiv papers...")
    docs = fetch_arxiv_docs()
    print(f"Fetched {len(docs)} unique papers.")

    if not docs:
        raise RuntimeError("No arXiv papers were fetched. Check your internet connection.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    sample_query = "transformer deepfake detection"
    print(f"\nTesting retrieval with: {sample_query}")
    results = vectorstore.similarity_search(sample_query, k=3)
    for idx, doc in enumerate(results, start=1):
        print(f"Result {idx}: {doc.metadata.get('title', 'Unknown')}")
        print(doc.page_content[:180].replace("\n", " ") + "...\n")

    print("Knowledge base is ready.")


if __name__ == "__main__":
    main()
