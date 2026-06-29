from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

load_dotenv()

from embeddings import get_embeddings
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
PERSIST_DIR = Path(os.getenv("CHROMA_DIR", "./deepfake_db"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))
MAX_CHARS_PER_DOC = int(os.getenv("MAX_CHARS_PER_DOC", "1200"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3200"))

_vectorstore = None
_groq_client = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Put it in your .env file before running."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def retrieve_papers(query: str, k: int | None = None) -> List[Document]:
    """Fetch the most relevant chunks from the local knowledge base."""
    top_k = k or DEFAULT_TOP_K
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": 10,
        },
    )
    return retriever.invoke(query)


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_docs_for_prompt(docs: Iterable[Document]) -> str:
    """Convert retrieved docs into a compact prompt-friendly context block."""
    blocks: list[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Unknown title")
        year = doc.metadata.get("year", "Unknown year")
        source = doc.metadata.get("source", "unknown")
        content = _clip(doc.page_content, MAX_CHARS_PER_DOC)

        block = (
            f"[{i}] Title: {title}\n"
            f"Year: {year}\n"
            f"Source: {source}\n"
            f"Excerpt: {content}"
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS and blocks:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(blocks)


def _sleep_backoff(attempt: int) -> None:
    # Small, free-tier-friendly retry delay.
    time.sleep(min(8, 2 ** attempt))


def generate_report(topic: str, docs: List[Document], max_tokens: int = 700) -> str:
    """
    Write a short research report using only the retrieved local context.
    This keeps token usage low enough for free-tier Groq limits.
    """
    if not docs:
        return (
            f"# Research Report: {topic}\n\n"
            "No relevant papers were found in the local knowledge base. "
            "Run `build_knowledge_base.py` first."
        )

    context = format_docs_for_prompt(docs)

    system_prompt = (
        "You are a careful research assistant. Use only the provided context. "
        "Do not invent paper titles, datasets, or benchmark numbers. "
        "If evidence is weak, say so plainly."
    )

    user_prompt = f"""
Topic: {topic}

Use the context below to write a concise markdown report with these sections:
1. Introduction
2. Key Papers
3. Methods Compared
4. Datasets / Benchmarks
5. Strengths and Limitations
6. Conclusion

Rules:
- Keep the report focused and practical.
- Mention paper titles inline when making claims.
- Do not fabricate citations or performance numbers.
- Prefer short paragraphs and bullet points.

Context:
{context}
""".strip()

    client = get_groq_client()

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "rate limit" in message or "rate_limit" in message or "429" in message:
                _sleep_backoff(attempt)
                continue
            raise

    raise RuntimeError(f"Groq request failed after retries: {last_error}")


def save_report(report: str, output_path: str | Path = "research_report.md") -> Path:
    path = Path(output_path)
    path.write_text(report, encoding="utf-8")
    return path
