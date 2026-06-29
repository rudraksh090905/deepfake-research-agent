from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

 #Keep all Hugging Face caches inside the project folder so Windows does not
 #fall back to a protected system cache path.
os.environ["HF_HOME"] = "./models/dir"
os.environ["TRANSFORMERS_CACHE"] = "./models/dir"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models/dir"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a single shared embedding model instance.

    This avoids downloading/loading the model more than once and keeps the
    project stable on constrained machines.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder="./models",
    )
