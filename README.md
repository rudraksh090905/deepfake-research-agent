# Deepfake Research Agent

An automated research assistant that finds, stores, and synthesizes academic research on deepfake detection (video and image) using a Retrieval-Augmented Generation (RAG) pipeline. It produces grounded, cited research reports on demand — built entirely on free tools, with no paid API dependency.

## What This Does

Instead of asking an AI model to answer questions about deepfakes from memory (which can be outdated or hallucinated), this system:

1. Pulls real, current research papers from arXiv
2. Stores them in a searchable vector database
3. Retrieves the most relevant papers for any given topic
4. Uses a free LLM to write a structured, cited report based only on that retrieved evidence

Every claim in the output can be traced back to a real paper in the knowledge base.

## Architecture

```
arXiv (research papers)
        |
        v
  Chunking & Embedding  (sentence-transformers, local, free)
        |
        v
   ChromaDB (vector store, persisted locally)
        |
        v
  MMR Retrieval  (relevant + diverse results)
        |
        v
  Groq LLM (Llama 3.1)  -->  Cited Markdown Report
```

## Project Structure

| File | Purpose |
|---|---|
| `build_knowledge_base.py` | Fetches deepfake research papers from arXiv, chunks them, embeds them, and stores them in ChromaDB |
| `embeddings.py` | Shared embedding model loader (cached, free, runs locally) |
| `agents.py` | Core retrieval and report-generation logic, calls Groq directly |
| `main.py` | Command-line entry point to generate a report on any topic |
| `requirements.txt` | Minimal dependency list, tested in a clean environment |

## How It Works

### 1. Building the Knowledge Base

`build_knowledge_base.py` searches arXiv using five different query variations related to deepfakes ("deepfake detection," "media forensics," "face manipulation detection," "GAN image forgery detection," "transformer deepfake detection") to get broader topic coverage than a single search would. Duplicate papers are filtered out by normalized title matching.

Each paper's title and abstract are split into overlapping ~350-character chunks, converted into vector embeddings using a free local model (`all-MiniLM-L6-v2`), and saved to a persistent ChromaDB store on disk.

### 2. Retrieval

When a research topic is requested, `agents.py` converts the query into the same vector space and retrieves the most relevant chunks using **MMR (Maximal Marginal Relevance)** rather than plain similarity search. MMR balances relevance with diversity, ensuring the report draws from multiple distinct papers rather than repeating one source.

### 3. Report Generation

The retrieved chunks are formatted into a token-budgeted context block (capped to stay within free-tier limits) and sent to Groq's Llama 3.1 model with an explicit instruction: use only the provided context, never invent citations or numbers, and state plainly when evidence is weak. The model returns a structured markdown report (Introduction, Key Papers, Methods Compared, Datasets/Benchmarks, Strengths and Limitations, Conclusion).

Requests automatically retry with exponential backoff if a rate limit is hit, which matters on a free API tier.

## Setup

### 1. Clone and create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add API keys

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-key-here
```

A free Groq API key can be obtained at [console.groq.com](https://console.groq.com).

### 4. Build the knowledge base (run once)

```bash
python build_knowledge_base.py
```

This fetches papers and builds the local vector database. Takes 1-3 minutes depending on connection speed.

## Usage

Generate a report on the default topic:

```bash
python main.py
```

Generate a report on a custom topic:

```bash
python main.py --topic "GAN-based deepfake image generation techniques"
```

Control how many source chunks are used and where the report is saved:

```bash
python main.py --topic "audio deepfake detection methods" --top-k 5 --output audio_report.md
```

### Available arguments

| Argument | Default | Description |
|---|---|---|
| `--topic` | `"deepfake detection methods using transformers and GANs"` | Research topic to investigate |
| `--top-k` | `3` | Number of source chunks retrieved as context |
| `--output` | `research_report.md` | Output file path |
| `--max-tokens` | `700` | Maximum length of the generated report |

## Design Decisions

**Why Groq instead of OpenAI?** The project needed to run entirely free. Groq provides fast, free-tier access to strong open models (Llama 3.1), avoiding any billing dependency.

**Why a direct pipeline instead of a multi-agent framework?** An earlier version used CrewAI to orchestrate multiple specialized agents. A library incompatibility between CrewAI's internal request handling and Groq's API (an unsupported prompt-caching parameter) caused persistent failures with no clean fix on the free tier. The system was rebuilt as a single, direct pipeline calling Groq's API natively — fewer moving parts, more reliable, same RAG principles.

**Why MMR over plain similarity search?** Plain similarity search can return multiple near-duplicate chunks from the same paper. MMR retrieves a wider candidate pool first, then selects for both relevance and diversity, producing reports grounded in multiple distinct sources.

**Why local embeddings instead of OpenAI embeddings?** Keeps the entire pipeline free to run, with no per-query cost for the retrieval step.

## Known Limitations

- Smaller free-tier LLMs (Llama 3.1 8B) occasionally blend general background knowledge with retrieved context rather than sticking strictly to it. Reports should be reviewed, not treated as infallible.
- Knowledge base coverage is limited to ~30 papers from five arXiv search queries; broader coverage would require more queries or additional sources (e.g. IEEE, ACM).
- No web search / real-time news integration in the current version — the system only knows what's in the local knowledge base.

## Future Improvements

- Expand the knowledge base with more source queries and additional academic databases
- Add a fact-checking step against current web sources
- Reintroduce multi-agent orchestration once a stable CrewAI–Groq integration is available, or switch to a paid LLM tier for that path
- Add a simple web UI for non-technical users to query the system
