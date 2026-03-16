"""
RAG pipeline: loads .txt docs, chunks them, embeds with sentence-transformers,
stores in Endee vector DB, and handles query + response generation.
"""

import logging
import os
import re
from dataclasses import dataclass, field

from backend.embeddings import get_embedding, get_embeddings, get_dimension, get_model_name
from backend.vector_store import create_vector_store

log = logging.getLogger(__name__)

# configurable via env vars
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
SIM_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.25"))
TOP_K = int(os.environ.get("TOP_K", "5"))


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int


@dataclass
class QueryResult:
    answer: str
    sources: list
    num_chunks_retrieved: int
    similarities: list = field(default_factory=list)


# --- document loading ---

def load_documents(data_dir):
    """Read all .txt files from a directory."""
    if not os.path.isdir(data_dir):
        log.warning("Data dir not found: %s", data_dir)
        return []

    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        docs.append({"filename": fname, "content": content})
        log.info("Loaded %s (%d chars)", fname, len(content))

    log.info("Total documents: %d", len(docs))
    return docs


# --- chunking (sentence-aware) ---

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text):
    paragraphs = re.split(r"\n\s*\n", text)
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) < 120 or not _SENT_SPLIT.search(para):
            sentences.append(para)
        else:
            for s in _SENT_SPLIT.split(para):
                s = s.strip()
                if s:
                    sentences.append(s)
    return sentences


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks at sentence boundaries."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        slen = len(sent)

        # single sentence bigger than chunk size -> own chunk
        if slen > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.append(sent)
            continue

        if current_len + slen + 1 > chunk_size and current:
            chunks.append(" ".join(current))

            # carry over some sentences for overlap
            overlap_sents = []
            olen = 0
            for s in reversed(current):
                if olen + len(s) + 1 > overlap:
                    break
                overlap_sents.insert(0, s)
                olen += len(s) + 1
            current = overlap_sents
            current_len = olen

        current.append(sent)
        current_len += slen + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(documents):
    all_chunks = []
    for doc in documents:
        parts = chunk_text(doc["content"])
        for i, text in enumerate(parts):
            all_chunks.append(Chunk(
                id=f"{doc['filename']}__chunk_{i}",
                text=text,
                source=doc["filename"],
                chunk_index=i,
            ))
    log.info("Created %d chunks", len(all_chunks))
    return all_chunks


# --- deduplication ---

def _deduplicate(texts, threshold=0.85):
    """Remove near-duplicate chunks using Jaccard similarity on word sets."""
    if len(texts) <= 1:
        return texts

    unique = [texts[0]]
    unique_sets = [set(texts[0].lower().split())]

    for t in texts[1:]:
        ws = set(t.lower().split())
        dup = False
        for us in unique_sets:
            if us and ws:
                jaccard = len(us & ws) / len(us | ws)
                if jaccard >= threshold:
                    dup = True
                    break
        if not dup:
            unique.append(t)
            unique_sets.append(ws)

    return unique


# --- main pipeline class ---

class RAGPipeline:
    def __init__(self, data_dir, endee_url="http://localhost:8080/api/v1"):
        self.data_dir = data_dir
        self.endee_url = endee_url
        self.vector_store = None
        self.chunks = []
        self._ready = False

    @property
    def is_ready(self):
        return self._ready

    def stats(self):
        """Pipeline metadata for the /api/stats endpoint."""
        return {
            "initialized": self._ready,
            "num_chunks": len(self.chunks),
            "embedding_model": get_model_name(),
            "embedding_dim": get_dimension() if self._ready else None,
            "store_type": self.vector_store.store_type if self.vector_store else None,
            "vector_count": self.vector_store.count() if self.vector_store else 0,
            "similarity_threshold": SIM_THRESHOLD,
        }

    def initialize(self):
        """Load docs -> chunk -> embed -> store. Called once at startup."""
        log.info("Initializing RAG pipeline...")

        docs = load_documents(self.data_dir)
        if not docs:
            raise FileNotFoundError(f"No .txt files in {self.data_dir}")

        self.chunks = chunk_documents(docs)

        # generate embeddings
        texts = [c.text for c in self.chunks]
        embeddings = get_embeddings(texts)
        dim = get_dimension()
        log.info("Generated %d embeddings (dim=%d)", len(embeddings), dim)

        # store vectors
        self.vector_store = create_vector_store(
            endee_url=self.endee_url, dimension=dim
        )
        items = [
            {"id": c.id, "vector": e,
             "meta": {"text": c.text, "source": c.source,
                      "chunk_index": c.chunk_index}}
            for c, e in zip(self.chunks, embeddings)
        ]
        self.vector_store.upsert(items)
        log.info("Stored %d vectors (%s)", len(items), self.vector_store.store_type)

        self._ready = True
        log.info("Pipeline ready!")

    def query(self, question, top_k=TOP_K):
        """Embed question -> search -> filter -> deduplicate -> generate response."""
        if not self._ready or not self.vector_store:
            return QueryResult(
                answer="System is still starting up, try again in a moment.",
                sources=[], num_chunks_retrieved=0,
            )

        q_vec = get_embedding(question)
        raw = self.vector_store.query(vector=q_vec, top_k=top_k)

        # filter by similarity threshold
        results = [r for r in raw if r.get("similarity", 0) >= SIM_THRESHOLD]

        context = []
        sims = []
        seen_sources = set()
        sources = []

        for r in results:
            meta = r.get("meta", {})
            text = meta.get("text", "")
            src = meta.get("source", "unknown")
            if text:
                context.append(text)
                sims.append(r.get("similarity", 0))
                if src not in seen_sources:
                    seen_sources.add(src)
                    sources.append(src)

        context = _deduplicate(context)
        answer = self._build_response(question, context)

        return QueryResult(
            answer=answer, sources=sources,
            num_chunks_retrieved=len(context), similarities=sims,
        )

    @staticmethod
    def _build_response(question, chunks):
        """Build answer text from retrieved chunks (template-based, no LLM)."""
        if not chunks:
            return (
                "I couldn't find relevant information for your question. "
                "Try rephrasing or ask something else about Galgotias University."
            )

        combined = "\n\n".join(chunks)
        combined = re.sub(r"[=]{3,}", "", combined)
        combined = re.sub(r"[-]{3,}", "", combined)

        parts = ["Based on the available information, here's what I found:\n"]

        for para in combined.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            for line in para.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # detect headings (short lines that don't end with punctuation)
                if line.endswith(":") or (len(line) < 60 and not line.endswith((".", "!", "?"))):
                    parts.append(f"\n**{line}**\n")
                else:
                    parts.append(f"{line}\n")

        response = "\n".join(parts).strip()
        response = re.sub(r"\n{3,}", "\n\n", response)
        response += (
            "\n\n---\n*This information is based on Galgotias University's "
            "official records. For the latest details, contact the relevant department.*"
        )
        return response
