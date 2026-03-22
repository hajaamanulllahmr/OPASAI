import math
import re
import os
from pathlib import Path
from typing import List, Dict

PROFILE_PATH = Path(__file__).parent / "data" / "profile.txt"

# Chunk tuning — each chunk is a sliding window of words.
# Larger chunks give more context per result; more overlap reduces
# the chance of a relevant sentence being split across two chunks.
CHUNK_SIZE    = 80   # words per chunk
CHUNK_OVERLAP = 20   # words shared between consecutive chunks
TOP_K_RESULTS = 5    # how many chunks to include in the LLM context


# ── Text utilities ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, and split into words.
    Short words (≤ 2 chars) are removed — they're almost always stop words.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [word for word in cleaned.split() if len(word) > 2]


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split `text` into overlapping word-window chunks.
    Overlap ensures that a sentence near the boundary of one chunk is
    also fully captured in the next chunk.
    """
    words  = text.split()
    step   = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


# ── TF-IDF index ─────────────────────────────────────────────────────────────

class TFIDFIndex:
    """
    A simple in-memory TF-IDF index over a list of text chunks.

    Build it once with `build(chunks)`, then call `search(query)` as many
    times as you like — it's fast enough that there's no need to cache
    search results.
    """

    def __init__(self):
        self.chunks: List[str]         = []
        self.chunk_tf: List[Dict]      = []   # TF vectors, one per chunk
        self.idf: Dict[str, float]     = {}   # IDF score for every known word
        self._is_built                 = False

    def build(self, chunks: List[str]):
        """Index all chunks. Call this once before searching."""
        self.chunks = chunks
        total_docs  = len(chunks)

        # Count how many chunks each word appears in (document frequency)
        doc_freq: Dict[str, int] = {}
        token_lists = []

        for chunk in chunks:
            tokens = _tokenize(chunk)
            token_lists.append(tokens)
            for word in set(tokens):                   # set() → count each word once per chunk
                doc_freq[word] = doc_freq.get(word, 0) + 1

        # IDF: rare words that appear in few chunks get a higher score
        self.idf = {
            word: math.log((total_docs + 1) / (count + 1)) + 1
            for word, count in doc_freq.items()
        }

        # TF: how often each word appears within its chunk (normalised by chunk length)
        self.chunk_tf = []
        for tokens in token_lists:
            freq: Dict[str, int] = {}
            for word in tokens:
                freq[word] = freq.get(word, 0) + 1
            total_words = max(len(tokens), 1)
            self.chunk_tf.append({word: count / total_words for word, count in freq.items()})

        self._is_built = True

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Convert a list of tokens into a TF-IDF weighted vector."""
        freq: Dict[str, int] = {}
        for word in tokens:
            freq[word] = freq.get(word, 0) + 1
        total = max(len(tokens), 1)
        return {word: (count / total) * self.idf.get(word, 1.0) for word, count in freq.items()}

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """
        Cosine similarity between two sparse TF-IDF vectors.
        Returns a value between 0 (no overlap) and 1 (identical).
        """
        dot_product  = sum(vec_a.get(word, 0.0) * vec_b.get(word, 0.0) for word in vec_a)
        magnitude_a  = math.sqrt(sum(v * v for v in vec_a.values())) or 1.0
        magnitude_b  = math.sqrt(sum(v * v for v in vec_b.values())) or 1.0
        return dot_product / (magnitude_a * magnitude_b)

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        """
        Find the `top_k` chunks most relevant to `query`.
        Falls back to returning the first few chunks if the index hasn't
        been built yet (shouldn't happen in normal use).
        """
        if not self._is_built:
            return self.chunks[:top_k]

        query_tokens = _tokenize(query)
        query_vector = self._tfidf_vector(query_tokens)

        # Score every chunk
        scores = []
        for i, chunk_tf in enumerate(self.chunk_tf):
            chunk_vector = {word: chunk_tf[word] * self.idf.get(word, 1.0) for word in chunk_tf}
            similarity   = self._cosine_similarity(query_vector, chunk_vector)
            scores.append((similarity, i))

        scores.sort(reverse=True)
        return [self.chunks[idx] for _, idx in scores[:top_k]]


# ── RAG engine (singleton) ────────────────────────────────────────────────────

class RAGEngine:
    """
    Loads the owner profile, builds the TF-IDF index, and answers
    "which chunks are relevant to this question?" on demand.

    This is a singleton — `get_rag_engine()` always returns the same
    instance so the index is built only once during the app's lifetime.
    """

    def __init__(self):
        self._index = TFIDFIndex()
        self._ready = False

    def load(self):
        """
        Read the profile file, split it into chunks, and build the index.
        Safe to call multiple times — subsequent calls are instant no-ops.
        """
        if self._ready:
            return

        if not PROFILE_PATH.exists():
            raise FileNotFoundError(
                f"Profile not found at {PROFILE_PATH}\n"
                "Make sure data/profile.txt exists."
            )

        profile_text = PROFILE_PATH.read_text(encoding="utf-8")
        chunks       = _chunk_text(profile_text)
        self._index.build(chunks)
        self._ready  = True

        print(f"[RAG] Index ready — {len(chunks)} chunks loaded from profile.")

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        """Return the `top_k` most relevant profile chunks for `query`."""
        self.load()
        return self._index.search(query, top_k)

    def build_context(self, query: str) -> str:
        """
        Retrieve relevant chunks and join them into a single context string
        ready to be inserted into the LLM prompt.
        """
        relevant_chunks = self.retrieve(query)
        return "\n\n".join(relevant_chunks)

    @property
    def ready(self) -> bool:
        """True once the index has been successfully built."""
        return self._ready


# ── Module-level singleton ────────────────────────────────────────────────────

_rag_engine = RAGEngine()

def get_rag_engine() -> RAGEngine:
    """Return the shared RAG engine instance."""
    return _rag_engine
