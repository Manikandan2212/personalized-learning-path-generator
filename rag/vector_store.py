"""
RAG Vector Store
Pure Python TF-IDF based semantic search engine.
No external ML dependencies required.
"""

import math
import re
import json
import os
import hashlib
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("rag.vector_store")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","being","have","has","had","do","does",
        "did","will","would","could","should","may","might","shall","can","need",
        "it","its","this","that","these","those","i","you","he","she","we","they",
        "me","him","her","us","them","my","your","his","our","their",
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]


class Document:
    def __init__(self, doc_id: str, content: str, metadata: Dict):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.tokens = tokenize(content)
        self.tf: Dict[str, float] = {}
        self._compute_tf()

    def _compute_tf(self):
        if not self.tokens:
            return
        freq: Dict[str, int] = {}
        for t in self.tokens:
            freq[t] = freq.get(t, 0) + 1
        total = len(self.tokens)
        self.tf = {t: c / total for t, c in freq.items()}


class VectorStore:
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.df: Dict[str, int] = {}   # document frequency per term
        self.N = 0

    def add_document(self, doc_id: str, content: str, metadata: Dict) -> str:
        doc = Document(doc_id, content, metadata)
        self.documents[doc_id] = doc
        self.N += 1
        for term in set(doc.tokens):
            self.df[term] = self.df.get(term, 0) + 1
        logger.info(f"Added document: {doc_id} ({len(doc.tokens)} tokens)")
        return doc_id

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.N + 1) / (df + 1)) + 1

    def _tfidf_vector(self, doc: Document) -> Dict[str, float]:
        return {term: tf * self._idf(term) for term, tf in doc.tf.items()}

    def _query_vector(self, query_tokens: List[str]) -> Dict[str, float]:
        freq: Dict[str, int] = {}
        for t in query_tokens:
            freq[t] = freq.get(t, 0) + 1
        total = max(len(query_tokens), 1)
        return {t: (c / total) * self._idf(t) for t, c in freq.items()}

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        terms = set(vec_a) & set(vec_b)
        if not terms:
            return 0.0
        dot = sum(vec_a[t] * vec_b[t] for t in terms)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        q_vec = self._query_vector(query_tokens)
        scores: List[Tuple[float, Document]] = []
        for doc in self.documents.values():
            d_vec = self._tfidf_vector(doc)
            score = self._cosine_similarity(q_vec, d_vec)
            if score >= min_score:
                scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scores[:top_k]:
            results.append({
                "doc_id": doc.doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": round(score, 4),
            })
        top_score = scores[0][0] if scores else 0
        logger.info(f"Search '{query[:40]}' → {len(results)} results (top score: {top_score:.3f})")      
        return results

    def get_document(self, doc_id: str) -> Optional[Dict]:
        doc = self.documents.get(doc_id)
        if not doc:
            return None
        return {"doc_id": doc.doc_id, "content": doc.content, "metadata": doc.metadata}

    def stats(self) -> Dict:
        return {
            "total_documents": self.N,
            "vocabulary_size": len(self.df),
            "doc_ids": list(self.documents.keys())[:10],
        }


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
