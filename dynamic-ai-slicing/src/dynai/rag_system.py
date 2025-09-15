#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) System with Advanced Compression
Combines vector search, caching, and compression for minimal memory footprint.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import time
import hashlib
import pickle
import zlib
from collections import OrderedDict, deque
import mmap
import os


@dataclass
class CompressedEmbedding:
    """Ultra-compressed embedding using quantization + compression."""
    compressed_data: bytes
    original_shape: Tuple[int, ...]
    quantization_bits: int
    compression_ratio: float
    
    def decompress(self) -> np.ndarray:
        """Decompress back to embedding."""
        # Decompress bytes
        decompressed = zlib.decompress(self.compressed_data)
        
        # Convert back to array
        if self.quantization_bits == 1:
            # Binary embeddings
            arr = np.frombuffer(decompressed, dtype=np.uint8)
            arr = np.unpackbits(arr)[:np.prod(self.original_shape)]
            arr = arr.reshape(self.original_shape).astype(np.float32)
            arr = arr * 2 - 1  # Convert 0/1 to -1/1
        elif self.quantization_bits == 2:
            # 2-bit quantization
            arr = np.frombuffer(decompressed, dtype=np.uint8)
            arr = arr.astype(np.float32) / 85 - 1  # Map to [-1, 1]
            arr = arr.reshape(self.original_shape)
        else:
            # 4-bit or 8-bit
            arr = np.frombuffer(decompressed, dtype=np.uint8)
            arr = arr.astype(np.float32) / 127.5 - 1
            arr = arr.reshape(self.original_shape)
        
        return arr


class VectorDatabase:
    """
    Memory-efficient vector database with compression and memory mapping.
    """
    
    def __init__(self, dim: int, max_vectors: int = 1000000, use_mmap: bool = True):
        self.dim = dim
        self.max_vectors = max_vectors
        self.use_mmap = use_mmap
        self.index = {}  # Maps doc_id to position
        self.metadata = {}  # Stores document metadata
        self.num_vectors = 0
        
        # Create memory-mapped file for vectors if requested
        if use_mmap:
            self.mmap_file = f"vectors_{dim}d_{max_vectors}.dat"
            self._init_mmap()
        else:
            self.vectors = np.zeros((max_vectors, dim), dtype=np.float16)  # Use float16
        
        # Locality-Sensitive Hashing for fast search
        self.lsh_tables = self._init_lsh(num_tables=8, hash_size=16)
        
        # Cache for frequently accessed vectors
        self.cache = OrderedDict()  # LRU cache
        self.cache_size = 1000
    
    def _init_mmap(self):
        """Initialize memory-mapped storage."""
        # Create file if doesn't exist
        file_size = self.max_vectors * self.dim * 2  # float16 = 2 bytes
        
        if not os.path.exists(self.mmap_file):
            with open(self.mmap_file, 'wb') as f:
                f.write(b'\x00' * file_size)
        
        # Memory map the file
        self.mmap_vectors = np.memmap(
            self.mmap_file,
            dtype='float16',
            mode='r+',
            shape=(self.max_vectors, self.dim)
        )
    
    def _init_lsh(self, num_tables: int, hash_size: int) -> List[Dict]:
        """Initialize LSH tables for approximate nearest neighbor."""
        tables = []
        for _ in range(num_tables):
            # Random projection matrix for this table
            projection = np.random.randn(self.dim, hash_size)
            projection /= np.linalg.norm(projection, axis=0)
            
            tables.append({
                'projection': projection,
                'buckets': {}
            })
        return tables
    
    def add_vector(self, doc_id: str, vector: np.ndarray, metadata: Dict = None):
        """Add vector with automatic compression."""
        if self.num_vectors >= self.max_vectors:
            # Evict oldest vectors
            self._evict_oldest()
        
        # Normalize and compress to float16
        vector = vector.astype(np.float16)
        vector /= np.linalg.norm(vector) + 1e-8
        
        # Store in memory map or array
        position = self.num_vectors
        if self.use_mmap:
            self.mmap_vectors[position] = vector
        else:
            self.vectors[position] = vector
        
        # Update index
        self.index[doc_id] = position
        if metadata:
            self.metadata[doc_id] = metadata
        
        # Add to LSH tables
        for table in self.lsh_tables:
            hash_val = self._compute_lsh_hash(vector, table['projection'])
            if hash_val not in table['buckets']:
                table['buckets'][hash_val] = []
            table['buckets'][hash_val].append(doc_id)
        
        self.num_vectors += 1
    
    def _compute_lsh_hash(self, vector: np.ndarray, projection: np.ndarray) -> str:
        """Compute LSH hash for vector."""
        projected = vector @ projection
        binary = (projected > 0).astype(int)
        return ''.join(map(str, binary))
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Fast approximate nearest neighbor search."""
        query_vector = query_vector.astype(np.float16)
        query_vector /= np.linalg.norm(query_vector) + 1e-8
        
        # Check cache first
        cache_key = hashlib.md5(query_vector.tobytes()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get candidates from LSH
        candidates = set()
        for table in self.lsh_tables:
            hash_val = self._compute_lsh_hash(query_vector, table['projection'])
            if hash_val in table['buckets']:
                candidates.update(table['buckets'][hash_val])
        
        # If not enough candidates, add random samples
        if len(candidates) < k * 3:
            random_ids = np.random.choice(
                list(self.index.keys()),
                min(k * 3, len(self.index)),
                replace=False
            )
            candidates.update(random_ids)
        
        # Score candidates
        scores = []
        for doc_id in candidates:
            position = self.index[doc_id]
            if self.use_mmap:
                doc_vector = self.mmap_vectors[position]
            else:
                doc_vector = self.vectors[position]
            
            # Cosine similarity
            similarity = np.dot(query_vector, doc_vector)
            scores.append((doc_id, float(similarity)))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = scores[:k]
        
        # Update cache
        self.cache[cache_key] = results
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        return results
    
    def _evict_oldest(self, fraction: float = 0.1):
        """Evict oldest vectors to make room."""
        num_to_evict = int(self.max_vectors * fraction)
        # Simple strategy: remove first entries
        # In production, use better eviction policy
        keys_to_remove = list(self.index.keys())[:num_to_evict]
        for key in keys_to_remove:
            del self.index[key]
            if key in self.metadata:
                del self.metadata[key]


class RAGPipeline:
    """
    Complete RAG pipeline with compression and optimization.
    """
    
    def __init__(self, embedding_dim: int = 384, use_compression: bool = True):
        self.embedding_dim = embedding_dim
        self.use_compression = use_compression
        
        # Vector database
        self.vector_db = VectorDatabase(embedding_dim, use_mmap=True)
        
        # Document store (compressed)
        self.doc_store = CompressedDocumentStore()
        
        # Query cache
        self.query_cache = LRUCache(capacity=100)
        
        # Embedding cache
        self.embedding_cache = {}
        
        # Statistics
        self.stats = {
            'queries': 0,
            'cache_hits': 0,
            'total_bytes_saved': 0
        }
    
    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Add document with automatic compression."""
        # Generate embedding (mock - in reality use a model)
        embedding = self._generate_embedding(text)
        
        # Compress document text
        compressed_doc = self.doc_store.add(doc_id, text, metadata)
        
        # Add to vector database
        self.vector_db.add_vector(doc_id, embedding, metadata)
        
        # Track compression savings
        original_size = len(text.encode('utf-8')) + embedding.nbytes
        compressed_size = compressed_doc['compressed_size'] + embedding.nbytes // 2  # float16
        self.stats['total_bytes_saved'] += original_size - compressed_size
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (mock implementation)."""
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Mock embedding generation
        # In reality, use sentence-transformers or similar
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        # Cache it
        self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def query(self, query_text: str, k: int = 5, use_reranking: bool = True) -> List[Dict]:
        """Query with retrieval and optional reranking."""
        self.stats['queries'] += 1
        
        # Check query cache
        cache_key = f"{query_text}_{k}"
        if cache_key in self.query_cache:
            self.stats['cache_hits'] += 1
            return self.query_cache[cache_key]
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query_text)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, k=k*2 if use_reranking else k)
        
        # Retrieve documents
        retrieved_docs = []
        for doc_id, score in results:
            doc_data = self.doc_store.get(doc_id)
            if doc_data:
                retrieved_docs.append({
                    'doc_id': doc_id,
                    'text': doc_data['text'],
                    'score': score,
                    'metadata': doc_data.get('metadata', {})
                })
        
        # Reranking (if enabled)
        if use_reranking and len(retrieved_docs) > k:
            retrieved_docs = self._rerank(query_text, retrieved_docs)[:k]
        
        # Cache results
        self.query_cache[cache_key] = retrieved_docs
        
        return retrieved_docs
    
    def _rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Rerank documents using cross-encoder (mock)."""
        # In reality, use a cross-encoder model
        # Here we just add some noise to scores and re-sort
        for doc in docs:
            # Mock reranking score
            doc['rerank_score'] = doc['score'] + np.random.uniform(-0.1, 0.1)
        
        docs.sort(key=lambda x: x.get('rerank_score', x['score']), reverse=True)
        return docs
    
    def get_statistics(self) -> Dict:
        """Get RAG system statistics."""
        return {
            **self.stats,
            'num_documents': self.vector_db.num_vectors,
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['queries']),
            'compression_ratio': self.doc_store.get_compression_ratio(),
            'memory_saved_mb': self.stats['total_bytes_saved'] / (1024 * 1024)
        }


class CompressedDocumentStore:
    """Document store with aggressive compression."""
    
    def __init__(self):
        self.documents = {}
        self.total_original_size = 0
        self.total_compressed_size = 0
    
    def add(self, doc_id: str, text: str, metadata: Dict = None) -> Dict:
        """Add document with compression."""
        original_size = len(text.encode('utf-8'))
        
        # Compress text
        compressed_text = zlib.compress(text.encode('utf-8'), level=9)
        compressed_size = len(compressed_text)
        
        # Store
        self.documents[doc_id] = {
            'compressed_text': compressed_text,
            'metadata': metadata,
            'original_size': original_size,
            'compressed_size': compressed_size
        }
        
        self.total_original_size += original_size
        self.total_compressed_size += compressed_size
        
        return {'compressed_size': compressed_size}
    
    def get(self, doc_id: str) -> Optional[Dict]:
        """Retrieve and decompress document."""
        if doc_id not in self.documents:
            return None
        
        doc = self.documents[doc_id]
        text = zlib.decompress(doc['compressed_text']).decode('utf-8')
        
        return {
            'text': text,
            'metadata': doc.get('metadata')
        }
    
    def get_compression_ratio(self) -> float:
        """Get overall compression ratio."""
        if self.total_compressed_size == 0:
            return 1.0
        return self.total_original_size / self.total_compressed_size


class LRUCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def __contains__(self, key: str) -> bool:
        if key in self.cache:
            # Move to end (most recent)
            self.cache.move_to_end(key)
            return True
        return False
    
    def __getitem__(self, key: str) -> Any:
        value = self.cache[key]
        self.cache.move_to_end(key)
        return value
    
    def __setitem__(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class HybridInference:
    """
    Combines RAG with model compression for efficient inference.
    """
    
    def __init__(self, model_params: int = 1_000_000_000):
        self.model_params = model_params
        self.rag = RAGPipeline(embedding_dim=384)
        
        # Adaptive computation: skip layers based on confidence
        self.layer_skip_threshold = 0.8
        self.early_exit_threshold = 0.9
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using RAG + compressed model."""
        # Retrieve relevant context
        context_docs = self.rag.query(prompt, k=3)
        
        # Build augmented prompt
        context = "\n".join([doc['text'][:200] for doc in context_docs])
        augmented_prompt = f"Context:\n{context}\n\nQuery: {prompt}\n\nResponse:"
        
        # Mock generation with early exit
        # In reality, this would use the actual model
        response = self._mock_generate(augmented_prompt, max_tokens)
        
        return response
    
    def _mock_generate(self, prompt: str, max_tokens: int) -> str:
        """Mock text generation."""
        # Simulate model generation
        words = ['The', 'RAG', 'system', 'efficiently', 'retrieves', 'relevant', 
                'information', 'reducing', 'model', 'size', 'requirements', 
                'through', 'external', 'memory', 'augmentation']
        
        response = []
        for i in range(min(max_tokens, len(words))):
            response.append(words[i % len(words)])
            
            # Simulate early exit based on confidence
            confidence = np.random.random()
            if confidence > self.early_exit_threshold and i > 5:
                break
        
        return ' '.join(response)


def demonstrate_rag_system():
    """Demonstrate the RAG system."""
    print("📚 ADVANCED RAG SYSTEM WITH COMPRESSION")
    print("=" * 60)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(embedding_dim=384)
    
    # Add sample documents
    documents = [
        ("doc1", "Neural networks can be compressed using quantization, pruning, and knowledge distillation."),
        ("doc2", "RAG systems reduce model size by storing knowledge externally in vector databases."),
        ("doc3", "Mixture of Experts allows conditional computation, activating only relevant model parts."),
        ("doc4", "Flash attention reduces memory usage through tiling and recomputation strategies."),
        ("doc5", "Binary embeddings can achieve 32x compression with minimal accuracy loss."),
    ]
    
    print("\n📥 Adding documents to RAG system...")
    for doc_id, text in documents:
        rag.add_document(doc_id, text, metadata={'source': 'research'})
    
    # Test queries
    queries = [
        "How to reduce model size?",
        "What is mixture of experts?",
        "Memory efficient attention mechanisms"
    ]
    
    print("\n🔍 Testing retrieval:")
    for query in queries:
        print(f"\nQuery: {query}")
        results = rag.query(query, k=2)
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['score']:.3f}] {result['text'][:60]}...")
    
    # Show statistics
    stats = rag.get_statistics()
    print("\n📊 Statistics:")
    print(f"  Documents stored: {stats['num_documents']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Memory saved: {stats['memory_saved_mb']:.2f} MB")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    # Test hybrid inference
    print("\n🤖 Testing hybrid inference:")
    hybrid = HybridInference()
    response = hybrid.generate("Explain model compression techniques", max_tokens=20)
    print(f"  Generated: {response}")
    
    print("\n✅ RAG system successfully demonstrates:")
    print("   - Vector database with LSH for fast search")
    print("   - Document compression (zlib)")
    print("   - Memory-mapped storage for large datasets")
    print("   - Caching for frequent queries")
    print("   - Hybrid inference combining RAG + model")

if __name__ == "__main__":
    demonstrate_rag_system()
