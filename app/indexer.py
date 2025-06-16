import faiss
import numpy as np
from typing import Optional

class FAISSIndexer:
    def __init__(self):
        self.index = None
        self.dimension = None
    
    def build_ivf_index(self, embeddings: np.ndarray, 
                       n_clusters: Optional[int] = None,
                       index_path: str = "index.ivf") -> None:
        """
        Build IVF FAISS index from embeddings
        """
        self.dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Auto-determine cluster count if not provided
        if n_clusters is None:
            n_clusters = min(max(int(np.sqrt(n_vectors)), 10), 1000)
        
        print(f"Building IVF index with {n_clusters} clusters for {n_vectors} vectors")
        
        # Create IVF index
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
        
        # Train the index
        print("Training index...")
        self.index.train(embeddings.astype(np.float32))
        
        # Add vectors to index
        print("Adding vectors to index...")
        self.index.add(embeddings.astype(np.float32))
        
        # Save index
        faiss.write_index(self.index, index_path)
        print(f"Index saved to {index_path}")
        print(f"Index contains {self.index.ntotal} vectors")
    
    def load_index(self, index_path: str = "index.ivf") -> None:
        """Load existing FAISS index"""
        print(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        print(f"Loaded index with {self.index.ntotal} vectors, dimension {self.dimension}")