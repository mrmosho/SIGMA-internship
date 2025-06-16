import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import time

class VectorSearcher:
    def __init__(self, indexer, metadata: List[Dict[str, Any]], 
                 model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize searcher with loaded index and metadata"""
        self.indexer = indexer
        self.metadata = metadata
        self.model = SentenceTransformer(model_name)
        
        if len(metadata) != indexer.index.ntotal:
            print(f"Warning: Metadata length ({len(metadata)}) doesn't match index size ({indexer.index.ntotal})")
    
    def search(self, query: str, k: int = 10, nprobe: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        Returns list of results with scores and metadata
        """
        start_time = time.time()
        
        # Encode query
        query_vector = self.model.encode([query])
        
        # Set search parameters
        self.indexer.index.nprobe = nprobe
        
        # Perform search
        distances, indices = self.indexer.index.search(
            query_vector.astype(np.float32), k
        )
        
        # Format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # Valid result
                result = {
                    'score': float(distance),
                    'metadata': self.metadata[idx]
                }
                results.append(result)
        
        search_time = time.time() - start_time
        
        print(f"Search completed in {search_time:.3f}s, found {len(results)} results")
        return results
