import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Tuple, List, Dict, Any

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedder with a sentence transformer model"""
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_csv(self, csv_path: str, text_column: str, 
                  batch_size: int = 100) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Convert CSV text column to embeddings
        Returns: (embeddings_array, metadata_list)
        """
        print(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")
        
        # Clean text data
        texts = df[text_column].fillna('').astype(str).tolist()
        print(f"Found {len(texts)} text entries")
        
        # Generate embeddings in batches
        print("Generating embeddings...")
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        embeddings_array = np.vstack(embeddings)
        
        # Prepare metadata (all columns except embeddings)
        metadata = df.to_dict('records')
        
        print(f"Generated {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
        return embeddings_array, metadata
