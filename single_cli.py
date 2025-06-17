# single_cli.py - All-in-one vector search CLI
import click
import json
import os
import pandas as pd
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_metadata(metadata: List[Dict[str, Any]], filepath: str = "metadata.json"):
    """Save metadata to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(metadata)} metadata entries to {filepath}")

def load_metadata(filepath: str = "metadata.json") -> List[Dict[str, Any]]:
    """Load metadata from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} metadata entries from {filepath}")
    return metadata

def check_files_exist(*filepaths):
    """Check if required files exist"""
    missing = [f for f in filepaths if not os.path.exists(f)]
    if missing:
        print(f"Missing files: {', '.join(missing)}")
        return False
    return True

# ============================================================================
# TEXT EMBEDDER CLASS
# ============================================================================

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

# ============================================================================
# FAISS INDEXER CLASS
# ============================================================================

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

# ============================================================================
# VECTOR SEARCHER CLASS
# ============================================================================

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

# ============================================================================
# CLI COMMANDS
# ============================================================================

@click.group()
def cli():
    """Simple Vector Search CLI"""
    pass

@cli.command()
@click.argument('csv_path')
@click.option('--batch-size', default=100, help='Batch size for embedding generation')
@click.option('--clusters', default=None, type=int, help='Number of IVF clusters (auto if not specified)')
@click.option('--text-column', default='DESCRIPTION', help='Name of the text column to embed')
def index(csv_path, batch_size, clusters, text_column):
    """Build search index from CSV file"""
    
    if not os.path.exists(csv_path):
        click.echo(f"Error: CSV file not found: {csv_path}")
        return
    
    try:
        # Generate embeddings
        embedder = TextEmbedder()
        embeddings, metadata = embedder.embed_csv(csv_path, text_column, batch_size)
        
        # Build index
        indexer = FAISSIndexer()
        indexer.build_ivf_index(embeddings, n_clusters=clusters)
        
        # Save metadata
        save_metadata(metadata)
        
        click.echo("✅ Indexing complete! Files created:")
        click.echo("  - index.ivf")
        click.echo("  - metadata.json")
        
    except Exception as e:
        click.echo(f"❌ Error during indexing: {e}")

@cli.command()
@click.argument('query')
@click.option('-k', '--results', default=5, help='Number of results to return')
@click.option('--nprobe', default=10, help='Number of clusters to search (higher = better recall)')
def search(query, results, nprobe):
    """Search the index for similar text"""
    
    # Check required files exist
    if not check_files_exist("index.ivf", "metadata.json"):
        click.echo("❌ Please run 'index' command first to build the search index")
        return
    
    try:
        # Load index and metadata
        indexer = FAISSIndexer()
        indexer.load_index()
        metadata = load_metadata()
        
        # Initialize searcher
        searcher = VectorSearcher(indexer, metadata)
        
        # Perform search
        click.echo(f"🔍 Searching for: '{query}'")
        search_results = searcher.search(query, k=results, nprobe=nprobe)
        
        # Display results
        click.echo(f"\n📋 Top {len(search_results)} results:")
        click.echo("-" * 50)
        
        for i, result in enumerate(search_results, 1):
            score = result['score']
            metadata = result['metadata']
            
            click.echo(f"{i}. Score: {score:.4f}")
            
            # Display key fields from your dataset
            if 'ID' in metadata:
                click.echo(f"   ID: {metadata['ID']}")
            
            if 'TITLE' in metadata:
                click.echo(f"   Title: {metadata['TITLE']}")
            
            if 'DESCRIPTION' in metadata:
                desc_preview = str(metadata['DESCRIPTION'])
                if len(desc_preview) > 150:
                    desc_preview = desc_preview[:150] + "..."
                click.echo(f"   Description: {desc_preview}")
            
            # Show additional useful fields
            extra_fields = ['SOURCE_NAME', 'LANGUAGE', 'REFERENCE_ID']
            for field in extra_fields:
                if field in metadata and metadata[field]:
                    value = str(metadata[field])
                    if len(value) > 80:
                        value = value[:80] + "..."
                    click.echo(f"   {field}: {value}")
            
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Search error: {e}")

@cli.command()
def status():
    """Show index status and statistics"""
    
    if not check_files_exist("index.ivf", "metadata.json"):
        click.echo("❌ No index found. Run 'index' command first.")
        return
    
    try:
        indexer = FAISSIndexer()
        indexer.load_index()
        metadata = load_metadata()
        
        click.echo("📊 Index Status:")
        click.echo(f"  Total vectors: {indexer.index.ntotal:,}")
        click.echo(f"  Dimensions: {indexer.dimension}")
        click.echo(f"  Index type: {type(indexer.index).__name__}")
        click.echo(f"  Metadata entries: {len(metadata):,}")
        
        if hasattr(indexer.index, 'nlist'):
            click.echo(f"  IVF clusters: {indexer.index.nlist}")
        
    except Exception as e:
        click.echo(f"❌ Status error: {e}")

@cli.command()
def interactive():
    """Interactive search mode"""
    
    if not check_files_exist("index.ivf", "metadata.json"):
        click.echo("❌ No index found. Run 'index' command first.")
        return
    
    try:
        # Load once
        click.echo("Loading index...")
        indexer = FAISSIndexer()
        indexer.load_index()
        metadata = load_metadata()
        searcher = VectorSearcher(indexer, metadata)
        
        click.echo("🔍 Interactive Search Mode (type 'quit' to exit)")
        click.echo("-" * 40)
        
        while True:
            query = input("\nEnter search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                click.echo("Goodbye! 👋")
                break
            
            if not query:
                continue
            
            results = searcher.search(query, k=3)  # Show top 3 in interactive mode
            
            for i, result in enumerate(results, 1):
                click.echo(f"\n{i}. Score: {result['score']:.4f}")
                metadata = result['metadata']
                
                # Show ID and Title if available
                if 'ID' in metadata:
                    click.echo(f"   ID: {metadata['ID']}")
                
                if 'TITLE' in metadata:
                    click.echo(f"   Title: {metadata['TITLE']}")
                
                # Show description preview
                if 'DESCRIPTION' in metadata:
                    desc_preview = str(metadata['DESCRIPTION'])
                    preview = desc_preview[:150] + "..." if len(desc_preview) > 150 else desc_preview
                    click.echo(f"   {preview}")
        
    except KeyboardInterrupt:
        click.echo("\n\nGoodbye! 👋")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

if __name__ == '__main__':
    cli()