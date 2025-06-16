import click
import os
from app.embedder import TextEmbedder
from app.indexer import FAISSIndexer
from app.searcher import VectorSearcher
from app.utils import save_metadata, load_metadata, check_files_exist

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