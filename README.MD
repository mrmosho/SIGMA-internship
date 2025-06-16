# Simple Vector Search CLI

A straightforward command-line tool for building and searching vector indexes using FAISS and sentence transformers.

## Setup

```bash
# Create project directory
mkdir vector-search-cli
cd vector-search-cli

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Build Index from CSV

```bash
python cli.py index data.csv
```

Options:
- `--batch-size 100`: Process embeddings in batches
- `--clusters 50`: Number of IVF clusters (auto-calculated if not specified)
- `--text-column DESCRIPTION`: Name of text column (defaults to 'DESCRIPTION')

Example:
```bash
python cli.py index products.csv --batch-size 50
```

### 2. Search the Index

```bash
python cli.py search "your search query"
```

Options:
- `-k 10`: Number of results to return
- `--nprobe 10`: Search precision (higher = better but slower)

Example:
```bash
python cli.py search "laptop computer" -k 5 --nprobe 20
```

### 3. Interactive Search Mode

```bash
python cli.py interactive
```

Lets you search repeatedly without reloading the index.

### 4. Check Status

```bash
python cli.py status
```

Shows index statistics and information.

## File Structure

```
vector-search-cli/
├── app/
│   ├── embedder.py      # Text embedding logic
│   ├── indexer.py       # FAISS index creation
│   ├── searcher.py      # Search functionality
│   └── utils.py         # Helper functions
├── cli.py               # Main CLI interface
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data.csv            # Your input dataset
├── index.ivf           # Generated FAISS index
└── metadata.json       # Generated metadata storage
```

## Quick Start Example

1. **Prepare your CSV file** with a text column to search
2. **Build the index**:
   ```bash
   python cli.py index data.csv "description"
   ```
3. **Search**:
   ```bash
   python cli.py search "what you're looking for"
   ```

## CSV Format

Your CSV should have at least one text column. All columns will be stored as metadata and returned with search results.

Example:
```csv
ID,TITLE,DESCRIPTION,SOURCE_NAME,LANGUAGE
1,"Sample Title","This is a sample description","Source1","EN"
2,"Another Title","Another sample description","Source2","EN"
```

## Tips

- **Larger datasets**: Use larger `--batch-size` for faster processing
- **Better search quality**: Increase `--nprobe` value
- **Faster search**: Decrease `--nprobe` value
- **Interactive mode**: Great for testing different queries quickly

## What happens when you run commands:

1. **`index`**: Reads CSV → generates embeddings → builds FAISS IVF index → saves files
2. **`search`**: Loads index → encodes query → finds similar vectors → returns results with metadata
3. **`interactive`**: Same as search but keeps running for multiple queries

That's it! Simple vector search without the complexity.