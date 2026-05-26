# Goodreads Information Retrieval System

A complete **Information Retrieval (IR) pipeline** for book discovery, implementing everything from near-duplicate detection to contextual snippet generation. Built as project for Modern Information Retrieval course at Sharif University of Technology.

## Features

- **Smart Search Engine** - Three ranking models (VSM, BM25, Unigram)
- **Spell Correction** - Hybrid Jaccard + TF scoring for typo-tolerant search
- **Near-Duplicate Detection** - MinHash LSH to remove redundant book summaries
- **Contextual Snippets** - Extracts optimal query-relevant text windows
- **Interactive UI** - Streamlit dashboard with real-time search
- **Comprehensive Evaluation** - MAP, NDCG, MRR, Precision, Recall metrics

## Core Implementation Details
1. Near-Duplicate Detection (MinHash LSH)
Shingling: 2-word shingles convert text to sets

MinHash Signatures: 100 permutations for compact representation

LSH Bands: 20 bands × 5 rows per band for candidate detection

Verification: Tested against LSHFakeData.json (every consecutive pair flagged as duplicate)

2. Spell Correction Pipeline
python
# Example: "whle" → "while"
1. Candidate generation via k-gram Jaccard similarity
2. Re-ranking using Normalized TF scores from corpus
3. Returns top correction with confidence
3. Scoring Models Implemented
Model	Description	Key Features
VSM	Vector Space Model	lnc.ltc weighting, cosine normalization
BM25	Okapi BM25	TF saturation + length normalization (k1=1.5, b=0.75)
Unigram	Language Model	Supports Naive / Bayes / Mixture smoothing
4. Smart Snippet Algorithm
Normalizes document for search, extracts from original raw text

Identifies optimal windows maximizing query term density

Merges windows with ... and highlights terms as ***term***

Returns missing query words separately

5. Indexing Strategy
Separate inverted indexes: descriptions, genres, characters

Structure: {term: {doc_id: term_frequency}}

Tiered indexing: Documents partitioned by importance for unsafe ranking optimization

Run once: Indexes are built and saved for reuse across search sessions

## Project Structure

| Module | Files |
|--------|-------|
| **UI** | `main.py` |
| **Logic** | `preprocess.py`, `LSH.py`, `spell_correction.py`, `Scorer.py`, `Search.py`, `snippet.py`, `Evaluation.py`, `utils.py` |
| **Indexer** | `index.py`, `tiered_index.py`, `index_reader.py`, `document_lengths_index.py`, `metadata_index.py` |
| **Data** | `crawled.json`, `preprocessed.json`, `top_3000_rated_books.csv`, `stopwords.txt` |
| **Indexes** | `index/` (JSON index files), `indexes/` (spell correction pickle) |

# Technologies Used
Python 3.8+ - Core logic

Streamlit - Interactive UI

NumPy - Vector operations for VSM

NLTK - Tokenization, stopwords, stemming

Datasketch - MinHash LSH implementation

📝 Notes
All modules communicate through well-defined interfaces

Original architecture preserved while adding custom optimizations

Docstrings included for all major functions

Tested with provided Goodreads dataset
