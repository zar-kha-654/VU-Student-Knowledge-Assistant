# Virtual University Student Knowledge Base

Precomputed RAG knowledge base.

## Source format

Markdown (.md)

## Documents

6

## Chunks

25

## Embedding model

BAAI/bge-small-en-v1.5

## Vector dimension

384

## Similarity

Cosine similarity using normalized embeddings
and FAISS Inner Product.

## Chunk size

1000 characters

## Chunk overlap

150 characters

## Source traceability

Each chunk stores:

- chunk ID
- source filename
- document title
- Markdown section
- chunk number
- file type
- character count

Note:

Markdown documents do not have PDF page numbers,
so section names are used for source location.