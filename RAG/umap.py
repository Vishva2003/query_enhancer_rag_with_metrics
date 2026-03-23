"""
Complete RAG Pipeline with UMAP Visualization
All steps in linear order - No functions, just sequential code
Uses: PyPDF, LangChain, SentenceTransformer, CrossEncoder, Gemini, UMAP
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import plotly.graph_objects as go
from config import DATA_RAW

from pypdf import PdfReader
from google import genai
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Load environment variables
load_dotenv()

# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================

print("\n" + "="*70)
print("📚 COMPLETE RAG PIPELINE WITH UMAP VISUALIZATION")
print("="*70 + "\n")

# File to process
PDF_FILE = DATA_RAW / "sample.pdf"  # ← CHANGE THIS TO YOUR PDF

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_LLM_KEY")  # or GEMINI_LLM_KEY

# Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 5
DISTANCE_THRESHOLD = 1.5

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

print(f"📄 Document: {PDF_FILE}")
print(f"🔢 Embedding Model: {EMBEDDING_MODEL}")
print(f"🔄 Reranker Model: {RERANKER_MODEL}")
print(f"🤖 LLM Model: {GEMINI_MODEL}")
print(f"✂️ Chunk Size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP}")

# ============================================================================
# STEP 2: DOCUMENT LOADING (PyPDF)
# ============================================================================

print("\n" + "="*70)
print("STEP 1: LOADING DOCUMENT")
print("="*70)


# Check if file exists
if not Path(PDF_FILE).exists():
    print(f"❌ File not found: {PDF_FILE}")
    print("Please update the PDF_FILE variable at the top of the script")
    exit()

# Load PDF
print(f"📖 Reading PDF: {PDF_FILE}")
pdf_reader = PdfReader(PDF_FILE)
total_pages = len(pdf_reader.pages)

# Extract text from all pages
full_text = ""
for page_num, page in enumerate(pdf_reader.pages, 1):
    page_text = page.extract_text()
    if page_text:
        full_text += page_text + "\n"
    print(f"  ✅ Processed page {page_num}/{total_pages}")

print(f"\n✅ Loaded {len(full_text):,} characters from {total_pages} pages")

# ============================================================================
# STEP 3: CHUNKING (LangChain Text Splitter)
# ============================================================================

print("\n" + "="*70)
print("STEP 2: CHUNKING TEXT")
print("="*70)

# Create text splitter
print(f"✂️ Creating chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    length_function=len
)

# Split text into chunks
chunks = text_splitter.split_text(full_text)

print(f"✅ Created {len(chunks)} chunks")
print(f"\n📝 Sample chunks:")
for i, chunk in enumerate(chunks[:3], 1):
    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
    print(f"  Chunk {i}: {preview}")

# ============================================================================
# STEP 4: EMBEDDING (SentenceTransformer)
# ============================================================================

print("\n" + "="*70)
print("STEP 3: GENERATING EMBEDDINGS")
print("="*70)


# Load embedding model
print(f"🔧 Loading SentenceTransformer: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Generate embeddings for all chunks
print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=True
)

print(f"✅ Generated embeddings with shape: {chunk_embeddings.shape}")
print(f"   Embedding dimension: {chunk_embeddings.shape[1]}")

# ============================================================================
# STEP 5: QUERY PROCESSING
# ============================================================================

print("\n" + "="*70)
print("STEP 4: QUERY PROCESSING")
print("="*70)

# Get query from user
query = input("\n❓ Enter your question: ").strip()

if not query:
    query = "What is this document about?"
    print(f"Using default query: {query}")

print(f"\n🔍 Query: {query}")

# Embed the query
print("🔢 Generating query embedding...")
query_embedding = embedding_model.encode([query], convert_to_numpy=True)

print(f"✅ Query embedding shape: {query_embedding.shape}")

# ============================================================================
# STEP 6: RETRIEVAL (Cosine Similarity)
# ============================================================================

print("\n" + "="*70)
print("STEP 5: RETRIEVING RELEVANT CHUNKS")
print("="*70)


# Calculate cosine similarity between query and all chunks
print(f"📊 Calculating similarity with {len(chunks)} chunks...")
similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

# Convert to distances (1 - similarity for consistency with ChromaDB)
distances = 1 - similarities

# Get top K results
print(f"🎯 Retrieving top {TOP_K_RETRIEVE} chunks...")
top_indices = np.argsort(distances)[:TOP_K_RETRIEVE]

retrieved_chunks = [chunks[i] for i in top_indices]
retrieved_distances = [distances[i] for i in top_indices]

print(f"\n✅ Retrieved {len(retrieved_chunks)} chunks:")
for i, (chunk, dist) in enumerate(zip(retrieved_chunks, retrieved_distances), 1):
    preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
    print(f"  {i}. Distance: {dist:.4f} | {preview}")

# ============================================================================
# STEP 7: FILTERING BY DISTANCE THRESHOLD
# ============================================================================

print("\n" + "="*70)
print("STEP 6: FILTERING BY DISTANCE THRESHOLD")
print("="*70)

print(f"🔍 Filtering chunks with distance < {DISTANCE_THRESHOLD}")

filtered_chunks = []
filtered_distances = []

for chunk, dist in zip(retrieved_chunks, retrieved_distances):
    if dist < DISTANCE_THRESHOLD:
        filtered_chunks.append(chunk)
        filtered_distances.append(dist)

print(f"✅ Filtered down to {len(filtered_chunks)} relevant chunks")

if not filtered_chunks:
    print("⚠️ No chunks passed the distance threshold")
    print("Continuing with all retrieved chunks...")
    filtered_chunks = retrieved_chunks
    filtered_distances = retrieved_distances

# ============================================================================
# STEP 8: RERANKING (CrossEncoder)
# ============================================================================

print("\n" + "="*70)
print("STEP 7: RERANKING WITH CROSS-ENCODER")
print("="*70)


# Load reranker model
print(f"🔧 Loading CrossEncoder: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL)

# Create query-document pairs
print(f"🔄 Reranking {len(filtered_chunks)} chunks...")
pairs = [[query, chunk] for chunk in filtered_chunks]

# Get reranking scores
rerank_scores = reranker.predict(pairs)

# Sort by reranking scores (higher is better)
reranked_indices = np.argsort(rerank_scores)[::-1][:TOP_K_RERANK]

reranked_chunks = [filtered_chunks[i] for i in reranked_indices]
reranked_scores = [rerank_scores[i] for i in reranked_indices]
reranked_distances = [filtered_distances[i] for i in reranked_indices]

print(f"\n✅ Top {len(reranked_chunks)} reranked chunks:")
for i, (chunk, score, dist) in enumerate(zip(reranked_chunks, reranked_scores, reranked_distances), 1):
    preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
    print(f"  {i}. Score: {score:.4f}, Dist: {dist:.4f} | {preview}")

# ============================================================================
# STEP 9: CONTEXT PREPARATION
# ============================================================================

print("\n" + "="*70)
print("STEP 8: PREPARING CONTEXT FOR LLM")
print("="*70)

# Format context from reranked chunks
context_parts = []
for i, (chunk, score) in enumerate(zip(reranked_chunks, reranked_scores), 1):
    context_parts.append(f"[Document {i}, Score: {score:.2f}]\n{chunk}\n")

context = "\n".join(context_parts)

print(f"✅ Context prepared with {len(reranked_chunks)} chunks")
print(f"📝 Total context length: {len(context):,} characters")

# ============================================================================
# STEP 10: LLM GENERATION (Google Gemini)
# ============================================================================

print("\n" + "="*70)
print("STEP 9: GENERATING ANSWER WITH GEMINI")
print("="*70)


# Configure Gemini
print(f"🔧 Configuring Gemini API...")
client = genai.Client(api_key=GEMINI_API_KEY)
# Create prompt
prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 

If you don't know the answer based on the context, say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {query}

Answer:"""

print(f"🤖 Generating answer with {GEMINI_MODEL}...")
print(f"📝 Prompt length: {len(prompt):,} characters")

# Generate response
response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
answer = response.text

print(f"\n{'='*70}")
print("💡 ANSWER:")
print('='*70)
print(answer)
print('='*70)

# ============================================================================
# STEP 11: UMAP VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("STEP 10: CREATING UMAP VISUALIZATION")
print("="*70)

import umap

# Ask if user wants UMAP
show_umap = input("\n🗺️ Show UMAP visualization? (y/n): ").strip().lower()

if show_umap == 'y':
    
    print("\n🗺️ Creating UMAP visualization...")
    
    # Prepare all embeddings (chunks + query)
    all_embeddings = np.vstack([chunk_embeddings, query_embedding])
    
    # Create labels
    labels = ["Document"] * len(chunks)
    labels.append("Query")
    
    # Create texts for hover
    texts = chunks.copy()
    texts.append(query)
    
    # Create colors
    colors = ["lightblue"] * len(chunks)
    colors.append("red")
    
    # Create sizes
    sizes = [6] * len(chunks)
    sizes.append(18)
    
    # Create symbols
    symbols = ["circle"] * len(chunks)
    symbols.append("star")
    
    # Mark retrieved chunks
    print("  🔷 Marking retrieved chunks...")
    retrieved_chunk_indices = top_indices[:len(filtered_chunks)]
    for idx in retrieved_chunk_indices:
        colors[idx] = "darkblue"
        sizes[idx] = 10
        symbols[idx] = "diamond"
        labels[idx] = "Retrieved"
    
    # Apply UMAP
    print("  ⚙️ Running UMAP dimensionality reduction...")
    n_neighbors = min(15, len(all_embeddings) - 1)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
        metric="cosine",
        verbose=True
    )
    
    embeddings_2d = reducer.fit_transform(all_embeddings)
    
    print("  ✅ UMAP reduction complete")
    
    # Create Plotly figure
    print("  🎨 Creating visualization...")
    fig = go.Figure()
    
    # Group by type for legend
    for label_type in set(labels):
        mask = [label == label_type for label in labels]
        indices = [i for i, m in enumerate(mask) if m]
        
        if not indices:
            continue
        
        x_coords = [embeddings_2d[i, 0] for i in indices]
        y_coords = [embeddings_2d[i, 1] for i in indices]
        hover_texts = [f"{labels[i]}: {texts[i][:120]}..." for i in indices]
        point_colors = [colors[i] for i in indices]
        point_sizes = [sizes[i] for i in indices]
        point_symbols = [symbols[i] for i in indices]
        
        mode = 'markers+text' if label_type == "Query" else 'markers'
        text = [label_type] if label_type == "Query" else None
        textposition = 'top center' if label_type == "Query" else None
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode=mode,
            marker=dict(
                size=point_sizes,
                color=point_colors,
                symbol=point_symbols,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=text,
            textposition=textposition,
            hovertext=hover_texts,
            hoverinfo='text',
            name=f"{label_type} ({len(indices)})"
        ))
    
    # Update layout
    query_preview = query[:60] + "..." if len(query) > 60 else query
    
    fig.update_layout(
        title=dict(
            text=f"📊 RAG Pipeline UMAP Visualization<br><sub>Query: '{query_preview}'</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        hovermode="closest",
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray', showgrid=True),
        yaxis=dict(gridcolor='lightgray', showgrid=True)
    )
    
    # Save and show
    output_file = "rag_pipeline_umap.html"
    fig.write_html(output_file)
    print(f"\n  💾 Saved to: {output_file}")
    
    print("  🌐 Opening in browser...")
    fig.show()
    
    print("\n✅ UMAP visualization complete!")
    print("\n📖 Legend:")
    print("  🔵 Light blue circles = All document chunks")
    print("  🔴 Red star = Your query")
    print("  🔷 Dark blue diamonds = Retrieved chunks")

# ============================================================================
# STEP 12: SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 PIPELINE SUMMARY")
print("="*70)

print(f"""
📄 Document: {Path(PDF_FILE).name}
  └─ Pages: {total_pages}
  └─ Characters: {len(full_text):,}

✂️ Chunking:
  └─ Total chunks: {len(chunks)}
  └─ Chunk size: {CHUNK_SIZE}
  └─ Overlap: {CHUNK_OVERLAP}

🔢 Embeddings:
  └─ Model: {EMBEDDING_MODEL}
  └─ Dimension: {chunk_embeddings.shape[1]}

🔍 Query: "{query}"

📊 Retrieval:
  └─ Retrieved: {len(retrieved_chunks)} chunks
  └─ Filtered: {len(filtered_chunks)} chunks (distance < {DISTANCE_THRESHOLD})

🔄 Reranking:
  └─ Model: {RERANKER_MODEL}
  └─ Top reranked: {len(reranked_chunks)} chunks

🤖 Generation:
  └─ Model: {GEMINI_MODEL}
  └─ Context length: {len(context):,} characters
  └─ Answer length: {len(answer):,} characters

✅ Pipeline completed successfully!
""")

print("="*70)
print("🎉 ALL STEPS COMPLETED!")
print("="*70 + "\n")