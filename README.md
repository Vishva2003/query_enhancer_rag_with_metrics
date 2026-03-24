 
# 📚 Advanced Multi-Model RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system with advanced query enhancement, multi-model support, and comprehensive evaluation metrics. Built with AI models via OpenRouter and featuring interactive 3D UMAP visualization.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Models](https://img.shields.io/badge/models-8-orange.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Models Supported](#-models-supported)
- [Architecture](#️-architecture)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [Components](#-components)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 📖 Overview

This advanced RAG system enables intelligent document-based question answering by combining:

- **🔍 Advanced Retrieval**: Vector search with distance filtering and cross-encoder reranking
- **🤖 Query Enhancement**: Generates 6 query variations (sub-queries, HyDE, step-back) via Grok 4
- **🎯 Multi-Model Support**: 8 AI models including GPT-5, Grok-4, Claude-4.5, DeepSeek-V3
- **📊 Real-Time Evaluation**: Precision, recall, distance metrics, and timing breakdown
- **🗺️ Interactive Visualization**: 3D UMAP embedding space with Plotly
- **💬 Streamlit UI**: Beautiful chat interface with document upload and metrics dashboard

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📄 Document Processing** | Upload PDF, DOCX, TXT files with automatic text extraction and intelligent chunking |
| **💾 Vector Storage** | ChromaDB with persistent storage and 384-dim SentenceTransformer embeddings |
| **🔍 Query Enhancement** | 6 query variations: original + 3 sub-queries + HyDE + step-back (via Grok) |
| **🎯 Multi-Stage Retrieval** | Vector search → Distance filtering → Cross-encoder reranking |
| **🤖 Multi-Model Support** | 8 models via OpenRouter (GPT-5, Grok-4, Claude-4.5, etc.) |
| **📊 Evaluation Metrics** | Precision, Recall, Avg Distance, Retrieval/Rerank/Generation timing |
| **🗺️ 3D Visualization** | Interactive 3D UMAP projection with Plotly |
| **📈 Metrics Dashboard** | Query history with CSV export and model performance comparison |

---

## 🤖 Models Supported

All models are Paid (there are tons of free models to choose from) and accessible via [OpenRouter](https://openrouter.ai):

| Model Name | Model ID | Provider | Best For |
|------------|----------|----------|----------|
| **ChatGPT-5** | `openai/gpt-5-chat` | OpenAI | General purpose, high quality |
| **Grok-4-Fast** | `x-ai/grok-4-fast` | xAI | Speed, real-time responses |
| **Claude-Sonnet-4.5** | `anthropic/claude-sonnet-4.5` | Anthropic | Detailed analysis, safety |
| **DeepSeek-V3** | `deepseek/deepseek-chat-v3.1` | DeepSeek | Code, technical reasoning |
| **Gemini-2.5-Flash-Lite** | `google/gemini-2.5-flash-lite` | Google | Fast, efficient, lightweight |
| **Qwen3.5-Flash** | `qwen/qwen3.5-flash-02-23` | Alibaba | Multilingual, balanced |
| **MiniMax-M1** | `minimax/minimax-m1` | MiniMax | Creative writing, dialogue |
| **Ministral-3B** | `mistralai/ministral-3b-2512` | Mistral | Smallest, fastest, low latency |

Users can select any model from the Streamlit sidebar dropdown.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG SYSTEM PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Document] → [Loader] → [Chunker] → [Embedder] → [ChromaDB]       │
│               (PDF/DOCX) (LangChain)  (SentenceT)   (Persistent)    │
│                                                                      │
│  [Query] → [QueryEnhancer] → [Multi-Query Retrieval]               │
│            (Gemini API)        (6 queries aggregated)               │
│                  ↓                      ↓                            │
│            [Sub-Queries]          [Vector Search]                   │
│            [HyDE]          →      [Distance Filter] → [Reranker]    │
│            [Step-Back]            (threshold<1.5)    (CrossEncoder) │
│                                                                      │
│  [Top-5 Context] → [Generator] → [Answer] + [Metrics]              │
│                    (Selected Model)                                 │
│                                                                      │
│  [UMAP 3D] ← [All Embeddings] ← [Queries + Chunks + Retrieved]    │
│  (Plotly)    (Dimensionality)                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Query Enhancement Flow

```
Original Query: "What is RAG?"
        ↓
    [Gemini API]
        ↓
    ┌─────────────────────────────────────────────┐
    │ 1. Original: "What is RAG?"                  │
    │ 2. Sub-Q1: "What does RAG stand for in NLP?" │
    │ 3. Sub-Q2: "How does RAG improve LLMs?"      │
    │ 4. Sub-Q3: "What are RAG's key components?"  │
    │ 5. HyDE: "RAG is a technique that..."        │
    │ 6. Step-Back: "What are LLM accuracy methods?"|
    └─────────────────────────────────────────────┘
        ↓
    [Embed All 6 Queries]
        ↓
    [Aggregate Results] → [Top-K Documents]
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/query_enhancer_rag_with_metrics.git
cd query_enhancer_rag_with_metrics
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# OpenRouter API (for LLM generation)
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=http://localhost:8501
OPENROUTER_SITE_NAME=AdvancedRAG

# Google Gemini API (Optional)
GEMINI_API_KEY=your-gemini-api-key-here

# LlamaParse API (for advanced PDF parsing)
LLAMA_API_KEY=your-llama-api-key-here
```

**Get your free API keys:**
- OpenRouter: [https://openrouter.ai](https://openrouter.ai)
- Google Gemini: [https://ai.google.dev](https://ai.google.dev)
- LlamaParse: [https://cloud.llamaindex.ai](https://cloud.llamaindex.ai)

---

## ⚙️ Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |
| `CROSS_ENCODER_MODEL` | `cross-encoder/qnli-distilroberta-base` | Reranking model |
| `VECTOR_DB_PATH` | `./vector_db` | ChromaDB persistent storage location |
| `BATCH_SIZE` | `32` | Batch size for embedding generation |
| `TOP_K_RESULTS` | `10` | Number of documents to retrieve |
| `CHUNK_SIZE` | `1000` | Default chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks (characters) |

**Models Configuration** (`config.py`):

```python
models = {
    "Chatgpt-5": "openai/gpt-5-chat",
    "Grok-4-Fast": "x-ai/grok-4-fast",
    "Claude-Sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "Deepseek-v3": "deepseek/deepseek-chat-v3.1",
    "Gemini-2.5-Flash-lite": "google/gemini-2.5-flash-lite",
    "Qwen3.5-flash-02-23": "qwen/qwen3.5-flash-02-23",
    "MiniMax-m1": "minimax/minimax-m1",
    "Ministral-3": "mistralai/ministral-3b-2512"
}
```

---

## 🎯 Usage

### Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Step-by-Step Usage

1. **📄 Upload Document**
   - Click "Upload a document" in sidebar
   - Supports: PDF, TXT, DOCX, DOC

2. **⚙️ Configure Settings**
   - Chunk size: 1000-2500 characters
   - Chunk overlap: 100-500 characters
   - Top K results: 1-40 documents
   - Distance threshold: Default 1.5

3. **🏷️ Enter Collection Name**
   - Name your vector database collection
   - Example: `research_papers`, `company_docs`

4. **🤖 Select Model**
   - Choose from 8 free models
   - Each optimized for different tasks

5. **▶️ Click Start**
   - System processes document
   - Creates chunks and embeddings
   - Stores in ChromaDB

6. **💬 Ask Questions**
   - Type question in chat input
   - System enhances query (6 variations)
   - Retrieves, reranks, and generates answer

7. **📊 View Metrics**
   - See precision, recall, distances
   - Retrieval/rerank/generation timing
   - Model performance stats

8. **🗺️ Visualize (Optional)**
   - Enable UMAP checkbox
   - See 3D embedding space
   - Interactive Plotly visualization

### Command-Line Script (Linear Pipeline)

```bash
python rag_pipeline_linear.py
```

This runs the complete pipeline step-by-step with no functions:
- Step 1: Document Loading (LlamaParse)
- Step 2: Chunking (LangChain)
- Step 3: Embedding (SentenceTransformer)
- Step 4: Query Enhancement (Gork)
- Step 5: Multi-Query Embedding
- Step 6: Multi-Query Retrieval
- Step 7: Distance Filtering
- Step 8: Reranking (CrossEncoder)
- Step 9: Context Preparation
- Step 10: LLM Generation (Gemini)
- Step 11: 3D UMAP Visualization
- Step 12: Summary & Metrics

---

## 📦 Components

### 1. **DocumentLoader** (`rag_agent/tools/document_loader.py`)

Extracts text from multiple formats:
- **PDF**: Uses LlamaParse (advanced)
- **DOCX/DOC**: Uses python-docx
- **TXT**: Plain text reader

```python
loader = DocumentLoader()
text = loader.load_file("document.pdf")
```

### 2. **Chunker** (`rag_agent/tools/chunker.py`)

Splits text using LangChain's `RecursiveCharacterTextSplitter`:
- Configurable chunk size and overlap
- Smart splitting on sentence boundaries

```python
chunker = Chunker()
chunks = chunker.chunk_text(text, chunk_size=1000, chunk_overlap=200)
```

### 3. **Embedder** (`rag_agent/tools/embedder.py`)

Generates 384-dim embeddings:
- Model: `all-MiniLM-L6-v2` (SentenceTransformer)
- Batch processing for efficiency
- Persistent ChromaDB storage

```python
embedder = Embedder()
embedder.add_collection(chunks, collection="my_docs")
```

### 4. **QueryEnhancer** (`query_enhancer/tools/query_agent.py`)

Generates 6 query variations using Gork 4:
- **Original query**
- **3 Sub-queries** (different aspects)
- **HyDE** (Hypothetical Document Embedding)
- **Step-back** (broader question)

```python
enhancer = QueryEnhancer()
enhanced = enhancer.enhance("What is RAG?")
# Returns: [original, sub1, sub2, sub3, hyde, stepback]
```

### 5. **Retriever** (`rag_agent/tools/retriever.py`)

Multi-query vector search:
- Embeds all enhanced queries
- Aggregates results (max similarity)
- Returns top-K with distances

```python
retriever = Retriever()
results = retriever.retrieve_multi(enhanced_queries, "my_docs", top_k=10)
```

### 6. **Reranker** (`rag_agent/tools/reranker.py`)

Cross-encoder reranking:
- Model: `cross-encoder/qnli-distilroberta-base`
- Scores query-document pairs
- Returns top-N ranked results

```python
reranker = Reranker()
ranked = reranker.rerank(query, retrieved_docs, top_k=5)
```

### 7. **Generator** (`rag_agent/tools/generator.py`)

OpenRouter-based LLM generation:
- Supports all 8 free models
- Configurable temperature
- Context-aware prompting

```python
generator = Generator()
answer = generator.generate(query, context, model="gpt-5-chat")
```

### 8. **RAGEvaluator** (`rag_agent/tools/evaluation.py`)

Calculates metrics:
- Precision, recall, MRR, NDCG
- Timing breakdown
- Query history tracking

```python
evaluator = RAGEvaluator()
metrics = evaluator.calculate_metrics(retrieved, relevant)
```

### 9. **UMAPVisualizer** (`rag_agent/tools/umap.py`)

3D embedding visualization:
- UMAP dimensionality reduction (3D)
- Interactive Plotly Scatter3D
- Color-coded: docs (blue), queries (red/orange), retrieved (dark blue)

```python
visualizer = UMAPVisualizer()
fig = visualizer.plot_umap_with_queries(
    collection="my_docs",
    original_query=query,
    enhanced_queries_list=enhanced,
    retrieved_docs=retrieved
)
```

---

## 📊 Evaluation Metrics

| Metric | Formula | Target | Description |
|--------|---------|--------|-------------|
| **Precision** | `Relevant ∩ Retrieved / Retrieved` | >0.70 | Accuracy of retrieval |
| **Recall** | `Relevant ∩ Retrieved / Relevant` | >0.80 | Coverage of relevant docs |
| **Avg Distance** | `Σ distances / n` | <1.0 | Semantic similarity |
| **Retrieval Time** | Vector search latency | <2s | Search speed |
| **Rerank Time** | Cross-encoder scoring | <5s | Reranking speed |
| **Generation Time** | LLM inference | <5s | Answer generation speed |
| **Total Time** | End-to-end latency | <12s | User experience |

Metrics are:
- Displayed per query in the UI
- Saved to query history
- Exportable to CSV
- Aggregated in dashboard

---

## 📁 Project Structure

```
RAG/
├── app1.py                         # Streamlit main application
├── config.py                       # Configuration settings
├── main.py                         # CLI example script
├── rag_pipeline_linear.py          # Linear pipeline (no functions)
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create this)
├── .gitignore                      # Git ignore file
│
├── rag_agent/
│   └── tools/
│       ├── __init__.py
│       ├── document_loader.py      # PDF/DOCX/TXT extraction
│       ├── chunker.py              # Text chunking (LangChain)
│       ├── embedder.py             # Vector embeddings (SentenceTransformer)
│       ├── retriever.py            # ChromaDB retrieval
│       ├── reranker.py             # Cross-encoder reranking
│       ├── generator.py            # OpenRouter LLM generation
│       ├── evaluation.py           # Metrics calculation
│       └── umap.py                 # 3D UMAP visualization
│
├── query_enhancer/
│   └── tools/
│       └── query_agent.py          # Query enhancement (Gemini)
│
├── data/
│   ├── raw/                        # Upload your documents here
│   └── processed/                  # Processed documents
│
├── vector_db/                      # ChromaDB persistent storage
│
└── README.md                       # This file
```

---

## 📦 Requirements

### Python Version
- Python 3.10 or higher

### Dependencies (`requirements.txt`)

```txt
# Core
streamlit>=1.28.0
python-dotenv>=1.0.0
pandas>=2.0.0
numpy>=1.24.0

# Document Processing
pypdf>=3.17.0
python-docx>=1.1.0
llama-parse>=0.4.0

# Text Processing
langchain-text-splitters>=0.0.1

# Embeddings & Retrieval
sentence-transformers>=2.2.0
chromadb>=0.4.0

# Visualization
umap-learn>=0.5.0
plotly>=5.17.0

# LLM APIs
openai>=1.0.0
google-genai>=0.3.0

# Utilities
scikit-learn>=1.3.0
```

### Install All Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **OpenRouter API Error** | Verify `OPENROUTER_API_KEY` in `.env` file |
| **Gemini API Error** | Verify `GEMINI_API_KEY` in `.env` file |
| **ChromaDB Permission Error** | Set `HF_HUB_DISABLE_SYMLINKS=1` environment variable |
| **UMAP Error** | Ensure data has >5 chunks; install `umap-learn` |
| **Out of Memory** | Reduce `BATCH_SIZE` in `config.py` |
| **Model Not Found** | Verify model ID matches OpenRouter exactly |
| **Import Errors** | Check virtual environment is activated |
| **Port Already in Use** | Change port: `streamlit run app.py --server.port 8502` |

### Debug Mode

Enable verbose logging:

```python
# In config.py
DEBUG = True
```

### Check Installation

```bash
python -c "import chromadb, sentence_transformers, umap; print('✅ All core packages installed')"
```

---

## 🎓 How It Works

### Query Enhancement Example

**Input Query:**
```
"What is RAG?"
```

**Gork Enhancement Output:**
```json
{
  "sub_queries": [
    "What does RAG stand for in NLP?",
    "How does RAG improve language model responses?",
    "What are the key components of RAG architecture?"
  ],
  "hyde": "RAG, or Retrieval-Augmented Generation, is a technique that combines information retrieval with language generation. It retrieves relevant documents from a knowledge base and uses them to enhance the context for generating accurate responses.",
  "step_back": "What are the general approaches to improving LLM accuracy and reducing hallucinations?"
}
```

**Embedding & Retrieval:**
```
1. Embed all 6 queries
2. Search ChromaDB for each query
3. Aggregate results (max similarity)
4. Filter by distance threshold (<1.5)
5. Rerank top results with CrossEncoder
6. Generate answer with selected LLM
```

---


## 🛣️ Roadmap

- [ ] Support for more document formats (Excel, CSV, HTML)
- [ ] Multi-document collections
- [ ] Advanced filtering (metadata, date range)
- [ ] Conversation memory
- [ ] Custom model fine-tuning
- [ ] API endpoint deployment
- [ ] Docker containerization
- [ ] Kubernetes deployment

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **[OpenRouter](https://openrouter.ai)** - access to 8+ AI models
- **[ChromaDB](https://www.trychroma.com/)** - Vector database
- **[Sentence-Transformers](https://www.sbert.net/)** - Embeddings
- **[LangChain](https://www.langchain.com/)** - Text splitting
- **[Streamlit](https://streamlit.io/)** - UI framework
- **[Plotly](https://plotly.com/)** - 3D visualization
- **[UMAP](https://umap-learn.readthedocs.io/)** - Dimensionality reduction

---

## 📧 Contact

For questions or support:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/query_enhancer_rag_with_metrics/issues)
- **Name**: Vishva MV
- **Email**: dev.vishvamv.com

---
