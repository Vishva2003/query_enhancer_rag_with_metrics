import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os
import tempfile as temp
import pandas as pd
import time
from datetime import datetime
from config import free_models

load_dotenv()


@st.cache_resource()
def get_rag_tools():
    from rag_agent.tools.document_loader import DocumentLoader
    from rag_agent.tools.chunker import Chunker
    from rag_agent.tools.embedder import Embedder
    from rag_agent.tools.retriever import Retriever
    from rag_agent.tools.reranker import Reranker
    from rag_agent.tools.generator import Generator
    from query_enhancer.tools.query_agent import QueryEnhancer
    from rag_agent.tools.evaluation import RAGEvaluator 

    loader = DocumentLoader()
    chunker = Chunker()
    embedder = Embedder()
    queryEnhancer = QueryEnhancer()
    retriever = Retriever()
    reranker = Reranker()
    generator = Generator()
    evaluator = RAGEvaluator()
    return loader, chunker, embedder, queryEnhancer, retriever, reranker, generator, evaluator


if "collection" not in st.session_state:
    st.session_state.collection = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Chatgpt-5"
    
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_history" not in st.session_state:
    st.session_state.query_history = []  # Track all queries and their metrics

if "metrics_enabled" not in st.session_state:
    st.session_state.metrics_enabled = True

st.set_page_config(page_title="Rag System", layout="wide")
st.title("RAG System")

with st.sidebar:
    st.header("Document Loader")
    upload_file = st.file_uploader("Upload a document [txt, pdf, docx, doc]", type=["pdf", "txt", "docx", "doc"])
    chunk_size = st.slider("Chunk size (Characters)", 1000,2500, 1000, step=15)
    overlap_size = st.slider("Chunk overlap (characters)", 100, 500, 200, step=10)
    top_k_results = st.slider("Top K results to retrieve", 1, 40, 10)
    st.session_state.collection = st.text_input("Collection name for the document", placeholder="Sample_collection")
    
    selected_model = st.selectbox(
        "Choose a free model:",
        options=free_models,
        index=0,
        format_func=lambda x: x.replace("-", "_").title()
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.rerun()
    
    st.divider()
    
    # Metrics toggle
    st.session_state.metrics_enabled = st.checkbox("📊 Enable RAG Metrics", value=True)
    
    # Show metrics summary if available
    if st.session_state.query_history and st.button("📈 Show Metrics Dashboard"):
        st.session_state.show_metrics = True
    else:
        st.session_state.show_metrics = False
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        start_processing = st.button("Start", type="primary", use_container_width=True)
    
    with col2:
        if st.button("Restart", type="secondary", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show model info for assistant messages
        if message["role"] == "assistant" and "model_used" in message:
            st.caption(f"🤖 Model: {message['model_used']}")


if start_processing:

    if not upload_file:
        st.sidebar.warning("Please upload a document.")
        st.stop()

    if not st.session_state.collection:
        st.sidebar.warning("Please enter a collection name.")
        st.stop()

    with temp.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(upload_file.getbuffer())
        temp_path = Path(f.name)

    with st.status("Processing document...", expanded=True) as status:
        try:
            loader, chunker, embedder, _, _, _, _, _ = get_rag_tools()

            text = loader.load_file(str(temp_path))

            chunk_list = chunker.chunk_text(text, chunk_size, overlap_size)

            embedder = embedder.add_collection(
                chunk_list,
                collection=st.session_state.collection
            )

            st.session_state["doc_processed"] = True
            st.session_state.total_chunks = len(chunk_list)

            status.update(
                label="Document processed successfully!",
                state='complete',
                expanded=False
            )

            st.success("Ready to answer questions about the document!")

            if chunk_list:
                with st.expander("Document Chunks"):
                    for i, chunk in enumerate(chunk_list[:3]):
                        st.markdown(f"**Chunk {i+1}:** {chunk[:200]}...")

        except Exception as e:
            status.update(
                label=f"Error processing document: {e}",
                state='error',
                expanded=False
            )

        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except:
                pass

if prompt := st.chat_input("Ask a question about the document..."):

    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner("generating answer..."):
            try:
                start_time = time.time()
                
                _, _, embedder, queryEnhancer, retriever, reranker, generator, evaluator = get_rag_tools()
                print(f"Retrieving with top_k={top_k_results}...")
                
                # Get model info
                current_model_key = selected_model.replace("_", "-")
                model_id = free_models[current_model_key]
                
                # Step 1: Enhance query
                enhanced_query = queryEnhancer.enhance(prompt)
                
                # Step 2: Retrieve documents
                retrieve_start = time.time()
                retrieved_data = retriever.retrieve_multi(enhanced_query, st.session_state.collection, top_k=top_k_results)
                retrieve_time = time.time() - retrieve_start
                
                filtered_docs = []
                filtered_distances = []
                for doc, dist in zip(retrieved_data['documents'], retrieved_data['distances']):
                    if dist < 1.0:  # Stricter threshold (adjust based on your data)
                        filtered_docs.append(doc)
                        filtered_distances.append(dist)
                print(f"Retrieved {len(filtered_docs)} documents")
                
                # Step 3: Rerank documents
                rerank_start = time.time()
                reranked_docs = reranker.rerank(prompt, filtered_docs, top_k=5)
                rerank_time = time.time() - rerank_start

                if not filtered_docs:
                    answer = "No relevant information found in the document."
                    
                    # Store metrics even for empty results
                    metrics = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'query': prompt,
                        'enhanced_query': enhanced_query,
                        'model_used': current_model_key,
                        'model_id': model_id,
                        'retrieved_count': 0,
                        'avg_distance': 0,
                        'retrieve_time': retrieve_time,
                        'rerank_time': rerank_time,
                        'total_time': time.time() - start_time,
                        'context_precision': 0,
                        'context_recall': 0,
                        'answer': answer
                    }
                else:
                    # Calculate metrics
                    context = reranker.format_context(reranked_docs)
                    
                    # Calculate average distance (lower is better)
                    avg_distance = sum(filtered_distances[:len(reranked_docs)]) / len(reranked_docs) if filtered_distances else 0
                    
                    # Generate answer
                    generate_start = time.time()
                    answer = generator.generate(
                        prompt,
                        context,
                        model=model_id,
                        temperature=0.3
                    )
                    generate_time = time.time() - generate_start
                    
                    # Calculate retrieval metrics
                    context_precision = 1.0 - (avg_distance / 2) if avg_distance else 0.5  # Normalized score
                    context_recall = min(1.0, len(reranked_docs) / top_k_results) if top_k_results > 0 else 0
                    
                    with st.expander("Retrieved context"):
                        for i, doc in enumerate(reranked_docs[:5]):
                            st.markdown(f"**Doc {i+1}:** {doc[:200]}...")
                    
                    # Store metrics
                    metrics = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'query': prompt,
                        'enhanced_query': enhanced_query,
                        'model_used': current_model_key,
                        'model_id': model_id,
                        'retrieved_count': len(filtered_docs),
                        'reranked_count': len(reranked_docs),
                        'avg_distance': round(avg_distance, 4),
                        'context_precision': round(context_precision, 4),
                        'context_recall': round(context_recall, 4),
                        'retrieve_time': round(retrieve_time, 3),
                        'rerank_time': round(rerank_time, 3),
                        'generate_time': round(generate_time, 3),
                        'total_time': round(time.time() - start_time, 3),
                        'answer': answer[:100] + "..."  # Preview
                    }
                    
                    # Show metrics if enabled
                    if st.session_state.metrics_enabled:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Precision", f"{context_precision:.2f}")
                        with col2:
                            st.metric("Recall", f"{context_recall:.2f}")
                        with col3:
                            st.metric("Docs", len(reranked_docs))
                        with col4:
                            st.metric("Time", f"{metrics['total_time']:.1f}s")
                        
                        # Show detailed metrics in expander
                        with st.expander("📊 Detailed Metrics"):
                            st.json({
                                "Model": current_model_key,
                                "Retrieval Time": f"{retrieve_time:.2f}s",
                                "Rerank Time": f"{rerank_time:.2f}s", 
                                "Generation Time": f"{generate_time:.2f}s",
                                "Avg Distance": f"{avg_distance:.4f}",
                                "Retrieved Docs": len(filtered_docs),
                                "Reranked Docs": len(reranked_docs)
                            })

                st.markdown(answer)
                
                # Add model info to answer
                st.caption(f"🤖 Generated by: {current_model_key}")
                
                # Add to query history
                st.session_state.query_history.append(metrics)

            except Exception as e:
                st.error(f"Error initializing RAG tools: {e}")
                answer = f"Error: {e}"
                st.markdown(answer)
                
                # Store error in history
                st.session_state.query_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'query': prompt,
                    'model_used': current_model_key if 'current_model_key' in locals() else 'unknown',
                    'error': str(e),
                    'answer': answer
                })

    # Add assistant message to history with model info
    st.session_state.messages.append({
        'role': 'assistant', 
        'content': answer,
        'model_used': current_model_key if 'current_model_key' in locals() else selected_model
    })

# Metrics Dashboard
if st.session_state.get('show_metrics', False) and st.session_state.query_history:
    with st.expander("📊 RAG Metrics Dashboard", expanded=True):
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.query_history)
        
        # Show summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", len(df))
        with col2:
            avg_time = df['total_time'].mean() if 'total_time' in df.columns else 0
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        with col3:
            avg_precision = df['context_precision'].mean() if 'context_precision' in df.columns else 0
            st.metric("Avg Precision", f"{avg_precision:.2f}")
        with col4:
            avg_recall = df['context_recall'].mean() if 'context_recall' in df.columns else 0
            st.metric("Avg Recall", f"{avg_recall:.2f}")
        
        # Model performance breakdown
        if 'model_used' in df.columns:
            st.subheader("🤖 Model Performance")
            model_stats = df.groupby('model_used').agg({
                'total_time': 'mean',
                'context_precision': 'mean',
                'context_recall': 'mean',
                'query': 'count'
            }).round(3)
            st.dataframe(model_stats)
        
        # Query history table
        st.subheader("📋 Query History")
        display_cols = ['timestamp', 'query', 'model_used', 'retrieved_count', 
                    'context_precision', 'context_recall', 'total_time']
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols])
        
        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics CSV",
            data=csv,
            file_name=f"rag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    if os.getenv("OPENROUTER_API_KEY"):
        st.success("✅ OpenRouter Connected")
with col2:
    if st.session_state.collection:
        st.success(f"✅ Document: {st.session_state.collection}")
    else:
        st.info("📄 No Document")
with col3:
    st.info(f"🤖 Model: {st.session_state.selected_model}")
    
# Show query count in footer
if st.session_state.query_history:
    st.caption(f"📊 Tracked {len(st.session_state.query_history)} queries with metrics")