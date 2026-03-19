import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os
import tempfile as temp
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

    loader = DocumentLoader()
    chunker = Chunker()
    embedder = Embedder()
    queryEnhancer = QueryEnhancer()
    retriever = Retriever()
    reranker = Reranker()
    generator = Generator()
    return loader, chunker, embedder, queryEnhancer, retriever, reranker, generator


if "collection" not in st.session_state:
    st.session_state.collection = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Chatgpt-5"
    
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

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
    

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

start_processing = st.sidebar.button("🚀 Start Processing")

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
            loader, chunker, embedder, _, _, _, _ = get_rag_tools()

            text = loader.load_file(str(temp_path))

            chunk_list = chunker.chunk_text(text, chunk_size, overlap_size)

            embedder = embedder.add_collection(
                chunk_list,
                collection=st.session_state.collection
            )

            st.session_state["doc_processed"] = True

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
                _, _, embedder, queryEnhancer, retriever, reranker, generator = get_rag_tools()
                print(f"Retrieving with top_k={top_k_results}...")
                enhanced_query = queryEnhancer.enhance(prompt)
                retrieved_data = retriever.retrieve_multi(enhanced_query, st.session_state.collection, top_k=top_k_results)
                retrieved_docs = retrieved_data['documents']
                print(len(retrieved_docs))
                reranked_docs = reranker.rerank(prompt, retrieved_docs, top_k=10)
                current_model = selected_model.replace("_", "-")

                if not retrieved_docs:
                    answer = "No relevant information found in the document."
                else:
                    context = reranker.format_context(reranked_docs)
                    answer = generator.generate(
                        prompt,
                        context,
                        model=free_models[current_model],
                        temperature=0.3
                        )

                    with st.expander("Retrieved context"):
                        for i, doc in enumerate(reranked_docs):
                            st.markdown(f"**Doc {i+1}:** {doc[:200]}...")
                            #print(f"Retrieved Doc {i+1}: {doc[:10]}.../n")

                st.markdown(answer)

            except Exception as e:
                st.error(f"Error initializing RAG tools: {e}")
                st.stop()

    st.session_state.messages.append({'role': 'assistant', 'content': answer})

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