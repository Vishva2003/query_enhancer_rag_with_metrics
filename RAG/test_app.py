import streamlit as st
from dotenv import load_dotenv
import os
import tempfile as temp



from rag_agent.tools.document_loader import DocumentLoader
from rag_agent.tools.chunker import Chunker
from rag_agent.tools.embedder import Embedder
from rag_agent.tools.retriever import Retriever
from rag_agent.tools.reranker import Reranker
from rag_agent.tools.generator import Generator
from query_enhancer.tools.query_agent import QueryEnhancer
from config import DATA_RAW, CHUNK_OVERLAP, CHUNK_SIZE, TOP_K_RESULTS


load_dotenv()

@st.cache_resource(show_spinner="Loading RAG tools...")
def get_rag_tools():
    loader = DocumentLoader()
    chunker = Chunker()
    embedder = Embedder()
    retriever = Retriever()
    reranker = Reranker()
    generator = Generator()
    return loader, chunker, embedder, retriever, reranker, generator


st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .step-header {
        background-color: #000;
        color: #fff
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# initialisation
if "file" not in st.session_state:
    st.session_state.file = None

if "top_k" not in st.session_state:
    st.session_state.top_k = None

if "collection" not in st.session_state:
    st.session_state.collection = None
    
if "rag_setup" not in st.session_state:
    st.session_state.rag_setup = None


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


st.set_page_config(page_title="Rag System", layout="wide")

st.markdown('<h1 class="main-header">Rag Agent</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your personal LLM </p>', unsafe_allow_html=True)

with st.sidebar:
    step = st.radio(
                "Select a tool",
                ["Upload File", "RAG Configuration", "QA Chat", "UMAP analysis"],
                index=0
        )

    if st.session_state.file is not None:
        st.success(f"File '{st.session_state.file.name}' uploaded successfully!")

    if st.session_state.collection is not None:
        st.success("collection is now added")


if step == "Upload File":
    st.header("Document Loader")
    upload_file = st.file_uploader("Upload a document...", type=["pdf", "txt", "docx", "doc"])

    if upload_file is not None:
        st.session_state.file = upload_file


if step == "RAG Configuration":

    if st.session_state.file is None:
        st.warning("Please upload a document in the 'Upload File' step to configure RAG tools.")
    else:
        st.header("RAG Settings")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Chunking")
            chunk_size = st.slider("Chunk size", 1000, 3000, 1000, step=50)
            overlap_size = st.slider("Overlap", 100, 500, 200, step=25)
            
        with col2:
            st.subheader("Collection & Retrieval")
            collection_name = st.text_input("Collection name", value=st.session_state.collection)
            st.session_state.collection = collection_name

            top_k = st.slider("Top-K chunks", 3, 20, 8)
        
        col3, col4 = st.columns([2, 1])
        with col4:
            rag_setup = st.button("🚀 Start Processing")
            st.session_state.rag_setup = rag_setup
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
                    loader, chunker, embedder, _, _, _ = get_rag_tools()

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


if step == "QA Chat":
    if not st.session_state.file and not st.session_state.rag_setup:
        st.warning("Please process a document first.")
    else:
        st.header("Ask questions")

        prompt = st.chat_input("Your question...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching & generating..."):
                    try:
                        _, _, embedder, retriever, reranker, generator = get_rag_tools()

                        retrieved = retriever.retrieve(prompt, st.session_state.collection, top_k=8)
                        docs = retrieved.get("documents", [])

                        if not docs:
                            answer = "No relevant chunks found."
                        else:
                            reranked = reranker.rerank(prompt, docs, top_k=5)
                            context = reranker.format_context(reranked)  # assuming reranker has it
                            answer = generator.generate(prompt, context)

                        st.markdown(answer)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

if step == "UMAP analysis":
    if st.session_state.file is None:
        st.warning("Please upload a document in the 'Upload File' step to start asking questions.")
    else:
        pass
        
