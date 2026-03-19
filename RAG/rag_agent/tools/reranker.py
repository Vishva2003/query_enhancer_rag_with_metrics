import sys
from pathlib import Path
from sentence_transformers import CrossEncoder


try:
    from config import TOP_K_RESULTS, CROSS_ENCODER_MODEL
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import TOP_K_RESULTS, CROSS_ENCODER_MODEL


class Reranker:

    def __init__(self):
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)

    def rerank(self, query, documents, top_k=None):
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)

        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return ranked_docs[:top_k]
    
    def format_context(self, ranked_docs):
        formatted_context = []
        for i, (doc, score) in enumerate(ranked_docs):
            formatted_context.append(f"Document {i+1} (score: {score:.2f}):\n{doc}\n")
        return '\n'.join(formatted_context)


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from retriever import Retriever
    from chunker import Chunker
    from embedder import Embedder
    
    # Test reranker
    loader = DocumentLoader()
    chunker = Chunker()
    embedder = Embedder()
    retriever = Retriever()
    reranker = Reranker()

    query = "what is bvae"
    collection_name = "test_collection"
    retrieved_data = retriever.retrieve(query, collection_name, 10)
    documents = retrieved_data['documents']
    print(f"Retrieved {len(documents)} documents.")
    reranked_docs = reranker.rerank(query, documents, top_k=5)
    print("Top 10 reranked documents:")
    for doc, score in reranked_docs:
        print(f"Score {score:.4f}: {doc[:100]}...")
    formatted_context = reranker.format_context(reranked_docs)
    print("\nFormatted Context:\n", formatted_context)
