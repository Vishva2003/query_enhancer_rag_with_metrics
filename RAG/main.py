from rag_agent.tools.document_loader import DocumentLoader
from rag_agent.tools.chunker import Chunker
from rag_agent.tools.embedder import Embedder
from rag_agent.tools.retriever import Retriever
from rag_agent.tools.reranker import Reranker
from rag_agent.tools.generator import Generator
from query_enhancer.tools.query_agent import QueryEnhancer
from config import DATA_RAW, CHUNK_OVERLAP, CHUNK_SIZE, TOP_K_RESULTS


# use your document path here ('Beta_vae.pdf' is just an example, you can replace it with any document you want to test with)
file_path = DATA_RAW / "sample.pdf"

if __name__ == "__main__":

    loader = DocumentLoader()
    chunker = Chunker()
    embedder = Embedder()
    retriever = Retriever()
    reranker = Reranker()
    generator = Generator()
    enhancer = QueryEnhancer()

    collection_name = 'bvae_collection'  # you can change this to any name you like related to the document

    if file_path.exists():
        text = loader.load_file(str(file_path))

        chunk_list = chunker.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        embedder = embedder.add_collection(
                chunk_list,
                collection=collection_name
            )
        query = 'what is bvae'
        enhanced_query = enhancer.enhance(query)
        print(f'the sub queries are{enhanced_query[:3]}')
        retrieved_data = retriever.retrieve_multi(enhanced_query, collection_name, TOP_K_RESULTS)
        filtered_docs = []
        filtered_distances = []
        for doc, dist in zip(retrieved_data['documents'], retrieved_data['distances']):
            if dist < 1.0:  # Stricter threshold (adjust based on your data)
                filtered_docs.append(doc)
                filtered_distances.append(dist)
        reranked_docs = reranker.rerank(query, filtered_docs, top_k=10)
        context = reranker.format_context(reranked_docs)
        print(f"\nRetrieved Context:\n{context}")
        answer = generator.generate(query, context)
        print(f"\nGenerated Answer:\n{answer}")  
