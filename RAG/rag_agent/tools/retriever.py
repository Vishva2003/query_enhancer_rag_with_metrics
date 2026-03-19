import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

try:
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K_RESULTS, DATA_RAW
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K_RESULTS, DATA_RAW


class Retriever:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
                            path=str(VECTOR_DB_PATH),
                            settings=Settings(anonymized_telemetry=False)
                            )
        self.collection = None

    def retrieve(self, query, collection_name, top_k=None):

        try:
            self.collection = self.client.get_collection(collection_name)
            print(f'Connected to collection: {collection_name}')
        except:
            print(f"Collection {collection_name} not found.")
            return []

        query_embedding = self.model.encode([query]).tolist()
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        return {
            'documents': result['documents'][0],
            'metadatas': result['metadatas'][0] if 'metadatas' in result else None,
            'distances': result['distances'][0],
            'ids': result['ids'][0]
        }

    def retrieve_multi(self, queries, collection_name, top_k=None):

        try:
            self.collection = self.client.get_collection(collection_name)
            print(f'Connected to collection: {collection_name}')
        except:
            print(f"Collection {collection_name} not found.")
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

        if not top_k:
            top_k = TOP_K_RESULTS

        query_embeddings = self.model.encode(queries).tolist()

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )

        seen_docs = {}

        for q_idx in range(len(queries)):

            docs = results["documents"][q_idx]
            metas = results.get("metadatas", [[]])[q_idx]
            distances = results["distances"][q_idx]
            ids = results["ids"][q_idx]

            for doc, meta, dist, doc_id in zip(docs, metas, distances, ids):

                if doc not in seen_docs:
                    seen_docs[doc] = (meta, dist, doc_id)

                else:
                    _, existing_dist, _ = seen_docs[doc]

                    if dist < existing_dist:
                        seen_docs[doc] = (meta, dist, doc_id)

        sorted_items = sorted(seen_docs.items(), key=lambda x: x[1][1])

        documents, metadatas, distances, ids = [], [], [], []

        for doc, (meta, dist, doc_id) in sorted_items:
            documents.append(doc)
            metadatas.append(meta)
            distances.append(dist)
            ids.append(doc_id)

        print(f"[Retriever] {len(queries)} queries → {len(documents)} unique docs")

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids
        }

    def format_context(self, retrieved_data):
        formatted_context = []
        for i, (doc, dist) in enumerate(zip(retrieved_data["documents"], retrieved_data["distances"])):
            formatted_context.append(f"Document {i+1} (Distance: {dist:.2f}):\n{doc}\n")
        return '\n'.join(formatted_context)


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunker import Chunker
    from embedder import Embedder

    loader = DocumentLoader()
    chunker = Chunker()
    embedder = Embedder()
    retriever = Retriever()

    file_path = DATA_RAW / "sample.pdf"
    text = loader.load_file(file_path)
    chunks = chunker.chunk_text(text, 500, 50)

    collection_name = 'test_collection'
    embedder.add_collection(chunks, collection_name)

    # Test retrieval
    query = "what is bvae"
    retrieved_data = retriever.retrieve_multi(query, collection_name)
    formatted_context = retriever.format_context(retrieved_data)
    print(formatted_context)
