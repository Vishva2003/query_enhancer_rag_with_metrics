import sys
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

try:
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, DATA_RAW, BATCH_SIZE
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, DATA_RAW, BATCH_SIZE


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        self.batch_size = BATCH_SIZE
        self.collection = None

    def create_collection(self, collection_name):
        try:
            if self.collection is None:
                print("Please Create a collection")
            else:
                self.collection = self.client.get_collection(collection_name)
                print(f'Using existing collection: {collection_name}')
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f'Created new collection: {collection_name}')

    def embed_batch(self, chunks):   # 100 chunks example
        all_embeddings = []
        total_batch = (len(chunks) + self.batch_size - 1) // self.batch_size # 4 batches
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batch} ({len(batch)} chunks)")
            
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True
            )
            
            all_embeddings.extend(batch_embeddings)
            
            if batch_num < total_batch:
                time.sleep(0.1)

        return all_embeddings

    def add_collection(self, chunks, collection, ids=None, metadatas=None):
        self.collection = collection
        
        total_batch = (len(chunks) + self.batch_size - 1) // self.batch_size 

        embeddings= self.embed_batch(chunks)

        self.create_collection(self.collection)

        if metadatas is None:
            metadatas = [{"index": chunk[:50]} for chunk in chunks]

        for i in range(0, len(chunks), self.batch_size):
            batch_end = min(i + self.batch_size, len(chunks))
            batch_num = i // self.batch_size + 1
            
            print(f"Adding batch {batch_num}/{total_batch} to ChromaDB")        
            
            batch_chunks = chunks[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_ids = [f"{collection}_chunk_{j}" for j in range(i, batch_end)]
            batch_metadata = metadatas[i:batch_end]
            
            # Add to collection
            self.collection.add(
                documents=batch_chunks,
                embeddings=batch_embeddings,
                ids=batch_ids,
                metadatas=batch_metadata
            )
        print(f"Added {len(chunks)} documents to collection: {self.collection}")
        return self.collection


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunker import Chunker

    # Test embedding
    loader = DocumentLoader()
    chunker = Chunker()
    embedder = Embedder()

    file_path = DATA_RAW / "sample.pdf"

    if file_path.exists():
        text = loader.load_file(file_path)
        chunks = chunker.chunk_text(text, 1500, 50)
        embedder.add_collection(chunks, "collection")
