import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


try:
    # Try relative import first (when imported as module)
    from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_RAW
except ImportError:
    # Fall back to absolute import (when run directly)
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_RAW


class Chunker:
    def __init__(self):
        self.chunk_size = None
        self.overlap_size = None

    def chunk_text(self, text, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.overlap_size = chunk_overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            length_function=len
        )
        return splitter.split_text(text)


if __name__ == "__main__":
    from document_loader import DocumentLoader

    loader = DocumentLoader()
    chunker = Chunker()

    file_path = DATA_RAW / "sample.pdf"

    if file_path.exists():
        text = loader.load_file(str(file_path))
        chunks = chunker.chunk_text(text, 700, 100)

        print(f"Total chunks: {len(chunks)}")
        print(f"\nFirst chunk ({len(chunks[0])} chars):")
        print(chunks[0])
    else:
        print("PDF not found.")