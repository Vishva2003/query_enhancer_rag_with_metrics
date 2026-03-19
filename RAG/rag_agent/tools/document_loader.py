from pypdf import PdfReader
from docx import Document
import sys
from pathlib import Path
from llama_parse import LlamaParse


try:
    # Try relative import first (when imported as module)
    from config import DATA_RAW, SUPPORTED_EXTENSIONS, LLAMA_API_KEY
except ImportError:
    # Fall back to absolute import (when run directly)
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import DATA_RAW, SUPPORTED_EXTENSIONS, LLAMA_API_KEY


class DocumentLoader:

    def __init__(self):

        self.llama_api = LLAMA_API_KEY
        self.llama_parser = LlamaParse(
                api_key=LLAMA_API_KEY,
                result_type="markdown",  # Output as markdown
                num_workers=4,  # Parallel processing
                verbose=True,
                language="en",
                extract_charts=True,
                auto_mode=True,
                auto_mode_trigger_on_image_in_page=True,
                auto_mode_trigger_on_table_in_page=True,
            )

    def text_processing(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def pdf_processing(self, file_path):
        try:
            return self._llamaparse_processing(file_path)
        except Exception as e:
            print(f"⚠️ LlamaParse failed: {e}, falling back to PyPDF")
            return self._pypdf_processing(file_path)

    def _pypdf_processing(self, file_path):
        """Fallback PyPDF processing"""
        reader = PdfReader(file_path)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    def _llamaparse_processing(self, file_path):
        """Advanced LlamaParse processing"""

        documents = self.llama_parser.load_data(file_path)
        # Combine all pages into one text
        full_text = ""
        for doc in documents:
            full_text += doc.text + "\n\n"

        return full_text

    def doc_processing(self, file_path):
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + '\n'
        return text

    def load_file(self, file_path):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = file_path.suffix.lower()

        if file_type not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{file_type}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        try:
            if file_type == '.pdf':
                return self.pdf_processing(file_path)
            elif file_type in {'.docx', '.doc'}:               # ← remove .doc or handle separately
                return self.doc_processing(file_path)
            elif file_type == '.txt':
                return self.text_processing(file_path)

        except Exception as e:
            raise RuntimeError(f"Failed to read content from {file_path.name}: {e}") from e


if __name__ == "__main__":

    loader = DocumentLoader()

    file = DATA_RAW / "sample.pdf"

    if file.exists():
        text = loader.load_file(file)
        print(f"Loaded {len(text)} characters from {file.name}")
