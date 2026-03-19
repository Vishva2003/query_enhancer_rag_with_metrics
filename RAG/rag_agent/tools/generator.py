from openai import OpenAI
import os
import sys
from pathlib import Path

try:
    from config import OPENROUTER_API_KEY, OPENROUTER_SITE_NAME, OPENROUTER_BASE_URL, OPENROUTER_SITE_URL, free_models
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import OPENROUTER_API_KEY, OPENROUTER_SITE_NAME, OPENROUTER_BASE_URL, OPENROUTER_SITE_URL, free_models



class Generator:
    def __init__(self, openrouter_api_key=None, site_url=None, site_name=None):

        self.api_key = OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Get one at https://openrouter.ai")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )
        
        # Optional headers for rankings
        self.site_url = OPENROUTER_SITE_URL
        self.site_name = OPENROUTER_SITE_NAME
        
        # Free models available on OpenRouter
        self.free_models = free_models
        
        # Set default model
        self.default_model = self.free_models["Gemini-2.5-Flash-lite"]
    
    def generate(self, query, context, model=None, temperature=0.7, max_tokens=1000, top_p=1):
        """
        Generate response using OpenRouter free models with OpenAI-compatible structure
        
        Args:
            query (str): The user's question
            context (str): Retrieved context from documents
            model (str): Model ID to use (defaults to llama-3.2-3b)
            temperature (float): Controls randomness (0-2)
            max_tokens (int): Maximum tokens in response
            top_p (float): Nucleus sampling parameter
        
        Returns:
            str: Generated answer
        """
        
        # Use specified model or default
        model_id = model or self.default_model
        
        # Create the prompt (keeping your exact prompt structure)
        prompt = f"""
        You are an expert AI assistant answering questions using retrieved documents.

        You MUST follow these rules:
        1. Use ONLY the information in the provided context.
        2. Do NOT use outside knowledge.
        3. If the answer is not fully supported by the context, say:
        "I don't have enough information to answer that question."
        4. If multiple documents disagree, report all viewpoints.

        ----------------------
        CONTEXT:
        {context}
        ----------------------

        QUESTION:
        {query}

        INSTRUCTIONS:
        - Read the context carefully.
        - Identify the most relevant parts.
        - Synthesize a clear, factual answer.
        - Quote or paraphrase directly from the context.
        - Do not guess or assume anything.

        ANSWER:
        """
        
        try:
            # Create completion using OpenAI-compatible structure
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"Error generating response: {str(e)}"


if __name__ == "__main__":

    from document_loader import DocumentLoader
    from retriever import Retriever
    from chunker import Chunker
    from embedder import Embedder
    from reranker import Reranker

    loader = DocumentLoader()
    chunker = Chunker()
    embedder = Embedder()
    retriever = Retriever()
    reranker = Reranker()

    query = "what is this document about?"
    collection_name = "test_collection"
    retrieved_data = retriever.retrieve(query, collection_name, 10)
    documents = retrieved_data['documents']
    print(f"Retrieved {len(documents)} documents.")
    reranked_docs = reranker.rerank(query, documents, top_k=10)
    print("Top 10 reranked documents:")
    for doc, score in reranked_docs:
        print(f"Score {score:.4f}: {doc[:100]}...")
    formatted_context = reranker.format_context(reranked_docs)
    print("\nFormatted Context:\n", formatted_context)

    generator = Generator()

    response = generator.generate(
        query,
        formatted_context,
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        temperature=0.3
    )
    print("\nResponse with Llama 3:", response)
