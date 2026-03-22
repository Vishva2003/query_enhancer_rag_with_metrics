import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH


class UMAPVisualizer:
    """
    UMAP visualizer for RAG document embeddings with query visualization
    """
    
    def __init__(self, embedder_model=EMBEDDING_MODEL):
        self.model = SentenceTransformer(embedder_model)
    
    def get_collection_embeddings(self, collection_name: str):
        """Retrieves embeddings, documents, and metadata from Chroma"""
        try:
            client = chromadb.PersistentClient(
                path=str(VECTOR_DB_PATH),
                settings=Settings(anonymized_telemetry=False)
            )
            collection = client.get_collection(name=collection_name)
            data = collection.get(include=["embeddings", "documents", "metadatas"])
            
            embeddings = np.array(data["embeddings"]) if data["embeddings"] else None
            documents = data["documents"]
            metadatas = data["metadatas"]
            
            return embeddings, documents, metadatas
            
        except Exception as e:
            print(f"UMAP Error accessing collection '{collection_name}': {e}")
            return None, None, None
    
    def create_enhanced_queries_dict(self, original_query: str, enhanced_queries_list: list):
        """
        Convert the list of enhanced queries from QueryEnhancer to a dictionary with types
        
        Args:
            original_query: The original user query
            enhanced_queries_list: List from QueryEnhancer.enhance()
                Format: [original, sub_query_1, sub_query_2, sub_query_3, hyde, step_back]
        """
        enhanced_dict = {}
        
        # Determine which components are present
        has_subqueries = len(enhanced_queries_list) > 1
        has_hyde = False
        has_stepback = False
        
        # Identify the type of each query
        for i, query_text in enumerate(enhanced_queries_list):
            if i == 0:
                enhanced_dict['original'] = {
                    'text': query_text,
                    'type': 'original'
                }
            elif i <= 3 and has_subqueries:
                enhanced_dict[f'sub_query_{i}'] = {
                    'text': query_text,
                    'type': 'sub_query'
                }
            elif 'HyDE' in query_text or i == 4 and 'retrieval' in query_text.lower():
                enhanced_dict['hyde'] = {
                    'text': query_text,
                    'type': 'hyde'
                }
                has_hyde = True
            else:
                enhanced_dict['step_back'] = {
                    'text': query_text,
                    'type': 'step_back'
                }
                has_stepback = True
        
        return enhanced_dict
    
    def get_embeddings_for_queries(self, queries_dict: dict):
        """Generate embeddings for all queries in the dictionary"""
        for key, data in queries_dict.items():
            embedding = self.model.encode([data['text']])[0]
            queries_dict[key]['embedding'] = embedding
        return queries_dict
    
    def get_retrieved_doc_embeddings(self, retrieved_docs, collection_embeddings, collection_docs):
        """Get embeddings for retrieved documents"""
        retrieved_embeddings = []
        retrieved_texts = []
        retrieved_indices = []
        
        for retrieved_doc in retrieved_docs:
            # Find matching document in collection
            for i, doc in enumerate(collection_docs):
                if doc == retrieved_doc:
                    retrieved_embeddings.append(collection_embeddings[i])
                    retrieved_texts.append(doc[:100])
                    retrieved_indices.append(i)
                    break
        
        return np.array(retrieved_embeddings) if retrieved_embeddings else None, retrieved_texts
    
    def plot_umap_with_queries(
        self,
        collection_name: str,
        original_query: str,
        enhanced_queries_list: list,
        retrieved_docs: list = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        max_docs: int = 500
    ):
        """
        Plot UMAP with document embeddings and query points
        
        Args:
            collection_name: Chroma collection name
            original_query: Original user query
            enhanced_queries_list: List from QueryEnhancer.enhance()
            retrieved_docs: List of retrieved document texts
        """
        
        # Get collection embeddings
        doc_embeddings, doc_texts, _ = self.get_collection_embeddings(collection_name)
        
        if doc_embeddings is None or len(doc_embeddings) < 5:
            return None, f"Not enough chunks in collection (need at least 5)"
        
        # Sample if too many documents
        if len(doc_embeddings) > max_docs:
            indices = np.random.choice(len(doc_embeddings), max_docs, replace=False)
            doc_embeddings = doc_embeddings[indices]
            doc_texts = [doc_texts[i] for i in indices]
        
        # Create enhanced queries dictionary
        queries_dict = self.create_enhanced_queries_dict(original_query, enhanced_queries_list)
        queries_dict = self.get_embeddings_for_queries(queries_dict)
        
        # Get embeddings for all queries
        query_embeddings = []
        query_labels = []
        query_types = []
        query_texts = []
        
        for name, data in queries_dict.items():
            query_embeddings.append(data['embedding'])
            # Format label nicely
            if name == 'original':
                label = 'Original Query'
            elif name.startswith('sub_query'):
                label = f"Sub-Query {name.split('_')[-1]}"
            elif name == 'hyde':
                label = 'HyDE (Hypothetical)'
            elif name == 'step_back':
                label = 'Step-Back Query'
            else:
                label = name.replace('_', ' ').title()
            
            query_labels.append(label)
            query_types.append(data['type'])
            query_texts.append(data['text'][:100])
        
        query_embeddings = np.array(query_embeddings)
        
        # Get retrieved document embeddings if provided
        retrieved_embeddings = None
        retrieved_texts = []
        if retrieved_docs:
            retrieved_embeddings, retrieved_texts = self.get_retrieved_doc_embeddings(
                retrieved_docs, doc_embeddings, doc_texts
            )
        
        # Combine all embeddings for UMAP fitting
        all_embeddings_list = [doc_embeddings, query_embeddings]
        if retrieved_embeddings is not None and len(retrieved_embeddings) > 0:
            all_embeddings_list.append(retrieved_embeddings)
        
        all_combined = np.vstack(all_embeddings_list)
        
        # UMAP reduction
        import umap as umap_lib
        reducer = umap_lib.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            metric="cosine"
        )
        all_2d = reducer.fit_transform(all_combined)
        
        # Split back
        doc_2d = all_2d[:len(doc_embeddings)]
        query_2d = all_2d[len(doc_embeddings):len(doc_embeddings) + len(query_embeddings)]
        
        offset = len(doc_embeddings) + len(query_embeddings)
        if retrieved_embeddings is not None and len(retrieved_embeddings) > 0:
            retrieved_2d = all_2d[offset:offset + len(retrieved_embeddings)]
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add document chunks (background)
        fig.add_trace(go.Scatter(
            x=doc_2d[:, 0], y=doc_2d[:, 1],
            mode='markers',
            marker=dict(size=6, color='lightblue', opacity=0.5, symbol='circle'),
            text=[text[:100] + "..." for text in doc_texts],
            hoverinfo='text',
            name=f'Document Chunks ({len(doc_2d)})'
        ))
        
        # Color mapping for query types
        color_map = {
            'original': '#FF0000',  # Red
            'sub_query': '#FF8C00',  # Orange
            'hyde': '#2E8B57',  # Sea Green
            'step_back': '#9370DB'  # Medium Purple
        }
        
        # Add queries
        for i, (name, data) in enumerate(queries_dict.items()):
            color = color_map.get(data['type'], 'gray')
            marker_size = 18 if data['type'] == 'original' else 12
            marker_symbol = 'star' if data['type'] == 'original' else 'circle'
            
            fig.add_trace(go.Scatter(
                x=[query_2d[i, 0]], y=[query_2d[i, 1]],
                mode='markers+text',
                marker=dict(size=marker_size, color=color, symbol=marker_symbol, line=dict(width=1, color='black')),
                text=query_labels[i],
                textposition='top center',
                hoverinfo='text',
                hovertext=f"<b>{query_labels[i]}</b><br>{data['text'][:150]}...",
                name=query_labels[i],
                showlegend=True
            ))
        
        # Add retrieved documents
        if retrieved_embeddings is not None and len(retrieved_embeddings) > 0:
            fig.add_trace(go.Scatter(
                x=retrieved_2d[:, 0], y=retrieved_2d[:, 1],
                mode='markers',
                marker=dict(size=12, color='#1E88E5', symbol='diamond', line=dict(width=1, color='black')),
                text=retrieved_texts,
                hoverinfo='text',
                name=f'Retrieved Docs ({len(retrieved_2d)})',
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title=f"RAG Embedding Space: '{original_query[:50]}...'",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            hovermode="closest",
            template="plotly_white",
            height=650,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            )
        )
        
        return fig, None


def create_query_visualization(
    collection_name: str,
    original_query: str,
    enhanced_queries_list: list,
    retrieved_docs: list = None
):
    """Helper function to create UMAP visualization"""
    visualizer = UMAPVisualizer()
    return visualizer.plot_umap_with_queries(
        collection_name=collection_name,
        original_query=original_query,
        enhanced_queries_list=enhanced_queries_list,
        retrieved_docs=retrieved_docs
    )