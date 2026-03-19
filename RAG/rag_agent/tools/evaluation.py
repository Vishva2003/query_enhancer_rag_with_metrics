# rag_agent/tools/evaluation.py
import numpy as np
from typing import List, Dict, Any, Optional

class RAGEvaluator:
    """
    Simple RAG metrics calculator
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate precision@k
        """
        if not retrieved_docs:
            return 0.0
        relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
        return len(relevant_retrieved) / len(retrieved_docs)
    
    def calculate_recall(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate recall@k
        """
        if not relevant_docs:
            return 0.0
        relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
        return len(relevant_retrieved) / len(relevant_docs)
    
    def calculate_mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank
        """
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 10) -> float:
        """
        Simplified NDCG calculation
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            if doc in relevant_docs:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def log_metrics(self, query: str, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for a query
        """
        self.metrics_history.append({
            'query': query,
            **metrics
        })
    
    def get_summary_stats(self) -> Dict[str, float]:
        """
        Get summary statistics of all logged metrics
        """
        if not self.metrics_history:
            return {}
        
        summary = {}
        metrics_keys = ['precision', 'recall', 'mrr', 'ndcg', 'retrieval_time', 'total_time']
        
        for key in metrics_keys:
            values = [m.get(key, 0) for m in self.metrics_history if key in m]
            if values:
                summary[f'avg_{key}'] = np.mean(values)
                summary[f'min_{key}'] = np.min(values)
                summary[f'max_{key}'] = np.max(values)
        
        return summary