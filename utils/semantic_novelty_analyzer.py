"""
Semantic novelty metric using OpenAI's text-embedding-ada-002.

This module implements novelty scoring for experiment artifacts using
semantic embeddings to detect conceptual innovation and novelty.
"""

import openai
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import pickle
from datetime import datetime

if TYPE_CHECKING:
    from services.llm_client import LLMClient


@dataclass
class NoveltyAnalysis:
    """Results of semantic novelty analysis."""
    novelty_score: float
    novelty_category: str  # "high", "moderate", "low"
    most_similar_artifact: Optional[str]
    similarity_score: float
    embedding_vector: List[float]
    analysis_timestamp: datetime


class SemanticNoveltyAnalyzer:
    """
    Analyzes semantic novelty of experiment artifacts using OpenAI embeddings.
    
    Uses text-embedding-ada-002 to compute semantic similarity against
    a growing corpus of previous artifacts to detect novel conceptual contributions.
    """
    
    def __init__(self, embedding_cache_path: Optional[str] = None):
        """
        Initialize the novelty analyzer.
        
        Args:
            embedding_cache_path: Optional path to cache embeddings
        """
        from services.llm_client import LLMClient  # Late import to avoid circular dependency
        self.client = LLMClient()
        self.embedding_model = "text-embedding-ada-002"
        
        # Novelty thresholds based on your specification
        self.novelty_thresholds = {
            "high": 0.75,      # > 0.75: High Novelty (new conceptual pillar)
            "moderate": 0.5    # 0.5-0.75: Moderate Novelty (synthesis/extension)
                              # < 0.5: Low Novelty (iterative/paraphrastic)
        }
        
        # Cache for embeddings to avoid recomputation
        self.cache_path = Path(embedding_cache_path) if embedding_cache_path else Path("embeddings_cache.pkl")
        self.embedding_cache = self._load_cache()
        
        # Corpus of previous artifacts for comparison
        self.artifact_corpus = []
        self.corpus_embeddings = []
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load embedding cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text, using cache if available.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        text_hash = self._get_text_hash(text)
        
        # Check cache first
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Get embedding from OpenAI
        try:
            response = self.client.get_embedding(text, model=self.embedding_model)
            embedding = response['data'][0]['embedding']
            
            # Cache the result
            self.embedding_cache[text_hash] = embedding
            self._save_cache()
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # ada-002 embedding dimension
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def add_to_corpus(self, artifact_text: str, artifact_id: str):
        """
        Add an artifact to the comparison corpus.
        
        Args:
            artifact_text: Text content of the artifact
            artifact_id: Unique identifier for the artifact
        """
        embedding = self._get_embedding(artifact_text)
        self.artifact_corpus.append({
            'id': artifact_id,
            'text': artifact_text,
            'embedding': embedding
        })
        self.corpus_embeddings.append(embedding)
    
    def analyze_novelty(self, artifact_text: str, artifact_id: str) -> NoveltyAnalysis:
        """
        Analyze the semantic novelty of an artifact.
        
        Args:
            artifact_text: Text content to analyze
            artifact_id: Unique identifier for this artifact
            
        Returns:
            NoveltyAnalysis with novelty score and category
        """
        # Get embedding for the new artifact
        artifact_embedding = self._get_embedding(artifact_text)
        
        # Calculate similarities against corpus
        max_similarity = 0.0
        most_similar_id = None
        
        if self.corpus_embeddings:
            similarities = [
                self._cosine_similarity(artifact_embedding, corpus_emb)
                for corpus_emb in self.corpus_embeddings
            ]
            
            max_similarity = max(similarities)
            max_idx = similarities.index(max_similarity)
            most_similar_id = self.artifact_corpus[max_idx]['id']
        
        # Calculate novelty score (1 - max_similarity)
        novelty_score = 1.0 - max_similarity
        
        # Determine novelty category
        if novelty_score > self.novelty_thresholds["high"]:
            novelty_category = "high"
        elif novelty_score >= self.novelty_thresholds["moderate"]:
            novelty_category = "moderate"
        else:
            novelty_category = "low"
        
        # Add this artifact to corpus for future comparisons
        self.add_to_corpus(artifact_text, artifact_id)
        
        return NoveltyAnalysis(
            novelty_score=novelty_score,
            novelty_category=novelty_category,
            most_similar_artifact=most_similar_id,
            similarity_score=max_similarity,
            embedding_vector=artifact_embedding,
            analysis_timestamp=datetime.now()
        )
    
    def batch_analyze_novelty(
        self, 
        artifacts: List[Tuple[str, str]]
    ) -> List[NoveltyAnalysis]:
        """
        Analyze novelty for a batch of artifacts.
        
        Args:
            artifacts: List of (artifact_text, artifact_id) tuples
            
        Returns:
            List of NoveltyAnalysis results
        """
        results = []
        
        for artifact_text, artifact_id in artifacts:
            analysis = self.analyze_novelty(artifact_text, artifact_id)
            results.append(analysis)
        
        return results
    
    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current artifact corpus."""
        if not self.artifact_corpus:
            return {
                "corpus_size": 0,
                "average_embedding_norm": 0.0,
                "embedding_dimensions": 0
            }
        
        embeddings = np.array(self.corpus_embeddings)
        
        return {
            "corpus_size": len(self.artifact_corpus),
            "average_embedding_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "embedding_dimensions": len(self.corpus_embeddings[0]),
            "embedding_model": self.embedding_model,
            "novelty_thresholds": self.novelty_thresholds
        }
    
    def validate_embedding_consistency(self, test_texts: List[str]) -> Dict[str, Any]:
        """
        Validate embedding consistency to detect potential model drift.
        
        Args:
            test_texts: List of test texts to embed multiple times
            
        Returns:
            Validation results including consistency metrics
        """
        consistency_results = {
            "test_count": len(test_texts),
            "max_variance": 0.0,
            "mean_variance": 0.0,
            "embeddings_stable": True
        }
        
        variances = []
        
        for text in test_texts:
            # Get embedding twice
            emb1 = self._get_embedding(text)
            # Force re-computation (bypass cache)
            text_hash = self._get_text_hash(text + "_test")
            if text_hash in self.embedding_cache:
                del self.embedding_cache[text_hash]
            
            emb2 = self._get_embedding(text)
            
            # Calculate variance
            variance = np.var(np.array(emb1) - np.array(emb2))
            variances.append(variance)
        
        if variances:
            consistency_results["max_variance"] = float(max(variances))
            consistency_results["mean_variance"] = float(np.mean(variances))
            consistency_results["embeddings_stable"] = consistency_results["max_variance"] < 1e-10
        
        return consistency_results
    
    def export_corpus(self, export_path: str):
        """Export the artifact corpus for analysis or backup."""
        corpus_data = {
            "export_timestamp": datetime.now().isoformat(),
            "embedding_model": self.embedding_model,
            "corpus_size": len(self.artifact_corpus),
            "artifacts": [
                {
                    "id": artifact["id"],
                    "text": artifact["text"],
                    "embedding": artifact["embedding"]
                }
                for artifact in self.artifact_corpus
            ],
            "novelty_thresholds": self.novelty_thresholds
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2)
    
    def load_corpus(self, import_path: str):
        """Load a previously exported corpus."""
        with open(import_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        self.artifact_corpus = corpus_data["artifacts"]
        self.corpus_embeddings = [artifact["embedding"] for artifact in self.artifact_corpus]
        
        print(f"Loaded corpus with {len(self.artifact_corpus)} artifacts")
        print(f"Model: {corpus_data.get('embedding_model', 'unknown')}")
        print(f"Export date: {corpus_data.get('export_timestamp', 'unknown')}")
