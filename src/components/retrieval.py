"""
HieQue: Multi-Level Text Retrieval Framework
Integrates GMM, GPT-4, BM25, and SPIDER semantic search for academic content retrieval
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from openai import OpenAI
import torch

from ..logger import get_logger
from ..exception import HieQueException

logger = get_logger(__name__)

@dataclass
class SearchResult:
    """Represents a single search result"""
    content: str
    document_id: str
    page_number: Optional[int]
    chapter: Optional[str]
    similarity_score: float
    retrieval_method: str
    metadata: Dict[str, Any]

@dataclass
class QueryResponse:
    """Represents a complete query response"""
    answer: str
    context: List[SearchResult]
    confidence_score: float
    processing_time: float
    retrieval_methods_used: List[str]

class HieQueRetrieval:
    """
    Main retrieval system integrating multiple search strategies
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4-turbo-preview",
                 chroma_persist_dir: str = "./chroma_db",
                 device: str = "auto"):
        """
        Initialize the HieQue retrieval system
        
        Args:
            openai_api_key: OpenAI API key for GPT-4 integration
            model_name: OpenAI model to use (default: gpt-4-turbo-preview)
            chroma_persist_dir: Directory for ChromaDB persistence
            device: Device for sentence transformers ('auto', 'cpu', 'cuda')
        """
        self.logger = logger
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._init_openai_client(openai_api_key, model_name)
        self._init_sentence_transformer()
        self._init_chroma_db(chroma_persist_dir)
        self._init_gmm_clustering()
        
        # Document storage
        self.documents = []
        self.document_embeddings = []
        self.gmm_clusters = None
        
        self.logger.info(f"HieQue retrieval system initialized on {self.device}")
    
    def _init_openai_client(self, api_key: Optional[str], model_name: str):
        """Initialize OpenAI client"""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HieQueException("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.gpt_model = model_name
        self.logger.info(f"OpenAI client initialized with model: {self.gpt_model}")
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer for SPIDER semantic search"""
        try:
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2', 
                device=self.device
            )
            self.logger.info("Sentence transformer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentence transformer: {e}")
            raise HieQueException(f"Sentence transformer initialization failed: {e}")
    
    def _init_chroma_db(self, persist_dir: str):
        """Initialize ChromaDB for vector storage"""
        try:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.chroma_client.create_collection(
                name="hieque_documents",
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"ChromaDB initialized at {persist_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise HieQueException(f"ChromaDB initialization failed: {e}")
    
    def _init_gmm_clustering(self):
        """Initialize GMM clustering parameters"""
        self.gmm = GaussianMixture(
            n_components=10,  # Default number of clusters
            random_state=42,
            covariance_type='full'
        )
        self.logger.info("GMM clustering initialized")
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for multi-level retrieval
        
        Args:
            documents: List of document dictionaries with 'content', 'id', 'metadata' keys
        """
        start_time = time.time()
        self.logger.info(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings for SPIDER semantic search
        self.document_embeddings = self.sentence_transformer.encode(
            texts, 
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        # Store in ChromaDB
        self._store_in_chromadb(documents)
        
        # Perform GMM clustering
        self._perform_gmm_clustering()
        
        # Initialize BM25 vectorizer
        self.bm25_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.bm25_matrix = self.bm25_vectorizer.fit_transform(texts)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Document indexing completed in {processing_time:.2f}s")
    
    def _store_in_chromadb(self, documents: List[Dict[str, Any]]):
        """Store documents in ChromaDB"""
        try:
            # Prepare data for ChromaDB
            ids = [str(doc['id']) for doc in documents]
            texts = [doc['content'] for doc in documents]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.info(f"Stored {len(documents)} documents in ChromaDB")
        except Exception as e:
            self.logger.error(f"Failed to store documents in ChromaDB: {e}")
            raise HieQueException(f"ChromaDB storage failed: {e}")
    
    def _perform_gmm_clustering(self):
        """Perform GMM clustering on document embeddings"""
        try:
            # Convert embeddings to numpy array
            embeddings_np = self.document_embeddings.cpu().numpy()
            
            # Fit GMM
            self.gmm_clusters = self.gmm.fit_predict(embeddings_np)
            
            # Store cluster assignments
            for i, doc in enumerate(self.documents):
                doc['cluster'] = int(self.gmm_clusters[i])
            
            self.logger.info(f"GMM clustering completed with {self.gmm.n_components} clusters")
        except Exception as e:
            self.logger.error(f"GMM clustering failed: {e}")
            raise HieQueException(f"GMM clustering failed: {e}")
    
    def query(self, 
              query_text: str, 
              max_results: int = 10,
              use_methods: List[str] = None,
              cluster_filter: Optional[int] = None) -> QueryResponse:
        """
        Perform multi-level retrieval query
        
        Args:
            query_text: The query text
            max_results: Maximum number of results to return
            use_methods: List of retrieval methods to use
            cluster_filter: Optional cluster ID to filter results
            
        Returns:
            QueryResponse object with answer and context
        """
        start_time = time.time()
        
        if use_methods is None:
            use_methods = ['spider', 'bm25', 'gmm']
        
        self.logger.info(f"Processing query: '{query_text}' with methods: {use_methods}")
        
        # Collect results from different methods
        all_results = []
        
        if 'spider' in use_methods:
            spider_results = self._spider_search(query_text, max_results)
            all_results.extend(spider_results)
        
        if 'bm25' in use_methods:
            bm25_results = self._bm25_search(query_text, max_results)
            all_results.extend(bm25_results)
        
        if 'gmm' in use_methods:
            gmm_results = self._gmm_search(query_text, max_results, cluster_filter)
            all_results.extend(gmm_results)
        
        # Remove duplicates and rank by similarity
        unique_results = self._deduplicate_and_rank(all_results)
        top_results = unique_results[:max_results]
        
        # Generate GPT-4 response
        answer = self._generate_gpt_response(query_text, top_results)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(top_results)
        
        processing_time = time.time() - start_time
        
        response = QueryResponse(
            answer=answer,
            context=top_results,
            confidence_score=confidence,
            processing_time=processing_time,
            retrieval_methods_used=use_methods
        )
        
        self.logger.info(f"Query completed in {processing_time:.2f}s with confidence {confidence:.2f}")
        return response
    
    def _spider_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform SPIDER semantic search"""
        try:
            # Encode query
            query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=True)
            
            # Calculate similarities
            similarities = cosine_similarity(
                query_embedding.cpu().numpy(),
                self.document_embeddings.cpu().numpy()
            )[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                results.append(SearchResult(
                    content=doc['content'],
                    document_id=str(doc['id']),
                    page_number=doc.get('metadata', {}).get('page'),
                    chapter=doc.get('metadata', {}).get('chapter'),
                    similarity_score=float(similarities[idx]),
                    retrieval_method='SPIDER',
                    metadata=doc.get('metadata', {})
                ))
            
            return results
        except Exception as e:
            self.logger.error(f"SPIDER search failed: {e}")
            return []
    
    def _bm25_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform BM25 keyword search"""
        try:
            # Transform query
            query_vector = self.bm25_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.bm25_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                results.append(SearchResult(
                    content=doc['content'],
                    document_id=str(doc['id']),
                    page_number=doc.get('metadata', {}).get('page'),
                    chapter=doc.get('metadata', {}).get('chapter'),
                    similarity_score=float(similarities[idx]),
                    retrieval_method='BM25',
                    metadata=doc.get('metadata', {})
                ))
            
            return results
        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
            return []
    
    def _gmm_search(self, query: str, max_results: int, cluster_filter: Optional[int] = None) -> List[SearchResult]:
        """Perform GMM cluster-based search"""
        try:
            # Encode query
            query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=True)
            
            # Find most similar cluster
            query_cluster = self.gmm.predict(query_embedding.cpu().numpy())[0]
            
            # Filter documents by cluster
            if cluster_filter is not None:
                cluster_docs = [i for i, doc in enumerate(self.documents) if doc.get('cluster') == cluster_filter]
            else:
                cluster_docs = [i for i, doc in enumerate(self.documents) if doc.get('cluster') == query_cluster]
            
            if not cluster_docs:
                return []
            
            # Calculate similarities within cluster
            cluster_embeddings = self.document_embeddings[cluster_docs]
            similarities = cosine_similarity(
                query_embedding.cpu().numpy(),
                cluster_embeddings.cpu().numpy()
            )[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            results = []
            for idx in top_indices:
                doc_idx = cluster_docs[idx]
                doc = self.documents[doc_idx]
                results.append(SearchResult(
                    content=doc['content'],
                    document_id=str(doc['id']),
                    page_number=doc.get('metadata', {}).get('page'),
                    chapter=doc.get('metadata', {}).get('chapter'),
                    similarity_score=float(similarities[idx]),
                    retrieval_method='GMM',
                    metadata=doc.get('metadata', {})
                ))
            
            return results
        except Exception as e:
            self.logger.error(f"GMM search failed: {e}")
            return []
    
    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicates and rank by similarity score"""
        # Remove duplicates based on document ID
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.document_id not in seen_ids:
                seen_ids.add(result.document_id)
                unique_results.append(result)
        
        # Sort by similarity score
        unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return unique_results
    
    def _generate_gpt_response(self, query: str, context: List[SearchResult]) -> str:
        """Generate GPT-4 response based on retrieved context"""
        try:
            # Prepare context for GPT
            context_text = "\n\n".join([
                f"Document {i+1} (Score: {result.similarity_score:.3f}):\n{result.content[:500]}..."
                for i, result in enumerate(context[:5])  # Limit context length
            ])
            
            prompt = f"""You are an expert academic research assistant. Based on the following context, provide a comprehensive and accurate answer to the user's question.

Context:
{context_text}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation. Cite specific parts of the context when relevant.

Answer:"""

            response = self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are an expert academic research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"GPT-4 response generation failed: {e}")
            return f"Unable to generate response due to error: {str(e)}"
    
    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """Calculate confidence score based on result quality"""
        if not results:
            return 0.0
        
        # Average similarity score
        avg_similarity = np.mean([r.similarity_score for r in results])
        
        # Diversity bonus (different retrieval methods)
        methods_used = set(r.retrieval_method for r in results)
        diversity_bonus = min(len(methods_used) * 0.1, 0.3)
        
        # Result count bonus
        count_bonus = min(len(results) * 0.05, 0.2)
        
        confidence = min(avg_similarity + diversity_bonus + count_bonus, 1.0)
        return confidence
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about GMM clusters"""
        if self.gmm_clusters is None:
            return {"error": "No clustering performed yet"}
        
        cluster_info = {}
        for i in range(self.gmm.n_components):
            cluster_docs = [j for j, doc in enumerate(self.documents) if doc.get('cluster') == i]
            cluster_info[f"cluster_{i}"] = {
                "document_count": len(cluster_docs),
                "documents": [self.documents[j]['id'] for j in cluster_docs]
            }
        
        return cluster_info
    
    def update_clustering(self, n_components: int):
        """Update GMM clustering with different number of components"""
        try:
            self.gmm = GaussianMixture(
                n_components=n_components,
                random_state=42,
                covariance_type='full'
            )
            self._perform_gmm_clustering()
            self.logger.info(f"GMM clustering updated to {n_components} components")
        except Exception as e:
            self.logger.error(f"Failed to update clustering: {e}")
            raise HieQueException(f"Clustering update failed: {e}")
    
    def export_results(self, results: List[SearchResult], format: str = "json") -> str:
        """Export search results in various formats"""
        if format == "json":
            import json
            return json.dumps([{
                "content": r.content,
                "document_id": r.document_id,
                "similarity_score": r.similarity_score,
                "retrieval_method": r.retrieval_method,
                "metadata": r.metadata
            } for r in results], indent=2)
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Content", "Document ID", "Score", "Method", "Metadata"])
            
            for r in results:
                writer.writerow([
                    r.content[:200] + "...",
                    r.document_id,
                    r.similarity_score,
                    r.retrieval_method,
                    str(r.metadata)
                ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
