"""
Comprehensive tests for the HieQue retrieval system
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from .retrieval import HieQueRetrieval, SearchResult, QueryResponse
from .data_extraction import DocumentProcessor
from ..exception import HieQueException

class TestHieQueRetrieval(unittest.TestCase):
    """Test cases for HieQueRetrieval class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock OpenAI API key
        self.mock_api_key = "sk-test1234567890abcdefghijklmnopqrstuvwxyz"
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample documents for testing
        self.sample_documents = [
            {
                'id': 'doc1',
                'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.',
                'metadata': {'chapter': 1, 'page': 1, 'topic': 'machine_learning'}
            },
            {
                'id': 'doc2',
                'content': 'Deep learning uses neural networks with multiple layers to model complex patterns in data.',
                'metadata': {'chapter': 2, 'page': 15, 'topic': 'deep_learning'}
            },
            {
                'id': 'doc3',
                'content': 'Natural language processing enables computers to understand and generate human language.',
                'metadata': {'chapter': 3, 'page': 30, 'topic': 'nlp'}
            },
            {
                'id': 'doc4',
                'content': 'Computer vision algorithms can identify objects and patterns in images and videos.',
                'metadata': {'chapter': 4, 'page': 45, 'topic': 'computer_vision'}
            },
            {
                'id': 'doc5',
                'content': 'Reinforcement learning agents learn through interaction with their environment.',
                'metadata': {'chapter': 5, 'page': 60, 'topic': 'reinforcement_learning'}
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_initialization(self, mock_chroma, mock_transformer, mock_openai):
        """Test system initialization"""
        # Mock the dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        # Test initialization
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        self.assertIsNotNone(retrieval_system)
        self.assertEqual(retrieval_system.gpt_model, "gpt-4-turbo-preview")
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_missing_api_key(self, mock_chroma, mock_transformer, mock_openai):
        """Test initialization without API key"""
        # Clear environment variable
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        with self.assertRaises(HieQueException):
            HieQueRetrieval(chroma_persist_dir=self.temp_dir)
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_document_indexing(self, mock_chroma, mock_transformer, mock_openai):
        """Test document indexing functionality"""
        # Mock dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value.encode.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        # Test indexing
        retrieval_system.index_documents(self.sample_documents)
        
        self.assertEqual(len(retrieval_system.documents), 5)
        self.assertIsNotNone(retrieval_system.document_embeddings)
        self.assertIsNotNone(retrieval_system.gmm_clusters)
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_spider_search(self, mock_chroma, mock_transformer, mock_openai):
        """Test SPIDER semantic search"""
        # Mock dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value.encode.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        # Index documents
        retrieval_system.index_documents(self.sample_documents)
        
        # Test SPIDER search
        results = retrieval_system._spider_search("machine learning", 3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        if results:
            self.assertIsInstance(results[0], SearchResult)
            self.assertEqual(results[0].retrieval_method, 'SPIDER')
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_bm25_search(self, mock_chroma, mock_transformer, mock_openai):
        """Test BM25 keyword search"""
        # Mock dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value.encode.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        # Index documents
        retrieval_system.index_documents(self.sample_documents)
        
        # Test BM25 search
        results = retrieval_system._bm25_search("neural networks", 3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        if results:
            self.assertIsInstance(results[0], SearchResult)
            self.assertEqual(results[0].retrieval_method, 'BM25')
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_gmm_search(self, mock_chroma, mock_transformer, mock_openai):
        """Test GMM cluster-based search"""
        # Mock dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value.encode.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        # Index documents
        retrieval_system.index_documents(self.sample_documents)
        
        # Test GMM search
        results = retrieval_system._gmm_search("artificial intelligence", 3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        if results:
            self.assertIsInstance(results[0], SearchResult)
            self.assertEqual(results[0].retrieval_method, 'GMM')
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_full_query(self, mock_chroma, mock_transformer, mock_openai):
        """Test complete query workflow"""
        # Mock dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value.encode.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        # Mock GPT response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Machine learning is a subset of AI that focuses on algorithms."
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        # Index documents
        retrieval_system.index_documents(self.sample_documents)
        
        # Test full query
        response = retrieval_system.query("What is machine learning?", max_results=3)
        
        self.assertIsInstance(response, QueryResponse)
        self.assertIsNotNone(response.answer)
        self.assertIsInstance(response.context, list)
        self.assertGreater(response.confidence_score, 0)
        self.assertGreater(response.processing_time, 0)
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_cluster_info(self, mock_chroma, mock_transformer, mock_openai):
        """Test cluster information retrieval"""
        # Mock dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value.encode.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        # Index documents
        retrieval_system.index_documents(self.sample_documents)
        
        # Get cluster info
        cluster_info = retrieval_system.get_cluster_info()
        
        self.assertIsInstance(cluster_info, dict)
        self.assertNotIn('error', cluster_info)
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_clustering_update(self, mock_chroma, mock_transformer, mock_openai):
        """Test GMM clustering update"""
        # Mock dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value.encode.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        # Index documents
        retrieval_system.index_documents(self.sample_documents)
        
        # Update clustering
        retrieval_system.update_clustering(5)
        
        self.assertEqual(retrieval_system.gmm.n_components, 5)
    
    @patch('openai.OpenAI')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_export_results(self, mock_chroma, mock_transformer, mock_openai):
        """Test result export functionality"""
        # Mock dependencies
        mock_openai.return_value = Mock()
        mock_transformer.return_value.encode.return_value = Mock()
        mock_chroma.return_value.create_collection.return_value = Mock()
        
        retrieval_system = HieQueRetrieval(
            openai_api_key=self.mock_api_key,
            chroma_persist_dir=self.temp_dir
        )
        
        # Create sample results
        results = [
            SearchResult(
                content="Sample content",
                document_id="doc1",
                page_number=1,
                chapter="1",
                similarity_score=0.85,
                retrieval_method="SPIDER",
                metadata={}
            )
        ]
        
        # Test JSON export
        json_export = retrieval_system.export_results(results, "json")
        self.assertIsInstance(json_export, str)
        self.assertIn("Sample content", json_export)
        
        # Test CSV export
        csv_export = retrieval_system.export_results(results, "csv")
        self.assertIsInstance(csv_export, str)
        self.assertIn("Sample content", csv_export)
        
        # Test unsupported format
        with self.assertRaises(ValueError):
            retrieval_system.export_results(results, "unsupported")


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DocumentProcessor()
        
        # Create sample text file
        self.sample_text = "This is a sample document for testing. It contains multiple sentences. Machine learning is a fascinating topic."
        self.text_file = os.path.join(self.temp_dir, "sample.txt")
        with open(self.text_file, 'w') as f:
            f.write(self.sample_text)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_text_processing(self):
        """Test text file processing"""
        documents = self.processor.process_document(self.text_file)
        
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        
        doc = documents[0]
        self.assertIn('id', doc)
        self.assertIn('content', doc)
        self.assertIn('metadata', doc)
        self.assertIn('chunk_number', doc)
    
    def test_unsupported_file_type(self):
        """Test processing of unsupported file types"""
        unsupported_file = os.path.join(self.temp_dir, "test.xyz")
        with open(unsupported_file, 'w') as f:
            f.write("test content")
        
        with self.assertRaises(DocumentProcessingError):
            self.processor.process_document(unsupported_file)
    
    def test_nonexistent_file(self):
        """Test processing of nonexistent file"""
        with self.assertRaises(DocumentProcessingError):
            self.processor.process_document("nonexistent.txt")
    
    def test_processing_stats(self):
        """Test processing statistics"""
        documents = self.processor.process_document(self.text_file)
        stats = self.processor.get_processing_stats(documents)
        
        self.assertIn('total_documents', stats)
        self.assertIn('total_characters', stats)
        self.assertIn('total_words', stats)
        self.assertIn('file_type_distribution', stats)
    
    def test_chunk_creation(self):
        """Test text chunking functionality"""
        long_text = "This is a very long text. " * 100  # Create long text
        
        chunks = self.processor._create_chunks(long_text, {})
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        
        for chunk in chunks:
            self.assertIn('content', chunk)
            self.assertIn('metadata', chunk)
            self.assertGreaterEqual(len(chunk['content']), self.processor.min_chunk_length)


class TestSearchResult(unittest.TestCase):
    """Test cases for SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test SearchResult object creation"""
        result = SearchResult(
            content="Test content",
            document_id="doc1",
            page_number=1,
            chapter="1",
            similarity_score=0.85,
            retrieval_method="SPIDER",
            metadata={"topic": "test"}
        )
        
        self.assertEqual(result.content, "Test content")
        self.assertEqual(result.document_id, "doc1")
        self.assertEqual(result.similarity_score, 0.85)
        self.assertEqual(result.retrieval_method, "SPIDER")


class TestQueryResponse(unittest.TestCase):
    """Test cases for QueryResponse dataclass"""
    
    def test_query_response_creation(self):
        """Test QueryResponse object creation"""
        results = [
            SearchResult(
                content="Test content",
                document_id="doc1",
                page_number=1,
                chapter="1",
                similarity_score=0.85,
                retrieval_method="SPIDER",
                metadata={}
            )
        ]
        
        response = QueryResponse(
            answer="Test answer",
            context=results,
            confidence_score=0.9,
            processing_time=0.5,
            retrieval_methods_used=["SPIDER"]
        )
        
        self.assertEqual(response.answer, "Test answer")
        self.assertEqual(len(response.context), 1)
        self.assertEqual(response.confidence_score, 0.9)
        self.assertEqual(response.processing_time, 0.5)


if __name__ == '__main__':
    unittest.main()
