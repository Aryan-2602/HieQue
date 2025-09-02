#!/usr/bin/env python3
"""
Basic usage example for HieQue framework
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components.retrieval import HieQueRetrieval
from components.data_extraction import DocumentProcessor
from utils import load_environment_variables

def main():
    """Demonstrate basic usage of HieQue framework"""
    
    # Load environment variables
    load_environment_variables()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. GPT-4 responses will not work.")
        print("Set OPENAI_API_KEY environment variable to enable full functionality.")
    
    print("🚀 HieQue Framework - Basic Usage Example")
    print("=" * 50)
    
    # Initialize document processor
    print("\n1. Initializing document processor...")
    processor = DocumentProcessor(
        chunk_size=800,
        overlap=150,
        min_chunk_length=100
    )
    
    # Sample documents (in real usage, these would come from files)
    sample_documents = [
        {
            'id': 'ai_intro',
            'content': '''Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

The field of AI has evolved significantly since its inception in the 1950s. Early AI research focused on symbolic reasoning and expert systems, while modern AI emphasizes machine learning and deep learning approaches. Machine learning enables computers to learn from data without being explicitly programmed, while deep learning uses neural networks with multiple layers to model complex patterns.

AI applications are now ubiquitous in our daily lives, from virtual assistants like Siri and Alexa to recommendation systems on streaming platforms and e-commerce websites. In healthcare, AI is used for medical image analysis, drug discovery, and personalized treatment plans. In transportation, AI powers autonomous vehicles and traffic optimization systems.''',
            'metadata': {
                'title': 'Introduction to Artificial Intelligence',
                'chapter': 1,
                'topic': 'AI fundamentals',
                'source': 'sample'
            }
        },
        {
            'id': 'ml_basics',
            'content': '''Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience. Unlike traditional programming, where rules are explicitly defined, machine learning systems learn patterns from data.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training on labeled data to make predictions or classifications. Unsupervised learning finds hidden patterns in unlabeled data through clustering and dimensionality reduction. Reinforcement learning agents learn optimal actions through trial and error, receiving rewards or penalties for their decisions.

Popular machine learning algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks. The choice of algorithm depends on the nature of the problem, the type of data available, and the desired outcome. Feature engineering, data preprocessing, and model evaluation are crucial steps in the machine learning pipeline.''',
            'metadata': {
                'title': 'Machine Learning Fundamentals',
                'chapter': 2,
                'topic': 'machine learning',
                'source': 'sample'
            }
        },
        {
            'id': 'deep_learning',
            'content': '''Deep Learning represents a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These neural networks are inspired by the structure and function of biological neural networks in the human brain.

Deep learning models can automatically learn hierarchical representations of data, starting from low-level features and building up to high-level abstractions. This capability makes deep learning particularly effective for tasks involving images, audio, text, and other high-dimensional data.

Convolutional Neural Networks (CNNs) excel at image recognition and computer vision tasks, while Recurrent Neural Networks (RNNs) and their variants like Long Short-Term Memory (LSTM) networks are well-suited for sequential data and natural language processing. Transformer architectures, such as BERT and GPT, have revolutionized language understanding and generation tasks.

The success of deep learning has been driven by several factors: the availability of large datasets, increased computational power, and advances in optimization algorithms. However, deep learning models also face challenges such as the need for extensive training data, computational requirements, and interpretability issues.''',
            'metadata': {
                'title': 'Deep Learning Concepts',
                'chapter': 3,
                'topic': 'deep learning',
                'source': 'sample'
            }
        }
    ]
    
    # Process documents
    print("\n2. Processing sample documents...")
    documents = []
    for doc in sample_documents:
        # In real usage, this would be done by the processor
        # Here we're just preparing the data structure
        documents.append(doc)
    
    print(f"   Processed {len(documents)} documents")
    
    # Initialize retrieval system
    print("\n3. Initializing retrieval system...")
    retrieval_system = HieQueRetrieval(
        openai_api_key=api_key,
        chroma_persist_dir="./example_db",
        device='auto'
    )
    
    # Index documents
    print("\n4. Indexing documents...")
    retrieval_system.index_documents(documents)
    
    # Get cluster information
    cluster_info = retrieval_system.get_cluster_info()
    print(f"   Created {len(cluster_info)} clusters")
    
    # Perform sample queries
    print("\n5. Performing sample queries...")
    
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the advantages of deep learning?",
        "Compare supervised and unsupervised learning"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query}")
        print("   " + "-" * 40)
        
        try:
            response = retrieval_system.query(
                query_text=query,
                max_results=3,
                use_methods=['spider', 'bm25', 'gmm']
            )
            
            print(f"   Answer: {response.answer[:200]}...")
            print(f"   Confidence: {response.confidence_score:.3f}")
            print(f"   Processing time: {response.processing_time:.3f}s")
            print(f"   Methods used: {', '.join(response.retrieval_methods_used)}")
            
            if response.context:
                print(f"   Top result: {response.context[0].content[:100]}...")
                print(f"   Score: {response.context[0].similarity_score:.3f}")
                print(f"   Method: {response.context[0].retrieval_method}")
        
        except Exception as e:
            print(f"   Error: {e}")
    
    # Export results
    print("\n6. Exporting results...")
    try:
        # Get results from last query
        response = retrieval_system.query("What is machine learning?", max_results=5)
        
        # Export to different formats
        json_export = retrieval_system.export_results(response.context, "ml_results", "json")
        csv_export = retrieval_system.export_results(response.context, "ml_results", "csv")
        
        print(f"   JSON export: {json_export}")
        print(f"   CSV export: {csv_export}")
    
    except Exception as e:
        print(f"   Export error: {e}")
    
    print("\n✅ Basic usage example completed!")
    print("\nNext steps:")
    print("1. Set OPENAI_API_KEY environment variable for GPT-4 responses")
    print("2. Process your own documents using the DocumentProcessor")
    print("3. Use the CLI interface: python -m src.cli --help")
    print("4. Explore the configuration options in config.json")

if __name__ == "__main__":
    main()
