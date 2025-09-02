"""
Command-line interface for HieQue framework
"""

import os
import sys
import json
import click
from pathlib import Path
from typing import Optional

from .retrieval import HieQueRetrieval
from .data_extraction import DocumentProcessor
from .logger import setup_logging, get_logger
from .utils import load_config, save_config, load_environment_variables
from .exception import HieQueException

logger = get_logger(__name__)

@click.group()
@click.version_option(version="1.0.0")
@click.option('--config', '-c', default='config.json', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """HieQue: Scalable Multi-Level Text Retrieval Framework
    
    A high-performance text retrieval system integrating GMM, GPT-4, BM25, and SPIDER semantic search.
    """
    # Load configuration
    config_data = load_config(config)
    if not config_data:
        click.echo(f"Warning: Could not load configuration from {config}", err=True)
        config_data = {}
    
    # Setup logging
    log_config = config_data.get('logging', {})
    if verbose:
        log_config['level'] = 'DEBUG'
        log_config['console_level'] = 'DEBUG'
    
    setup_logging(log_config)
    
    # Load environment variables
    load_environment_variables()
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['config'] = config_data
    ctx.obj['config_file'] = config

@cli.command()
@click.option('--input', '-i', required=True, help='Input file or directory path')
@click.option('--output', '-o', default='./chroma_db', help='Output directory for vector database')
@click.option('--chunk-size', default=1000, help='Text chunk size in characters')
@click.option('--overlap', default=200, help='Overlap between chunks')
@click.pass_context
def index(ctx, input, output, chunk_size, overlap):
    """Index documents for retrieval"""
    try:
        config = ctx.obj['config']
        
        # Initialize document processor
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_length=config.get('retrieval', {}).get('min_chunk_length', 100)
        )
        
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"Error: Input path does not exist: {input}", err=True)
            sys.exit(1)
        
        click.echo(f"Processing documents from: {input}")
        
        # Process documents
        if input_path.is_file():
            documents = processor.process_document(str(input_path))
        else:
            documents = processor.process_directory(str(input_path))
        
        if not documents:
            click.echo("No documents were processed successfully.", err=True)
            sys.exit(1)
        
        # Get processing statistics
        stats = processor.get_processing_stats(documents)
        click.echo(f"\nProcessing completed:")
        click.echo(f"  Total chunks: {stats['total_documents']}")
        click.echo(f"  Total characters: {stats['total_characters']:,}")
        click.echo(f"  Total words: {stats['total_words']:,}")
        click.echo(f"  File types: {', '.join(stats['file_type_distribution'].keys())}")
        
        # Initialize retrieval system
        retrieval_system = HieQueRetrieval(
            chroma_persist_dir=output,
            device='auto'
        )
        
        # Index documents
        click.echo("\nIndexing documents...")
        retrieval_system.index_documents(documents)
        
        # Get cluster information
        cluster_info = retrieval_system.get_cluster_info()
        click.echo(f"\nClustering completed:")
        for cluster_id, info in cluster_info.items():
            click.echo(f"  {cluster_id}: {info['document_count']} documents")
        
        click.echo(f"\nDocuments indexed successfully in: {output}")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--database', '-d', default='./chroma_db', help='Vector database directory')
@click.option('--max-results', default=10, help='Maximum number of results')
@click.option('--methods', default='spider,bm25,gmm', help='Retrieval methods to use (comma-separated)')
@click.option('--output', '-o', help='Output file for results')
@click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'csv', 'txt']), help='Output format')
@click.pass_context
def query(ctx, query, database, max_results, methods, output, output_format):
    """Query the indexed documents"""
    try:
        config = ctx.obj['config']
        
        # Initialize retrieval system
        retrieval_system = HieQueRetrieval(
            chroma_persist_dir=database,
            device='auto'
        )
        
        # Parse methods
        method_list = [m.strip() for m in methods.split(',')]
        
        click.echo(f"Querying: {query}")
        click.echo(f"Methods: {', '.join(method_list)}")
        click.echo(f"Max results: {max_results}")
        
        # Perform query
        response = retrieval_system.query(
            query_text=query,
            max_results=max_results,
            use_methods=method_list
        )
        
        # Display results
        click.echo(f"\nAnswer: {response.answer}")
        click.echo(f"\nConfidence: {response.confidence_score:.3f}")
        click.echo(f"Processing time: {response.processing_time:.3f}s")
        click.echo(f"Methods used: {', '.join(response.retrieval_methods_used)}")
        
        # Display context
        if response.context:
            click.echo(f"\nTop {len(response.context)} results:")
            for i, result in enumerate(response.context, 1):
                click.echo(f"\n{i}. {result.retrieval_method} (Score: {result.similarity_score:.3f})")
                click.echo(f"   Document: {result.document_id}")
                if result.chapter:
                    click.echo(f"   Chapter: {result.chapter}")
                if result.page_number:
                    click.echo(f"   Page: {result.page_number}")
                click.echo(f"   Content: {result.content[:200]}...")
        
        # Export results if requested
        if output:
            export_path = retrieval_system.export_results(
                response.context, 
                output, 
                output_format
            )
            click.echo(f"\nResults exported to: {export_path}")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', default='./chroma_db', help='Vector database directory')
@click.pass_context
def info(ctx, database):
    """Get information about the indexed documents"""
    try:
        # Initialize retrieval system
        retrieval_system = HieQueRetrieval(
            chroma_persist_dir=database,
            device='auto'
        )
        
        if not retrieval_system.documents:
            click.echo("No documents indexed yet.")
            return
        
        # Display document information
        click.echo(f"Indexed Documents: {len(retrieval_system.documents)}")
        
        # Get cluster information
        cluster_info = retrieval_system.get_cluster_info()
        if 'error' not in cluster_info:
            click.echo(f"\nClusters: {len(cluster_info)}")
            for cluster_id, info in cluster_info.items():
                click.echo(f"  {cluster_id}: {info['document_count']} documents")
        
        # Display file type distribution
        file_types = {}
        for doc in retrieval_system.documents:
            file_type = doc['metadata'].get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        click.echo(f"\nFile Types:")
        for file_type, count in file_types.items():
            click.echo(f"  {file_type}: {count} chunks")
        
        # Display total statistics
        total_chars = sum(len(doc['content']) for doc in retrieval_system.documents)
        total_words = sum(len(doc['content'].split()) for doc in retrieval_system.documents)
        
        click.echo(f"\nTotal Content:")
        click.echo(f"  Characters: {total_chars:,}")
        click.echo(f"  Words: {total_words:,}")
        
    except Exception as e:
        logger.error(f"Info retrieval failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--database', '-d', default='./chroma_db', help='Vector database directory')
@click.option('--components', default=10, help='Number of GMM components')
@click.pass_context
def recluster(ctx, database, components):
    """Update GMM clustering with different number of components"""
    try:
        # Initialize retrieval system
        retrieval_system = HieQueRetrieval(
            chroma_persist_dir=database,
            device='auto'
        )
        
        if not retrieval_system.documents:
            click.echo("No documents indexed yet. Please index documents first.")
            return
        
        click.echo(f"Updating clustering to {components} components...")
        
        # Update clustering
        retrieval_system.update_clustering(components)
        
        # Get new cluster information
        cluster_info = retrieval_system.get_cluster_info()
        click.echo(f"\nNew clustering completed:")
        for cluster_id, info in cluster_info.items():
            click.echo(f"  {cluster_id}: {info['document_count']} documents")
        
    except Exception as e:
        logger.error(f"Reclustering failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', default='config.json', help='Configuration file path')
@click.pass_context
def config_show(ctx, config):
    """Show current configuration"""
    try:
        config_data = load_config(config)
        if config_data:
            click.echo(json.dumps(config_data, indent=2))
        else:
            click.echo(f"No configuration found at: {config}")
    except Exception as e:
        click.echo(f"Error reading configuration: {e}", err=True)

@cli.command()
@click.option('--config', '-c', default='config.json', help='Configuration file path')
@click.pass_context
def config_edit(ctx, config):
    """Edit configuration file"""
    try:
        config_path = Path(config)
        if not config_path.exists():
            click.echo(f"Configuration file not found: {config}")
            return
        
        # Try to open in default editor
        editor = os.environ.get('EDITOR', 'nano')
        os.system(f"{editor} {config_path}")
        
        click.echo(f"Configuration file opened in {editor}")
        
    except Exception as e:
        click.echo(f"Error editing configuration: {e}", err=True)

@cli.command()
@click.option('--database', '-d', default='./chroma_db', help='Vector database directory')
@click.option('--yes', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clear(ctx, database, yes):
    """Clear all indexed documents"""
    try:
        if not yes:
            if not click.confirm(f"Are you sure you want to clear all documents from {database}?"):
                return
        
        # Remove database directory
        db_path = Path(database)
        if db_path.exists():
            import shutil
            shutil.rmtree(db_path)
            click.echo(f"Cleared database: {database}")
        else:
            click.echo(f"Database not found: {database}")
        
    except Exception as e:
        logger.error(f"Clear operation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()
