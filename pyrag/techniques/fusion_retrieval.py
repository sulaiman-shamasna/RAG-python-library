"""
Hybrid Retrieval System combining BM25 and Vector Search

This system implements a hybrid retrieval approach that combines:
- BM25 (keyword-based search)
- Vector similarity search (semantic search)

The combination provides benefits of both keyword matching and semantic understanding.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# Configuration
@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval system"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_k: int = 5
    fusion_alpha: float = 0.5  # Weight for vector search in fusion


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hybrid Retrieval System (BM25 + Vector Search)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # File handling arguments
    parser.add_argument(
        "--input", 
        type=str,
        default="data/Understanding_Climate_Change.pdf",
        help="Path to input PDF file"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Directory to persist/load vector store (skip processing if exists)"
    )
    
    # Retrieval parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for vector search in hybrid score (0=BM25 only, 1=Vector only)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks for processing"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between text chunks"
    )
    
    # Query handling
    parser.add_argument(
        "--query",
        type=str,
        default="What are the impacts of climate change on the environment?",
        help="Query to search for"
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="File containing multiple queries (one per line)"
    )
    
    # Output control
    parser.add_argument(
        "--show-full",
        action="store_true",
        help="Show full document content instead of snippets"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_environment() -> None:
    """
    Load environment variables and validate configuration
    
    Raises:
        SystemExit: If required environment variables are missing
    """
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


class DocumentProcessor:
    """Handles document loading and preprocessing"""
    
    @staticmethod
    def load_and_split_pdf(
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_dir: Optional[str] = None,
        verbose: bool = False
    ) -> Tuple[FAISS, List[Document]]:
        """
        Load and split PDF document into chunks
        
        Args:
            file_path: Path to PDF file
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
            persist_dir: Directory to persist/load vector store
            verbose: Enable verbose logging
            
        Returns:
            Tuple containing:
                - FAISS vector store
                - List of cleaned document chunks
        """
        try:
            if persist_dir and os.path.exists(persist_dir):
                if verbose:
                    print(f"Loading vector store from {persist_dir}")
                return FAISS.load_local(persist_dir, OpenAIEmbeddings()), []
            
            if verbose:
                print(f"Processing PDF file: {file_path}")
            
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split and clean text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            cleaned_texts = DocumentProcessor._clean_texts(texts)
            
            # Create vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
            
            if persist_dir:
                if verbose:
                    print(f"Saving vector store to {persist_dir}")
                vectorstore.save_local(persist_dir)
            
            return vectorstore, cleaned_texts
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

    @staticmethod
    def _clean_texts(documents: List[Document]) -> List[Document]:
        """Clean document texts by replacing special characters"""
        for doc in documents:
            doc.page_content = doc.page_content.replace('\t', ' ')
        return documents


class HybridRetriever:
    """Combines BM25 and vector similarity search"""
    
    def __init__(self, vectorstore: FAISS, documents: List[Document], verbose: bool = False):
        """
        Initialize the hybrid retriever
        
        Args:
            vectorstore: FAISS vector store
            documents: List of documents for BM25 index
            verbose: Enable verbose logging
        """
        self.vectorstore = vectorstore
        self.verbose = verbose
        self.bm25_index = self._create_bm25_index(documents)
        
    def _create_bm25_index(self, documents: List[Document]) -> BM25Okapi:
        """
        Create BM25 index from documents
        
        Args:
            documents: List of documents to index
            
        Returns:
            BM25Okapi index
        """
        if self.verbose:
            print("Creating BM25 index...")
            
        tokenized_docs = [doc.page_content.split() for doc in documents]
        return BM25Okapi(tokenized_docs)
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5
    ) -> List[Document]:
        """
        Perform hybrid search combining BM25 and vector similarity
        
        Args:
            query: Search query
            k: Number of documents to return
            alpha: Weight for vector search (0-1)
            
        Returns:
            List of top k documents
        """
        if self.verbose:
            print(f"\nProcessing query: '{query}'")
            print(f"Retrieval parameters: k={k}, alpha={alpha}")
            
        # Get all documents from vectorstore
        all_docs = self.vectorstore.similarity_search("", k=self.vectorstore.index.ntotal)
        
        # BM25 search
        bm25_scores = self.bm25_index.get_scores(query.split())
        
        # Vector search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=len(all_docs))
        
        # Normalize and combine scores
        combined_scores = self._combine_scores(
            bm25_scores,
            [score for _, score in vector_results],
            alpha
        )
        
        # Sort and return top results
        sorted_indices = np.argsort(combined_scores)[::-1]
        return [all_docs[i] for i in sorted_indices[:k]]
    
    def _combine_scores(
        self,
        bm25_scores: np.ndarray,
        vector_scores: List[float],
        alpha: float
    ) -> np.ndarray:
        """
        Normalize and combine BM25 and vector scores
        
        Args:
            bm25_scores: Array of BM25 scores
            vector_scores: List of vector similarity scores
            alpha: Weight for vector search
            
        Returns:
            Array of combined scores
        """
        if self.verbose:
            print("Combining BM25 and vector scores...")
        
        # Normalize vector scores (higher is better)
        vector_scores = np.array(vector_scores)
        norm_vector = 1 - ((vector_scores - np.min(vector_scores)) / 
                          (np.max(vector_scores) - np.min(vector_scores)))
        
        # Normalize BM25 scores
        norm_bm25 = ((bm25_scores - np.min(bm25_scores)) / 
                    (np.max(bm25_scores) - np.min(bm25_scores)))
        
        return alpha * norm_vector + (1 - alpha) * norm_bm25


def display_results(documents: List[Document], show_full: bool = False) -> None:
    """
    Display retrieval results
    
    Args:
        documents: List of retrieved documents
        show_full: Whether to show full content or snippets
    """
    print("\nRetrieval Results:")
    print("=" * 50)
    
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        if show_full:
            print(doc.page_content)
        else:
            print(doc.page_content[:500] + "...")
        print("-" * 50)


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Load configuration
    load_environment()
    config = RetrievalConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        default_k=args.top_k,
        fusion_alpha=args.alpha
    )
    
    try:
        # Process documents
        vectorstore, documents = DocumentProcessor.load_and_split_pdf(
            args.input,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            persist_dir=args.persist_dir,
            verbose=args.verbose
        )
        
        # Initialize retriever
        retriever = HybridRetriever(vectorstore, documents, args.verbose)
        
        # Handle queries
        if args.query_file:
            with open(args.query_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            queries = [args.query]
        
        # Process each query
        for query in queries:
            if not query:
                continue
                
            # Perform hybrid search
            results = retriever.hybrid_search(
                query,
                k=config.default_k,
                alpha=config.fusion_alpha
            )
            
            # Display results
            print(f"\n{'='*30} QUERY: {query} {'='*30}")
            display_results(results, args.show_full)
            
    except Exception as e:
        print(f"\nError in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


