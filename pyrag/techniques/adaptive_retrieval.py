"""
Adaptive Retrieval-Augmented Generation (RAG) System

This system dynamically selects retrieval strategies based on query classification,
then uses the retrieved documents to generate answers. It supports four query types:
- Factual: Direct factual questions
- Analytical: Complex analysis requiring multiple perspectives
- Opinion: Seeking diverse viewpoints
- Contextual: Questions requiring user-specific context
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field


# Configuration Dataclasses
@dataclass
class AppConfig:
    """Application configuration settings"""
    openai_api_key: str
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4000
    chunk_size: int = 800
    chunk_overlap: int = 0
    factual_multiplier: int = 2  # Added this missing attribute


@dataclass
class RetrievalConfig:
    """Retrieval strategy configuration"""
    default_k: int = 4
    opinion_k: int = 3
    analytical_subqueries: int = 2


# Pydantic Models for Structured Output
class CategoriesOptions(BaseModel):
    """Query classification options"""
    category: str = Field(
        description="The category of the query: Factual, Analytical, Opinion, or Contextual",
        example="Factual"
    )


class RelevantScore(BaseModel):
    """Document relevance score"""
    score: float = Field(
        description="The relevance score of the document to the query (1-10)",
        example=8.0
    )


class SelectedIndices(BaseModel):
    """Selected document indices"""
    indices: List[int] = Field(
        description="Indices of selected documents",
        example=[0, 1, 2, 3]
    )


class SubQueries(BaseModel):
    """Generated sub-queries for analytical queries"""
    sub_queries: List[str] = Field(
        description="List of sub-queries for comprehensive analysis",
        example=["What is X?", "How does Y affect Z?"]
    )


def load_configuration() -> AppConfig:
    """
    Load and validate application configuration from environment variables
    
    Returns:
        AppConfig: Loaded configuration object
        
    Raises:
        SystemExit: If required configuration is missing
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in the .env file.")
        sys.exit(1)
        
    os.environ["OPENAI_API_KEY"] = api_key
    
    return AppConfig(
        openai_api_key=api_key,
        model_name=os.getenv('MODEL_NAME', 'gpt-4o'),
        temperature=float(os.getenv('TEMPERATURE', '0.0')),
        max_tokens=int(os.getenv('MAX_TOKENS', '4000')),
        chunk_size=int(os.getenv('CHUNK_SIZE', '800')),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '0')),
        factual_multiplier=int(os.getenv('FACTUAL_MULTIPLIER', '2'))
    )


class QueryClassifier:
    """Classifies queries into categories to determine retrieval strategy"""
    
    def __init__(self, config: AppConfig):
        """
        Initialize the query classifier
        
        Args:
            config: Application configuration
        """
        self.llm = ChatOpenAI(
            temperature=config.temperature,
            model_name=config.model_name,
            max_tokens=config.max_tokens
        )
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Classify the following query into one of these categories: "
                "Factual, Analytical, Opinion, or Contextual.\n"
                "Query: {query}\nCategory:"
            )
        )
        self.chain = self.prompt | self.llm.with_structured_output(CategoriesOptions)

    def classify(self, query: str) -> str:
        """
        Classify a query into one of the supported categories
        
        Args:
            query: The input query to classify
            
        Returns:
            str: The determined category (Factual, Analytical, Opinion, or Contextual)
        """
        print("Classifying query...")
        try:
            return self.chain.invoke(query).category
        except Exception as e:
            print(f"Error classifying query: {e}")
            return "Factual"  # Default to factual on error


class BaseRetrievalStrategy:
    """Base class for all retrieval strategies"""
    
    def __init__(self, texts: List[str], config: AppConfig):
        """
        Initialize the base retrieval strategy
        
        Args:
            texts: List of text documents to index
            config: Application configuration
        """
        self.embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.documents = text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = ChatOpenAI(
            temperature=config.temperature,
            model_name=config.model_name,
            max_tokens=config.max_tokens
        )
        self.config = config

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Base retrieval method - should be overridden by subclasses
        
        Args:
            query: The query to retrieve documents for
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Retrieved documents
        """
        return self.db.similarity_search(query, k=k)


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    """Retrieval strategy optimized for factual queries"""
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve documents for factual queries with query enhancement and relevance ranking
        
        Args:
            query: The factual query
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Top k most relevant documents
        """
        print("Retrieving factual information...")
        
        # Enhance the query for better retrieval
        enhanced_query = self._enhance_query(query)
        print(f'Enhanced Query: {enhanced_query}')

        # Retrieve and rank documents
        docs = self.db.similarity_search(enhanced_query, k=k * self.config.factual_multiplier)
        ranked_docs = self._rank_documents(enhanced_query, docs)
        
        return [doc for doc, _ in ranked_docs[:k]]

    def _enhance_query(self, query: str) -> str:
        """Enhance factual query for better retrieval"""
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        return query_chain.invoke(query).content

    def _rank_documents(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        """Rank documents by relevance to the query"""
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template=(
                "On a scale of 1-10, how relevant is this document to the query: '{query}'?\n"
                "Document: {doc}\nRelevance score:"
            )
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)

        ranked_docs = []
        print("Ranking documents...")
        for doc in docs:
            score = float(ranking_chain.invoke({
                "query": query,
                "doc": doc.page_content
            }).score)
            ranked_docs.append((doc, score))

        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    """Retrieval strategy optimized for analytical queries requiring multiple perspectives"""
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve documents for analytical queries using sub-query generation
        
        Args:
            query: The analytical query
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Diverse set of documents covering multiple aspects
        """
        print("Retrieving analytical information...")
        
        # Generate sub-queries for comprehensive analysis
        sub_queries = self._generate_sub_queries(query, k)
        print(f'Sub-queries for comprehensive analysis: {sub_queries}')

        # Retrieve documents for each sub-query
        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))

        # Select diverse and relevant documents
        return self._select_diverse_documents(query, all_docs, k)

    def _generate_sub_queries(self, query: str, k: int) -> List[str]:
        """Generate sub-queries for comprehensive analysis"""
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Generate {k} sub-questions for: {query}"
        )
        sub_queries_chain = sub_queries_prompt | self.llm.with_structured_output(SubQueries)
        return sub_queries_chain.invoke({"query": query, "k": k}).sub_queries

    def _select_diverse_documents(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Select diverse and relevant documents"""
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template=(
                "Select the most diverse and relevant set of {k} documents for: '{query}'\n"
                "Documents: {docs}\n"
                "Return only the indices of selected documents as a list of integers."
            )
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)
        
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(docs)])
        selected_indices = diversity_chain.invoke({
            "query": query,
            "docs": docs_text,
            "k": k
        }).indices

        print('Selected diverse and relevant documents.')
        return [docs[i] for i in selected_indices if i < len(docs)]


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    """Retrieval strategy optimized for opinion-based queries"""
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve documents representing diverse opinions on a topic
        
        Args:
            query: The opinion-seeking query
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Documents representing diverse viewpoints
        """
        print("Retrieving opinion-based information...")
        
        # Identify distinct viewpoints
        viewpoints = self._identify_viewpoints(query, k)
        print(f'Identified viewpoints: {viewpoints}')

        # Retrieve documents for each viewpoint
        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k=2))

        # Select representative opinion documents
        return self._select_opinion_documents(query, all_docs, k)

    def _identify_viewpoints(self, query: str, k: int) -> List[str]:
        """Identify distinct viewpoints on the topic"""
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Identify {k} distinct viewpoints or perspectives on: {query}"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        return viewpoints_chain.invoke({"query": query, "k": k}).content.split('\n')

    def _select_opinion_documents(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Select diverse opinion documents"""
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template=(
                "Classify these documents into distinct opinions on '{query}' "
                "and select the {k} most representative and diverse viewpoints:\n"
                "Documents: {docs}\nSelected indices:"
            )
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)

        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(docs)])
        selected_indices = opinion_chain.invoke({
            "query": query,
            "docs": docs_text,
            "k": k
        }).indices

        print('Selected diverse opinion documents.')
        return [docs[i] for i in selected_indices if isinstance(i, int) and i < len(docs)]


class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    """Retrieval strategy optimized for context-dependent queries"""
    
    def retrieve(self, query: str, k: int = 4, user_context: Optional[str] = None) -> List[Document]:
        """
        Retrieve documents considering user-specific context
        
        Args:
            query: The context-dependent query
            k: Number of documents to retrieve
            user_context: Additional context about the user's situation
            
        Returns:
            List[Document]: Contextually relevant documents
        """
        print("Retrieving contextual information...")
        
        # Contextualize the query
        contextualized_query = self._contextualize_query(query, user_context)
        print(f'Contextualized Query: {contextualized_query}')

        # Retrieve and rank documents considering context
        docs = self.db.similarity_search(contextualized_query, k=k * 2)
        ranked_docs = self._rank_contextual_documents(query, user_context, docs)
        
        return [doc for doc, _ in ranked_docs[:k]]

    def _contextualize_query(self, query: str, context: Optional[str]) -> str:
        """Adapt query based on user context"""
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Given the user context: {context}\n"
                "Reformulate the query to best address the user's needs: {query}"
            )
        )
        context_chain = context_prompt | self.llm
        return context_chain.invoke({
            "query": query,
            "context": context or "No specific context provided"
        }).content

    def _rank_contextual_documents(self, query: str, context: Optional[str], docs: List[Document]) -> List[Tuple[Document, float]]:
        """Rank documents by contextual relevance"""
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template=(
                "Given the query: '{query}' and user context: '{context}', "
                "rate the relevance of this document on a scale of 1-10:\n"
                "Document: {doc}\nRelevance score:"
            )
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)
        print("Ranking contextual documents...")

        ranked_docs = []
        for doc in docs:
            score = float(ranking_chain.invoke({
                "query": query,
                "context": context or "No specific context provided",
                "doc": doc.page_content
            }).score)
            ranked_docs.append((doc, score))

        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs


class AdaptiveRetriever:
    """Router that selects the appropriate retrieval strategy based on query classification"""
    
    def __init__(self, texts: List[str], config: AppConfig):
        """
        Initialize the adaptive retriever
        
        Args:
            texts: List of text documents to index
            config: Application configuration
        """
        self.classifier = QueryClassifier(config)
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts, config),
            "Analytical": AnalyticalRetrievalStrategy(texts, config),
            "Opinion": OpinionRetrievalStrategy(texts, config),
            "Contextual": ContextualRetrievalStrategy(texts, config)
        }
        self.retrieval_config = RetrievalConfig()

    def get_relevant_documents(self, query: str, user_context: Optional[str] = None) -> List[Document]:
        """
        Retrieve documents using the appropriate strategy for the query
        
        Args:
            query: The query to retrieve documents for
            user_context: Additional context for contextual queries
            
        Returns:
            List[Document]: Retrieved documents
        """
        category = self.classifier.classify(query)
        print(f"Using {category} retrieval strategy...")
        
        strategy = self.strategies.get(category)
        if not strategy:
            print(f"Unknown category '{category}'. Using default retrieval strategy.")
            return []

        if category == "Opinion":
            return strategy.retrieve(query, k=self.retrieval_config.opinion_k)
        elif category == "Contextual":
            return strategy.retrieve(query, k=self.retrieval_config.default_k, user_context=user_context)
        else:
            return strategy.retrieve(query, k=self.retrieval_config.default_k)


class PydanticAdaptiveRetriever(BaseRetriever):
    """LangChain compatible wrapper for the adaptive retriever"""
    
    adaptive_retriever: AdaptiveRetriever = Field(exclude=True)
    user_context: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Sync document retrieval"""
        return self.adaptive_retriever.get_relevant_documents(query, self.user_context)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async document retrieval"""
        return self.get_relevant_documents(query)


class AdaptiveRAG:
    """End-to-end Adaptive RAG system combining retrieval and generation"""
    
    def __init__(self, texts: List[str], config: AppConfig):
        """
        Initialize the RAG system
        
        Args:
            texts: List of text documents to index
            config: Application configuration
        """
        adaptive_retriever = AdaptiveRetriever(texts, config)
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)
        self.llm = ChatOpenAI(
            temperature=config.temperature,
            model_name=config.model_name,
            max_tokens=config.max_tokens
        )

        self.prompt = PromptTemplate(
            template=(
                "Use the following pieces of context to answer the question at the end.\n"
                "If you don't know the answer, just say that you don't know, "
                "don't try to make up an answer.\n\n"
                "{context}\n\n"
                "Question: {question}\nAnswer:"
            ),
            input_variables=["context", "question"]
        )
        self.llm_chain = self.prompt | self.llm

    def answer(self, query: str, user_context: Optional[str] = None) -> Any:
        """
        Generate an answer to a query using adaptive retrieval
        
        Args:
            query: The question to answer
            user_context: Additional context for contextual queries
            
        Returns:
            Any: The generated answer
        """
        self.retriever.user_context = user_context
        docs = self.retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant documents found to answer this question."
            
        context = "\n".join([doc.page_content for doc in docs])
        return self.llm_chain.invoke({
            "context": context,
            "question": query
        })


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Adaptive RAG System")
    parser.add_argument('--query', type=str, help="Single query to process")
    parser.add_argument('--file', type=str, help="File containing queries (one per line)")
    parser.add_argument('--context', type=str, help="User context for contextual queries")
    parser.add_argument('--texts', type=str, default="sample_texts.txt",
                       help="File containing documents to index")
    return parser.parse_args()


def load_texts(file_path: str) -> List[str]:
    """Load documents from a file"""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Text file {file_path} not found. Using sample text.")
        return [
            "The Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
            "The distance between Earth and the Sun is about 93 million miles (150 million kilometers).",
            "Earth's climate is influenced by its distance from the Sun, axial tilt, and atmospheric composition.",
            "There are several theories about the origin of life on Earth including abiogenesis and panspermia.",
            "Earth's position in the Solar System's habitable zone allows for liquid water to exist on its surface.",
            "The average distance from Earth to the Moon is about 238,855 miles (384,400 kilometers).",
            "The Moon's distance from Earth varies due to its elliptical orbit, ranging from about 225,623 miles (363,104 km) at perigee to 252,088 miles (405,696 km) at apogee."
        ]


def main():
    """Main execution function"""
    args = parse_arguments()
    config = load_configuration()
    
    # Load documents
    texts = load_texts(args.texts)
    rag_system = AdaptiveRAG(texts, config)

    # Process queries
    if args.query:
        result = rag_system.answer(args.query, args.context)
        print(f"Question: {args.query}\nAnswer: {result.content}\n")
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Query file {args.file} not found.")
            return
            
        for query in queries:
            result = rag_system.answer(query, args.context)
            print(f"Question: {query}\nAnswer: {result.content}\n")
    else:
        # Default queries if none provided
        queries = [
            "What is the distance between the Earth and the Sun?",
            "How does the Earth's distance from the Sun affect its climate?",
            "What are the different theories about the origin of life on Earth?",
            "How does the Earth's position in the Solar System influence its habitability?",
            "What is the distance to the Moon?"
        ]
        
        for query in queries:
            result = rag_system.answer(query)
            print(f"Question: {query}\nAnswer: {result.content}\n")


if __name__ == "__main__":
    main()