"""
Self-RAG Implementation

This system implements a Self-Reflective Retrieval Augmented Generation pipeline that:
1. Determines if retrieval is necessary
2. Retrieves relevant documents when needed
3. Evaluates document relevance
4. Generates responses with self-assessment
5. Selects the best response based on support and utility
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.vectorstores import FAISS

from pyrag.utils.helper_functions import encode_pdf 

# Configuration
@dataclass
class SelfRAGConfig:
    """Configuration for Self-RAG system"""
    top_k: int = 3
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000
    verbose: bool = False


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(
        description="Self-Reflective RAG System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output parameters
    parser.add_argument(
        "--input", 
        type=str,
        # required=True,
        default="data/the_intelligent_investor_ch_8.pdf",
        help="Path to input PDF file"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to process"
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="File containing multiple queries (one per line)"
    )
    
    # Retrieval parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve"
    )
    
    # LLM configuration
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model to use for generation"
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation"
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for LLM generation"
    )
    
    # Output control
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def load_environment() -> None:
    """Load environment variables and validate configuration"""
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


class RetrievalResponse(BaseModel):
    """Response model for retrieval decision"""
    response: str = Field(..., description="Output only 'Yes' or 'No'")


class RelevanceResponse(BaseModel):
    """Response model for relevance assessment"""
    response: str = Field(..., description="Output only 'Relevant' or 'Irrelevant'")


class GenerationResponse(BaseModel):
    """Response model for generated response"""
    response: str = Field(..., description="The generated response")


class SupportResponse(BaseModel):
    """Response model for support assessment"""
    response: str = Field(..., description="Output 'Fully supported', 'Partially supported', or 'No support'")


class UtilityResponse(BaseModel):
    """Response model for utility assessment"""
    response: int = Field(..., description="Rate the utility of the response from 1 to 5")


class SelfRAG:
    """Self-Reflective RAG system implementation with detailed logging"""
    
    def __init__(self, config: SelfRAGConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens
        )
        self._initialize_prompt_chains()
    
    def _initialize_prompt_chains(self):
        """Initialize all prompt templates and chains"""
        # Retrieval decision chain
        self.retrieval_prompt = PromptTemplate(
            input_variables=["query"],
            template="Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
        )
        self.retrieval_chain = self.retrieval_prompt | self.llm.with_structured_output(RetrievalResponse)
        
        # Relevance assessment chain
        self.relevance_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
        )
        self.relevance_chain = self.relevance_prompt | self.llm.with_structured_output(RelevanceResponse)
        
        # Generation chain
        self.generation_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the query '{query}' and the context '{context}', generate a response."
        )
        self.generation_chain = self.generation_prompt | self.llm.with_structured_output(GenerationResponse)
        
        # Support assessment chain
        self.support_prompt = PromptTemplate(
            input_variables=["response", "context"],
            template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
        )
        self.support_chain = self.support_prompt | self.llm.with_structured_output(SupportResponse)
        
        # Utility assessment chain
        self.utility_prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
        )
        self.utility_chain = self.utility_prompt | self.llm.with_structured_output(UtilityResponse)
    
    def process_query(self, query: str, vectorstore: FAISS) -> str:
        """Process a single query through the Self-RAG pipeline with detailed logging"""
        print(f"\nProcessing query: {query}")
        
        # Step 1: Determine if retrieval is necessary
        print("Step 1: Determining if retrieval is necessary...")
        retrieval_decision = self._determine_retrieval(query)
        
        if retrieval_decision == 'yes':
            # Step 2: Retrieve relevant documents
            print("Step 2: Retrieving relevant documents...")
            docs = vectorstore.similarity_search(query, k=self.config.top_k)
            contexts = [doc.page_content for doc in docs]
            print(f"Retrieved {len(contexts)} documents")
            
            # Step 3: Evaluate relevance
            print("Step 3: Evaluating relevance of retrieved documents...")
            relevant_contexts = []
            for i, context in enumerate(contexts, 1):
                relevance = self._evaluate_relevance(query, context)
                print(f"Document {i} relevance: {relevance}")
                if relevance == 'relevant':
                    relevant_contexts.append(context)
            
            print(f"Number of relevant contexts: {len(relevant_contexts)}")
            
            if not relevant_contexts:
                print("No relevant contexts found. Generating without retrieval...")
                return self._generate_response(query, "No relevant context found.")
            
            # Step 4-6: Generate and evaluate responses
            print("Step 4: Generating responses using relevant contexts...")
            responses = []
            for i, context in enumerate(relevant_contexts, 1):
                print(f"\nGenerating response for context {i}...")
                response = self._generate_response(query, context)
                
                print(f"Step 5: Assessing support for response {i}...")
                support = self._assess_support(response, context)
                print(f"Support assessment: {support}")
                
                print(f"Step 6: Evaluating utility for response {i}...")
                utility = self._evaluate_utility(query, response)
                print(f"Utility score: {utility}")
                
                responses.append((response, support, utility))
            
            # Select best response
            print("\nSelecting the best response...")
            best_response = max(responses, key=lambda x: (
                x[1] == 'fully supported', 
                x[2]
            ))
            print(f"Best response support: {best_response[1]}, utility: {best_response[2]}")
            
            return best_response[0]
        else:
            print("Generating without retrieval...")
            return self._generate_response(query, "No retrieval necessary.")
    
    def _determine_retrieval(self, query: str) -> str:
        """Determine if retrieval is needed for the query"""
        result = self.retrieval_chain.invoke({"query": query})
        return result.response.strip().lower()
    
    def _evaluate_relevance(self, query: str, context: str) -> str:
        """Evaluate relevance of a single context"""
        result = self.relevance_chain.invoke({
            "query": query,
            "context": context
        })
        return result.response.strip().lower()
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response for given query and context"""
        result = self.generation_chain.invoke({
            "query": query,
            "context": context
        })
        return result.response
    
    def _assess_support(self, response: str, context: str) -> str:
        """Assess how well the response is supported by the context"""
        result = self.support_chain.invoke({
            "response": response,
            "context": context
        })
        return result.response.strip().lower()
    
    def _evaluate_utility(self, query: str, response: str) -> int:
        """Evaluate utility of the response"""
        result = self.utility_chain.invoke({
            "query": query,
            "response": response
        })
        return int(result.response)


def main():
    """Main execution function with formatted output"""
    # Parse arguments and load environment
    args = parse_arguments()
    load_environment()
    
    # Create configuration
    config = SelfRAGConfig(
        top_k=args.top_k,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        verbose=args.verbose
    )
    
    # Load vector store
    try:
        print(f"\nLoading documents from: {args.input}")
        vectorstore = encode_pdf(args.input) 
    except Exception as e:
        print(f"Error loading vector store: {e}")
        sys.exit(1)
    
    # Initialize Self-RAG
    self_rag = SelfRAG(config)
    
    # Handle queries
    if args.query_file:
        with open(args.query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    elif args.query:
        queries = [args.query]
    else:
        queries = [
            "What is Mr. Market?",
            "Give me 5 wise quotes of Charlie Munger!"
        ]
    
    # Process each query
    for query in queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        response = self_rag.process_query(query, vectorstore)
        
        print(f"\n{'='*80}")
        print("FINAL RESPONSE:")
        print(response)
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

""" Example output

================================================================================
QUERY: What is Mr. Market?
================================================================================

Processing query: What is Mr. Market?
Step 1: Determining if retrieval is necessary...
Step 2: Retrieving relevant documents...
Retrieved 3 documents
Step 3: Evaluating relevance of retrieved documents...
Document 1 relevance: relevant
Document 2 relevance: relevant
Document 3 relevance: relevant
Number of relevant contexts: 3
Step 4: Generating responses using relevant contexts...

Generating response for context 1...
Step 5: Assessing support for response 1...
Support assessment: partially supported
Step 6: Evaluating utility for response 1...
Utility score: 5

Generating response for context 2...
Step 5: Assessing support for response 2...
Support assessment: fully supported
Step 6: Evaluating utility for response 2...
Utility score: 5

Generating response for context 3...
Step 5: Assessing support for response 3...
Support assessment: partially supported
Step 6: Evaluating utility for response 3...
Utility score: 5

Selecting the best response...
Best response support: fully supported, utility: 5

================================================================================
FINAL RESPONSE:
Mr. Market is a metaphor used in investing to describe the unpredictable and sometimes irrational behavior of the stock market. In this context, Mr. Market represents the daily fluctuations in stock prices and offers to buy or sell investments based on his changing perceptions of value. As a prudent investor or businessman, it is advised not to let Mr. Market's daily communication determine your view of the value of an investment. Instead, it is recommended to form your own ideas of the value of your holdings based on full reports from the company and not be swayed by Mr. Market's emotional swings.
================================================================================

================================================================================
QUERY: Give me 5 wise quotes of Charlie Munger!
================================================================================

Processing query: Give me 5 wise quotes of Charlie Munger!
Step 1: Determining if retrieval is necessary...
Step 2: Retrieving relevant documents...
Retrieved 3 documents
Step 3: Evaluating relevance of retrieved documents...
Document 1 relevance: relevant
Document 2 relevance: irrelevant
Document 3 relevance: relevant
Number of relevant contexts: 2
Step 4: Generating responses using relevant contexts...

Generating response for context 1...
Step 5: Assessing support for response 1...
Support assessment: partially supported
Step 6: Evaluating utility for response 1...
Utility score: 5

Generating response for context 2...
Step 5: Assessing support for response 2...
Support assessment: partially supported
Step 6: Evaluating utility for response 2...
Utility score: 5

Selecting the best response...
Best response support: partially supported, utility: 5

================================================================================
FINAL RESPONSE:
Here are 5 wise quotes of Charlie Munger:
1.
"Spend each day trying to be a little wiser than you were when you woke up."
2.
"In my whole life, I have known no wise people who didn't read all the time - none, zero."
3.
"The best thing a human being can do is to help another human being know more."
4.
"The big money is not in the buying and selling, but in the waiting."
5.
"The first rule is not to lose. The second rule is not to forget the first rule."
================================================================================

"""