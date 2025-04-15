import os
import sys
from dotenv import load_dotenv
from typing import List, Any

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Error: OPENAI_API_KEY not found. Please set it in the .env file.")
    sys.exit(1)
os.environ["OPENAI_API_KEY"] = api_key

from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

from pyrag.utils.helper_functions import *
from pyrag.evaluation.evaluate_rag import *


class CategoriesOptions(BaseModel):
    category: str = Field(
        description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual",
        example="Factual"
    )


class QueryClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\n"
                "Query: {query}\nCategory:"
            )
        )
        self.chain = self.prompt | self.llm.with_structured_output(CategoriesOptions)

    def classify(self, query: str) -> str:
        print("Classifying query...")
        return self.chain.invoke(query).category


class RelevantScore(BaseModel):
    score: float = Field(description="The relevance score of the document to the query", example=8.0)


class BaseRetrievalStrategy:
    def __init__(self, texts: List[str]):
        self.embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        self.documents = text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        return self.db.similarity_search(query, k=k)


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        print("Retrieving factual information...")
        # Enhance the query using LLM
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f'Enhanced Query: {enhanced_query}')

        # Retrieve documents using the enhanced query
        docs = self.db.similarity_search(enhanced_query, k=k*2)

        # Rank the relevance of retrieved documents
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
            input_data = {"query": enhanced_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]


class SelectedIndices(BaseModel):
    indices: List[int] = Field(description="Indices of selected documents", example=[0, 1, 2, 3])


class SubQueries(BaseModel):
    sub_queries: List[str] = Field(
        description="List of sub-queries for comprehensive analysis",
        example=["What is the population of New York?", "What is the GDP of New York?"]
    )


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        print("Retrieving analytical information...")
        # Generate sub-queries using LLM
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Generate {k} sub-questions for: {query}"
        )

        sub_queries_chain = sub_queries_prompt | self.llm.with_structured_output(SubQueries)

        input_data = {"query": query, "k": k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries
        print(f'Sub-queries for comprehensive analysis: {sub_queries}')

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))

        # Ensure diversity and relevance using LLM
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template=(
                "Select the most diverse and relevant set of {k} documents for the query: '{query}'\n"
                "Documents: {docs}\n"
                "Return only the indices of selected documents as a list of integers."
            )
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices_result = diversity_chain.invoke(input_data).indices
        print('Selected diverse and relevant documents.')

        return [all_docs[i] for i in selected_indices_result if i < len(all_docs)]


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        print("Retrieving opinion-based information...")
        # Identify potential viewpoints using LLM
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Identify {k} distinct viewpoints or perspectives on the topic: {query}"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": query, "k": k}
        viewpoints = viewpoints_chain.invoke(input_data).content.split('\n')
        print(f'Identified viewpoints: {viewpoints}')

        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k=2))

        # Classify and select diverse opinions using LLM
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template=(
                "Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints:\n"
                "Documents: {docs}\nSelected indices:"
            )
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)

        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices = opinion_chain.invoke(input_data).indices
        print('Selected diverse and relevant opinion documents.')

        return [all_docs[i] for i in selected_indices if isinstance(i, int) and i < len(all_docs)]


class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query: str, k: int = 4, user_context: str = None) -> List[Document]:
        print("Retrieving contextual information...")
        # Incorporate user context into the query using LLM
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Given the user context: {context}\n"
                "Reformulate the query to best address the user's needs: {query}"
            )
        )
        context_chain = context_prompt | self.llm
        input_data = {"query": query, "context": user_context or "No specific context provided"}
        contextualized_query = context_chain.invoke(input_data).content
        print(f'Contextualized Query: {contextualized_query}')

        # Retrieve documents using the contextualized query
        docs = self.db.similarity_search(contextualized_query, k=k*2)

        # Rank the relevance of retrieved documents considering user context
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template=(
                "Given the query: '{query}' and user context: '{context}', rate the relevance of this document on a scale of 1-10:\n"
                "Document: {doc}\nRelevance score:"
            )
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)
        print("Ranking contextual documents...")

        ranked_docs = []
        for doc in docs:
            input_data = {
                "query": contextualized_query,
                "context": user_context or "No specific context provided",
                "doc": doc.page_content
            }
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]


class AdaptiveRetriever:
    def __init__(self, texts: List[str]):
        self.classifier = QueryClassifier()
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts),
            "Analytical": AnalyticalRetrievalStrategy(texts),
            "Opinion": OpinionRetrievalStrategy(texts),
            "Contextual": ContextualRetrievalStrategy(texts)
        }

    def get_relevant_documents(self, query: str) -> List[Document]:
        category = self.classifier.classify(query)
        strategy = self.strategies.get(category)
        if not strategy:
            print(f"Unknown category '{category}'. Using default retrieval strategy.")
            return []
        return strategy.retrieve(query)


class PydanticAdaptiveRetriever(BaseRetriever):
    adaptive_retriever: AdaptiveRetriever = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.adaptive_retriever.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


class AdaptiveRAG:
    def __init__(self, texts: List[str]):
        adaptive_retriever = AdaptiveRetriever(texts)
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

        prompt_template = (
            "Use the following pieces of context to answer the question at the end. \n"
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            "{context}\n\n"
            "Question: {question}\nAnswer:"
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        self.llm_chain = prompt | self.llm

    def answer(self, query: str) -> Any:
        docs = self.retriever.get_relevant_documents(query)
        input_data = {
            "context": "\n".join([doc.page_content for doc in docs]),
            "question": query
        }
        return self.llm_chain.invoke(input_data)


def main():
    texts = [
        "The Earth is the third planet from the Sun and the only astronomical object known to harbor life."
    ]

    rag_system = AdaptiveRAG(texts)

    queries = [
        "What is the distance between the Earth and the Sun?",
        "How does the Earth's distance from the Sun affect its climate?",
        "What are the different theories about the origin of life on Earth?",
        "How does the Earth's position in the Solar System influence its habitability?"
    ]

    for query in queries:
        result = rag_system.answer(query)
        print(f"Question: {query}\nAnswer: {result.content}\n")


if __name__ == "__main__":
    main()