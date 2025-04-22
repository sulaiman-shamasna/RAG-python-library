import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
from pyrag.utils.helper_functions import *
from pyrag.evaluation.evaluate_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

path = "data/the_intelligent_investor_ch_8.pdf"
vectorstore = encode_pdf(path)

llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1000, temperature=0)

class RetrievalResponse(BaseModel):
    response: str = Field(..., title="Determines if retrieval is necessary", description="Output only 'Yes' or 'No'.")
retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
)

class RelevanceResponse(BaseModel):
    response: str = Field(..., title="Determines if context is relevant", description="Output only 'Relevant' or 'Irrelevant'.")
relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)

class GenerationResponse(BaseModel):
    response: str = Field(..., title="Generated response", description="The generated response.")
generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)

class SupportResponse(BaseModel):
    response: str = Field(..., title="Determines if response is supported", description="Output 'Fully supported', 'Partially supported', or 'No support'.")
support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)

class UtilityResponse(BaseModel):
    response: int = Field(..., title="Utility rating", description="Rate the utility of the response from 1 to 5.")
utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)

# Create LLMChains for each step
retrieval_chain = retrieval_prompt | llm.with_structured_output(RetrievalResponse)
relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)
generation_chain = generation_prompt | llm.with_structured_output(GenerationResponse)
support_chain = support_prompt | llm.with_structured_output(SupportResponse)
utility_chain = utility_prompt | llm.with_structured_output(UtilityResponse)

def self_rag(query, vectorstore, top_k=3):
    print(f"\nProcessing query: {query}")
    
    # Step 1: Determine if retrieval is necessary
    print("Step 1: Determining if retrieval is necessary...")
    input_data = {"query": query}
    retrieval_decision = retrieval_chain.invoke(input_data).response.strip().lower()
    print(f"Retrieval decision: {retrieval_decision}")
    
    if retrieval_decision == 'yes':
        # Step 2: Retrieve relevant documents
        print("Step 2: Retrieving relevant documents...")
        docs = vectorstore.similarity_search(query, k=top_k)
        contexts = [doc.page_content for doc in docs]
        print(f"Retrieved {len(contexts)} documents")
        
        # Step 3: Evaluate relevance of retrieved documents
        print("Step 3: Evaluating relevance of retrieved documents...")
        relevant_contexts = []
        for i, context in enumerate(contexts):
            input_data = {"query": query, "context": context}
            relevance = relevance_chain.invoke(input_data).response.strip().lower()
            print(f"Document {i+1} relevance: {relevance}")
            if relevance == 'relevant':
                relevant_contexts.append(context)
        
        print(f"Number of relevant contexts: {len(relevant_contexts)}")
        
        # If no relevant contexts found, generate without retrieval
        if not relevant_contexts:
            print("No relevant contexts found. Generating without retrieval...")
            input_data = {"query": query, "context": "No relevant context found."}
            return generation_chain.invoke(input_data).response
        
        # Step 4: Generate response using relevant contexts
        print("Step 4: Generating responses using relevant contexts...")
        responses = []
        for i, context in enumerate(relevant_contexts):
            print(f"Generating response for context {i+1}...")
            input_data = {"query": query, "context": context}
            response = generation_chain.invoke(input_data).response
            
            # Step 5: Assess support
            print(f"Step 5: Assessing support for response {i+1}...")
            input_data = {"response": response, "context": context}
            support = support_chain.invoke(input_data).response.strip().lower()
            print(f"Support assessment: {support}")
            
            # Step 6: Evaluate utility
            print(f"Step 6: Evaluating utility for response {i+1}...")
            input_data = {"query": query, "response": response}
            utility = int(utility_chain.invoke(input_data).response)
            print(f"Utility score: {utility}")
            
            responses.append((response, support, utility))
        
        # Select the best response based on support and utility
        print("Selecting the best response...")
        best_response = max(responses, key=lambda x: (x[1] == 'fully supported', x[2]))
        print(f"Best response support: {best_response[1]}, utility: {best_response[2]}")
        return best_response[0]
    else:
        # Generate without retrieval
        print("Generating without retrieval...")
        input_data = {"query": query, "context": "No retrieval necessary."}
        return generation_chain.invoke(input_data).response
    

def main():
    query = "What is Mr. Market?"
    response = self_rag(query, vectorstore)

    print("\nFinal response:")
    print(response)


    query = "Give me 5 wise quotes of Charlie Munger!"
    response = self_rag(query, vectorstore)

    print("\nFinal response:")
    print(response)

if __name__ == "__main__":
    main()

""" Example output

Processing query: What is Mr. Market?
Step 1: Determining if retrieval is necessary...
Retrieval decision: yes
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

Final response:
Mr. Market is a metaphor used by Benjamin Graham to describe the stock market's behavior. In Graham's analogy, Mr. Market is a person who offers to buy or sell stocks at different prices every day. As an investor, it is important to not let Mr. Market's daily fluctuations dictate your view of the value of your investments. Instead, it is wise to form your own opinions based on thorough analysis and company reports.

Processing query: Give me 5 wise quotes of Charlie Munger!
Step 1: Determining if retrieval is necessary...
Retrieval decision: yes
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
Utility score: 4
Selecting the best response...
Best response support: partially supported, utility: 5

Final response:
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

"""