from typing import Dict, Any, Optional
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import ollama
from vector_store import get_retriever, get_docstore, get_vectorstore

llm = ChatOllama(model="qwen2.5vl:7b", temperature=0.1)
citation_llm = ChatOllama(model="qwen2.5vl:7b", temperature=0.0, format="json")
print("RAG chain components initialized successfully.")

def analyze_query_image(image_b64: str) -> str:
    """
    Use a multimodal LLM to generate a detailed description for the user's uploaded query image.
    """
    print("Analyzing user's query image...")
    try:
        response = ollama.chat(
            model='qwen2.5vl:7b',
            messages=[{
                'role': 'user',
                'content': 'Describe this image in detail. Focus on key objects, text, charts, and the overall context. This description will be used to find relevant information in a database.',
                'images': [image_b64]
            }],
            options={"temperature": 0.1}
        )
        description = response['message']['content']
        print(f"Query image description generated: {description[:100]}...")
        return description
    except Exception as e:
        print(f"ERROR: Failed to analyze query image: {e}")
        return ""

def rag_chain_with_source_retrieval(input_question: str, query_image_b64: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrates a RAG process that accepts an optional query image.
    """
    print(f"--- Running RAG chain for question: '{input_question[:50]}...' ---")
    search_query = input_question
    if query_image_b64:
        print("Image provided in query. Analyzing image to enhance search query...")
        image_description = analyze_query_image(query_image_b64)
        search_query = f"{input_question}\n\n[Information from uploaded image]:\n{image_description}"
        
    print(f"Step 2: Retrieving documents with enhanced query: '{search_query[:100]}...'")
    direct_results = get_vectorstore().similarity_search(search_query, k=10)
    unique_results = []
    seen_contents = set()
    for doc in direct_results:
        content = doc.page_content.strip()
        if content not in seen_contents:
            unique_results.append(doc)
            seen_contents.add(content)
    print(f"Found {len(unique_results)} unique documents.")
    
    if not unique_results:
        print("No relevant documents found. Generating a direct response using the VLM.")
        messages = [{'role': 'user', 'content': input_question}]
        if query_image_b64:
            messages[0]['images'] = [query_image_b64]
        response = ollama.chat(model="qwen2.5vl:7b", temperature=0.1, messages=messages)
        return {"answer": response['message']['content'], "sources": []}   
    
    print("Step 3: Creating prompt for answer generation...")
    full_context_str = "\n\n---\n\n".join([doc.page_content for doc in unique_results])
    
    answer_generation_prompt_template = ChatPromptTemplate.from_template(
        """
        **Your Role**: You are a document analysis assistant.
        **Task**: Based ONLY on the "Context Information" below, answer the "User's Question".
        ---
        **Context Information**:
        {context}
        ---
        **User's Question**:
        {question}
        ---
        **Your Answer**:
        """
    )
    answer_generation_chain = answer_generation_prompt_template | llm | StrOutputParser()
    generated_answer = answer_generation_chain.invoke({
        "context": full_context_str,
        "question": search_query 
    })
    print(f"Generated Answer: {generated_answer[:100]}...")
    
    print("Step 4: Identifying relevant sources...")
    source_options = [f"<SOURCE_{i+1}>\n{doc.page_content}\n</SOURCE_{i+1}>" for i, doc in enumerate(unique_results)]
    all_sources_str = "\n\n".join(source_options)
    
    citation_prompt_template = ChatPromptTemplate.from_template(
    """
    You are a highly analytical and skeptical citation-finding assistant. Your ONLY job is to determine which of the provided sources were *actually used* to create the given answer.
    
    You must follow these STRICT rules:
    1. Compare the "Generated Answer" against each "Source" provided below.
    2. Identify ONLY the sources that contain the EXACT information, facts, or data points present in the answer.
    3. **CRITICAL RULE**: Do NOT cite a source just because it shares some keywords with the answer. The source must SEMANTICALLY support the claims in the answer.
    4. If the answer makes a specific claim (e.g., "Revenue was $5M"), the source MUST contain that exact fact.
    5. Be extremely skeptical. It is better to return an empty list than to cite an irrelevant source.
    6. Respond with a JSON object containing a single key "cited_sources". The value should be a list of the IDs of the relevant sources (e.g., {{{{"cited_sources": ["SOURCE_1", "SOURCE_3"]}}}}).
    7. If no source directly supports the answer, you MUST return an empty list: {{{{"cited_sources": []}}}}.
    8. Do not add any explanation or text outside of the JSON object.
    ---
    **Generated Answer**:
    {answer}
    ---
    **Available Sources**:
    {sources}
    ---
    **Your JSON Response**:
    """
    )
    citation_chain = citation_prompt_template | citation_llm | JsonOutputParser()
    
    try:
        citation_response = citation_chain.invoke({
            "answer": generated_answer,
            "sources": all_sources_str
        })
        cited_source_ids = citation_response.get("cited_sources", [])
    except Exception as e:
        print(f"ERROR: Failed to get citations from LLM. Returning empty sources. Error: {e}")
        cited_source_ids = []
        
    final_sources = []
    for i, doc in enumerate(unique_results):
        source_id = f"SOURCE_{i+1}"
        if source_id in cited_source_ids:
            metadata = doc.metadata
            final_sources.append({
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "summary": doc.page_content,
                "type": metadata.get("type", "text"),
                "image_b64": metadata.get("content_b64") if metadata.get("type") == "image" else None
            })
    print(f"Chain finished. Returning answer and {len(final_sources)} cited sources.")
    
    return {
        "answer": generated_answer,
        "sources": final_sources
    }

if __name__ == '__main__':
    print("\n--- Running standalone test for the RAG chain ---")
    print("--- Make sure you have run vector_store.py first to build the database! ---")
    while True:
        try:
            test_question = input("\nEnter your question (or type 'quit' to exit): ")
            if test_question.lower() == 'quit':
                break
            if not test_question:
                continue
            response_dict = rag_chain_with_source_retrieval(test_question)
            print("\n" + "="*50)
            print(" RAG RESPONSE")
            print("="*50)
            print(f"\n[ANSWER]:\n{response_dict['answer']}")
            if response_dict['sources']:
                print("\n[CITED SOURCES]:")
                for i, source in enumerate(response_dict['sources'], 1):
                    print(f"  Source {i}: File Name: {source['source']}, Page: {source['page']}")
            else:
                print("\n[CITED SOURCES]: None")
            print("="*50 + "\n")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break
