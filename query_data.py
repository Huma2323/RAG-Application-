import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful research assistant. Use the following context to answer the question as accurately and coherently as possible. If the context does not contain enough information, respond with "The context does not provide enough information to answer the question."

Context:
{context}

---

Question: {question}

Review the entire context, determine and use the most relevant information and return a direct, coherent and clear answer to the question. 
Answer:
"""


def main():
    load_dotenv()
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


# def query_rag(query_text: str):
#     # Prepare the DB.
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     print(prompt)

#     model = OpenAI(model="gpt-4o-mini", temperature=0)
#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
    # return response_text


def query_rag(query_text: str):
    """Query the RAG system and return the response."""
    try:
        # Prepare the DB
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Search the DB
        results = db.similarity_search_with_score(query_text, k=5)
        
        # Prepare the context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Format the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Invoke the model
        model = OpenAI(model="gpt-4o-mini", temperature=0)
        response_text = model.invoke(prompt)
        
        # Extract sources
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        
        # Format the response with sources
        return f"{response_text}\n\nSources:\n" + "\n".join(sources)
    
    except Exception as e:
        return f"Error: {str(e)}"
    

# Define a function to format source content with italics
def format_sources_with_styles(sources_text):
    """Format source content to have italicized text using Markdown"""
    formatted_text = ""
    lines = sources_text.split('\n')
    
    in_content = False
    for line in lines:
        if line.startswith("Content:"):
            # Start of content section
            formatted_text += "Content: *"
            in_content = True
            # Add the content after "Content:" label
            if ":" in line:
                content_part = line.split(":", 1)[1]
                formatted_text += f"{content_part.strip()}*\n"
        elif line.startswith("Source") or line.startswith("Relevance") or line.startswith("-"):
            # End of content section if we encounter a new source or separator
            if in_content:
                in_content = False
            formatted_text += line + "\n"
        else:
            # Continue content or add non-content lines
            if in_content:
                formatted_text += f"*{line.strip()}*\n"
            else:
                formatted_text += line + "\n"
    
    return formatted_text

def process_query(query_text):
    """
    Process the query and split the response and sources into separate outputs.
    Returns tuple of (clean_response, sources_text)
    """
    try:
        # Get the raw response from query_rag
        full_response = query_rag(query_text)
        
        # Split response if it contains "Sources:"
        if "\nSources:" in full_response:
            clean_response, sources_part = full_response.split("\nSources:", 1)
            sources_text = f"Sources:\n{sources_part.strip()}"
        else:
            clean_response = full_response
            sources_text = "No sources available"
            
        # Additionally, get the actual source passages
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=5)
        
        # Format source passages with content
        detailed_sources = "Source Passages:\n\n"
        for i, (doc, score) in enumerate(results):
            source_id = doc.metadata.get("id", "Unknown")
            content = doc.page_content
            detailed_sources += f"Source {i+1}: {source_id}\n"
            detailed_sources += f"Content: {content}\n"
            detailed_sources += f"Relevance: {score:.4f}\n\n"
            detailed_sources += "-" * 40 + "\n\n"
        
        return clean_response, detailed_sources
        
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        return error_message, "No sources available due to an error."

# Modify process_query to apply formatting
def enhanced_process_query(query_text):
    response, sources = process_query(query_text)
    formatted_sources = format_sources_with_styles(sources)
    return response, formatted_sources


if __name__ == "__main__":
    main()