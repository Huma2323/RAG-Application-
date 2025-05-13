import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator 
from get_embedding_function import get_embedding_function
from typing import List, Optional
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful research assistant. Use the following context to answer the question as accurately and coherently as possible. Keep your answer in a single paragraph and as related to the context as possible.

Context:
{context}

---

Question: {question}

Review the entire context, determine and use the most relevant information and return a direct, coherent and clear answer to the question. 

{format_instructions}

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

# Define Pydantic model for structured response parsing
class RAGResponse(BaseModel):
    """Schema for a structured RAG response."""
    answer: str = Field(description="The answer to the user's question")
    sources: List[str] = Field(description="List of document sources that support the answer")
    confidence_score: Optional[float] = Field(
        default=None, 
        description="Confidence score between 0 and 1"
    )
    
    # Update validator to use field_validator syntax for Pydantic v2
    @field_validator('confidence_score')
    def check_confidence_range(cls, v):
        """Ensure confidence score is between 0 and 1"""
        if v is not None and (v < 0 or v > 1):
            return 0.5  # Default to middle value if out of range
        return v
# Create the parser
parser = PydanticOutputParser(pydantic_object=RAGResponse)

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
        # Get format instructions
        format_instructions = parser.get_format_instructions()
        
        # Format the prompt with format instructions
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text, 
            question=query_text,
            format_instructions=format_instructions
        )
        
        # Invoke the model
        model = OpenAI(model="gpt-4o-mini", temperature=0)
        response_text = model.invoke(prompt)
        
        # Extract sources
        sources = [doc.metadata.get("id", None) for doc, _score in results]
         
        try:
            # Try to parse the response
            parsed_response = parser.parse(response_text)
            
            # If source information is missing, add it manually
            if not parsed_response.sources:
                parsed_response.sources = sources
                
            # Add a default confidence if not provided
            if parsed_response.confidence_score is None:
                parsed_response.confidence_score = 0.85
                
            return parsed_response
            
        except Exception as parsing_error:
            # Fallback if parsing fails
            print(f"Error parsing response: {parsing_error}")
            return RAGResponse(
                answer=response_text,
                sources=sources,
                confidence_score=0.7  # Default fallback confidence
            )
    
    except Exception as e:
        return RAGResponse(
            answer=f"Error: {str(e)}",
            sources=[],
            confidence_score=0.0
        )
    

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

# Update the process_query function to highlight the source content more clearly
def process_query(query_text):
    """
    Process the query and split the response and sources into separate outputs.
    Returns tuple of (clean_response, sources_text)
    """
    try:
        # Get the structured response from query_rag
        response_obj = query_rag(query_text)
        
        # Extract the answer text
        clean_response = response_obj.answer
        
        # Format sources
        sources_list = response_obj.sources
        
        # Additionally, get the actual source passages
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=5)
        
        # Format source passages with content
        detailed_sources = "<h3>Source Passages Used for Response</h3>"
        for i, (doc, score) in enumerate(results):
            source_id = doc.metadata.get("id", "Unknown")
            content = doc.page_content
            
            # Make it clearer which source contains which text
            detailed_sources += f"<div style='margin:15px 0; padding:10px; border:1px solid #ddd; border-radius:5px;'>"
            detailed_sources += f"<div style='font-weight:bold; color:#333;'>Source {i+1}: {source_id}</div>"
            detailed_sources += f"<div style='margin:8px 0; padding:8px; background-color:#f9f9f9; border-left:4px solid #007bff; font-style:italic;'>{content}</div>"
            detailed_sources += f"<div style='font-size:0.8em; color:#666;'>Relevance Score: {score:.4f}</div>"
            detailed_sources += "</div>"
        
        # Add confidence score if available
        if response_obj.confidence_score is not None:
            detailed_sources += f"<div style='margin-top:15px; padding:8px; background-color:#e6f3ff; border-radius:5px;'><b>Response Confidence Score:</b> {response_obj.confidence_score:.2f}</div>"
        
        return clean_response, detailed_sources
        
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        return error_message, "No sources available due to an error."
        

    
# Modify process_query to apply formatting
def enhanced_process_query(query_text):
    """Process a query and return formatted response and sources"""
    response, sources = process_query(query_text)
    return response, sources 


if __name__ == "__main__":
    main()