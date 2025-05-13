

import os
import gradio as gr
from dotenv import load_dotenv

# Import constants and functions from existing modules
from query_data import CHROMA_PATH, query_rag, RAGResponse, enhanced_process_query
from populate_database import (upload_and_process_files, 
    clear_database
)
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
# Load environment variables
load_dotenv()

# Update this function to show full source text content
def process_query_for_ui(query_text):
    """Process a query and format it nicely for the Gradio UI with detailed source content"""
    try:
        # Get structured response directly from query_rag
        response_obj = query_rag(query_text)
        
        # Safety check - make sure we have a proper object
        if not isinstance(response_obj, RAGResponse):
            # Try to convert dict to RAGResponse if needed
            if isinstance(response_obj, dict):
                response_obj = RAGResponse(**response_obj)
        
        # Extract the answer
        answer_text = response_obj.answer
        
        # Format confidence score if available
        confidence_html = ""
        if response_obj.confidence_score is not None:
            confidence_html = f"<div style='margin-top:10px; padding:5px; background-color:#e6f3ff; border-radius:5px;'><b>Confidence Score:</b> {response_obj.confidence_score:.2f}</div>"
        
        # Format the complete answer with confidence
        formatted_answer = f"{answer_text}{confidence_html}"
        
        # Retrieve actual source documents with content
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=5)
        
        # Format source passages with complete content
        sources_html = "<h3>Source Passages</h3>"
        for i, (doc, score) in enumerate(results):
            source_id = doc.metadata.get("id", "Unknown")
            content = doc.page_content
            
            # Nicely format each source with its content
            sources_html += f"<div style='margin:15px 0; padding:10px; border:1px solid #ddd; border-radius:5px;'>"
            sources_html += f"<div style='font-weight:bold; color:#333;'>Source {i+1}: {source_id}</div>"
            sources_html += f"<div style='margin:8px 0; padding:8px; background-color:#f9f9f9; border-left:4px solid #007bff; font-style:italic;'>{content}</div>"
            sources_html += f"<div style='font-size:0.8em; color:#666;'>Relevance Score: {score:.4f}</div>"
            sources_html += "</div>"
        
        return formatted_answer, sources_html
        
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(f"Error in process_query_for_ui: {str(e)}")
        
        # Fall back to legacy processing
        try:
            answer, sources = enhanced_process_query(query_text)
            return answer, sources
        except:
            return error_message, "No sources available due to an error."

# Define the Gradio interface with tabs
def create_interface():
    with gr.Blocks(title="RAG Question Answering System", css="""
        #response-box {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            background-color: #f9f9f9;
        }
        #sources-sidebar {
            max-height: 400px;
            overflow-y: auto;
            padding: 8px;
            border-left: 1px solid #eee;
        }
        .confidence-indicator {
            margin-top: 8px;
            height: 10px;
            border-radius: 5px;
        }
        """) as interface:
        
        with gr.Tabs():
            # Database Population Tab
            with gr.TabItem("Populate Database"):
                gr.Markdown("## Add Documents to the Database")
                gr.Markdown("Upload PDF files to add to the knowledge base.")
                
                # File upload component
                file_upload = gr.File(
                    file_count="multiple",
                    file_types=[".pdf"],
                    label="Upload PDF Files"
                )
                
                # Process and reset buttons
                with gr.Row():
                    process_btn = gr.Button("Process Files", variant="primary")
                    reset_btn = gr.Button("Reset Database", variant="secondary")
                
                # Status output
                status_output = gr.Textbox(
                    label="Status",
                    lines=3,
                    interactive=False
                )
                
                # Set up button actions
                process_btn.click(
                    fn=upload_and_process_files,
                    inputs=file_upload,
                    outputs=status_output
                )
                
                reset_btn.click(
                    fn=clear_database,
                    inputs=None,
                    outputs=status_output
                )
            
            # Question Answering Tab
            with gr.TabItem("Ask Questions"):
                gr.Markdown("# Document Question Answering")
                gr.Markdown("Ask a question about the documents in the database.")
                
                # Question input at the top
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question here...",
                    lines=2
                )
                submit_btn = gr.Button("Submit Question", variant="primary")
                
                # Answer box and sources side by side
                with gr.Row():
                    # Answer box on the left (wider)
                    with gr.Column(scale=3):
                        response_output = gr.HTML(
                            label="Answer",
                            elem_id="response-box"
                        )
                    
                    # Collapsible sources on the right (narrower)
                    with gr.Column(scale=1):
                        with gr.Accordion("View Sources", open=True):  # Default to open so users see sources
                            sources_output = gr.HTML(
                                elem_id="sources-sidebar",
                                label="Source Documents"
                            )
                
                # Set up the submission action to update both outputs
                submit_btn.click(
                    fn=process_query_for_ui,  # Use the new function
                    inputs=question_input,
                    outputs=[response_output, sources_output]
                )
                
                # # Add a checkbox to switch between using Pydantic and legacy
                # with gr.Accordion("Settings", open=False):
                #     use_legacy = gr.Checkbox(
                #         label="Use Legacy Processing (No Pydantic)",
                #         value=False
                #     )
                    
                    # # Add callback to change processing method
                    # def update_processing_method(use_legacy_mode):
                    #     if use_legacy_mode:
                    #         submit_btn.click(
                    #             fn=enhanced_process_query,
                    #             inputs=question_input,
                    #             outputs=[response_output, sources_output],
                    #             api_name="legacy_query"
                    #         )
                    #     else:
                    #         submit_btn.click(
                    #             fn=process_query_for_ui,
                    #             inputs=question_input,
                    #             outputs=[response_output, sources_output],
                    #             api_name="pydantic_query"
                    #         )
                    #     return "Processing method updated"
                    
                    # use_legacy.change(
                    #     fn=update_processing_method,
                    #     inputs=use_legacy,
                    #     outputs=gr.Textbox(visible=False)
                    # )
        
    return interface

# Launch the app
if __name__ == "__main__":
    # Verify the Chroma database exists
    if not os.path.exists(CHROMA_PATH):
        print(f"Warning: Chroma database not found at {CHROMA_PATH}. Please run populate_database.py first.")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(share=False)