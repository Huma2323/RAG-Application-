# import argparse
# import os
# import shutil
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_unstructured import UnstructuredLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from get_embedding_function import get_embedding_function
# from langchain_chroma import Chroma
# from dotenv import load_dotenv


# CHROMA_PATH = "chroma"
# DATA_PATH = "documents"


# def main(): 
#     # Load environment variables
#     load_dotenv()

#     # Check if the database should be cleared (using the --clear flag).
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
#     if args.reset:
#         print("✨ Clearing Database")
#         clear_database()

#     # Create (or update) the data store.
#     documents = load_documents()
#     chunks = split_documents(documents)
#     add_to_chroma(chunks)


# def load_documents():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()


# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=200,
#         chunk_overlap=40,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_splitter.split_documents(documents)


# def add_to_chroma(chunks: list[Document]):
#     # Load the existing database.
#     db = Chroma(
#         persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
#     )

#     # Calculate Page IDs.
#     chunks_with_ids = calculate_chunk_ids(chunks)

#     # Add or Update the documents.
#     existing_items = db.get(include=[])  # IDs are always included by default
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")

#     # Only add documents that don't exist in the DB.
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)

#     if len(new_chunks):
#         print(f"👉 Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#     else:
#         print("✅ No new documents to add")


# def calculate_chunk_ids(chunks):

#     # This will create IDs like "data/monopoly.pdf:6:2"
#     # Page Source : Page Number : Chunk Index

#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"

#         # If the page ID is the same as the last one, increment the index.
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#         # Calculate the chunk ID.
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#         # Add it to the page meta-data.
#         chunk.metadata["id"] = chunk_id

#     return chunks


# def clear_database():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)


# if __name__ == "__main__":
#     main()
import argparse
import os
import shutil
import tempfile
import logging
from typing import List, Optional, Dict, Any
from enum import Enum


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from dotenv import load_dotenv


# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "documents"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80




def main():
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Populate the document database")
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Size of text chunks (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--api", 
        action="store_true", 
        help="Use Unstructured API instead of local processing (requires API key)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Clear the database if requested
    if args.reset:
        logger.info("✨ Clearing Database")
        clear_database()

    # Create or update the data store
    try:
        documents = load_documents(
            strategy=args.strategy,
            use_api=args.api
        )
        logger.info(f"Loaded {len(documents)} document segments")
        
        chunks = split_documents(
            documents, 
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap
        )
        logger.info(f"Split into {len(chunks)} chunks")
        
        add_to_chroma(chunks)
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return 1
    
    return 0





def split_documents(documents, chunk_size=800, chunk_overlap=100):
    """
    Enhanced document splitting with improved chunking strategy.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked Document objects
    """
    
    
    # Create a splitter with smart defaults for PDF content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Handle different paragraph styles
    )
    
    # Process each document to maintain metadata
    all_chunks = []
    
    for doc in documents:
        # Extract metadata before splitting
        metadata = doc.metadata
        
        # Split the document content
        chunks = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[metadata]
        )
        
        all_chunks.extend(chunks)
    
    return all_chunks
    



def add_to_chroma(chunks: List[Document]):
    """
    Add document chunks to Chroma vectorstore.
    
    Args:
        chunks: List of document chunks to add
    """
    # Load the existing database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate IDs for chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing document IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"]) if existing_items and "ids" in existing_items else set()
    logging.info(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        logging.info(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        logging.info("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    """
    Generate unique IDs for document chunks based on content and position.
    
    Args:
        chunks: List of document chunks
    
    Returns:
        List of document chunks with IDs added to metadata
    """
    last_source_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # Build a source identifier based on available metadata
        metadata = chunk.metadata
        source = metadata.get("source_file", metadata.get("source", "unknown"))
        
        # For structured documents
        if "category" in metadata:
            category = metadata.get("category", "text")
            page_num = metadata.get("page_number", 0)
            element_id = metadata.get("element_id", "")
            source_id = f"{source}:{page_num}:{category}:{element_id}"
        # For standard documents
        else:
            page = metadata.get("page", 0)
            source_id = f"{source}:{page}"

        # If the source ID is the same as the last one, increment the index
        if source_id == last_source_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            last_source_id = source_id

        # Generate the chunk ID
        chunk_id = f"{source_id}:{current_chunk_index}"
        
        # Add it to the metadata
        chunk.metadata["id"] = chunk_id

    return chunks

def upload_and_process_files(files):
    """
    Process uploaded PDF files and add them to the database using simple PDF extraction.
    
    Args:
        files: List of uploaded file objects from Gradio
    
    Returns:
        Status message
    """
    try:
        if not files:
            return "No files were uploaded."
        
        # Ensure the data directory exists
        os.makedirs(DATA_PATH, exist_ok=True)
        
        # Process each file individually to avoid memory issues
        total_elements = 0
        processed_files = []
        
        for file in files:
            # Get just the filename from the path
            filename = os.path.basename(file.name)
            dest_path = os.path.join(DATA_PATH, filename)
            
            # Copy the file (Gradio already saved it at a temp location)
            shutil.copy(file.name, dest_path)
            processed_files.append(dest_path)
            
            # Use simple PDF extraction - always reliable
            try:
                
                logging.info(f"Processing {filename} with simple PDF extraction")
                loader = PyPDFLoader(dest_path)
                documents = loader.load()
                
                # Add source filename to metadata
                for doc in documents:
                    doc.metadata["source_file"] = filename
                    # Add page number if not present
                    if "page" not in doc.metadata:
                        doc.metadata["page"] = doc.metadata.get("page_number", 0)
                
                total_elements += len(documents)
                
                # Split documents into chunks
                chunks = split_documents(documents)
                
                # Add to Chroma
                add_to_chroma(chunks)
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                return f"❌ Error processing {filename}: {str(e)}"
        
        return f"✅ Successfully processed {len(processed_files)} files and extracted {total_elements} document elements."
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Error details: {error_details}")
        return f"❌ Error processing files: {str(e)}"


def clear_database():
    """Remove the Chroma directory to clear the database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info(f"Database at {CHROMA_PATH} has been cleared")


if __name__ == "__main__":
    main()