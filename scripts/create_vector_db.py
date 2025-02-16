"""Ingest a JSONL file of Dorar lectures into a vector store (Chroma)
and store relevant artifacts in Weights & Biases.
"""
import json
import time
import logging
import re
import os
import pathlib
from typing import List, Union
import shutil
import langchain
import openai
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.cache import SQLiteCache
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import wandb
from wandb import Artifact



########################################
# CONFIG
########################################
# Path to the JSONL file containing the scraped lecture data.
LECTURES_JSONL = "../Data/bak/lectures.jsonl"

# Directory where the Chroma vector database will be persisted.
CHROMA_PERSIST_DIR = "../Data/vector_store"

# Option to overwrite the existing vector database. If True, any existing data in CHROMA_PERSIST_DIR
# will be removed before creating a new vector database. If False, new documents will be appended.
OVERWRITE_VECTOR_DB = True

# Chunking parameters: chunk size (approximate token count) and overlap.
CHUNK_SIZE = 8096
CHUNK_OVERLAP = 200

# Additional config for controlling request size & retries
EMBED_BATCH_SIZE = 100
EMBED_MAX_RETRIES = 5


# OpenAI Embedding model to use.
OPENAI_EMBED_MODEL = "text-embedding-ada-002"    # or "text-embedding-3-small"

LANGCHAIN_DATABASE = "../Data/langchain.db"

# Weights & Biases project name.
WANDB_PROJECT = "llmapps"
OPENAI_KEY_FILE = "../key.txt"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
langchain.llm_cache = SQLiteCache(database_path=LANGCHAIN_DATABASE)


########################################
# Configure OpenAI API Key
########################################
def configure_openai_api_key():
    """
    Ensure that the OPENAI_API_KEY is set in the environment.
    If not, attempt to read it from a local 'key.txt' file.
    """
    if os.getenv("OPENAI_API_KEY") is None:
        # fallback: read from local 'key.txt' if you have it
        if os.path.exists(OPENAI_KEY_FILE):
            with open(OPENAI_KEY_FILE, "r", encoding="utf-8") as f:
                os.environ["OPENAI_API_KEY"] = f.read().strip()
    if not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        raise ValueError("Please set a valid OPENAI_API_KEY environment variable or key.txt")


########################################
# Utility to Remove Arabic Diacritics (Harakat)
########################################
def remove_arabic_diacritics(text: str) -> str:
    """Remove common Arabic diacritics (harakat) from the text."""
    return re.sub(r"[\u0617-\u061A\u064B-\u0652]+", "", text)


########################################
# A Custom OpenAIEmbeddings Subclass to Handle RateLimitError
########################################
class RateLimitRetryOpenAIEmbeddings(OpenAIEmbeddings):
    """
    Subclass of OpenAIEmbeddings that intercepts RateLimitError.
    When encountered, sleeps for 60 seconds, then retries.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = EMBED_BATCH_SIZE
        all_embeddings = []
        # Process the texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            success = False
            while not success:
                try:
                    # Call the original embed_documents for the current batch
                    batch_embeddings = super().embed_documents(batch)
                    success = True
                except openai.RateLimitError as e:
                    logger.warning(
                        f"Rate limit error on batch {i//batch_size}: {e}. Waiting 60 seconds before retrying..."
                    )
                    time.sleep(60)
            all_embeddings.extend(batch_embeddings)
            # Optional: add a short pause between batches to help with rate limits
            time.sleep(1)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        while True:
            try:
                return super().embed_query(text)
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit error on embed_query: {e}. Waiting 60 seconds before retrying...")
                time.sleep(60)


########################################
# Parse the Lectures from JSONL
########################################
def load_jsonl_documents(jsonl_path: str) -> List[Document]:
    """
    Load Dorar lecture data from a JSONL file and convert it into a list of Document objects.

    Each JSON line is expected to be a nested structure, for example:
    {
        "Category_Name": {
            "SubSection_Name": [
                {
                  "lecture_title": "...",
                  "lecture_url": "...",
                  "content": "...",
                  "path": "Category_Name > SubSection_Name",
                  "title": "..."
                },
                ...
            ]
        }
    }

    Returns:
        List[Document]: List of Document objects with page_content and metadata.
    """
    all_docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line.")
                continue

            # Convert this "data" into documents
            docs_from_line = extract_docs_from_object(data)
            all_docs.extend(docs_from_line)

    return all_docs


def extract_docs_from_object(obj: Union[dict, list], parent_path: str = "ROOT") -> List[Document]:
    """
    Recursively traverse the nested JSON structure to create Document objects.

    - If obj is a dict, iterate over key-value pairs and update the parent path.
    - If obj is a list, check each element:
        * If the element is a lecture object (has 'content', 'lecture_url', 'title'), create a Document.
        * Otherwise, process recursively.

    Args:
        obj: The JSON object (dict or list) to process.
        parent_path: The string representing the hierarchical path for metadata.

    Returns:
        List[Document]: Extracted Document objects.
    """
    docs = []

    if isinstance(obj, dict):
        for key, val in obj.items():
            new_path = f"{parent_path} > {key}"
            docs.extend(extract_docs_from_object(val, new_path))
    elif isinstance(obj, list):
        # Possibly a list of "lecture" dicts or further sub-lists
        for item in obj:
            if isinstance(item, dict):
                # If item has 'content' => treat it as a lecture
                if "content" in item and "lecture_url" in item:
                    content_text = item["content"]
                    metadata = {
                        "title": item.get("title", "NoTitle"),
                        "url": item.get("lecture_url", "NoURL"),
                        "path": item.get("path", parent_path),
                    }
                    docs.append(Document(page_content=content_text, metadata=metadata))
                else:
                    # Nested structure
                    docs.extend(extract_docs_from_object(item, parent_path))
            else:
                # Not a dict => skip or handle differently
                pass
    else:
        # Some other data type => skip
        pass

    return docs


########################################
# Chunk Documents
########################################
def chunk_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> \
List[Document]:
    """
    Split each Document into smaller chunks using MarkdownTextSplitter.

    Args:
        documents: List of Document objects.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.

    Returns:
        List[Document]: List of chunked Document objects.
    """
    markdown_text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = markdown_text_splitter.split_documents(documents)
    return split_docs


########################################
# Create a Chroma DB from Documents
########################################
def create_vector_store(documents: List[Document], persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    """
    Create a Chroma vector store from a list of documents using OpenAI embeddings.

    Args:
        documents: List of Document objects to be embedded.
        persist_dir: Directory to persist the Chroma DB.

    Returns:
        Chroma: The created vector store.
    """
    logger.info(f"Creating embeddings for {len(documents)} documents...")

    # Remove harakat from each document before embedding
    for doc in documents:
        doc.page_content = remove_arabic_diacritics(doc.page_content)
        pass

    # Create the embedding function using the global API key.
    embedding_function = RateLimitRetryOpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=OPENAI_EMBED_MODEL,
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_dir,
    )
    logger.info(f"Vector store persisted at '{persist_dir}'.")
    return vector_store


########################################
# Weights & Biases Logging Functions
########################################
def log_dataset(documents: List[Document], run: wandb.run):
    """
    Log the dataset to Weights & Biases as a dataset artifact.

    Each Document is written as a JSON line.
    """

    # Open file in binary write mode to force UTF-8 encoding.
    doc_artifact = Artifact(name="documentation_dataset", type="dataset")
    with doc_artifact.new_file("documents.json", mode="wb") as f:
        for doc in documents:
            # Use model_dump_json() (Pydantic V2) instead of the deprecated json() method.
            json_str = doc.model_dump_json() + "\n"
            f.write(json_str.encode("utf-8"))
    run.log_artifact(doc_artifact)
    logger.info("Dataset logged to W&B.")


def log_index(vector_store_dir: str, run: wandb.run):
    """
    Log the vector store directory as a search index artifact to Weights & Biases.
    """

    index_artifact = Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)
    logger.info("Vector store logged to W&B.")


########################################
# Main Ingestion Function
########################################
def ingest_data() -> tuple[List[Document], Chroma]:
    """
    Ingest data from the lectures JSONL file:
      1. Load lectures into Document objects.
      2. Split large documents into chunks.
      3. Create embeddings and store them in a Chroma vector store.

    Returns:
        Tuple[List[Document], Chroma]: The list of (chunked) documents and the vector store.
    """
    # Load documents from JSONL.
    documents = load_jsonl_documents(LECTURES_JSONL)
    logger.info(f"Parsed {len(documents)} documents from '{LECTURES_JSONL}'.")

    # Chunk the documents.
    split_documents = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info(f"Split into {len(split_documents)} document chunks.")

    # If OVERWRITE_VECTOR_DB is True, remove the existing directory to avoid duplication.
    if os.path.exists(CHROMA_PERSIST_DIR) and OVERWRITE_VECTOR_DB:
        logger.info(f"Overwriting existing vector store in '{CHROMA_PERSIST_DIR}'.")
        shutil.rmtree(CHROMA_PERSIST_DIR)
    # Create the vector store.
    vector_store = create_vector_store(split_documents, CHROMA_PERSIST_DIR)
    return split_documents, vector_store


def main():
    # 1. Configure the OpenAI API key.
    configure_openai_api_key()

    # 2. Initialize a Weights & Biases run.
    '''run = wandb.init(project=WANDB_PROJECT, config={
        "jsonl_file": LECTURES_JSONL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_store": CHROMA_PERSIST_DIR
    })'''

    # 3. Ingest the data and create the vector store.
    documents, vector_store = ingest_data()

    # 4. Log the dataset, vector store, and prompt (if provided) to W&B.
    #log_dataset(documents, run)
    #log_index(CHROMA_PERSIST_DIR, run)

    #run.finish()
    logger.info("Ingestion and logging complete.")


if __name__ == "__main__":
    main()
