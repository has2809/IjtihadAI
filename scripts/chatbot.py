"""
chatbot.py

This script:
1. Loads a Chroma vector store containing embedded fiqh lectures (Step 4).
2. Loads a chat prompt that defines the system/human prompts for Islamic Q&A.
3. Uses the vector store + LLM in a retrieval-based pipeline to answer user queries.
"""

import os
import json
import logging
import wandb
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


########################################
# CONFIG
########################################

# Weights & Biases project name.
WANDB_PROJECT = "llmapps"

# openAI API key location
OPENAI_KEY_FILE = "../key.txt"

# Path to the local Chroma DB folder.
CHROMA_DB_DIR = "../Data/vector_store"
# Path to your custom prompts JSON
PROMPTS_FILE = "../prompts.json"

K_NEAREST = 6

MODEL_NAME = "gpt-4o-mini"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


################################################
# Load or Configure Vector Store
################################################
def load_local_chroma_db(db_dir: str, openai_api_key: str) -> Chroma:
    """
    Loads a local Chroma database from db_dir using OpenAIEmbeddings.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    return vectorstore


################################################
# Load Custom Prompts
################################################
def load_chat_prompt_from_file(file_path: str):
    """
    Loads system/human prompts from a JSON file.
    Example:
    {
      "system_template": "You are an Islamic chatbot. ...",
      "human_template": "{question}\n================\nFinal Answer in Arabic:"
    }
    Returns a ChatPromptTemplate object.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Prompt file '{file_path}' not found. Using default prompts.")
        # fallback default
        system_template = "You are a helpful Islamic chatbot..."
        human_template = "{question}\n---\nAnswer:"
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        system_template = data.get("system_template", "You are a helpful Islamic chatbot...")
        human_template = data.get("human_template", "{question}\n---\nAnswer:")

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
    return ChatPromptTemplate.from_messages(messages)


################################################
# Create a ConversationalRetrievalChain
################################################
def create_conversational_chain(db: Chroma, prompt_template: ChatPromptTemplate, openai_api_key: str):
    """
    Creates a retrieval-based chain that:
      - Takes user input + chat history
      - Retrieves relevant chunks from the vector store
      - Uses the LLM to generate an answer
    """
    retriever = db.as_retriever(search_kwargs={"k": K_NEAREST})
    llm = ChatOpenAI(
        temperature=0.3,
        openai_api_key=openai_api_key,
        model_name=MODEL_NAME
    )
    # The 'chain_type="stuff"' means we simply stuff retrieved chunks into the final prompt.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        chain_type="stuff",
        return_source_documents=True
    )
    return chain


################################################
# Main: Interactive Chat
################################################
def main():
    # Configure the OpenAI API key.
    configure_openai_api_key()

    # Load local Chroma DB
    db = load_local_chroma_db(CHROMA_DB_DIR, os.environ["OPENAI_API_KEY"])

    # Load chat prompt
    prompt_template = load_chat_prompt_from_file(PROMPTS_FILE)

    # Build the retrieval-based chain
    chain = create_conversational_chain(db, prompt_template, os.environ["OPENAI_API_KEY"])

    print("Welcome to the Islamic Chatbot. Type 'exit' or Ctrl+C to quit.")
    chat_history = []  # List of (user_query, bot_answer)

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Prepare the chain input
            result = chain({"question": user_input, "chat_history": chat_history})
            bot_answer = result["answer"]
            source_docs = result["source_documents"]

            # Print the answer
            print(f"Bot: {bot_answer}\n")

            # Optionally show sources:
            if source_docs:
                print("Sources:")
                for i, doc in enumerate(source_docs, start=1):
                    print(f"[{i}] {doc.metadata.get('url', 'No URL')} (Path={doc.metadata.get('path','NoPath')})")

            # Save the conversation history if you want multi-turn context
            chat_history.append((user_input, bot_answer))

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
