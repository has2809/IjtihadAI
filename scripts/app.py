"""
app.py

Build a Gradio web interface for your Islamic fiqh chatbot.

This script:
1. Loads a vector store + retrieval chain from your existing chatbot.py code.
2. Creates a Gradio Blocks interface for user interaction.
3. Optionally logs the run with wandb if you want to keep usage logs.

Usage:
  $ pip install gradio
  $ python app.py
"""

import os
from types import SimpleNamespace

import gradio as gr
import wandb

# ---------------------------
# Adjust these imports to match your chatbot.py
# If your file or function names differ, adapt accordingly.
# ---------------------------
from chatbot import (
    load_local_chroma_db,
    load_chat_prompt_from_file,
    create_conversational_chain,
    configure_openai_api_key,
    CHROMA_DB_DIR,
    PROMPTS_FILE
)

###############################
# 1) Configure Logging / wandb
###############################
# Optionally track usage with wandb
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "llmapps"  # Adjust if you want a different project name

###############################
# 2) Chat Class for Gradio
###############################
class Chat:
    """
    A wrapper that:
      - keeps track of a wandb run
      - loads the vector store & chain only once
      - answers queries from the user
    """

    def __init__(self, config: SimpleNamespace):
        """
        Initialize the chatbot with wandb run + placeholders.
        """
        self.config = config
        # Start a wandb run (optional)
        self.wandb_run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            job_type=self.config.job_type,
            config=self.config,
        )
        self.vector_store = None
        self.chain = None

    def __call__(self, question: str, history=None, openai_api_key=None):
        """
        Gradio callback for each user message.
        Returns updated chat history (list of (user, bot) pairs).

        1) Ensure we have an OpenAI API key.
        2) Load vector store if needed.
        3) Load chain if needed.
        4) Generate an answer, store in the chat history.
        """
        # 1) Determine API key
        configure_openai_api_key()
        openai_api_key = os.environ["OPENAI_API_KEY"]

        # 2) Load local Chroma DB if not loaded yet
        if self.vector_store is None:
            self.vector_store = load_local_chroma_db(CHROMA_DB_DIR, openai_api_key)

        # 3) Load or create the chain if not loaded
        if self.chain is None:
            prompt_template = load_chat_prompt_from_file(PROMPTS_FILE)
            self.chain = create_conversational_chain(self.vector_store, prompt_template, openai_api_key)

        # 4) Prepare the question & chat history
        history = history or []
        user_input = question.strip()

        result = self.chain({"question": user_input, "chat_history": history})
        bot_answer = result["answer"]

        # Retrieve source docs if needed
        source_docs = result.get("source_documents", [])

        # Append to history
        history.append((user_input, bot_answer))

        if source_docs:
            src_text = "\n".join(f"[{i}] {doc.metadata.get('url', 'No URL')} (Path={doc.metadata.get('path','NoPath')})" for i, doc in enumerate(source_docs, start=1))
            history.append(("Context", src_text))

        return history, history


###############################
# 3) Build the Gradio UI
###############################
with gr.Blocks() as demo:
    # Some HTML header
    gr.HTML(
        """
        <div style="text-align: center; max-width: 700px; margin: 0 auto;">
          <h1 style="font-weight: 900;">IjtihadAI</h1>
          <p>
            اسأل عن الفقه أو الأحكام الإسلامية بناءً على دروس موقع الدرر السنية.
            أدخل مفتاح OpenAI الخاص بك إذا لم يكن مكتوبًا في ملف key.text، ثم اكتب سؤالك واضغط على Enter.
            <br>
            تم بناء هذا الموقع باستخدام Gradio وLangChain.
          </p>
        </div>
        """
    )

    with gr.Row():
        # Textbox for user question
        question = gr.Textbox(
            label="اكتب سؤالك الشرعي هنا",
            placeholder="e.g. ما هي أقسام المياه في الإسلام؟"
        )
        # Optional password field for API key
        openai_api_key = gr.Textbox(
            type="password",
            label="OpenAI API key (optional)",
        )

    # Gradio uses state to store the conversation across turns
    state = gr.State()
    # A Chatbot display
    chatbot = gr.Chatbot(label="IjtihadAI")

    # We tie the question input to a Chat object
    # config is a minimal SimpleNamespace so we can pass W&B info
    config = SimpleNamespace(
        project="llmapps",
        entity=None,
        job_type="production",
    )

    # The Chat object is constructed once for the entire session
    # The user enters a question => calls Chat(...)
    question.submit(
        fn=Chat(config),
        inputs=[question, state, openai_api_key],
        outputs=[chatbot, state],
    )

# Finally, launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=8884,
        show_error=True
    )
