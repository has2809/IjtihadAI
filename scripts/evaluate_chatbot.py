"""
evaluate_chatbot.py

Evaluate your chatbot on a local evaluation dataset by:
 1) Loading question + reference answers from a local CSV or JSON.
 2) Generating answers from your chatbot chain for each question.
 3) Using QAEvalChain or a simpler approach to label them CORRECT or INCORRECT.
 4) Calculating metrics like accuracy.
 5) Writing a text-based report and a CSV with final results.

Adjust the code to match your local chain loading method.
"""

import os
import json
import csv
import pandas as pd
from typing import List
from langchain.evaluation.qa import QAEvalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# If you have a local chain loader or something like chain.py, import it:
# from chain import load_chain, load_vector_store  # Example
# OR from chatbot import create_conversational_chain, load_local_chroma_db, etc.

########################
# Configuration
########################
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
# For example, you might have:
# from your_chatbot_file import create_conversational_chain, load_local_chroma_db, ...
CHROMA_DB_DIR = "../Data/vector_store"
EVAL_DATA_PATH = "../Data/evaluation_dataset.csv"   # or JSONL if you prefer
RESULTS_CSV = "eval_results.csv"
REPORT_FILE = "evaluation_report.txt"

########################
# 1) Load or configure chain
########################
def configure_chain():
    """
    Load or create a retrieval-based chain from your code.
    For example:
      1) load a local Chroma db
      2) create a ConversationalRetrievalChain
    """
    # Pseudocode:
    # db = load_local_chroma_db(CHROMA_DB_DIR, OPENAI_API_KEY)
    # prompt_template = load_chat_prompt_from_file("prompts.json")
    # chain = create_conversational_chain(db, prompt_template, OPENAI_API_KEY)
    # return chain

    # For demonstration, let's just create a dummy chain or an LLM instance:
    # If your chain is a simple Q&A, you can do something like:
    # dummy_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo")
    # But we want a chain that takes: {"question": q, "chat_history": [...]}

    # The simplest fallback, if you have no retrieval, might be:
    # from langchain.chains import LLMChain
    # from langchain.prompts import PromptTemplate
    # ...
    # But let's assume you have a real retrieval chain from your code.

    raise NotImplementedError("Please implement chain loading logic from your own code.")


########################
# 2) Load the evaluation dataset
########################
def load_evaluation_dataset(path: str) -> pd.DataFrame:
    """
    Loads CSV or JSONL. For demonstration, let's assume CSV with columns:
      question, answer
    If your dataset has other columns, adjust accordingly.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return df


########################
# 3) Generate answers from chain
########################
def generate_answers(df: pd.DataFrame, chain: ConversationalRetrievalChain) -> pd.DataFrame:
    """
    Use the chain to get answers for each question in the dataset.
    We store them in a new column 'model_answer'.
    """
    answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_q = row["question"]
        # You can reset chat_history to [] for each new question
        result = chain({"question": user_q, "chat_history": []})
        model_answer = result["answer"]
        answers.append(model_answer)
    df["model_answer"] = answers
    return df


########################
# 4) Evaluate answers (Using QAEvalChain or a naive approach)
########################
def evaluate_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    We'll use a QAEvalChain to label each row as CORRECT or INCORRECT.
    This requires a short prompt for the evaluator.
    """
    # If you have your own prompt, you can define it or load it from file:
    eval_prompt = PromptTemplate(
        template="""
You are an evaluator. You will be given a question, a reference answer, and a chatbot's answer.
Decide if the chatbot's answer is CORRECT or INCORRECT based on the reference.

QUESTION: {query}
REFERENCE ANSWER: {answer}
CHATBOT ANSWER: {result}

GRADE: CORRECT or INCORRECT
""",
        input_variables=["query", "answer", "result"],
    )

    # Create an LLM for evaluation
    evaluator_llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo"  # or whichever model
    )
    eval_chain = QAEvalChain.from_llm(evaluator_llm, prompt=eval_prompt)

    examples = []
    predictions = []
    for i, row in df.iterrows():
        q = str(row["question"])
        ref_a = str(row["answer"])
        model_a = str(row["model_answer"])
        # The QAEvalChain expects "query", "answer", "result"
        examples.append({"query": q, "answer": ref_a})
        predictions.append({"query": q, "answer": ref_a, "result": model_a})

    # Evaluate
    graded_outputs = eval_chain.evaluate(examples, predictions)

    # 'graded_outputs' is a list of dicts: e.g. [{"text": "GRADE: CORRECT", "results":"GRADE: CORRECT"}...]
    # We'll parse the "results" to see if it's CORRECT or INCORRECT.
    final_grades = []
    for item in graded_outputs:
        txt = item.get("results", "None")
        # Usually it might look like "GRADE: CORRECT" or "GRADE: INCORRECT"
        # Let's parse the last word
        if "CORRECT" in txt:
            final_grades.append("CORRECT")
        elif "INCORRECT" in txt:
            final_grades.append("INCORRECT")
        else:
            final_grades.append("UNKNOWN")

    df["model_score"] = final_grades
    return df


########################
# 5) Summarize & Write Report
########################
def write_report(df: pd.DataFrame) -> None:
    """
    Calculate accuracy, write CSV of results, write a text summary with examples.
    """
    # Write the CSV
    df.to_csv(RESULTS_CSV, index=False)

    # Calculate accuracy
    total = len(df)
    correct = sum(df["model_score"] == "CORRECT")
    accuracy = correct / total if total > 0 else 0.0

    # Possibly gather examples of correct & incorrect
    correct_examples = df[df["model_score"] == "CORRECT"].head(3)
    incorrect_examples = df[df["model_score"] == "INCORRECT"].head(3)

    # Prepare text summary
    lines = []
    lines.append(f"Evaluation Report\n=================\n")
    lines.append(f"Total samples: {total}")
    lines.append(f"Correct: {correct}")
    lines.append(f"Accuracy: {accuracy:.2%}\n")

    lines.append("Some correct examples:\n-------------------\n")
    for _, row in correct_examples.iterrows():
        lines.append(f"Q: {row['question']}")
        lines.append(f"Reference A: {row['answer']}")
        lines.append(f"Model A: {row['model_answer']}\n")

    lines.append("Some incorrect examples:\n-------------------\n")
    for _, row in incorrect_examples.iterrows():
        lines.append(f"Q: {row['question']}")
        lines.append(f"Reference A: {row['answer']}")
        lines.append(f"Model A: {row['model_answer']}\n")

    lines.append("Suggestions:\n-------------\n")
    lines.append("- Provide more context from the vector store.\n")
    lines.append("- Fine-tune or adjust prompt instructions.\n")
    lines.append("- Filter ambiguous queries more effectively.\n")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Report written to {REPORT_FILE}")


########################
# 6) Main
########################
def main():
    # 1) Load or create chain
    chain = configure_chain()  # Implement in your code

    # 2) Load dataset
    df = load_evaluation_dataset(EVAL_DATA_PATH)

    # 3) Generate answers
    df = generate_answers(df, chain)

    # 4) Evaluate answers
    df = evaluate_answers(df)

    # 5) Summarize & write out results
    write_report(df)


if __name__ == "__main__":
    main()
