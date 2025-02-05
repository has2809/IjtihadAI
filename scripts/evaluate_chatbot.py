"""
evaluate_chatbot.py

Evaluate your chatbot on *two* local evaluation datasets (e.g.,
 - generated_evaluation.jsonl
 - islamqa_evaluation.jsonl)

Procedure:
 1) For each dataset:
    - Load question+reference from JSONL
    - Generate answers using your retrieval-based chain
    - Auto-label certain obviously-wrong answers (like "I'm sorry" if reference is non-empty)
    - Use QAEvalChain to label the rest as CORRECT or INCORRECT
    - Calculate metrics (accuracy, etc.)
    - Write partial results (CSV, snippet of text)
2) Combine all summaries in a single final REPORT_FILE.

Adjust as needed for your environment.
"""

import os
import json
import pandas as pd
from typing import List, Tuple
from langchain.evaluation.qa import QAEvalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# import your existing functions from chatbot.py
from chatbot import (
    create_conversational_chain,
    load_chat_prompt_from_file,
    load_local_chroma_db,
    configure_openai_api_key,
    CHROMA_DB_DIR,
    PROMPTS_FILE
)

########################
# Configuration
########################
EVAL_DATASETS = [
    ("Generated QAs", "../Data/generated_evaluation.jsonl"),
    ("IslamQA QAs", "../Data/islamqa_evaluation.jsonl"),
]

RESULTS_DIR = "../reports"
MODEL_NAME = "gpt-4o-mini"

# We'll place each dataset's CSV in RESULTS_DIR and create a single combined report
REPORT_FILE = os.path.join(RESULTS_DIR, "evaluation_report.txt")


########################
# 1) Load or configure chain
########################
def configure_chain():
    """
    Load a retrieval-based chain from your code.
      1) load a local Chroma db
      2) create a ConversationalRetrievalChain
    """
    db = load_local_chroma_db(CHROMA_DB_DIR, os.environ["OPENAI_API_KEY"])
    prompt_template = load_chat_prompt_from_file(PROMPTS_FILE)
    chain = create_conversational_chain(db, prompt_template, os.environ["OPENAI_API_KEY"])
    return chain


########################
# 2) Load the evaluation dataset
########################
def load_evaluation_dataset(path: str) -> pd.DataFrame:
    """
    Loads CSV or JSONL.
    - If JSONL, we expect items with keys ["question", "answer"] at minimum.
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

    # We assume each record has at least "question" and "answer".
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError(f"Dataset {path} must have 'question' and 'answer' columns/fields.")
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
    sources = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
        user_q = row["question"]
        # reset chat_history to [] for each new question
        result = chain({"question": user_q, "chat_history": []})
        model_answer = result.get("answer", "")
        model_sources = result.get("source_documents", [])
        answers.append(model_answer)
        sources.append(model_sources)
    df["model_answer"] = answers
    df["model_sources"] = sources
    return df


########################
# 4) Evaluate answers (Using QAEvalChain + auto-check)
########################
def evaluate_answers(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """
    We'll do two steps:
     1) Auto-label obviously wrong answers (like "I'm sorry..." if reference is non-empty).
     2) Use QAEvalChain to label the rest as CORRECT or INCORRECT.

    We'll store the final label in "model_score".
    """
    # 4.1) Auto-label pass:
    auto_labels = []
    for i, row in df.iterrows():
        ref_answer = str(row["answer"]).strip()
        model_answer = str(row["model_answer"]).strip()

        # If there's a non-empty reference, but the model just disclaim "I don't have info..."
        # or "I'm sorry" => label INCORRECT
        # We'll look for some keywords in both English/Arabic
        disclaim_phrases = [
            "i'm sorry", "i am sorry", "i don't have", "i do not have",
            "أنا آسف", "ليس لدي المعلومات", "لا أمتلك المعلومات"
        ]

        is_disclaim = any(phrase.lower() in model_answer.lower() for phrase in disclaim_phrases)
        if ref_answer and is_disclaim:
            auto_labels.append("INCORRECT")
        else:
            auto_labels.append("UNLABELED")  # We'll let QAEvalChain decide

    df["auto_label"] = auto_labels

    # 4.2) QAEvalChain for the rest
    eval_prompt = PromptTemplate(
        template="""
You are an evaluator. You will be given a question, the reference answer, and the chatbot's answer. 
If the chatbot's answer does not match or is missing essential info from the reference, label it INCORRECT.
Otherwise, label it CORRECT.

QUESTION: {query}
REFERENCE ANSWER: {answer}
CHATBOT ANSWER: {result}

GRADE: CORRECT or INCORRECT
""",
        input_variables=["query", "answer", "result"],
    )

    evaluator_llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name=MODEL_NAME
    )
    eval_chain = QAEvalChain.from_llm(evaluator_llm, prompt=eval_prompt)

    examples = []
    predictions = []
    # We'll only run the chain for rows that are UNLABELED
    # then combine them with the auto_label
    for i, row in df.iterrows():
        if row["auto_label"] == "UNLABELED":
            q = str(row["question"])
            ref_a = str(row["answer"])
            model_a = str(row["model_answer"])
            examples.append({"query": q, "answer": ref_a})
            predictions.append({"query": q, "answer": ref_a, "result": model_a})
        else:
            # we won't run the chain, we already have auto_label
            pass

    if len(examples) > 0:
        graded_outputs = eval_chain.evaluate(examples, predictions)
    else:
        graded_outputs = []

    # We'll map them back to rows
    # We'll keep a pointer or index
    chain_idx = 0

    final_labels = []
    for i, row in df.iterrows():
        if row["auto_label"] != "UNLABELED":
            # use auto_label
            final_labels.append(row["auto_label"])
        else:
            txt = graded_outputs[chain_idx].get("results", "")
            chain_idx += 1
            if "CORRECT" in txt:
                final_labels.append("CORRECT")
            elif "INCORRECT" in txt:
                final_labels.append("INCORRECT")
            else:
                final_labels.append("UNKNOWN")

    df["model_score"] = final_labels
    return df


########################
# 5) Summarize & Write CSV and partial report
########################
def summarize_dataset(df: pd.DataFrame, dataset_label: str, out_dir: str) -> str:
    """
    Calculate metrics for a single dataset, write a CSV, return a summary text to be appended
    to the final combined report.
    """
    # We'll define a dataset-specific CSV
    csv_path = os.path.join(out_dir, f"eval_results_{dataset_label.replace(' ', '_')}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Calculate metrics
    total = len(df)
    correct_count = sum(df["model_score"] == "CORRECT")
    incorrect_count = sum(df["model_score"] == "INCORRECT")
    unknown_count = sum(df["model_score"] == "UNKNOWN")
    accuracy = correct_count / total if total > 0 else 0.0

    # Gather samples
    correct_examples = df[df["model_score"] == "CORRECT"].head(3)
    incorrect_examples = df[df["model_score"] == "INCORRECT"].head(3)

    lines = []
    lines.append(f"=== Evaluation for {dataset_label} ===")
    lines.append(f"Total samples: {total}")
    lines.append(f"Correct: {correct_count}")
    lines.append(f"Incorrect: {incorrect_count}")
    lines.append(f"Unknown: {unknown_count}")
    lines.append(f"Accuracy: {accuracy:.2%}\n")

    # Some correct examples
    lines.append("Some correct examples:")
    lines.append("-------------------")
    for _, row in correct_examples.iterrows():
        lines.append(f"Q: {row['question']}")
        lines.append(f"Reference: {row['answer']}")
        lines.append(f"Model: {row['model_answer']}\n")

    # Some incorrect
    lines.append("Some incorrect examples:")
    lines.append("-------------------")
    for _, row in incorrect_examples.iterrows():
        lines.append(f"Q: {row['question']}")
        lines.append(f"Reference: {row['answer']}")
        lines.append(f"Model: {row['model_answer']}\n")

    lines.append("Suggestions:")
    lines.append("------------")
    lines.append("- Provide more or better chunked context from the vector store.")
    lines.append("- Improve system/human prompts to reduce disclaimers if data is present.")
    lines.append("- Possibly tune or refine the LLM for better coverage.\n")

    return "\n".join(lines)


########################
# 6) Main
########################
def main():
    # Ensure we have an API key
    configure_openai_api_key()

    # Build retrieval chain
    chain = configure_chain()

    # We'll store text summaries for each dataset, then combine them in final file
    all_summaries = []

    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    for dataset_label, dataset_path in EVAL_DATASETS:
        print(f"\n=== Evaluating dataset: {dataset_label} | {dataset_path} ===")
        # Load
        df = load_evaluation_dataset(dataset_path)
        # Generate answers
        df = generate_answers(df, chain)
        # Evaluate
        df = evaluate_answers(df, dataset_label)
        # Summarize
        summary_text = summarize_dataset(df, dataset_label, out_dir)
        all_summaries.append(summary_text)

    # Write final combined report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        report_content = "\n\n".join(all_summaries)
        f.write(report_content)

    print(f"\nAll evaluation done. See combined report at {REPORT_FILE}")


if __name__ == "__main__":
    main()
