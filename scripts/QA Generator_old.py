import os
import json
import random
import tiktoken
from openai import OpenAI

########################################
# User Configs
########################################
OPENAI_KEY_FILE = '../key.txt'
# 1) The name of your scraped data file (JSONL)
SCRAPED_DATA_FILE = "../Data/lectures.jsonl"

# 2) Output file for the generated Q&A pairs
OUTPUT_EVAL_FILE = "../Data/evaluation_dataset.jsonl"

# 3) Global token usage limit (input + output tokens)
MAX_TOKEN_USAGE = 5000000

# 4) Minimum length of a valid answer
MIN_ANSWER_LENGTH = 40

# 5) Whether to check duplicates using a simple string-similarity function
USE_STRING_SIMILARITY = False

# 6) Whether to generate multiple variations of each question
GENERATE_VARIATIONS = False

# 7) Model & Prompt Configs
OPENAI_MODEL = "gpt-4o-mini"
#QUESTION_TYPES = ["factual", "procedural", "comparative", "edge_case"]
QUESTION_TYPES = ["General"]

########################################
# Setup OpenAI
########################################

# Make sure to set your API key.txt as an env variable:
# export OPENAI_API_KEY="sk-xxxx"
if os.getenv("OPENAI_API_KEY") is None:
    with open(OPENAI_KEY_FILE, 'r') as f:
        os.environ["OPENAI_API_KEY"] = f.readline().strip()
assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key.txt"
client = OpenAI()
########################################
# Token Counting Helper
########################################

def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Use tiktoken to estimate token usage for a given text prompt or completion.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

########################################
# (Optional) String Similarity
########################################

def levenshtein_distance(s1, s2):
    """
    Classic DP approach to compute Levenshtein distance (edit distance).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def similarity_ratio(q1, q2):
    """
    Return a ratio [0..1], higher = more similar.
    1 => identical strings
    """
    dist = levenshtein_distance(q1.lower(), q2.lower())
    max_len = max(len(q1), len(q2))
    return 1.0 - (dist / max_len)

########################################
# LLM Call
########################################

def generate_question_answer(text_chunk, question_type="factual"):
    """
    Prompts the LLM with the lecture text + question instructions.
    Returns a dict: {
      'question': ...,
      'answer': ...,
      'confidence': ...,
      'references': [...],
    } or None if invalid.
    """
    system_prompt = (
        "You are an AI assistant specialized in Islamic fiqh. "
        "You ONLY use the provided text to create a question and answer, "
        "without adding external info. Provide references if available. "
    )
    user_prompt = f"""
TEXT:
{text_chunk}

TASK:
1. Generate ONE question (only in Arabic).
2. Provide a correct answer derived purely from the text.
3. Provide a numeric confidence score from 1 to 100.
4. Include references in a JSON array, if relevant. 
5. Return valid JSON with keys: question, answer, confidence, references.

QUESTION_TYPE: {question_type}
"""
    # Estimate input tokens
    inp_tokens = count_tokens(system_prompt+user_prompt)
    if generate_question_answer.global_token_usage + inp_tokens > MAX_TOKEN_USAGE:
        # We can't proceed, would exceed usage
        return None, inp_tokens, 0

    # Call the ChatCompletion
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=4096,
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None, inp_tokens, 0

    # The LLM's completion
    out_text = response.choices[0].message.content
    #out_text = out_text.strip("```json\n").strip("```")  # clean the output
    inp_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens

    # Attempt to parse the model output as JSON
    try:
        # The model may have wrapped the JSON in triple backticks.
        # We'll do a quick cleanup
        out_text = out_text.strip("```json").strip("```")
        data = json.loads(out_text)
    except Exception:
        # If it's not valid JSON, we skip
        return None, inp_tokens, out_tokens


    # Check minimal conditions (answer length, etc.)
    if len(data.get("answer", "")) < MIN_ANSWER_LENGTH:
        return None, inp_tokens, out_tokens

    # Return the Q&A dict
    return data, inp_tokens, out_tokens

# keep track of the global usage
generate_question_answer.global_token_usage = 0

########################################
# MAIN: Generate Q&A from JSONL
########################################

def process_data_structure(
    data_obj,
    parent_path=None,
    generated_questions=None
):
    """
    Recursively traverse 'data_obj' which can be:
      - A dict: { 'sub_section': <list or dict or further structures> }
      - A list: [ { 'content':..., 'title':..., 'lecture_url':... }, ...]
      - A "lecture object": a single dictionary with keys "content", etc. (rare)

    We gather texts from lectures in each sub-section and call 'generate_question_answer'.
    :param data_obj: the current piece of data to process
    :param parent_path: a string describing the nested path (e.g. "كتاب الصوم > الفصل 1")
    :param generated_questions: list of QAs so far
    :return: None (the function appends QAs to generated_questions)
    """

    # Ensure we have a place to store QAs
    if generated_questions is None:
        generated_questions = []

    if parent_path is None:
        parent_path = "ROOT"

    # Case 1: If it's a list, assume it's a list of "lecture" objects or sub-objects
    if isinstance(data_obj, list):
        # Combine all text from this list of lectures (or sub-items)
        combined_text = ""
        references = set()

        # We'll check if the items look like "lecture objects"
        for item in data_obj:
            if isinstance(item, dict):
                # If item has "content" + "title", treat it as a lecture
                if "content" in item and "title" in item and "lecture_url" in item:
                    # This is a lecture
                    combined_text += (
                        f"\nTitle: {item['title']}\nURL: {item['lecture_url']}\n"
                        f"{item['content']}\n"
                    )
                    references.add(f"{item['title']} ({item['lecture_url']})")
                else:
                    # Possibly it's a nested structure, let's recursively process
                    process_data_structure(
                        item,
                        parent_path=parent_path,
                        generated_questions=generated_questions
                    )
            else:
                # Not a dict, skip or handle differently
                pass

        # If we have enough combined_text, we can generate Q&As
        if combined_text.strip():
            question_types_copy = QUESTION_TYPES[:]
            random.shuffle(question_types_copy)

            for qtype in question_types_copy:
                if generate_question_answer.global_token_usage >= MAX_TOKEN_USAGE:
                    print("[Info] Exceeded MAX_TOKEN_USAGE, stopping generation for this chunk.")
                    break

                qa_data, in_tok, out_tok = generate_question_answer(combined_text, question_type=qtype)
                if qa_data is None:
                    continue

                # optional check duplicates
                if USE_STRING_SIMILARITY:
                    # compare with existing
                    is_duplicate = False
                    for g in generated_questions:
                        ratio = similarity_ratio(qa_data["question"], g["question"])
                        if ratio > 0.85:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue

                # update usage
                generate_question_answer.global_token_usage += (in_tok + out_tok)

                record = {
                    "category_or_path": parent_path,
                    "question_type": qtype,
                    "question": qa_data["question"],
                    "answer": qa_data["answer"],
                    "confidence": qa_data.get("confidence", "N/A"),
                    "references_from_llm": qa_data.get("references", []),
                    "source_lectures": list(references),
                    "token_usage": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok
                    }
                }
                generated_questions.append(record)
                # Show progress
                print(f"[Q#{len(generated_questions)}] path='{parent_path}', tokens={generate_question_answer.global_token_usage}")

    # Case 2: If it's a dict, we iterate over key-value pairs
    elif isinstance(data_obj, dict):
        for key, val in data_obj.items():
            # e.g. key might be "chapter 1", val might be a list of lectures
            new_path = f"{parent_path} > {key}"
            process_data_structure(val, parent_path=new_path, generated_questions=generated_questions)

    else:
        # unknown structure, skip
        pass

    # end of function

########################################
# MAIN: Generate Q&A from the JSONL
########################################

def main():
    generated_questions = []

    # Read each line from the scraped data file
    with open(SCRAPED_DATA_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Each line could be:
    # {
    #   "كتابُ الطَّهارةِ": {
    #       "sub_section": [...],
    #       ...
    #   }
    # }
    # or it might be a list, etc.
    total_lines_processed = 0

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            data_entry = json.loads(line)
        except json.JSONDecodeError:
            print(f"[Warning] Line {idx} is not valid JSON. Skipping.")
            continue

        # 'data_entry' might be a dict with one or more categories
        # or possibly a list, etc.

        process_data_structure(
            data_entry,
            parent_path="ROOT",
            generated_questions=generated_questions
        )
        total_lines_processed += 1

        if generate_question_answer.global_token_usage >= MAX_TOKEN_USAGE:
            print("[Info] Stopped early due to token limit.")
            break

    # Write all Q&As to OUTPUT_EVAL_FILE
    with open(OUTPUT_EVAL_FILE, "w", encoding="utf-8") as out_f:
        for item in generated_questions:
            json.dump(item, out_f, ensure_ascii=False)
            out_f.write("\n")

    print("\n✅ Generation Complete.")
    print(f"Processed {total_lines_processed} lines from '{SCRAPED_DATA_FILE}'.")
    print(f"Total Q&A generated: {len(generated_questions)}")
    print(f"Total tokens used: {generate_question_answer.global_token_usage}")


if __name__ == "__main__":
    main()
