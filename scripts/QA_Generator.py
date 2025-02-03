import os
import json
import random
import tiktoken
from openai import OpenAI

########################################
# User Configs
########################################

OPENAI_KEY_FILE = '../key.txt'

# 1) The name of your scraped data file (JSONL).
SCRAPED_DATA_FILE = "../Data/lectures.jsonl"

# 2) Output file for the generated Q&A pairs.
OUTPUT_EVAL_FILE = "../Data/evaluation_dataset.jsonl"

# 3) Global token usage limit (input + output tokens).
MAX_TOKEN_USAGE = 1_000_000

# 4) Minimum length of a valid answer.
MIN_ANSWER_LENGTH = 40

# 5) Whether to check duplicates using string-similarity.
USE_STRING_SIMILARITY = False

# 6) Whether to generate multiple variations of each question.
GENERATE_VARIATIONS = False

# 7) Model & Prompt Configs
OPENAI_MODEL = "gpt-4o-mini"

# 8) Types of questions to generate
QUESTION_TYPES = ["General"]

# 9) Max # of Q&As per chunk
MAX_QUESTIONS_PER_CHUNK = 2

# 10) Max # of Q&As total for each category
MAX_QUESTIONS_PER_CATEGORY = 10

# 11) Global max # of Q&As across entire dataset
MAX_QUESTIONS_TOTAL = 100

########################################
# Setup OpenAI
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
        "without adding external info. Provide references if available."
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
    # Estimate tokens
    inp_tokens = count_tokens(system_prompt + user_prompt)
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
    # usage
    inp_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens

    # parse JSON
    try:
        # The model may have wrapped the JSON in triple backticks.
        # We'll do a quick cleanup
        out_text = out_text.strip("```json").strip("```")
        data = json.loads(out_text)
    except Exception:
        # If it's not valid JSON, we skip
        return None, inp_tokens, out_tokens

    # Check minimal conditions
    if len(data.get("answer", "")) < MIN_ANSWER_LENGTH:
        return None, inp_tokens, out_tokens

    # Return the Q&A dict
    return data, inp_tokens, out_tokens

generate_question_answer.global_token_usage = 0


########################################
# Recursively generate Q&As
########################################
def process_data_structure(
    data_obj,
    parent_path,
    generated_questions,
    category_name,
    cat_q_count,
    global_q_count
):
    """
        Recursively traverse 'data_obj' which can be:
      - A dict: { 'sub_section': <list or dict or further structures> }
      - A list: [ { 'content':..., 'title':..., 'lecture_url':... }, ...]
      - A "lecture object": a single dictionary with keys "content", etc. (rare)
    We gather texts from lectures in each sub-section and call 'generate_question_answer'.
    For each chunk, produce up to MAX_QUESTIONS_PER_CHUNK Q&As.
    We also enforce MAX_QUESTIONS_PER_CATEGORY (cat_q_count < ...),
    plus overall usage or global_q_count < MAX_QUESTIONS_TOTAL.
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
                    combined_text += (
                        f"\nTitle: {item['title']}\nURL: {item['lecture_url']}\n"
                        f"{item['content']}\n"
                    )
                    references.add(f"{item['title']} ({item['lecture_url']})")
                else:
                    # Possibly it's a nested structure, let's recursively proces
                    cat_q_count, global_q_count = process_data_structure(
                        item,
                        parent_path,
                        generated_questions,
                        category_name,
                        cat_q_count,
                        global_q_count
                    )
            else:
                # Not a dict, skip or handle differently
                pass

        # If we have enough combined_text, we can generate Q&As
        if combined_text.strip():
            local_q_generated = 0
            # Shuffle question types
            qtypes = QUESTION_TYPES[:]
            random.shuffle(qtypes)

            for qtype in qtypes:
                if local_q_generated >= MAX_QUESTIONS_PER_CHUNK:
                    break
                if cat_q_count[category_name] >= MAX_QUESTIONS_PER_CATEGORY:
                    break
                if generate_question_answer.global_token_usage >= MAX_TOKEN_USAGE:
                    print("[Info] Stopped: max token usage reached.")
                    break
                if global_q_count >= MAX_QUESTIONS_TOTAL:
                    print("[Info] Stopped: max total Q reached.")
                    break

                qa_data, in_tok, out_tok = generate_question_answer(combined_text, question_type=qtype)
                if qa_data is None:
                    continue
                # optional check duplicates
                if USE_STRING_SIMILARITY:
                    # compare with existing
                    is_dup = False
                    new_q = qa_data["question"]
                    for existing in generated_questions:
                        ratio = similarity_ratio(new_q, existing["question"])
                        if ratio > 0.85:
                            is_dup = True
                            break
                    if is_dup:
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
                cat_q_count[category_name] += 1
                global_q_count += 1
                local_q_generated += 1

                print(f"[Q#{global_q_count}] category='{category_name}', path='{parent_path}', tokens={generate_question_answer.global_token_usage}")
    # Case 2: If it's a dict, we iterate over key-value pairs
    elif isinstance(data_obj, dict):
        for k, v in data_obj.items():
            # e.g. key might be "chapter 1", val might be a list of lectures
            new_path = f"{parent_path} > {k}"
            cat_q_count, global_q_count = process_data_structure(
                v,
                new_path,
                generated_questions,
                category_name,
                cat_q_count,
                global_q_count
            )
    else:
        # unknown structure, skip
        pass

    return cat_q_count, global_q_count


########################################
# MAIN
########################################
def main():
    configure_openai_api_key()
    global client
    client = OpenAI()
    # We'll store Q&As in a list
    generated_questions = []
    # We'll track how many Qs per category
    cat_q_count = {}
    # We'll track total Q across dataset
    global_q_count = 0

    # Read each line from the scraped data file
    with open(SCRAPED_DATA_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines_processed = 0

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            data_entry = json.loads(line)
        except json.JSONDecodeError:
            print(f"[Warning] Invalid JSON on line {idx}. Skipping.")
            continue

        # Each line is typically { "كتابُ الطَّهارةِ": {...} } for example
        # We assume only 1 top-level key
        cat_keys = list(data_entry.keys())
        if not cat_keys:
            continue

        category_name = cat_keys[0]
        if category_name not in cat_q_count:
            cat_q_count[category_name] = 0

        # If this category already reached MAX_QUESTIONS_PER_CATEGORY, skip
        if cat_q_count[category_name] >= MAX_QUESTIONS_PER_CATEGORY:
            print(f"[Info] Already reached max Q for category: {category_name}. Skipping line {idx}.")
            continue
        if global_q_count >= MAX_QUESTIONS_TOTAL:
            print("[Info] Reached global Q limit. Stopping overall.")
            break
        if generate_question_answer.global_token_usage >= MAX_TOKEN_USAGE:
            print("[Info] Reached max token usage. Stopping overall.")
            break

        cat_data = data_entry[category_name]

        cat_q_count, global_q_count = process_data_structure(
            cat_data,
            parent_path=category_name,
            generated_questions=generated_questions,
            category_name=category_name,
            cat_q_count=cat_q_count,
            global_q_count=global_q_count
        )

        total_lines_processed += 1

        # Check again if we are done
        if generate_question_answer.global_token_usage >= MAX_TOKEN_USAGE:
            print("[Info] Stopped due to token usage limit.")
            break
        if global_q_count >= MAX_QUESTIONS_TOTAL:
            print("[Info] Stopped due to total question limit.")
            break

    # Write final Q&As
    with open(OUTPUT_EVAL_FILE, "w", encoding="utf-8") as out_f:
        for item in generated_questions:
            json.dump(item, out_f, ensure_ascii=False)
            out_f.write("\n")

    print("\n✅ Generation Complete.")
    print(f"Processed {total_lines_processed} lines from '{SCRAPED_DATA_FILE}'.")
    print(f"Total Q&A generated: {len(generated_questions)} (global_q_count={global_q_count}).")
    print(f"Token usage so far: {generate_question_answer.global_token_usage}")
    # Print category stats
    print("\nCategory Q&A Counts:")
    for cat, count in cat_q_count.items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
