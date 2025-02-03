"""
IslamQA_Scraper.py

Scrapes a certain number of Q&A pairs from categories on islamqa.info/ar,
and saves them in JSONL format, with the same structure as QA_Generator.py
so they can be used for the same evaluation mechanism.

Usage:
  1) Adjust CATEGORIES, MAX_QUESTIONS_PER_CATEGORY, and OUTPUT_JSONL as needed.
  2) Run: python IslamQA_Scraper.py
  3) A JSONL file is produced, each line is a dict with keys:
     {
       "category_or_path": ...,
       "question_type": "scraped",
       "question": "...",
       "answer": "...",
       "confidence": "N/A",
       "references_from_llm": [],
       "source_lectures": [...],
       "token_usage": {"input_tokens": 0, "output_tokens": 0}
     }

Dependencies:
  pip install requests beautifulsoup4
"""

import os
import time
import json
import requests
from bs4 import BeautifulSoup

# -- Adjust these as needed -----------------------
CATEGORIES = {
    "الطهارة": "https://islamqa.info/ar/categories/topics/58/%D8%A7%D9%84%D8%B7%D9%87%D8%A7%D8%B1%D8%A9",
    "الصلاة":   "https://islamqa.info/ar/categories/topics/70/%D8%A7%D9%84%D8%B5%D9%84%D8%A7%D8%A9",
    "الزكاة":   "https://islamqa.info/ar/categories/topics/99/%D8%A7%D9%84%D8%B2%D9%83%D8%A7%D8%A9",
    "الصوم":   "https://islamqa.info/ar/categories/topics/108/%D8%A7%D9%84%D8%B5%D9%88%D9%85",
    "الحج":    "https://islamqa.info/ar/categories/topics/126/%D8%A7%D9%84%D8%AD%D8%AC"
}
MAX_QUESTIONS_PER_CATEGORY = 5   # how many Q&As to scrape from each category
OUTPUT_JSONL = "../Data/islamqa_evaluation.jsonl"
# -----------------------------------------------


def get_soup(url):
    """
    Simple GET request + parse with BeautifulSoup.
    """
    time.sleep(1)  # a small delay to avoid hammering the server
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                      " (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_question_links(category_url, max_to_collect):
    """
    Given a category page (like https://islamqa.info/ar/categories/topics/58/...),
    iterate through pagination links, gather question URLs until we reach max_to_collect.

    Returns a list of question-page URLs.
    """
    collected_links = []
    page_url = category_url
    while True:
        if len(collected_links) >= max_to_collect:
            break

        print(f"Scraping category page: {page_url}")
        soup = get_soup(page_url)

        # Each question link has <a class="card post-card " href="...">
        question_cards = soup.select("div.single-topic > a.card.post-card")
        for card in question_cards:
            href = card.get("href")
            if href and href.startswith("https://islamqa.info/ar/answers/"):
                collected_links.append(href)
                if len(collected_links) >= max_to_collect:
                    break
        if len(collected_links) >= max_to_collect:
            break

        # Find next page link
        next_a = soup.select_one("ul.pagination-list li.next > a[rel='next']")
        if next_a:
            next_url = next_a.get("href")
            if next_url and next_url != page_url:
                page_url = next_url
            else:
                # no more pages
                break
        else:
            # no next page
            break

    return collected_links[:max_to_collect]


def parse_question_page(question_url):
    """
    Given a link to a question page, parse out the question text and answer text.

    We'll store them in a structure that is compatible with QA_Generator.
    """
    print(f"  Parsing question page: {question_url}")
    soup = get_soup(question_url)

    # The question is in <section class="single_fatwa__question ..."> <div> or <p>
    q_section = soup.select_one("section.single_fatwa__question")
    question_text = ""
    if q_section:
        # attempt to get all paragraphs or sub text
        paragraphs = q_section.select("div, p")
        question_text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    # The answer is in <section class="single_fatwa__answer__body ..."> <div class="content">
    a_section = soup.select_one("section.single_fatwa__answer__body div.content")
    answer_text = ""
    if a_section:
        # gather all paragraphs or sub elements
        paragraphs = a_section.select("p, div")
        # to keep it simpler, we'll just join their text
        answer_text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    # Return question, answer
    return question_text, answer_text


def main():
    """
    Main scraping function:
      1. For each category, gather up to MAX_QUESTIONS_PER_CATEGORY question-page links.
      2. Parse each question page, produce data in the QA_Generator-like format.
      3. Append lines to OUTPUT_JSONL.
    """
    total_scraped = 0
    out_f = open(OUTPUT_JSONL, "a", encoding="utf-8")

    for cat_name, cat_url in CATEGORIES.items():
        print(f"--- Scraping category: {cat_name} ---")
        question_links = find_question_links(cat_url, MAX_QUESTIONS_PER_CATEGORY)

        for link in question_links:
            q_text, a_text = parse_question_page(link)

            if not q_text.strip() or not a_text.strip():
                # skip empty or partial
                continue

            # Build the record that matches QA_Generator style
            record = {
                "category_or_path": cat_name,
                "question_type": "scraped",
                "question": q_text,
                "answer": a_text,
                "confidence": "N/A",
                "references_from_llm": [],
                "source_lectures": [link],
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            }

            json.dump(record, out_f, ensure_ascii=False)
            out_f.write("\n")

            total_scraped += 1

        print(f"Collected {len(question_links)} questions for category {cat_name}.")

    out_f.close()
    print(f"\nAll done! Total Q&As scraped = {total_scraped}.\nSaved in {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
