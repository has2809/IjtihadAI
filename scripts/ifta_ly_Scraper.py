"""
IslamQA_Scraper.py

Scrapes a certain number of Q&A pairs from categories on ifta.ly
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
    "الطهارة": "https://ifta.ly/category/%d8%a7%d9%84%d9%81%d8%aa%d8%a7%d9%88%d9%89/%d8%a7%d9%84%d8%b9%d8%a8%d8%a7%d8%af%d8%a7%d8%aa/%d8%a7%d9%84%d8%b7%d9%87%d8%a7%d8%b1%d8%a9/",
    "الصلاة":   "https://ifta.ly/category/%d8%a7%d9%84%d9%81%d8%aa%d8%a7%d9%88%d9%89/%d8%a7%d9%84%d8%b9%d8%a8%d8%a7%d8%af%d8%a7%d8%aa/%d8%a7%d9%84%d8%b5%d9%84%d8%a7%d8%a9/",
    "الزكاة":   "https://ifta.ly/category/%d8%a7%d9%84%d9%81%d8%aa%d8%a7%d9%88%d9%89/%d8%a7%d9%84%d8%b9%d8%a8%d8%a7%d8%af%d8%a7%d8%aa/%d8%a7%d9%84%d8%b2%d9%83%d8%a7%d8%a9/",
    "الصوم":   "https://ifta.ly/category/%d8%a7%d9%84%d9%81%d8%aa%d8%a7%d9%88%d9%89/%d8%a7%d9%84%d8%b9%d8%a8%d8%a7%d8%af%d8%a7%d8%aa/%d8%a7%d9%84%d8%b5%d9%88%d9%85/",
    "الحج":    "https://ifta.ly/category/%d8%a7%d9%84%d9%81%d8%aa%d8%a7%d9%88%d9%89/%d8%a7%d9%84%d8%b9%d8%a8%d8%a7%d8%af%d8%a7%d8%aa/%d8%a7%d9%84%d8%ad%d8%ac-%d9%88%d8%a7%d9%84%d9%87%d8%af%d9%8a-%d9%88%d8%a7%d9%84%d8%a3%d8%b6%d8%a7%d8%ad%d9%8a/"
}
MAX_QUESTIONS_PER_CATEGORY = 10   # how many Q&As to scrape from each category
OUTPUT_JSONL = "../Data/ifta_evaluation.jsonl"
# -----------------------------------------------


def get_soup(url):
    """
    Simple GET request + parse with BeautifulSoup.
    """
    time.sleep(1)  # a small delay to avoid hammering the server
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_question_links(category_url, max_to_collect):
    """
    Given a category page on ifta.ly,
    iterate through pagination, gather question URLs until we reach max_to_collect.
    Returns a list of question-page URLs.
    """
    collected_links = []
    page_url = category_url
    while True:
        if len(collected_links) >= max_to_collect:
            break

        print(f"Scraping category page: {page_url}")
        soup = get_soup(page_url)

        # Each question link is in: <h3 class="post-title"><a href="...">title</a></h3>
        question_cards = soup.select("h3.post-title > a")
        for card in question_cards:
            href = card.get("href")
            if href and href.startswith("https://ifta.ly/"):
                collected_links.append(href)
                if len(collected_links) >= max_to_collect:
                    break
        if len(collected_links) >= max_to_collect:
            break

        # Find "next" page link.
        # The snippet indicates: <div class="pages-nav"><ul class="pages-numbers"> <li class="the-next-page"><a href="...">»</a></li>
        next_a = soup.select_one("li.the-next-page a")
        if next_a:
            next_url = next_a.get("href")
            if next_url and next_url != page_url:
                page_url = next_url
            else:
                break
        else:
            break

    return collected_links[:max_to_collect]


def parse_question_page(question_url):
    """
    Given a link to a question page on ifta.ly, parse out the question text and answer text.
    We store them in a structure that is compatible with QA_Generator/IslamQA approach,
    but we separate question and answer by the occurrence of 'الجواب:' in the text.
    """
    print(f"  Parsing question page: {question_url}")
    soup = get_soup(question_url)

    # We remove the '.is-expanded' to avoid None
    content_div = soup.select_one("div.entry-content.entry.clearfix")
    if not content_div:
        return "", ""

    full_text = content_div.get_text("\n", strip=True)

    # We'll separate by 'الجواب:' if present
    if "الجواب" in full_text:
        parts = full_text.split("الجواب", 1)
        question_text = parts[0].strip()
        answer_text = parts[1].strip()
    elif "أما بعد" in full_text:
        parts = full_text.split("أما بعد", 1)
        question_text = parts[0].strip()
        answer_text = parts[1].strip()
    else:
        question_text = full_text
        answer_text = full_text

    return question_text, answer_text


def main():
    """
    Main scraping function:
      1. For each category, gather up to MAX_QUESTIONS_PER_CATEGORY question-page links.
      2. Parse each question page, produce data in QA_Generator-like format.
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

            # Build the record in the same style as QA_Generator
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
