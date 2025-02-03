import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Base page
BASE_URL = "https://dorar.net/feqhia"
# Where we'll save the final data
SAVE_PATH = "../Data/lectures.jsonl"
METRICS_FILE_PATH = "../Data/Dorar_scraping_metrics.json"

# The top-level categories we want to scrape.
TARGET_CATEGORIES = [
    "كتابُ الطَّهارةِ",
    "كِتابُ الصَّلاةِ",
    "كتابُ الزَّكاةِ",
    "كتابُ الصَّوم",
    "كتابُ الحَجِّ"
]
'''
TARGET_CATEGORIES = [
    "كتابُ الطَّهارةِ",
    "كِتابُ الصَّلاةِ",
    "كتابُ الزَّكاةِ",
    "كتابُ الصَّوم",
    "كتابُ الحَجِّ"
]
'''
# Metrics storage
metrics = {}

def fetch_lecture_text(url):
    """
    Fetch the lecture text (Arabic) from a given dorar.net/feqhia/<ID> URL
    using requests and BeautifulSoup.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    main_text_div = soup.select_one("div.w-100.mt-4")
    if main_text_div:
        return main_text_div.get_text(strip=True)
    return "Lecture content not found."


def expand_and_collect_links(driver, li_element, path_so_far):
    """
    Recursively expand folders (li elements) and collect any lecture links within.

    :param driver: Selenium WebDriver
    :param li_element: the <li> node we’re expanding
    :param path_so_far: list of folder/category names leading to this point
    :return: list of dicts: [{ 'path': [...], 'title': ..., 'url': ... }, ...]
    """
    collected = {}
    num_lectures = 0
    # Expand the current <li> if it's closed
    # We look for the clickable <a style="cursor: pointer;">
    # or the folder icon. If we don't find it, it might already be open or be a leaf.
    try:
        clickable = li_element.find_element(By.CSS_SELECTOR, "a[style='cursor: pointer;']")
        driver.execute_script("arguments[0].click();", clickable)
        time.sleep(2)  # Small delay for the DOM to update
        #WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, ":scope > ul > li.mtree-node")))
    except Exception:
        # Possibly no clickable element => might be a leaf
        pass

    # Now that we've clicked, sub-items should appear (sub-<ul> with child <li>)
    # 1) Gather direct links in this <li> (which might be the final lecture)
    #    These are typically <a href="/feqhia/123"> elements
    links_in_li = li_element.find_elements(By.CSS_SELECTOR, "a[href*='/feqhia/']")
    for link_el in links_in_li:
        href = link_el.get_attribute("href")
        title = link_el.text.strip()
        # In some structures, the text might be empty or repeated. Filter if needed:
        if href and title:
            # We store the link with the path + the link's own title
            num_lectures += 1
            lecture_content = fetch_lecture_text(href)
            section = path_so_far[-1] if path_so_far else "General"
            collected.setdefault(section, []).append({
                "lecture_title": title,
                "lecture_url": href,
                "content": lecture_content,
                "path": " > ".join(path_so_far),
                "title": title
            })

    # 2) Recursively expand child <li> under this node
    sub_li_elements = li_element.find_elements(By.CSS_SELECTOR, ":scope > ul > li.mtree-node")
    # For each child <li>, the "name" might be in the <a> text, so we read it:
    for sub_li in sub_li_elements:
        # The text for the subfolder is often in the <a> or combined with icons
        # We'll do sub_li.text.splitlines()[0], or we can find the first <a> child
        try:
            sub_title_el = sub_li.find_element(By.CSS_SELECTOR, "a[style='cursor: pointer;']")
            sub_title = sub_title_el.text.strip()
        except Exception:
            # Fallback if there's no clickable subfolder link
            sub_title = sub_li.text.strip().split("\n")[0]

        new_path = path_so_far + [sub_title]
        deeper_data, num_sub = expand_and_collect_links(driver, sub_li, new_path)
        num_lectures += num_sub
        # Merge deeper results
        for key, value in deeper_data.items():
            collected.setdefault(key, []).extend(value)

    return collected, num_lectures


def scrape_one_category(driver, category_name):
    """
    1) Find the top-level <li> whose text EXACTLY matches `category_name`.
    2) Expand it recursively and collect lectures.
    3) Save immediately in JSONL.
    Returns the number of lectures found.
    """
    # search top-level <li> nodes under #mtree
    top_level_lis = driver.find_elements(By.CSS_SELECTOR, "ul#mtree > li.mtree-node")
    for li_el in top_level_lis:
        li_text = li_el.text.strip()
        if li_text == category_name:
            # found the correct top-level node
            cat_data, count_lectures = expand_and_collect_links(driver, li_el, [category_name])
            # write to JSONL
            save_data_as_jsonl({category_name: cat_data})
            return count_lectures

    # If not found exactly, try partial match or return 0
    return 0



def save_data_as_jsonl(data, save_path=SAVE_PATH):
    """Save structured lecture data as JSONL (newline-separated JSON)."""
    with open(save_path, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n") # Newline to separate JSON objects


def main():
    start_time = time.time()

    driver = webdriver.Chrome()
    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul#mtree")))

    for cat in TARGET_CATEGORIES:
        print(f"🔍 Processing category: {cat}")
        count_found = scrape_one_category(driver, cat)
        metrics[cat] = count_found

    driver.quit()

    elapsed_time = time.time() - start_time

    # Print final metrics
    print("\n✅ Scraping Completed!")
    print(f"⏳ Total Execution Time: {elapsed_time:.2f} seconds")
    print("📊 Number of Lectures per Category:")
    for category, count in metrics.items():
        print(f"   - {category}: {count} lectures")

    with open(METRICS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()
