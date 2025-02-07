# IjtihadAI
## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Disclaimer](#disclaimer)
4. [Directory Structure](#directory-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
    - [Data Scraping and Preparation](#data-scraping-and-preparation)
    - [Vector Database Creation](#vector-database-creation)
    - [Running the Chatbot (CLI or Web)](#running-the-chatbot-cli-or-web)
    - [Generating & Evaluating Q&A Datasets](#generating--evaluating-qa-datasets)
7. [Prompts and Customization](#prompts-and-customization)
8. [Weights & Biases Integration (Optional)](#weights--biases-integration-optional)
9. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
10. [Future Improvements](#future-improvements)
11. [License](#license)

---

## Project Overview
This repository contains an **Islamic fiqh chatbot** that answers user queries based on a **local knowledge base** scraped from an Islamic teachings website. The system uses:

- **Dorar.net** data for **lecture-style fiqh content** (scraped via `dorar_scraper.py`)
- **IslamQA** and **ifta.ly** for **Q&A-style data** (used only for evaluation)
- **OpenAI LLM embeddings**
- **LangChain** for retrieval-based Q&A
- **A Gradio web interface** to chat with the model in **Arabic**

The core objective is to enable an **Islamic fiqh Q&A system** that strictly references only the scraped context. If relevant data is not found in the knowledge base, the chatbot responds with a polite disclaimer.

---

## Features
### **Automated Web Scraping**
- Collect fiqh content from **dorar.net** (lectures) and **Q&A data** from `islamqa.info/ar` and `ifta.ly`.
- Save the scraped data in **structured JSONL format**.

### **Embedding & Vector Database**
- Convert collected data into **embeddings** via `create_vector_db.py` and store them in a **local Chroma vector database**.

### **Retrieval-based Q&A**
- A specialized **retrieval chain** using LangChain’s `ConversationalRetrievalChain` ensures the LLM **references only the scraped content**.

### **Evaluation Pipeline**
- Scripts to generate **synthetic Q&A** from the scraped data (`QA_Generator.py`).
- Scrape real **user Q&A** from external Islamic QA sites (`IslamQA_Scraper.py` & `Ifta_ly_Scraper.py`).
- Evaluate correctness using `evaluate_chatbot.py` with a **custom evaluation prompt** for automatic labeling (`CORRECT` / `INCORRECT`).

### **Gradio Web Interface**
- A **simple, user-friendly interface** (`app.py`) enabling direct chat **in Arabic**.
- Supports manually **entering an OpenAI API key** on the webpage or using a local `key.txt`.

---

## Disclaimer
This application is for **demonstration and educational purposes only**. It is **not an official fatwa** or absolute Islamic ruling service. The answers are based strictly on the textual data scraped from the indicated websites; they **are not guaranteed** to represent a final religious opinion. Always consult **qualified scholars** for authoritative guidance.

---

## Directory Structure
```
.
├── scripts/
│   ├── dorar_scraper.py         # Scrapes lecture data from dorar.net
│   ├── IslamQA_scraper.py       # Scrapes Q&A from islamqa.info/ar
│   ├── Ifta_ly_Scraper.py       # Scrapes Q&A from ifta.ly
│   ├── QA_Generator.py          # Generates synthetic Q&A from scraped dataset
│   ├── create_vector_db.py      # Embeds the data, creates ChromaDB
│   ├── chatbot.py               # Main retrieval-based chain (CLI usage)
│   ├── evaluate_chatbot.py      # Evaluates chatbot on local Q&A datasets
│   ├── app.py                   # Gradio web app for chatbot
│   └── ...
├── data/
│   ├── lectures.jsonl           # Output from dorar_scraper.py
│   ├── islamqa_evaluation.jsonl # Output from IslamQA_scraper.py
│   ├── ifta_evaluation.jsonl    # Output from Ifta_ly_Scraper.py
│   ├── generated_evaluation.jsonl # Output from QA_Generator.py
│   ├── vector_store/            # Chroma database files
│   ├── langchain.db             # Optional local caching DB
│   └── ...
├── reports/
│   ├── evaluation_report.txt    # Combined evaluation summary
│   ├── eval_results_*.csv       # CSV with labeled results
│   ├── Dorar_scraping_metrics.json
│   └── ...
├── prompts.json                 # JSON-based system/human/evaluation prompts
├── requirements.txt             # Python dependencies
├── key.txt                      # (Ignored by Git) OpenAI API key storage
├── LICENSE
├── README.md
└── ...
```

---

## Installation & Setup
### **Install Python 3.11.9 (recommended).**
### **Install dependencies:**
```bash
pip install -r requirements.txt
```
*(You may also use a virtual environment: `python -m venv venv && source venv/bin/activate`.)*

### **OpenAI Key**
- Put your **OpenAI API key** in a file named **`key.txt`** in the project root.
- Ensure this file **is not tracked by Git** (it is already `.gitignored`).
- Alternatively, **set your `OPENAI_API_KEY`** environment variable or enter it manually in the **Gradio web UI**.

---

## Usage
### **1. Data Scraping and Preparation**
(A) **Scrape Dorar.net**
```bash
cd scripts
python dorar_scraper.py  # → lectures.jsonl
```
- This script navigates the site using Selenium & ChromeDriver.
- It collects Islamic fiqh lectures from the categories specified in `TARGET_CATEGORIES`, storing them in `../data/lectures.jsonl`.
- Some links might occasionally be skipped; you can rerun the script if needed or adjust the wait times.

(B) **Scrape Additional Q&A**
```bash
python IslamQA_scraper.py  # → islamqa_evaluation.jsonl
python Ifta_ly_Scraper.py  # → ifta_evaluation.jsonl
```
(C) **Generate Synthetic Q&A**

```bash
python QA_Generator.py  # → generated_evaluation.jsonl
```
*Each script has a `MAX_QUESTIONS_PER_CATEGORY` global variable you can adjust to retrieve more data.*
### **2. Vector Database Creation**
After scraping, convert the main data (usually `lectures.jsonl`) into embeddings:
```bash
python create_vector_db.py
```
- Reads from `LECTURES_JSONL` (by default `../data/lectures.jsonl`).
- Splits texts into chunks.
- Persists them in a local Chroma database (`../data/vector_store`).

### **3. Running the Chatbot (CLI or Web)**
#### **CLI Mode**
```bash
python chatbot.py
```
- Loads the local Chroma DB, sets up a retrieval chain, and starts a command-line loop.
- Type your questions in Arabic.
- Type `exit` or press `Ctrl+C` to quit.
#### **Web Interface (Gradio)**
```bash
python app.py
```

### **4. Evaluating Q&A Datasets**
#### **Run Evaluation**
```bash
python evaluate_chatbot.py
```
- Open the provided Gradio URL (e.g., `http://0.0.0.0:8884`).
  - Enter your question in Arabic.
  - *(Optional)* Provide an alternative OpenAI API key if not using `key.txt`.
- The chatbot will display relevant sources from the local database.
---
## Weights & Biases Integration (Optional)
Certain scripts (e.g., `create_vector_db.py`, `app.py`) contain references or placeholders for Weights & Biases logging. By default, the code has them **disabled or commented out**.

If you want to track **metrics online**, follow these steps:
1. **Sign up for Weights & Biases** at [wandb.ai](https://wandb.ai/).
2. Run the following command to log in:
   ```bash
   wandb login
   ```
3. **Uncomment or re-enable** the logging calls inside the relevant scripts (`create_vector_db.py`, `app.py`).

---
## Common Issues & Troubleshooting
### **Dorar Scraper Skips Some Lectures**
- The site’s JavaScript or network delays can cause **partial expansions**.
- Increase **sleep times** or add more robust `WebDriverWait(...)` calls in `dorar_scraper.py`.
- It might be due to some other reason
### **Rate Limits / Token Usage**
- Scripts like `create_vector_db.py` or large Q&A generation in `QA_Generator.py` can **hit OpenAI rate limits**.
- The code includes **partial handling for `RateLimitError`**, but you may still need to **slow down** or **upgrade your OpenAI plan** if usage is heavy.

### **ChromeDriver / Selenium**
- You must have **Chrome (or a ChromeDriver-compatible browser) installed**.
- Ensure the version of **ChromeDriver** matches your **local Chrome version**.

### **Memory / Large Data**
- If you gather **extremely large datasets**, chunking in `create_vector_db.py` or `QA_Generator.py` can be adjusted (look for `CHUNK_SIZE` or `MAX_QUESTIONS_TOTAL`).


---

## Future Improvements
### **Remote Execution & Hosting**
- Containerize with **Docker** or host the **Gradio app** on a cloud platform.
- Possibly integrate full **Weights & Biases online logging** for advanced usage metrics.

### **Extended Website Coverage**
- Include **more categories** or **sources**.
- Potentially **unify the Q&A** from multiple websites into a **single embed**.

### **Advanced Fine-Tuning**
- Instead of **raw completion-based usage**, consider **fine-tuned LLM models**.
- Evaluate **advanced approaches** for better **source attribution**.

### **Comprehensive Testing**
- Expand the **evaluation dataset** or adopt **manual expert review**.
- Possibly implement **partial correctness scoring** for more nuanced results.


---

## License
This project is under the **MIT License**. See `LICENSE` for details.

