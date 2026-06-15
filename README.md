
# Multilingual FAQ Retrieval System

A smart FAQ search engine that uses NLP models to detect user language, translate it to English, summarize long queries, and find the best answer using semantic similarity.

##  Features
- **Language Detection:** Automatically detects English, French, and Spanish queries.
- **Translation & Summarization:** Translates inputs to English and summarizes them for optimal semantic matching.
- **Semantic Search:** Uses Sentence Transformers (`all-mpnet-base-v2`) and Cosine Similarity to find the closest FAQ item.
- **Multilingual Output:** Automatically translates the English answer back to the user's original language (French/Spanish).

##  Requirements
To run this project, you will need to install the following dependencies:
- `torch`
- `transformers`
- `sentence-transformers`
- `sacremoses`

##  How to Use
Run the script using Python:
```bash
python FAQ_code.py
```
