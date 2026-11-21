## ShopBot: a multilingual laptop assistant (English / French / Arabic)



ShopBot answers questions about laptops in English, French or Arabic. The data is scraped automatically from an e-commerce practice site, cleaned, and searched with a mix of simple rules (for prices, RAM, brands...) and a multilingual embedding model (for meaning).



## What it does

1. **Scrapes** every product page of a laptop catalogue (name, price, description, rating, reviews, link).
2. **Cleans** the data and extracts RAM, storage and screen size from the description text.
3. **Answers** questions typed in English, French or Arabic, in a terminal or in a web interface.



## Project structure

mini-shopbot/

├── scraper.py          # scrapes the site and writes the CSV files

├── chatbot.py          # the bot logic + a terminal chat (python chatbot.py)

├── app.py              # Streamlit web interface

├── requirements.txt    # Python dependencies

├── data/

│   ├── laptops\_raw.csv     # exactly as scraped

│   └── laptops\_clean.csv   # cleaned data used by the bot

├── scraper.ipynb       # notebook used to build and test the scraper

└── chatbot.ipynb       # notebook used to build and test the bot



## Setup

conda create -n shopbot python=3.11 -y

conda activate shopbot

pip install -r requirements.txt



## Usage

* Scrape the data : python scraper.py
* Chat in the terminal : python chatbot.py
* Open the web interface : streamlit run app.py
* 



## Tech stack

Python, requests, BeautifulSoup, pandas, sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2), Hugging Face transformers (Helsinki-NLP/opus-mt), lingua, Streamlit.

