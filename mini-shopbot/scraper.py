import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://webscraper.io"
URL = f"{BASE}/test-sites/e-commerce/static/computers/laptops"
HEADERS = {"User-Agent": "Mozilla/5.0 (learning project)"}
MAX_PAGES = 30     
MAX_RETRIES = 3
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

session = requests.Session()
session.headers.update(HEADERS)


def text_or_none(card, selector):
    el = card.select_one(selector)
    return el.get_text(strip=True) if el else None


def attr_or_none(card, selector, attr):
    el = card.select_one(selector)
    return el.get(attr) if el else None


def fetch_page(page_number):

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(URL, params={"page": page_number}, timeout=10)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            print(f"Page {page_number}, attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 * attempt)


def scrape_page(page_number):
    html = fetch_page(page_number)
    if html is None:
        return []
    soup = BeautifulSoup(html, "html.parser")
    products = []
    for card in soup.select("div.thumbnail"):
        href = attr_or_none(card, "a.title", "href")
        products.append({
            "name": attr_or_none(card, "a.title", "title"),
            "price": text_or_none(card, "h4.price"),
            "description": text_or_none(card, "p.description"),
            "rating": attr_or_none(card, "p[data-rating]", "data-rating"),
            "reviews": text_or_none(card, "span[itemprop='reviewCount']"),
            "url": BASE + href if href else None,
        })
    return products


all_products = []
for page in range(1, MAX_PAGES + 1):
    if page > 1:
        time.sleep(1)  
    products = scrape_page(page)
    if not products:   
        break
    all_products.extend(products)
    print(f"Page {page}: {len(products)} products (total: {len(all_products)})")

df = pd.DataFrame(all_products)
print(df.shape)

df.to_csv(f"{DATA_DIR}/laptops_raw.csv", index=False)
print("Saved raw data")


df_clean = df.copy()

# Price: "$416.99" -> 416.99
df_clean["price"] = (
    df_clean["price"].str.replace(r"[^\d.]", "", regex=True).astype(float)
)

# Rating and reviews: text -> integers
df_clean["rating"] = df_clean["rating"].astype(int)
df_clean["reviews"] = df_clean["reviews"].astype(int)

# Remove duplicates (same url = same product)
df_clean = df_clean.drop_duplicates(subset="url")


STORAGE_TYPES = r"(?:SSD|HDD|eMMC|Flash)"


def extract_specs(desc):
    if not isinstance(desc, str):
        return pd.Series({"screen_inch": None, "ram_gb": None, "storage_gb": None})

    screen = re.search(r'([\d.]+)"', desc)
    storage = re.search(rf"\b(\d+)\s*(GB|TB)\s*{STORAGE_TYPES}", desc, re.I)
    ram = re.search(rf"\b(\d+)\s*GB\b(?!\s*{STORAGE_TYPES})", desc, re.I)

    storage_gb = None
    if storage:
        storage_gb = int(storage.group(1)) * (1000 if storage.group(2).upper() == "TB" else 1)

    return pd.Series({
        "screen_inch": float(screen.group(1)) if screen else None,
        "ram_gb": int(ram.group(1)) if ram else None,
        "storage_gb": storage_gb,
    })


specs = df_clean["description"].apply(extract_specs)
df_clean = pd.concat([df_clean, specs], axis=1)

#print(df_clean[["screen_inch", "ram_gb", "storage_gb"]].isna().sum())
#print("\nDescriptions with missing specs:")
#print(df_clean[df_clean[["ram_gb", "storage_gb"]].isna().any(axis=1)][["description"]])
#print("\nSuspicious RAM values (> 64 GB):")
#print(df_clean[df_clean["ram_gb"] > 64][["description", "ram_gb"]])

df_clean.to_csv(f"{DATA_DIR}/laptops_clean.csv", index=False)

#print(df_clean.dtypes)
#print(df_clean.shape)
#print(df_clean[df_clean["ram_gb"] >= 8].sort_values("price").head())