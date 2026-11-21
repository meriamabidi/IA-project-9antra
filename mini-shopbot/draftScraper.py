import requests
import pandas as pd
from bs4 import BeautifulSoup

URL = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops"
response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (learning project)"}, timeout=10)
response.raise_for_status()
print(response.status_code)

soup = BeautifulSoup(response.text, "html.parser")
cards = soup.select("div.thumbnail")
print(f"Found {len(cards)} products")

#Func extract products min specific page

def scrape_page(page_number):
    url = f"https://webscraper.io/test-sites/e-commerce/static/computers/laptops?page={page_number}"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (learning project)"}, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    cards = soup.select("div.thumbnail")

    products = []
    for card in cards:
        products.append({
            "name": card.select_one("a.title")["title"],
            "price": card.select_one("h4.price").get_text(strip=True),
            "description": card.select_one("p.description").get_text(strip=True),
            "rating": card.select_one("p[data-rating]")["data-rating"],
            "reviews": card.select_one("span[itemprop='reviewCount']").get_text(strip=True),
            "url": "https://webscraper.io" + card.select_one("a.title")["href"],
        })
    return products

#testtt fel page 2

#page2 = scrape_page(2)
#print(len(page2), "products")
#print(page2[0]["name"])


import time

all_products = []
page = 1

while page <= 30:                     
    try:
        products = scrape_page(page)
    except requests.HTTPError:         
        break
    if not products:                  
        break

    all_products.extend(products)
    print(f"Page {page}: {len(products)} products (total: {len(all_products)})")
    page += 1
    time.sleep(1)                      

df = pd.DataFrame(all_products)
print(df.shape)
df.head()
#Next etape clean il data
df_clean = df.copy()

#Price: "$416.99" -> 416.99
df_clean["price"] = df_clean["price"].str.replace("$", "", regex=False).astype(float)

#Rating and reviews: text -> integers
df_clean["rating"] = df_clean["rating"].astype(int)
df_clean["reviews"] = df_clean["reviews"].astype(int)

#Remove duplicate products(2 rows with same url = same product)
df_clean = df_clean.drop_duplicates(subset="url")

#Save the cleaned version
df_clean.to_csv("data/laptops_clean.csv", index=False)

print(df_clean.dtypes) #shows the data type of each column
print(df_clean.shape)
df_clean.sort_values("price").head()
import re

def extract_specs(desc):
    screen = re.search(r'([\d.]+)"', desc)
    sizes = re.findall(r'(\d+)\s*(GB|TB)', desc)   

    def to_gb(value, unit):
        return int(value) * (1000 if unit == "TB" else 1)

    ram = to_gb(*sizes[0]) if len(sizes) > 0 else None
    storage = to_gb(*sizes[1]) if len(sizes) > 1 else None
    return pd.Series({
        "screen_inch": float(screen.group(1)) if screen else None,
        "ram_gb": ram,
        "storage_gb": storage,
    })

specs = df_clean["description"].apply(extract_specs)
df_clean = pd.concat([df_clean, specs], axis=1)

print(df_clean[["screen_inch", "ram_gb", "storage_gb"]].isna().sum())   # 9adeh adna min missing valeur f column ram,storage,screen..
df_clean.to_csv("data/laptops_clean.csv", index=False)
df_clean[df_clean["ram_gb"] >= 8].sort_values("price").head()

import os

print(df.shape)                     
print(df["url"].nunique())          

os.makedirs("data", exist_ok=True)
df.to_csv("data/laptops_raw.csv", index=False)
print("Saved")