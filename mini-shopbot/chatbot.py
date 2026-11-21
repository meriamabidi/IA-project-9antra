from pathlib import Path
import re

import pandas as pd

# check that everything is installed
try:
    from lingua import Language, LanguageDetectorBuilder
    from sentence_transformers import SentenceTransformer, util
    from transformers import MarianMTModel, MarianTokenizer
    from transformers import logging as hf_logging
except ImportError as e:
    raise SystemExit(f"Missing package: {e.name}. Run: pip install -r requirements.txt")

hf_logging.set_verbosity_error()   

# Settings 

CSV_PATH = Path(__file__).parent / "data" / "laptops_clean.csv"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3               
MIN_SCORE = 0.25          
DEBUG = True              

SORTS = {                 
    "price_asc": (["price"], [True]),
    "price_desc": (["price"], [False]),
    "rating_desc": (["rating", "reviews"], [False, False]),
    "reviews_desc": (["reviews"], [False]),
}

MESSAGES = {
    "en": {
        "intro": "Here are the best matches:",
        "none": "Sorry, I couldn't find a matching laptop in the data.",
        "hello": "Hello! Ask me about laptops. Type 'help' to see what I can do.",
        "thanks": "You're welcome!",
        "help": ("I can search the laptops by:\n"
                 "- price: 'under 500', 'over 300', 'between 300 and 600'\n"
                 "- RAM: '16gb ram'\n"
                 "- storage: '512gb ssd', '1tb'\n"
                 "- screen: '15.6 inch'\n"
                 "- brand: 'Lenovo', 'Asus'\n"
                 "- sorting: 'cheapest', 'most expensive', 'best rated', 'most popular'\n"
                 "You can combine them: 'cheapest Lenovo with 8gb ram under 600'."),
    },
    "fr": {
        "intro": "Voici les meilleurs résultats :",
        "none": "Désolé, je n'ai trouvé aucun ordinateur correspondant dans les données.",
        "hello": "Bonjour ! Posez-moi vos questions sur les ordinateurs portables. Tapez 'aide' pour voir ce que je sais faire.",
        "thanks": "Avec plaisir !",
        "help": ("Je peux chercher par :\n"
                 "- prix : 'moins de 500', 'plus de 300', 'entre 300 et 600'\n"
                 "- RAM : '16 go de ram'\n"
                 "- stockage : '512 go ssd', '1 to'\n"
                 "- écran : '15.6 pouces'\n"
                 "- marque : 'Lenovo', 'Asus'\n"
                 "- tri : 'le moins cher', 'le plus cher', 'le mieux noté'\n"
                 "Vous pouvez les combiner."),
    },
    "ar": {
        "intro": "هذه أفضل النتائج:",
        "none": "عذرًا، لم أجد حاسوبًا مطابقًا في البيانات.",
        "hello": "مرحبًا! اسألني عن الحواسيب المحمولة. اكتب 'مساعدة' لمعرفة ما يمكنني فعله.",
        "thanks": "عفوًا!",
        "help": ("يمكنني البحث حسب: السعر، الذاكرة (RAM)، التخزين، حجم الشاشة، العلامة التجارية، "
                 "والترتيب (الأرخص، الأغلى، الأعلى تقييمًا).\n"
                 "مثال: أرخص لينوفو بذاكرة 8 جيجا تحت 600."),
    },
}

# Language detection 


detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.FRENCH, Language.ARABIC
).build()


def detect_lang(text):
    lang = detector.detect_language_of(text)          # None if it can't decide
    return lang.iso_code_639_1.name.lower() if lang else "en"


# Offline translation to English
_models = {}   # loaded on first use, then kept in memory


def _load(lang):
    if lang not in _models:
        name = f"Helsinki-NLP/opus-mt-{lang}-en"
        _models[lang] = (MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name))
    return _models[lang]


def to_english(text, lang):
    """Translate a question from 'fr' or 'ar' to English, offline. On any error, return the text unchanged."""
    if lang == "en":
        return text
    try:
        tokenizer, mt_model = _load(lang)
        batch = tokenizer([text], return_tensors="pt", padding=True)
        output = mt_model.generate(**batch, max_new_tokens=64)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print("   [translation error]", e)
        return text


# Rules: price, RAM, storage, screen, rating, brand, sorting 
NUM = r"(\d+(?:\.\d+)?)"
NOT_SPEC = r"(?![\d.]*\s*(?:gb|tb|inch|in\b|\"|reviews?|stars?))"
STORAGE_WORDS = r"(?:ssd|hdd|emmc|flash|storage|disk)"


def get_brands(frame):
    """Brand = first word of the product name, e.g. 'Lenovo ThinkPad ...' -> 'lenovo'."""
    first = frame["name"].fillna("").str.split().str[0].str.lower()
    return {b for b in first if b.isalpha() and len(b) >= 2}


def parse_query(question_en, brands=frozenset()):
    q = re.sub(r"(?<=\d),(?=\d{3})", "", question_en.lower())
    f = {"min_price": None, "max_price": None, "min_ram": None, "min_storage": None,
         "screen": None, "min_rating": None, "brands": [], "sort": None}


    m = (re.search(r"(?:rating|rated)\D{0,15}?(\d(?:\.\d)?)(?![\d.])", q)
         or re.search(r"(\d(?:\.\d)?)\s*(?:\+\s*)?stars?", q))
    if m:
        f["min_rating"] = float(m.group(1))
        q = q[:m.start()] + " " + q[m.end():]

    m = re.search(rf"(?:between|from)\s*\$?{NUM}\s*(?:and|to|-|–)\s*\$?{NUM}{NOT_SPEC}", q)
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        f["min_price"], f["max_price"] = lo, hi

    if f["max_price"] is None:
        m = re.search(
            r"(?:under|below|less than|cheaper than|at most|up to|max(?:imum)?|budget(?: of)?)"
            rf"\D{{0,12}}?{NUM}{NOT_SPEC}",
            q,
        )
        if m:
            f["max_price"] = float(m.group(1))

    if f["min_price"] is None:
        m = re.search(
            r"(?:over|above|more than|greater than|at least|from|\bmin(?:imum)?\b|starting (?:at|from))"
            rf"\D{{0,12}}?{NUM}{NOT_SPEC}",
            q,
        )
        if m:
            f["min_price"] = float(m.group(1))

    if f["min_price"] is None and f["max_price"] is None:
        m = re.search(rf"(?:around|about|approximately|roughly)\D{{0,6}}?{NUM}{NOT_SPEC}", q)
        if m:
            p = float(m.group(1))
            f["min_price"], f["max_price"] = p * 0.8, p * 1.2

    m = (re.search(rf"{NUM}\s*(gb|tb)\s*(?:of\s*)?{STORAGE_WORDS}", q)
         or re.search(rf"{STORAGE_WORDS}\D{{0,12}}?{NUM}\s*(gb|tb)", q)
         or re.search(rf"{NUM}\s*(tb)\b", q))                       # a bare "1tb" is storage
    if m:
        value, unit = float(m.group(1)), m.group(2)
        f["min_storage"] = value * (1000 if unit == "tb" else 1)
        q = q[:m.start()] + " " + q[m.end():]                       # so it isn't read as RAM

    m = (re.search(r"(\d+)\s*gb\s*(?:of\s*)?ram", q)
         or re.search(r"ram\D{0,12}?(\d+)\s*gb", q)
         or re.search(rf"(\d+)\s*gb(?!\s*(?:of\s*)?{STORAGE_WORDS})", q))
    if m:
        f["min_ram"] = int(m.group(1))

    m = re.search(rf"{NUM}\s*(?:-?\s*inch(?:es)?|\")", q)
    if m:
        f["screen"] = float(m.group(1))

    f["brands"] = sorted(b for b in brands if re.search(rf"\b{re.escape(b)}\b", q))

    if re.search(
        r"\bcheap(?:est)?\b|inexpensive|not expensive|less expensive|least expensive|"
        r"lowest price|lowest cost|most affordable|affordable|budget[- ]friendly|low[- ]cost",
        q,
    ):
        f["sort"] = "price_asc"
    elif re.search(
        r"expensive|pricey|priciest|highest price|most costly|costly|premium|high[- ]end|top[- ]priced",
        q,
    ):
        f["sort"] = "price_desc"
    elif re.search(r"best[- ]rated|top[- ]rated|highest[- ]rated|highest rating|best rating", q):
        f["sort"] = "rating_desc"
    elif re.search(r"most reviewed|most popular", q):
        f["sort"] = "reviews_desc"
    return f


def has_filter(f):
    return any(f[k] is not None for k in
               ("min_price", "max_price", "min_ram", "min_storage", "screen", "min_rating")) or bool(f["brands"])


def apply_filters(frame, f):
    out = frame
    if f["min_price"] is not None:
        out = out[out["price"] >= f["min_price"]]
    if f["max_price"] is not None:
        out = out[out["price"] <= f["max_price"]]
    if f["min_ram"] is not None:
        out = out[out["ram_gb"] >= f["min_ram"]]
    if f["min_storage"] is not None:
        out = out[out["storage_gb"] >= f["min_storage"]]
    if f["screen"] is not None:
        out = out[(out["screen_inch"] - f["screen"]).abs() <= 0.5]
    if f["min_rating"] is not None:
        out = out[out["rating"] >= f["min_rating"]]
    if f["brands"]:
        first = out["name"].fillna("").str.split().str[0].str.lower()
        out = out[first.isin(f["brands"])]
    return out


# Greetings thanks help 

def detect_intent(original, question_en):
    """Return 'hello', 'thanks', 'help' or None. Only whole-message matches count."""
    def clean(t):
        return re.sub(r"[^\w\s]", "", t.lower()).strip()

    en, raw = clean(question_en), clean(original)
    if re.fullmatch(r"(?:hi|hello|hey|good (?:morning|afternoon|evening)|greetings)(?: there| bot| shopbot)?", en) \
            or raw in {"bonjour", "salut", "bonsoir", "مرحبا", "أهلا", "اهلا", "السلام عليكم"}:
        return "hello"
    if re.fullmatch(r"(?:thanks|thank you|thx|many thanks|thank you very much)", en) \
            or raw in {"merci", "merci beaucoup", "شكرا", "شكرًا"}:
        return "thanks"
    if re.fullmatch(r"(?:help|what can you do|what do you do|how does this work|how can you help)", en) \
            or raw in {"aide", "aidemoi", "مساعدة"}:
        return "help"
    return None


# Search: rules + meaning 
def search(question, question_en, df, model, embeddings, brands):
    f = parse_query(question_en, brands)

    # similarity between the question and every laptop (the model is multilingual)
    q_emb = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()
    scored = df.assign(score=scores)

    candidates = apply_filters(scored, f)
    if candidates.empty:
        return candidates

    if f["sort"]:
        cols, ascending = SORTS[f["sort"]]
        return candidates.sort_values(cols, ascending=ascending).head(TOP_K)

    candidates = candidates.sort_values("score", ascending=False).head(TOP_K)
    if not has_filter(f):
        candidates = candidates[candidates["score"] >= MIN_SCORE]   # refuse weak matches
    return candidates

# Build the reply 
def format_product(row):
    return (
        f"- {row['name']} | ${row['price']:.2f} | rating {row['rating']}/5 "
        f"({row['reviews']} reviews)\n  {row['description']}\n  {row['url']}"
    )


def answer(user_input, df, model, embeddings, brands):
    lang = detect_lang(user_input)
    question_en = to_english(user_input, lang)

    msgs = MESSAGES.get(lang, MESSAGES["en"])

    intent = detect_intent(user_input, question_en)
    if intent:
        return msgs[intent]

    results = search(user_input, question_en, df, model, embeddings, brands)
    if results.empty:
        return msgs["none"]

    lines = [format_product(row) for _, row in results.iterrows()]
    return msgs["intro"] + "\n" + "\n".join(lines)

# Run 

def main():
    print("Loading data and model (the first run downloads them)...")
    df = pd.read_csv(CSV_PATH)
    df["text"] = df["name"].fillna("") + ". " + df["description"].fillna("")
    brands = get_brands(df)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["text"].tolist(), convert_to_tensor=True)

    for lang in ("fr", "ar"):      
        _load(lang)

    print(f"\nShopBot ready: {len(df)} laptops. Type 'help' for examples, 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bye!")
            break
        if not user_input:
            continue
        print(f"\nBot: {answer(user_input, df, model, embeddings, brands)}\n")


if __name__ == "__main__":
    main()