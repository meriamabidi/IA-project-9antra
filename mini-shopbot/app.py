#ShopBot web interface ( bel Streamlit)

import re

import pandas as pd
import streamlit as st

import chatbot as cb

cb.DEBUG = False   

# examples to try 
EXAMPLES = {
    "English": ["Cheapest laptop with 8GB RAM", "Laptops under 500", "A laptop for travel",
                "Cheapest Lenovo with 8gb ram under 600", "help"],
    "Français": ["Quel est l'ordinateur le moins cher", "Meilleur ordinateur le mieux noté"],
    "العربية": ["أرخص حاسوب"],
}


# logic
def is_arabic(text):
    return bool(re.search(r"[؀-ۿ]", text))


def ask(question):
    """Everything the interface needs to answer one question."""
    lang = cb.detect_lang(question)
    question_en = cb.to_english(question, lang)
    rules = cb.parse_query(question_en, brands)
    intent = cb.detect_intent(question, question_en)       
    results = None if intent else cb.search(question, question_en, df, model, embeddings, brands)
    return {
        "lang": lang,
        "question_en": question_en,
        "rules": {k: v for k, v in rules.items() if v not in (None, [])},  
        "intent": intent,
        "results": results,
        "msgs": cb.MESSAGES.get(lang, cb.MESSAGES["en"]),
    }


def to_table(results):
    cols = {"name": "Name", "price": "Price ($)", "rating": "Rating", "reviews": "Reviews",
            "ram_gb": "RAM (GB)", "storage_gb": "Storage (GB)", "screen_inch": "Screen (in)",
            "url": "Link"}
    out = results[list(cols)].copy()
    for c in ("ram_gb", "storage_gb"):
        out[c] = out[c].astype("Int64")          
    return out.rename(columns=cols).reset_index(drop=True)


def build_reply(question):
    info = ask(question)
    results = info.pop("results")
    reply = {"role": "assistant", "info": info, "rtl": info["lang"] == "ar"}
    if info["intent"]:                                       
        reply.update(text=info["msgs"][info["intent"]], table=None)
    elif results.empty:
        reply.update(text=info["msgs"]["none"], table=None)
    else:
        reply.update(text=info["msgs"]["intro"], table=to_table(results))
    return reply



def show(text, rtl=False):
    if rtl:   # right-to-left for ar
        safe = text.replace("$", "&#36;")
        st.markdown(f'<div dir="rtl" style="text-align:right">{safe}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text.replace("$", r"\$"))


def render(msg):
    with st.chat_message(msg["role"]):
        show(msg["text"], msg.get("rtl", False))
        if msg.get("table") is not None:
            st.dataframe(
                msg["table"],
                hide_index=True,
                column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")},
            )
      #  if msg.get("info") and st.session_state.show_info:
        #    with st.expander("How I understood your question"):
           #     info = msg["info"]
           #     st.write(f"**Language detected:** {info['lang']}")
           #     st.write(f"**English version:** {info['question_en']}")
           #     st.write(f"**Rules found:** {info['rules'] or 'none (meaning search only)'}")

# il page mte3na 

st.set_page_config(page_title="ShopBot", page_icon="💻")
st.title("💻 ShopBot")
st.caption("Ask about laptops in English, French or Arabic")


@st.cache_resource(show_spinner="Loading models...")
def load():

    data = pd.read_csv(cb.CSV_PATH)
    data["text"] = data["name"].fillna("") + ". " + data["description"].fillna("")
    brand_set = cb.get_brands(data)
    embedding_model = cb.SentenceTransformer(cb.MODEL_NAME)
    vectors = embedding_model.encode(data["text"].tolist(), convert_to_tensor=True)
    for lang in ("fr", "ar"):          # load the translation models 
        cb._load(lang)
    return data, embedding_model, vectors, brand_set


df, model, embeddings, brands = load()

if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None

st.session_state.setdefault("show_info", True)

# sidebar
#def set_pending(q):
#    st.session_state.pending = q


#def clear_history():
#    st.session_state.history = []


#with st.sidebar:
#    st.header("Try an example")
#    for language, questions in EXAMPLES.items():
#        st.caption(language)
 #       for q in questions:
 #           st.button(q, on_click=set_pending, args=(q,), key=f"example_{q}")
 #   st.divider()
#    st.toggle("Show how I understood the question", value=True, key="show_info")
#    st.button("Clear conversation", on_click=clear_history)

# Handle a new question 
typed = st.chat_input("Ask about laptops...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.history.append(
        {"role": "user", "text": question, "rtl": is_arabic(question), "table": None, "info": None}
    )
    with st.spinner("Thinking..."):
        st.session_state.history.append(build_reply(question))

for msg in st.session_state.history:
    render(msg)