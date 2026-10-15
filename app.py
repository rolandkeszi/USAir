"""US airline tweets: interactive sentiment dashboard (Streamlit).

The layout follows the original dashboard: every setting lives in the sidebar.
Language: Hungarian by default, English with ?lang=en in the URL.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from wordcloud import STOPWORDS, WordCloud

DATA_PATH = Path(__file__).parent / "data" / "Tweets.csv"

SENTIMENTS = ["positive", "neutral", "negative"]
AIRLINES = ["US Airways", "United", "American", "Southwest", "Delta", "Virgin America"]
COLORS = {"negative": "#e34948", "neutral": "#b8b7b1", "positive": "#2a78d6"}
MASKS = re.compile(r"\[(?:name|email|phone|number|code)\]")  # placeholders left by the anonymisation

TEXT = {
    "hu": {
        "page_title": "Légitársasági tweetek – szentimentelemzés",
        "title": "Amerikai légitársaságok utasainak szentimentelemzése interaktív dashboardon",
        "intro": "Tweetek szentimentelemzése: ezen a dashboardon különféle szempontok szerint lehet "
                 "adattudományi elemzéseket végezni a bal oldali oszlop beállításai alapján, "
                 "egy N={n} elemű adatbázison.",
        "sidebar_title": "Twitter-adatok szentimentelemzése",
        "sidebar_intro": "Interaktív beállítások 🦃 🦤 🐦‍⬛ 🦜 🦚 🪽 🐦‍🔥 🐦",
        "language": "Nyelv",
        "random_head": "Véletlen tweet a kiválasztott szentiment szerint",
        "random_q": "Milyen szentimentű tweetet jelenítsünk meg véletlenszerűen?",
        "count_head": "Tweetek száma szentimenttípusonként",
        "chart_type": "Adatvizualizáció típusa",
        "count_hide": "Elrejtés – tweetek száma szentimenttípusonként",
        "hour_head": "Mikor tweeteltek az utasok?",
        "hour_slider": "A tweet időpontja (egyórás időszakok, csendes-óceáni idő)",
        "hour_hide": "Bezárás – mikor tweeteltek ábra",
        "hour_title": "Tweetek a beállított időszakban",
        "hour_text": "{n} tweet {h}:00 és {h2}:00 között",
        "raw": "Mutassa a nyers adattáblát",
        "airline_head": "Tweetek száma légitársaságonként",
        "airline_hide": "Bezárás – tweetek száma légitársaságonként",
        "breakdown_head": "Légitársaságok bontása szentiment szerint",
        "pick": "Légitársaságok kiválasztása",
        "wc_head": "Szófelhő",
        "wc_q": "Melyik szentimenttípus szófelhőjét mutassuk?",
        "wc_hide": "Bezárás – szófelhő",
        "wc_title": "Szófelhő: {s} szentiment",
        "bar": "Oszlopdiagram",
        "pie": "Kördiagram",
        "sentiment": "Szentiment",
        "tweets": "Tweetek",
        "airline": "Légitársaság",
        "labels": {"negative": "negatív", "neutral": "semleges", "positive": "pozitív"},
        "source": "Adat: Twitter US Airline Sentiment (Crowdflower / Kaggle), anonimizálva · Készítette: Roland Keszi",
    },
    "en": {
        "page_title": "US airline tweets – sentiment dashboard",
        "title": "Sentiment analysis of US airline passengers on an interactive dashboard",
        "intro": "Sentiment analysis of tweets: on this dashboard you can explore the data from several "
                 "angles using the settings in the left sidebar, on a dataset of N={n} tweets.",
        "sidebar_title": "Sentiment analysis of Twitter data",
        "sidebar_intro": "Interactive settings 🦃 🦤 🐦‍⬛ 🦜 🦚 🪽 🐦‍🔥 🐦",
        "language": "Language",
        "random_head": "Random tweet of the selected sentiment",
        "random_q": "Which sentiment should the random tweet have?",
        "count_head": "Number of tweets by sentiment",
        "chart_type": "Visualisation type",
        "count_hide": "Hide – number of tweets by sentiment",
        "hour_head": "When did passengers tweet?",
        "hour_slider": "Time of the tweet (one-hour slots, Pacific time)",
        "hour_hide": "Close – when did passengers tweet",
        "hour_title": "Tweets in the selected time slot",
        "hour_text": "{n} tweets between {h}:00 and {h2}:00",
        "raw": "Show the raw data table",
        "airline_head": "Number of tweets by airline",
        "airline_hide": "Close – number of tweets by airline",
        "breakdown_head": "Breakdown of airlines by sentiment",
        "pick": "Pick airlines",
        "wc_head": "Word cloud",
        "wc_q": "Word cloud for which sentiment?",
        "wc_hide": "Close – word cloud",
        "wc_title": "Word cloud for {s} sentiment",
        "bar": "Bar plot",
        "pie": "Pie chart",
        "sentiment": "Sentiment",
        "tweets": "Tweets",
        "airline": "Airline",
        "labels": {"negative": "negative", "neutral": "neutral", "positive": "positive"},
        "source": "Data: Twitter US Airline Sentiment (Crowdflower / Kaggle), anonymised · Built by Roland Keszi",
    },
}

lang = st.query_params.get("lang", "hu")
if lang not in TEXT:
    lang = "hu"
T = TEXT[lang]
LABELS = T["labels"]

st.set_page_config(page_title=T["page_title"], page_icon="✈️")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["tweet_created"] = pd.to_datetime(df["tweet_created"], format="%Y-%m-%d %H:%M:%S %z")
    return df


data = load_data()


def chart_type(key):
    return st.sidebar.selectbox(T["chart_type"], ["bar", "pie"], key=key, format_func=lambda c: T[c])


def count_chart(counts, name_col, kind, colors=None):
    if kind == "bar":
        fig = px.bar(counts, x=name_col, y="count", color=name_col if colors else None,
                     color_discrete_map=colors, height=500,
                     labels={"count": T["tweets"], name_col: ""})
        fig.update_layout(showlegend=False)
    else:
        fig = px.pie(counts, values="count", names=name_col, color=name_col if colors else None,
                     color_discrete_map=colors)
    return fig


# ---------- header ----------
st.title(T["title"])
n_fmt = f"{len(data):,}".replace(",", " " if lang == "hu" else ",")
st.markdown(T["intro"].format(n=n_fmt))

st.sidebar.title(T["sidebar_title"])
chosen_lang = st.sidebar.segmented_control(
    T["language"], ["hu", "en"], default=lang, format_func=lambda x: {"hu": "Magyar", "en": "English"}[x],
)
if chosen_lang and chosen_lang != lang:
    st.query_params["lang"] = chosen_lang
    st.rerun()
st.sidebar.markdown(T["sidebar_intro"])

# ---------- random tweet ----------
st.sidebar.subheader(T["random_head"])
random_tweet = st.sidebar.radio(T["random_q"], SENTIMENTS, format_func=lambda s: LABELS[s])
st.sidebar.markdown(
    data.loc[data["airline_sentiment"] == random_tweet, "text"].sample(n=1).iat[0].replace("$", "\\$")
)

# ---------- tweets by sentiment ----------
st.sidebar.markdown(f"### {T['count_head']}")
select = chart_type("1")
sentiment_count = data["airline_sentiment"].value_counts().rename_axis("sentiment").reset_index()
sentiment_count["sentiment"] = sentiment_count["sentiment"].map(LABELS)
if not st.sidebar.checkbox(T["count_hide"], False):
    st.markdown(f"### {T['count_head']}")
    colors = {LABELS[s]: c for s, c in COLORS.items()}
    st.plotly_chart(count_chart(sentiment_count, "sentiment", select, colors))

# ---------- tweets by hour ----------
st.sidebar.subheader(T["hour_head"])
hour = st.sidebar.slider(T["hour_slider"], 0, 23)
modified_data = data[data["tweet_created"].dt.hour == hour]
if not st.sidebar.checkbox(T["hour_hide"], True, key="2"):
    st.markdown(f"### {T['hour_title']}")
    st.markdown(T["hour_text"].format(n=len(modified_data), h=hour, h2=(hour + 1) % 24))
    hour_count = (modified_data.groupby(["airline", "airline_sentiment"]).size()
                  .rename("count").reset_index())
    hour_count["airline_sentiment"] = hour_count["airline_sentiment"].map(LABELS)
    fig = px.bar(hour_count, x="airline", y="count", color="airline_sentiment",
                 color_discrete_map={LABELS[s]: c for s, c in COLORS.items()},
                 category_orders={"airline": AIRLINES, "airline_sentiment": [LABELS[s] for s in SENTIMENTS]},
                 labels={"count": T["tweets"], "airline": "", "airline_sentiment": T["sentiment"]})
    st.plotly_chart(fig)
    if st.sidebar.checkbox(T["raw"], False):
        st.write(modified_data[["tweet_created", "airline", "airline_sentiment", "negativereason", "text"]])

# ---------- tweets by airline ----------
st.sidebar.subheader(T["airline_head"])
each_airline = chart_type("3")
airline_count = data["airline"].value_counts().rename_axis("airline").reset_index()
if not st.sidebar.checkbox(T["airline_hide"], True, key="4"):
    st.subheader(T["airline_head"])
    st.plotly_chart(count_chart(airline_count, "airline", each_airline))


# ---------- breakdown by airline ----------
@st.cache_data
def plot_sentiment(airline):
    count = data.loc[data["airline"] == airline, "airline_sentiment"].value_counts()
    return count.reindex(SENTIMENTS, fill_value=0)


st.sidebar.subheader(T["breakdown_head"])
choice = st.sidebar.multiselect(T["pick"], AIRLINES, key="5")
if choice:
    st.subheader(T["breakdown_head"])
    breakdown_type = st.sidebar.selectbox(T["chart_type"], ["pie", "bar"], key="6", format_func=lambda c: T[c])
    names = [LABELS[s] for s in SENTIMENTS]
    colors = [COLORS[s] for s in SENTIMENTS]
    if breakdown_type == "bar":
        fig = make_subplots(rows=1, cols=len(choice), subplot_titles=choice, shared_yaxes=True)
        for j, airline in enumerate(choice):
            fig.add_trace(go.Bar(x=names, y=plot_sentiment(airline).values, marker_color=colors,
                                 showlegend=False), row=1, col=j + 1)
    else:
        fig = make_subplots(rows=1, cols=len(choice), specs=[[{"type": "domain"}] * len(choice)],
                            subplot_titles=choice)
        for j, airline in enumerate(choice):
            fig.add_trace(go.Pie(labels=names, values=plot_sentiment(airline).values, marker_colors=colors,
                                 sort=False, showlegend=j == 0), 1, j + 1)
    fig.update_layout(height=600)
    st.plotly_chart(fig)

# ---------- word cloud ----------
st.sidebar.header(T["wc_head"])
word_sentiment = st.sidebar.radio(T["wc_q"], SENTIMENTS, format_func=lambda s: LABELS[s], key="wc")
if not st.sidebar.checkbox(T["wc_hide"], True, key="7"):
    st.subheader(T["wc_title"].format(s=LABELS[word_sentiment]))
    words = MASKS.sub(" ", " ".join(data.loc[data["airline_sentiment"] == word_sentiment, "text"]))
    processed_words = " ".join(w for w in words.split() if "http" not in w and not w.startswith("@") and w != "RT")
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640,
                          random_state=42).generate(processed_words)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

st.divider()
st.caption(T["source"])
