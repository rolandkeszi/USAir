"""US airline tweets: interactive sentiment dashboard (Streamlit).

Language: Hungarian by default, English with ?lang=en in the URL.
"""
import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

DATA_PATH = Path(__file__).parent / "data" / "Tweets.csv"

SENTIMENTS = ["negative", "neutral", "positive"]
EXTRA_STOPWORDS = {
    "flight", "flights", "amp", "u", "now", "will", "get", "got", "im", "us", "one", "still",
    "even", "just", "can", "go", "going", "make", "need", "2", "dm", "plane",
}
COLORS = {"negative": "#e34948", "neutral": "#b8b7b1", "positive": "#2a78d6"}
REASONS_HU = {
    "Customer Service Issue": "Ügyfélszolgálat",
    "Late Flight": "Késés",
    "Can't Tell": "Nem azonosítható",
    "Cancelled Flight": "Törölt járat",
    "Lost Luggage": "Elveszett poggyász",
    "Bad Flight": "Rossz repülési élmény",
    "Flight Booking Problems": "Foglalási gondok",
    "Flight Attendant Complaints": "Légiutas-kísérők",
    "longlines": "Hosszú sorok",
    "Damaged Luggage": "Sérült poggyász",
}

TEXT = {
    "hu": {
        "page_title": "Légitársasági tweetek – szentimentelemzés",
        "title": "Mit mondanak az utasok az amerikai légitársaságokról?",
        "subtitle": "{n:,} kézzel címkézett tweet hat amerikai légitársaságról · 2015. február 16–24.",
        "filters": "Szűrők",
        "airlines": "Légitársaságok",
        "sentiments": "Szentiment",
        "hours": "Napszak (csendes-óceáni idő, óra)",
        "reset_hint": "Minden ábra és mutató a szűrt adatokra vonatkozik.",
        "kpi_tweets": "Tweetek",
        "kpi_neg": "Negatív",
        "kpi_pos": "Pozitív",
        "kpi_ratio": "Neg. / poz. arány",
        "tab_overview": "📊 Áttekintés",
        "tab_complaints": "⚠️ Panaszok",
        "tab_time": "🕐 Időbeli minta",
        "tab_text": "💬 Szövegek",
        "tab_data": "📄 Adatok",
        "by_airline": "Szentiment-megoszlás légitársaságonként",
        "by_airline_note": "A légitársaságok a negatív tweetek aránya szerint vannak sorba rendezve.",
        "volume": "Tweetek száma légitársaságonként",
        "reasons": "A negatív tweetek okai",
        "heatmap": "Panaszprofil: az okok aránya légitársaságonként (%)",
        "heatmap_note": "Soronként 100%: megmutatja, mi a jellemző panasz az adott légitársaságnál.",
        "by_hour": "Tweetek óránként, szentiment szerint",
        "by_day": "Tweetek naponta, szentiment szerint",
        "wordcloud": "Szófelhő",
        "wordcloud_for": "Melyik szentiment szavai?",
        "samples": "Példa tweetek",
        "new_sample": "Új példák",
        "data_title": "Szűrt adattábla",
        "download": "Letöltés CSV-ben",
        "empty": "A beállított szűrőkre nincs találat. Bővítsd a szűrőket a bal oldalon.",
        "share": "Arány (%)",
        "count": "Tweetek",
        "airline": "Légitársaság",
        "reason": "Ok",
        "hour": "Óra",
        "day": "Nap",
        "sentiment": "Szentiment",
        "labels": {"negative": "Negatív", "neutral": "Semleges", "positive": "Pozitív"},
        "source": "Adat: Twitter US Airline Sentiment (Crowdflower / Kaggle) · Készítette: Roland Keszi",
        "language": "Nyelv",
    },
    "en": {
        "page_title": "US airline tweets – sentiment dashboard",
        "title": "What do passengers say about US airlines?",
        "subtitle": "{n:,} human-labelled tweets about six US airlines · 16–24 February 2015",
        "filters": "Filters",
        "airlines": "Airlines",
        "sentiments": "Sentiment",
        "hours": "Time of day (Pacific time, hour)",
        "reset_hint": "Every chart and figure reflects the filtered data.",
        "kpi_tweets": "Tweets",
        "kpi_neg": "Negative",
        "kpi_pos": "Positive",
        "kpi_ratio": "Neg. / pos. ratio",
        "tab_overview": "📊 Overview",
        "tab_complaints": "⚠️ Complaints",
        "tab_time": "🕐 Time patterns",
        "tab_text": "💬 Text",
        "tab_data": "📄 Data",
        "by_airline": "Sentiment split by airline",
        "by_airline_note": "Airlines sorted by share of negative tweets.",
        "volume": "Tweet volume by airline",
        "reasons": "Why passengers complain",
        "heatmap": "Complaint profile: share of each reason by airline (%)",
        "heatmap_note": "Each row adds up to 100%, showing what is typical for that airline.",
        "by_hour": "Tweets by hour of day, by sentiment",
        "by_day": "Tweets by day, by sentiment",
        "wordcloud": "Word cloud",
        "wordcloud_for": "Words of which sentiment?",
        "samples": "Sample tweets",
        "new_sample": "New samples",
        "data_title": "Filtered data",
        "download": "Download as CSV",
        "empty": "No tweets match the current filters. Widen the filters in the sidebar.",
        "share": "Share (%)",
        "count": "Tweets",
        "airline": "Airline",
        "reason": "Reason",
        "hour": "Hour",
        "day": "Day",
        "sentiment": "Sentiment",
        "labels": {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"},
        "source": "Data: Twitter US Airline Sentiment (Crowdflower / Kaggle) · Built by Roland Keszi",
        "language": "Language",
    },
}

lang = st.query_params.get("lang", "hu")
if lang not in TEXT:
    lang = "hu"
T = TEXT[lang]
LABELS = T["labels"]

st.set_page_config(page_title=T["page_title"], page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px;}
    [data-testid="stMetricValue"] {font-size: 1.9rem; font-weight: 700;}
    [data-testid="stMetric"] {background: var(--secondary-background-color, #f6f8fa);
        border-radius: 10px; padding: 0.9rem 1rem;}
    .subtitle {color: #59636e; margin-top: -0.6rem; margin-bottom: 1.2rem;}
    .tweet {border-left: 4px solid var(--c); background: rgba(127,127,127,0.08);
        border-radius: 6px; padding: 0.6rem 0.9rem; margin-bottom: 0.6rem;}
    .tweet small {color: #59636e;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["tweet_created"] = pd.to_datetime(df["tweet_created"], format="%Y-%m-%d %H:%M:%S %z")
    df["hour"] = df["tweet_created"].dt.hour
    df["day"] = df["tweet_created"].dt.strftime("%m-%d")
    df["reason_hu"] = df["negativereason"].map(REASONS_HU)
    return df


data = load_data()
ALL_AIRLINES = sorted(data["airline"].unique())

# ---------- sidebar ----------
with st.sidebar:
    chosen_lang = st.segmented_control(
        T["language"], ["hu", "en"], default=lang,
        format_func=lambda x: {"hu": "Magyar", "en": "English"}[x],
    )
    if chosen_lang and chosen_lang != lang:
        st.query_params["lang"] = chosen_lang
        st.rerun()

    st.header(T["filters"])
    airlines = st.multiselect(T["airlines"], ALL_AIRLINES, default=ALL_AIRLINES)
    sentiments = st.pills(
        T["sentiments"], SENTIMENTS, selection_mode="multi", default=SENTIMENTS,
        format_func=lambda s: LABELS[s],
    )
    hours = st.slider(T["hours"], 0, 23, (0, 23))
    st.caption(T["reset_hint"])

df = data[
    data["airline"].isin(airlines)
    & data["airline_sentiment"].isin(sentiments or [])
    & data["hour"].between(*hours)
]

# ---------- header ----------
st.title(T["title"])
st.markdown(f'<p class="subtitle">{T["subtitle"].format(n=len(data))}</p>', unsafe_allow_html=True)

if df.empty:
    st.warning(T["empty"])
    st.stop()

n = len(df)
counts = df["airline_sentiment"].value_counts()
neg, pos = counts.get("negative", 0), counts.get("positive", 0)

k1, k2, k3, k4 = st.columns(4)
k1.metric(T["kpi_tweets"], f"{n:,}")
k2.metric(T["kpi_neg"], f"{neg / n:.0%}")
k3.metric(T["kpi_pos"], f"{pos / n:.0%}")
k4.metric(T["kpi_ratio"], f"{neg / pos:.1f}×" if pos else "–")


def section(title):
    st.markdown(f"#### {title}")


def style(fig, height=420):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        font=dict(size=13),
    )
    return fig


tabs = st.tabs([T["tab_overview"], T["tab_complaints"], T["tab_time"], T["tab_text"], T["tab_data"]])

# ---------- overview ----------
with tabs[0]:
    left, right = st.container(), st.container()
    share = (
        df.groupby("airline")["airline_sentiment"].value_counts(normalize=True)
        .mul(100).rename("share").reset_index()
    )
    order = (
        share[share["airline_sentiment"] == "negative"]
        .sort_values("share", ascending=False)["airline"].tolist()
    )
    order += [a for a in share["airline"].unique() if a not in order]
    share["label"] = share["airline_sentiment"].map(LABELS)
    with left:
        section(T["by_airline"])
        fig = px.bar(
            share, x="share", y="airline", color="airline_sentiment", orientation="h",
            category_orders={"airline": order, "airline_sentiment": SENTIMENTS},
            color_discrete_map=COLORS, text=share["share"].map(lambda v: f"{v:.0f}%" if v >= 10 else ""),
            labels={"share": T["share"], "airline": "", "airline_sentiment": T["sentiment"]},
            custom_data=["label"],
        )
        fig.update_traces(hovertemplate="%{y}<br>%{customdata[0]}: %{x:.1f}%<extra></extra>",
                          textposition="inside", insidetextanchor="start", textangle=0)
        fig.for_each_trace(lambda t: t.update(name=LABELS[t.name]))
        fig.update_layout(barmode="stack", xaxis_range=[0, 100], bargap=0.3)
        st.plotly_chart(style(fig, 380), use_container_width=True)
        st.caption(T["by_airline_note"])
    with right:
        section(T["volume"])
        vol = df["airline"].value_counts().reset_index()
        vol.columns = ["airline", "count"]
        fig = px.bar(
            vol, x="count", y="airline", orientation="h", text="count",
            category_orders={"airline": vol["airline"].tolist()},
            labels={"count": T["count"], "airline": ""},
        )
        fig.update_traces(marker_color="#59636e", texttemplate="%{text:,}", textposition="outside",
                          hovertemplate="%{y}: %{x:,}<extra></extra>", cliponaxis=False)
        st.plotly_chart(style(fig, 300), use_container_width=True)

# ---------- complaints ----------
with tabs[1]:
    reason_col = "reason_hu" if lang == "hu" else "negativereason"
    cant_tell = REASONS_HU["Can't Tell"] if lang == "hu" else "Can't Tell"
    negatives = df[df["airline_sentiment"] == "negative"].dropna(subset=[reason_col])
    if negatives.empty:
        st.info(T["empty"])
    else:
        left, right = st.container(), st.container()
        with left:
            section(T["reasons"])
            reasons = negatives[reason_col].value_counts(normalize=True).mul(100).reset_index()
            reasons.columns = ["reason", "share"]
            fig = px.bar(
                reasons, x="share", y="reason", orientation="h",
                category_orders={"reason": reasons["reason"].tolist()},
                labels={"share": T["share"], "reason": ""},
            )
            fig.update_traces(
                marker_color=["#b8b7b1" if r == cant_tell else COLORS["negative"] for r in reasons["reason"]],
                texttemplate="%{x:.1f}%", textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>", cliponaxis=False,
            )
            st.plotly_chart(style(fig, 460), use_container_width=True)
        with right:
            section(T["heatmap"])
            profile = pd.crosstab(negatives["airline"], negatives[reason_col], normalize="index").mul(100)
            profile = profile[profile.sum().sort_values(ascending=False).index]
            fig = px.imshow(
                profile.round(1), text_auto=".0f", aspect="auto",
                color_continuous_scale=["#fdf0f0", "#e34948", "#8f1d1c"],
                labels=dict(x=T["reason"], y=T["airline"], color="%"),
            )
            fig.update_xaxes(tickangle=-35, title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(style(fig, 460), use_container_width=True)
            st.caption(T["heatmap_note"])

# ---------- time ----------
with tabs[2]:
    left, right = st.container(), st.container()
    for col, key, title in ((left, "hour", T["by_hour"]), (right, "day", T["by_day"])):
        with col:
            section(title)
            agg = df.groupby([key, "airline_sentiment"]).size().rename("count").reset_index()
            fig = px.bar(
                agg, x=key, y="count", color="airline_sentiment",
                category_orders={"airline_sentiment": SENTIMENTS},
                color_discrete_map=COLORS,
                labels={key: T[key], "count": T["count"], "airline_sentiment": T["sentiment"]},
            )
            fig.for_each_trace(lambda t: t.update(name=LABELS[t.name]))
            fig.update_layout(barmode="stack", bargap=0.15)
            st.plotly_chart(style(fig, 340), use_container_width=True)

# ---------- text ----------
with tabs[3]:
    left, right = st.container(), st.container()
    with left:
        section(T["wordcloud"])
        options = [s for s in SENTIMENTS if s in df["airline_sentiment"].unique()]
        wc_sent = st.segmented_control(T["wordcloud_for"], options, default=options[0], format_func=lambda s: LABELS[s])
        wc_sent = wc_sent or options[0]
        words = " ".join(df.loc[df["airline_sentiment"] == wc_sent, "text"])
        words = " ".join(w for w in words.split() if "http" not in w and not w.startswith("@") and w != "RT")
        if words.strip():
            palette = {
                "negative": ["#8f1d1c", "#b8302f", "#e34948"],
                "neutral": ["#3d444d", "#59636e", "#818b98"],
                "positive": ["#0b3f8a", "#1c5fb4", "#2a78d6"],
            }[wc_sent]
            cloud = WordCloud(
                stopwords=STOPWORDS | EXTRA_STOPWORDS, background_color="white",
                color_func=lambda *a, **k: palette[len(a[0]) % len(palette)],
                width=1000, height=520, max_words=120, random_state=42,
            ).generate(words)
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.imshow(cloud, interpolation="bilinear")
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig)
    with right:
        section(T["samples"])
        if st.button(T["new_sample"], icon="🔄"):
            st.session_state["seed"] = st.session_state.get("seed", 0) + 1
        sample = df.sample(min(5, len(df)), random_state=st.session_state.get("seed", 0))
        for _, row in sample.iterrows():
            s = row["airline_sentiment"]
            st.markdown(
                f'<div class="tweet" style="--c:{COLORS[s]}">{html.escape(row["text"])}<br>'
                f'<small>{row["airline"]} · {LABELS[s]} · {row["tweet_created"]:%Y-%m-%d %H:%M}</small></div>',
                unsafe_allow_html=True,
            )

# ---------- data ----------
with tabs[4]:
    section(T["data_title"])
    cols = ["tweet_created", "airline", "airline_sentiment", "negativereason", "text"]
    st.dataframe(df[cols], use_container_width=True, hide_index=True, height=480)
    st.download_button(
        T["download"], df[cols].to_csv(index=False).encode("utf-8"),
        file_name="airline_tweets_filtered.csv", mime="text/csv", icon="⬇️",
    )

st.divider()
st.caption(T["source"])
