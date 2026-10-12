# ✈️ US Airline Tweets — Interactive Sentiment Dashboard

[![Live demo](https://img.shields.io/badge/live%20demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://rolandkeszi-sentiment-usair.streamlit.app/?lang=en) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) ![Tweets](https://img.shields.io/badge/data-14%2C640%20tweets-59636e?style=flat-square)

An interactive **Streamlit** dashboard for exploring what passengers say about six US airlines on Twitter. It works on 14,640 human-labelled tweets and supports filtering by sentiment, airline and hour of day. It also shows a complaint profile per airline, time patterns, and word clouds of what people actually write. The interface is bilingual (English / Hungarian).

**▶ [Open the live dashboard](https://rolandkeszi-sentiment-usair.streamlit.app/?lang=en)** <sub>(If the app has been idle, Streamlit Cloud needs about 30 seconds to wake it up.)</sub>

<p align="center">
  <img src="assets/sentiment-by-airline.svg" width="100%" alt="100% stacked bars of tweet sentiment per airline. US Airways 78% negative and 9% positive; American 71% negative; United 69% negative; Southwest 49%; Delta 43%; Virgin America 36% negative and 30% positive.">
</p>

## Key findings

- **Twitter is where passengers complain.** 63% of all tweets are negative, and only 16% are positive.
- **There is a two-tier market.** US Airways, American and United draw 69–78% negative tweets. Virgin America, Delta and Southwest stay between 36% and 49%.
- **US Airways is the outlier.** It receives 8.4 negative tweets for every positive one, while Virgin America's ratio is close to 1:1.
- **Service, not the flight itself, is the main pain point.** Customer service accounts for almost a third of complaints, ahead of delays and cancellations combined.

<p align="center">
  <img src="assets/negative-reasons.svg" width="100%" alt="Horizontal bars of complaint reasons among 9,178 negative tweets: customer service issue 31.7%, late flight 18.1%, cancelled flight 9.2%, lost luggage 7.9%, bad flight 6.3%, flight booking problems 5.8%, flight attendant complaints 5.2%, long lines 1.9%, damaged luggage 0.8%; 13.0% could not be classified.">
</p>

## Dashboard features

| Tab | What you get |
|---|---|
| 📊 **Overview** | Headline KPIs, the sentiment split per airline (sorted by negative share) and tweet volume. |
| ⚠️ **Complaints** | Ranked complaint reasons, plus a heatmap of each airline's complaint profile. |
| 🕐 **Time patterns** | Tweets by hour of day and by day, stacked by sentiment. |
| 💬 **Text** | Word clouds per sentiment, with noise words removed, and a set of sample tweets you can refresh. |
| 📄 **Data** | The filtered table, downloadable as CSV. |

The sidebar filters (airlines, sentiment, hour range) apply to every tab. Add `?lang=en` or `?lang=hu` to the URL to choose the language.

## Data

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (Crowdflower via Kaggle). It contains tweets about six US airlines from 16 to 24 February 2015. Contributors labelled each tweet as positive, neutral or negative and tagged a reason for each negative one.

| Field | Used for |
|---|---|
| `airline_sentiment` | Sentiment label (positive / neutral / negative) |
| `negativereason` | Complaint category |
| `airline` | Airline filter and comparison |
| `tweet_created` | Hour-of-day filter and time patterns |
| `text` | Random tweet and word clouds |

## Run locally

```bash
git clone https://github.com/rolandkeszi/USAir.git
cd USAir
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Anonymisation

The published file is fully anonymised by [`scripts/anonymize_tweets.py`](scripts/anonymize_tweets.py), so no tweet can be linked to a person:

- the author name and tweet id columns are removed;
- @mentions of private accounts are replaced with `@user`, and only the airlines' official accounts are kept;
- e-mail addresses, phone numbers, long numeric ids and booking codes are masked (`[email]`, `[phone]`, `[number]`, `[code]`);
- person names found by named-entity recognition (spaCy) are replaced with `[name]`.

> **Note on location data:** the CSV contains `latitude`/`longitude` columns, but they are synthetic (spread uniformly across the US, with points in the ocean). The original dataset has almost no geotags. The dashboard therefore does not show a map.

## Repository structure

```text
├── app.py              # Streamlit dashboard (HU/EN)
├── .streamlit/         # theme
├── data/Tweets.csv     # labelled tweets, anonymised
├── scripts/            # anonymisation script
├── assets/             # README figures (adapt to light/dark theme)
└── requirements.txt
```

---

<sub>Part of my data science portfolio · see also the <a href="https://github.com/rolandkeszi/llm-sentiment-benchmark">LLM vs. classical sentiment benchmark</a>, where 12 models are compared on the same kind of task.</sub>
