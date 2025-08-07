
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Update the DATA_URL with the Windows path
#DATA_URL = r"C:\DS\PE\2fe\Bea\01_2fe\Cou\MasodikProba20240925\Tweets.csv"
DATA_URL = 'data/Tweets.csv'


st.title("Amerikai légitásaságok utasainak sentiment analyzise interactive dashbord-on")
st.sidebar.title("Twitter adatok Sentiment Analysise")
st.markdown("Tweetek sentiment elemzése - ezen a dashboard-on különféle szempontok szerint lehet adattudomáyi elemzéseket végezni, a bal odali oszlopban megadott beállítások alapján egy N=14485 elemű adatbázis alapján.")
st.sidebar.markdown("Interactive beállítások 🦃 🦤 🐦‍⬛ 🦜 🦚 🪽 🐦‍🔥🐦")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

st.sidebar.subheader("Véletlen tweet kiválasztása a beállíott vélemény minták adatai alapján")
random_tweet = st.sidebar.radio('Milyen típusú sentiment szöveget jelenítsük meg, véletlenszerűen?', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

st.sidebar.markdown("### Tweetek száma sentiment típusonként")
select = st.sidebar.selectbox('Adatvizualizáció típusa', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if not st.sidebar.checkbox("Elrejtés - Tweetek száma sentiment típusonként", True):
    st.markdown("### Tweetek száma sentiment típusonként")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)
st.markdown("### Tweetek hely szerint")

st.map(data) 

st.sidebar.subheader("Mikor és honnan tweeteltek az utasok?")
hour = st.sidebar.slider("A tweet időpontjának beállítása (1 órás időszakonként)", 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]

if not st.sidebar.checkbox("Bezárás - Mikor és honnan tweeteltek ábra", True, key='2'):
    st.markdown(f"### Tweet helyszíne a beállított idő periódusok szerint\n{len(modified_data)} tweet a következő idószakban:  {hour}:00 és {(hour + 1) % 24}:00")
    st.map(modified_data)
    if st.sidebar.checkbox("Mutassa a nyers adattáblát", False):
        st.write(modified_data)

st.sidebar.subheader("Tweet-ek száma légitársaságonként")
each_airline = st.sidebar.selectbox('Vizualizáció típusa', ['Bar plot', 'Pie chart'], key='3')
airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_sentiment_count = pd.DataFrame({'Airline': airline_sentiment_count.index, 'Tweets': airline_sentiment_count.values.flatten()})

if not st.sidebar.checkbox("Bezárás - Tweet-ek száma légitársaságonként", True, key='4'):
    st.subheader("Tweet-ek száma légitársaságonként")
    if each_airline == 'Bar plot':
        fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_1)
    elif each_airline == 'Pie chart':
        fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
        st.plotly_chart(fig_2)

@st.cache_data
def plot_sentiment(airline):
    df = data[data['airline'] == airline]
    count = df['airline_sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment': count.index, 'Tweets': count.values.flatten()})
    return count

st.sidebar.subheader("Breakdown airline by sentiment")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key='5')
if len(choice) > 0:
    st.subheader("Breakdown airline by sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot'], key='6')
    fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
    
    if breakdown_type == 'Bar plot':
        for j in range(len(choice)):
            fig_3.add_trace(
                go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
                row=1, col=j+1
            )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
    
    elif breakdown_type == 'Pie chart':
        fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type': 'domain'}] * len(choice)], subplot_titles=choice)
        for j in range(len(choice)):
            fig_3.add_trace(
                go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
                1, j+1
            )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)

st.sidebar.header("Szófelhő")
word_sentiment = st.sidebar.radio('Sentiment típusonkénti szófelhő megjelenítése', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close", True, key='7'):
    st.subheader(f'Word cloud for {word_sentiment} sentiment')
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)

    # Update: Create figure before passing to st.pyplot()
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")  # No ticks
    st.pyplot(fig)
