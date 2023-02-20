import spacy
import base64
import pandas as pd
import matplotlib.pyplot as plt
import snscrape.modules.twitter as sntwitter
from io import BytesIO
from collections import Counter
from LSTM_model import predict_sentiment
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, render_template, flash, redirect, session

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

def get_tweet(query):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == 100:
            break
        else:
            if tweet.lang=='en':
                tweets.append(str(tweet.rawContent))
    
    return tweets

def count_sentiment(sentiments):
    neg_counts = 0
    pos_counts = 0
    for sentiment in sentiments:
        if sentiment == 0:
            neg_counts += 1
        elif sentiment == 1:
            pos_counts += 1
    
    sentiment_counts = {'negative': neg_counts, 'positive': pos_counts}
    
    return sentiment_counts

def preprocessing(sentence):
    nlp = spacy.load("en_core_web_sm")
    stopwords = list(STOP_WORDS)
    doc = nlp(sentence)
    cleaned_tokens = []
    for token in doc:
        if token.text.lower() not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
            token.pos_ != 'SYM' and token.text not in list('0123456789+-*^~%$#@&/\|[]<>(){}') and \
            token.text.startswith('@') == False:
            cleaned_tokens.append(token.lemma_.lower().strip())

    return " ".join(cleaned_tokens)

def get_top_words(tweets, sentiments):
    dict = {'Tweet': tweets, 'Sentiment': sentiments}
    df = pd.DataFrame(dict)
    df['Tokens'] = df['Tweet'].apply(preprocessing)

    neg_corpus = [sent.split(" ") for sent in df.Tokens[df.Sentiment == 0].to_numpy()]
    pos_corpus = [sent.split(" ") for sent in df.Tokens[df.Sentiment == 1].to_numpy()]

    neg_vocab = []
    for i in range(len(neg_corpus)):
        for word in neg_corpus[i]:
            neg_vocab.append(word)
    
    pos_vocab = []
    for i in range(len(pos_corpus)):
        for word in pos_corpus[i]:
            pos_vocab.append(word)

    top_neg_words = []
    for i in range(10):
        neg_word = Counter(neg_vocab).most_common(10)[i][0]
        top_neg_words.append(neg_word)
    
    top_pos_words = []
    for i in range(10):
        pos_word = Counter(pos_vocab).most_common(10)[i][0]
        top_pos_words.append(pos_word)

    return top_neg_words, top_pos_words

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/search_query', methods=["POST"])
def search_query():
    if 'Query' not in request.form:
        return redirect('/')

    session["query"] = request.form['Query']

    return redirect('/sentiment_analysis')

@app.route('/sentiment_analysis', methods=["GET"])
def sentiment_analysis():
    query = session.get('query', None)
    tweets = get_tweet(query)

    sentiments = []
    for tweet in tweets:
        prediction = int(predict_sentiment(tweet))
        sentiments.append(prediction)
    
    sentiment_counts = count_sentiment(sentiments)
    
    labels = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    plt.bar(labels, counts)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    sentiment_plot = base64.b64encode(buf.getbuffer()).decode("ascii")

    top_neg_words, top_pos_words = get_top_words(tweets, sentiments)

    return render_template('analysis.html', query=query, tweets=tweets, 
                            sentiments=sentiments, sentiment_plot=sentiment_plot,
                            top_neg_words=top_neg_words, top_pos_words=top_pos_words)

if __name__ == "__main__":
    app.run(debug=True)