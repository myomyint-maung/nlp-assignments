from flask import Flask, request, render_template, flash, redirect, session
import snscrape.modules.twitter as sntwitter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

def get_tweets(query):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():

        if len(tweets) == 100:
            break
        else:
            if tweet.lang=='en':
                tweets.append(str(tweet.rawContent))
    
    return tweets

def predict(text):
    text = torch.tensor(text_pipeline(text))
    text = text.reshape(1, -1)
    text_length = torch.tensor([text.size(1)]).to(dtype=torch.int64)
    with torch.no_grad():
        output = model(text, text_length).squeeze(1)
        predicted = torch.max(output.data, 1)[1]
        return int(predicted)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/search_query', methods=["POST"])
def search_query():
    if 'query' not in request.form:
        return redirect('/')

    session["query"] = request.form['query']

    return redirect('/sentiment_analysis')

@app.route('/sentiment_analysis', methods=["GET"])
def sentiment_analysis():
    query = session.get('query', None)
    tweets = get_tweets(query)

    sentiments = []

    for tweet in tweets:
        prediction = predict(tweet)
        sentiments.append(prediction)

if __name__ == "__main__":
    app.run(debug=True)