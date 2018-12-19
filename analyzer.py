import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import emoji_detect as emoji
import cleaner
import re


# pickle.dump(tvec, open('trans.sav', 'wb'))
# pickle.dump(model, open('lr.sav', 'wb'))


def cleanTweet(twitter):
    # Remove URL, Links, Special Characters etc from tweet
    twitter = re.sub(r"http\S+", "", twitter)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", twitter).split())


loaded_model = pickle.load(open('lr.sav', 'rb'))
loaded_transformer = pickle.load(open('trans.sav', 'rb'))

def get_sentiment(tweet):
	text = cleaner.spacy_cleaner(tweet)
	sentiment_value = loaded_model.predict(loaded_transformer.transform([text]))[0]
	return sentiment_value
