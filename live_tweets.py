from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
# import MySQLdb
import time
import json
import analyzer as s
import cleaner
import emoji_detect as emoji



#        replace mysql.server with "localhost" if you are running via your own server!
#                        server       MySQL username	MySQL pass  Database name.
# conn = MySQLdb.connect("mysql.server","beginneraccount","cookies","beginneraccount$tutorial")

# c = conn.cursor()

#consumer key, consumer secret, access token, access secret.
ckey="J53DqZWH1ISE4DrtHK8RbGGOU"
csecret="Fv4Ho1xLg29UwpjbE7IgZQGnVn27zywilV49QoxBw4sj5Pbc5S"
atoken="1064761402919809024-Y1Oq3Vem2u1TPsgobVdb8Zw8BpKyK5"
asecret="en38EAim8hoFgomkXaX4sfv7hDgzIyAD62rhiD4q5Cive"


emoji_pos, emoji_neg = emoji.get_emoji()
emoticons_pos, emoticons_neg = emoji.get_emoticon()

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        # tweet = all_data["text"]

        # username = all_data["user"]["screen_name"]

        # c.execute("INSERT INTO taula (time, username, tweet) VALUES (%s,%s,%s)",
        #     (time.time(), username, tweet))

        # conn.commit()

        try:
            tweet = all_data["text"]
            # out = open('verizon_twitter_data.txt', 'a+')
            # tweet = tweet.encode('utf-8')
            # out.write(str(tweet)+"\n")
            emoji_sentiment = emoji.emoji_detect(tweet, emoji_pos, emoji_neg)
            if emoji_sentiment != 100:
                print(tweet, emoji_sentiment)
                print("-------THIS IS A EMOJI SENTIMENT!-------")
            else:
                clean_tweet = s.cleanTweet(tweet)
                emoticon_sentiment = emoji.emoticon_detect(clean_tweet, emoticons_pos, emoticons_neg)
                if emoticon_sentiment != 100:
                    print(tweet, emoticon_sentiment)
                    print("======THIS IS A EMOTICON SENTIMENT!======")
                else:
                    sentiment_value = s.get_sentiment(tweet)
                    print(tweet, sentiment_value)

        except KeyError:
            pass

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"], languages=['en'])
