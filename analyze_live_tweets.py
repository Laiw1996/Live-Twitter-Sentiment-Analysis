from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import cleaner
import emo_detect as emoji
import re
import json
import server
import pickle


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

loaded_model = pickle.load(open('lr.sav', 'rb'))
loaded_transformer = pickle.load(open('trans.sav', 'rb'))



def cleanTweet(twitter):
    # Remove URL, Links, Special Characters etc from tweet
    twitter = re.sub(r"http\S+", "", twitter)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", twitter).split())


class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        
        f = open("./flask_url/twitter-sentences.txt", "a")
        output = open("./flask_url/twitter-output.txt", "a")
        
        try:
            tweet = all_data["text"]
        
            f.write(tweet + '\n')
            #print(tweet)
            #output.write(tweet)
            # out = open('verizon_twitter_data.txt', 'a+')
            # tweet = tweet.encode('utf-8')
            #output.write(tweet+"\n")
            
            
            emoji_sentiment = emoji.emoji_detect(tweet, emoji_pos, emoji_neg)
            if emoji_sentiment != 100:
                #print(tweet, emoji_sentiment)
                output.write(str(emoji_sentiment))
                #output.write("emoji_sentiment")
                output.write('\n')
            #print("-------THIS IS A EMOJI SENTIMENT!-------")
            else:
                clean_tweet = cleanTweet(tweet)
                emoticon_sentiment = emoji.emoticon_detect(clean_tweet, emoticons_pos, emoticons_neg)
                if emoticon_sentiment != 100:
                    #print(tweet, emoticon_sentiment)
                    output.write(str(emoticon_sentiment))
                    #output.write("emoticon_sentiment")
                    output.write('\n')
                #print("======THIS IS A EMOTICON SENTIMENT!======")
                else:
                    text = cleaner.spacy_cleaner(tweet)
                    
                    #print("======================================")
                    sentiment = loaded_model.predict(loaded_transformer.transform([text]))[0]
                    # print(tweet, sentiment)
                    # sentiment = server.predict(text)
                    #output.write(tweet)
                    #output.write('\n')
                    output.write(str(sentiment))
                    #output.write("sentiment")
                    output.write('\n')
                    #f.write(text + '\n')
        
            f.close()
            output.close()
        
        except KeyError:
            pass
            
        return True

def on_error(self, status):
    print(status)



auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

# search_topic = input("Enter Keyword to search about: ")
twitterStream = Stream(auth, listener())
#twitterStream.filter(track=["Donald Trump"], languages=['en'])
#twitterStream.filter(track=["happy"], languages=['en'])
twitterStream.filter(track=['Trump'], languages=['en'])
