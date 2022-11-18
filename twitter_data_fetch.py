import twitter_credentials as credentials
import tweepy

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import re
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.tokenize import SpaceTokenizer
from textblob.taggers import NLTKTagger

import sys


def clean_text(tweet_text):
    replacements = [
        (r'@[A-Za-z0-9]+', ''),
        (r'#', ''),
        (r'https?://\S+', ' '),
        (r'\n', ''),
        (' +', ' '),
        ('^ +', '')
    ]

    for old, new in replacements:
        tweet_text = re.sub(old, new, tweet_text)
    return tweet_text


def main(hashtag, output_file_name):
    # parameters
    ACTUAL_TIME = datetime.utcnow()
    TWITTER_QUERY = f'#{hashtag} -is:retweet'
    START_TIME = ACTUAL_TIME - timedelta(days=1)
    BATCH_LIMIT = 100
    MAX_TWEETS = 1000000

    # client initialization
    client = tweepy.Client(bearer_token=credentials.BEARER_TOKEN,
                           consumer_key=credentials.CONSUMER_KEY,
                           consumer_secret=credentials.CONSUMER_SECRET,
                           access_token=credentials.ACCESS_TOKEN,
                           access_token_secret=credentials.ACCESS_TOKEN_SECRET)

    # tweets retrieval
    paginator = tweepy.Paginator(client.search_recent_tweets, query=TWITTER_QUERY,
                                 tweet_fields=['created_at'],
                                 max_results=BATCH_LIMIT,
                                 user_fields=['public_metrics'],
                                 expansions='author_id',
                                 start_time=START_TIME).flatten(limit=MAX_TWEETS)

    users_from_curr_batch = []
    tweets_list = []

    for i, tweet in enumerate(paginator):
        clean_tweet = clean_text(tweet.text)

        if not clean_tweet:
            break

        followers = -1
        try:
            if not bool(i % BATCH_LIMIT):
                users_from_curr_batch = {
                    u["id"]: u for u in paginator.gi_frame.f_locals['response'].includes['users']}

            if users_from_curr_batch and users_from_curr_batch[tweet.author_id]:
                user = users_from_curr_batch[tweet.author_id]
                followers = user.public_metrics['followers_count']
        except:
            print('Exception at iteration ' + str(i))

        if followers < 1:
            break

        tb = TextBlob(clean_tweet, tokenizer=SpaceTokenizer(),
                      pos_tagger=NLTKTagger(), analyzer=NaiveBayesAnalyzer())

        sentiment = 'Positive' if tb.polarity > 0 else 'Negative' if tb.polarity < 0 else 'Neutral'

        tweets_list.append([tweet.id, tweet.created_at, clean_tweet,
                            followers, tb.polarity, tb.subjectivity, sentiment])

    # create an empty dataframe for tweets
    df = pd.DataFrame(tweets_list,
                      columns=['id', 'created_at', 'clean_text', 'followers', 'polarity', 'subjectivity', 'sentiment'])

    # Log transformation of the followers column
    df['log_followers'] = df['followers'].apply(np.log)

    # weighed polarity calculation
    df['weighed_polarity'] = df.apply(
        lambda row: row['log_followers'] * row['polarity'], axis=1)

    # write to CSV
    df.to_csv(output_file_name, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('script requires 2 input parameters: <twitter hashtag> and <output file name>')
        sys.exit(0)
    else:
        HASHTAG = sys.argv[1]
        OUTPUT_FILE_NAME = sys.argv[2]
        main(HASHTAG, OUTPUT_FILE_NAME)
