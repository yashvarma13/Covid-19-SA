import pandas as pd
import numpy as np
import re
import emoji
import nltk

# Download necessary NLTK data
nltk.download('words')

tweets_data = pd.read_csv("covid-19_tweets.csv",encoding="utf-8")

print(tweets_data.head())  # Number of

# Define a function to clean tweets
def clean_tweet(tweet):
    # Remove @ mentions
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

# Apply the cleaning function to the 'tweets_text' column
tweets_data['tweet_text'] = tweets_data['tweet_text'].apply(clean_tweet)

# Save the cleaned data
tweets_data.to_csv('cleaned_tweets.csv', index=False)