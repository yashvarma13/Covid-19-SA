import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download necessary NLTK data
nltk.download('words')
nltk.download('stopwords')

tweets_data = pd.read_csv("covid-19_tweets.csv", encoding="utf-8")

print(tweets_data.head())

# Define a function to clean tweets
import string
import re

def clean_tweet(tweet):
    # Remove @ mentions, URLs, and extra spaces
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    # Additional cleaning operations:
    # 1. Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))

    # 2. Convert to lowercase
    tweet = tweet.lower()

    # 3. Remove numbers
    tweet = re.sub(r'\d+', '', tweet)

    # 4. Remove hashtags (optional, depending on your analysis)
    tweet = re.sub(r'#', '', tweet)

    return tweet

# Apply the cleaning function to the 'tweet_text' column
tweets_data['tweet_text'] = tweets_data['tweet_text'].apply(clean_tweet)

# Save the cleaned data
tweets_data.to_csv('cleaned_tweets.csv', index=False)

df = pd.read_csv("cleaned_tweets.csv", encoding="iso-8859-1")

print(df.info())

df_neutral=df[df['label']==2]
df_positive=df[df['label']==3]
df_negative=df[df['label']==1]
df1=df[df['label']!=2]

sns.countplot(x='label', data=df1)
plt.title('Distribution of Sentiment Labels')
plt.show()

# Drop rows with neutral sentiment as we want to predict for 1 (negative) and 3 (positive)
df = df[df['label'].isin([1, 3])]

vect = CountVectorizer(lowercase=True, stop_words="english")
x = df.tweet_text
y = df.label
x = vect.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted\n")
print(results)

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

