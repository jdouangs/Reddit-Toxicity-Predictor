# Import Dependencies
import json
import nltk
import praw
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Import text blobs
from textblob import TextBlob, Word

# Loading pickle vects
toxic_vect = pickle.load(open('pickle/vect.sav', 'rb'))
severetoxic_vect = pickle.load(open('pickle/vect2.sav', 'rb'))
obscene_vect = pickle.load(open('pickle/vect3.sav', 'rb'))
threat_vect = pickle.load(open('pickle/vect4.sav', 'rb'))
insult_vect = pickle.load(open('pickle/vect5.sav', 'rb'))
identityhate_vect = pickle.load(open('pickle/vect6.sav', 'rb'))

# Loading pickle savs
toxic_model = pickle.load(open('pickle/toxic.sav', 'rb'))
severetoxic_model = pickle.load(open('pickle/severetoxic.sav', 'rb'))
obscene_model = pickle.load(open('pickle/obscene.sav', 'rb'))
threat_model = pickle.load(open('pickle/threat.sav', 'rb'))
insult_model = pickle.load(open('pickle/insult.sav', 'rb'))
identityhate_model = pickle.load(open('pickle/identityhate.sav', 'rb'))

# Reddit API (PRAW) link: https://praw.readthedocs.io/en/latest/getting_started/quick_start.html
reddit = praw.Reddit(client_id='WGIz87QjfTIqPA',
                     client_secret='Im_VLbyINey3d9WfwCltAzD4vqw',
                     user_agent='Python Comments for Bootcamp Class v. 1.0 (by /u/jdouangs)')

# Function to detect sentiment of reddit comments
def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Master scrape function for reddit comments into a pandas dataframe
def scrape(choice):
    df = pd.DataFrame(columns=['Date', 'Username', 'Score', 'Comment', 'Submission Title', 'Link'])
    subreddit = reddit.subreddit(choice.lower())
    i = 1
    for submission in subreddit.hot(limit=5):
        if "megathread".lower() not in submission.title.lower():
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                if comment.author is not None:
                    Username = comment.author.name
                    Score = int(comment.score)
                    Date = str(datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'))
                    CommentText = comment.body.replace(',', '').replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u201c", "'").replace(u"\u201d", "'")
                    SubmissionTitle = submission.title.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u201c", "'").replace(u"\u201d", "'")
                    Link = submission.url
                    df.loc[i] = [Date, Username, Score, CommentText, SubmissionTitle, Link]
                    i += 1
    toxic = []
    severe_toxic = []
    obscene = []
    threat = []
    insult = []
    identity_hate = []
    sentiment = []

    toxic_predict_prob = []
    severe_toxic_predict_prob = []
    obscene_predict_prob = []
    threat_predict_prob = []
    insult_predict_prob = []
    identity_hate_predict_prob = []

    for index, row in df.iterrows():
        toxic.append(toxic_model.predict(toxic_vect.transform(pd.Series(row['Comment'])))[0])
        toxic_predict_prob.append(round(np.max(toxic_model.predict_proba(toxic_vect.transform(pd.Series(row['Comment'])))[0]),4))
        
        severe_toxic.append(severetoxic_model.predict(severetoxic_vect.transform(pd.Series(row['Comment'])))[0])
        severe_toxic_predict_prob.append(round(np.max(severetoxic_model.predict_proba(severetoxic_vect.transform(pd.Series(row['Comment'])))[0]),4))
        
        obscene.append(obscene_model.predict(obscene_vect.transform(pd.Series(row['Comment'])))[0])
        obscene_predict_prob.append(round(np.max(obscene_model.predict_proba(obscene_vect.transform(pd.Series(row['Comment'])))[0]),4))
        
        threat.append(threat_model.predict(threat_vect.transform(pd.Series(row['Comment'])))[0])
        threat_predict_prob.append(round(np.max(threat_model.predict_proba(threat_vect.transform(pd.Series(row['Comment'])))[0]),4))
        
        insult.append(insult_model.predict(insult_vect.transform(pd.Series(row['Comment'])))[0])
        insult_predict_prob.append(round(np.max(insult_model.predict_proba(insult_vect.transform(pd.Series(row['Comment'])))[0]),4))
        
        identity_hate.append(identityhate_model.predict(identityhate_vect.transform(pd.Series(row['Comment'])))[0])
        identity_hate_predict_prob.append(round(np.max(identityhate_model.predict_proba(identityhate_vect.transform(pd.Series(row['Comment'])))[0]),4))
        
        sentiment.append(round(detect_sentiment(row['Comment']),3))
        
    df['Toxic Prediction'] = toxic
    df["Toxic Prediction Probability"] = toxic_predict_prob
    df['Severe Toxic Prediction'] = severe_toxic
    df["Severe Toxic Prediction Probability"] = severe_toxic_predict_prob
    df['Obscene Prediction'] = obscene
    df["Obscene Prediction Probability"] = obscene_predict_prob
    df['Threat Prediction'] = threat
    df["Threat Prediction Probability"] = threat_predict_prob
    df['Insult Prediction'] = insult
    df["Insult Prediction Probability"] = insult_predict_prob
    df['Identity Hate Prediction'] = identity_hate
    df["Identity Hate Prediction Probability"] = identity_hate_predict_prob
    df['Sentiment'] = sentiment

    toxic_data = df[['Date', 'Username', 'Score', 'Comment', 'Submission Title', 'Link', 'Toxic Prediction', 'Toxic Prediction Probability', 'Sentiment']]
    toxic_only = toxic_data.loc[toxic_data['Toxic Prediction'] == 1]
    toxic_only = toxic_only.sort_values(by="Toxic Prediction Probability", ascending=False)

    severe_data = df[['Date', 'Username', 'Score', 'Comment', 'Submission Title', 'Link', 'Severe Toxic Prediction', 'Severe Toxic Prediction Probability', 'Sentiment']]
    severe_only = severe_data.loc[severe_data['Severe Toxic Prediction'] == 1]
    severe_only = severe_only.sort_values(by="Severe Toxic Prediction Probability", ascending=False)

    obscene_data = df[['Date', 'Username', 'Score', 'Comment', 'Submission Title', 'Link', 'Obscene Prediction', 'Obscene Prediction Probability', 'Sentiment']]
    obscene_only = obscene_data.loc[obscene_data['Obscene Prediction'] == 1]
    obscene_only = obscene_only.sort_values(by="Obscene Prediction Probability", ascending=False)

    threat_data = df[['Date', 'Username', 'Score', 'Comment', 'Submission Title', 'Link', 'Threat Prediction', 'Threat Prediction Probability', 'Sentiment']]
    threat_only = threat_data.loc[threat_data['Threat Prediction'] == 1]
    threat_only = threat_only.sort_values(by="Threat Prediction Probability", ascending=False)

    insult_data = df[['Date', 'Username', 'Score', 'Comment', 'Submission Title', 'Link', 'Insult Prediction', 'Insult Prediction Probability', 'Sentiment']]
    insult_only = insult_data.loc[insult_data['Insult Prediction'] == 1]
    insult_only = insult_only.sort_values(by="Insult Prediction Probability", ascending=False)

    identityhate_data = df[['Date', 'Username', 'Score', 'Comment', 'Submission Title', 'Link', 'Identity Hate Prediction', 'Identity Hate Prediction Probability', 'Sentiment']]
    identityhate_only = identityhate_data.loc[identityhate_data['Identity Hate Prediction'] == 1]
    identityhate_only = identityhate_only.sort_values(by="Identity Hate Prediction Probability", ascending=False)

    df_array = [
        df.to_dict('records'),
        toxic_only.to_dict('records'),
        severe_only.to_dict('records'),
        obscene_only.to_dict('records'),
        threat_only.to_dict('records'),
        insult_only.to_dict('records'),
        identityhate_only.to_dict('records')
    ]
    return df_array