# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 20:41:30 2020

@author: BILAL khan
"""

import numpy as np 
import pandas as pd 
import re  
import nltk  
nltk.download('stopwords')  
from nltk.corpus import stopwords  

tweets = pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")

tweets.head()

tweets.shape


import seaborn as sns
sns.countplot(x='airline_sentiment', data=tweets)
 
sns.countplot(x='airline', data=tweets)
 
sns.countplot(x='airline', hue="airline_sentiment", data=tweets)
 
 
X = tweets.iloc[:, 10].values  
y = tweets.iloc[:, 1].values
print(y) 
processed_tweets = []
 
for tweet in range(0, len(X)):  
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))
 
    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
 
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
 
    # Substituting multiple spaces with single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
 
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
 
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
 
    processed_tweets.append(processed_tweet)
    
    
 
from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
x = tfidfconverter.fit_transform(processed_tweets).toarray()
 
print(len(processed_tweets))

#print(processed_tweets[3])
#print(x[0])
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
 



from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
text_classifier.fit(X_train, y_train)
 
 
predictions = text_classifier.predict(X_test)
 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))




import nltk 
word_features = []

def buildVocabulary(processed_tweets):
    all_words = []
    
    for words in processed_tweets:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features

def buildVocabulary2(preprocessedTrainingData):
    all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features



def extract_features(tweet):
    """all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()"""
    
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features 

# Now we can extract the features and train the classifier 
"""
tup = [('bilal is average student','neutral'), ('hamza is a bad boy', 'negative')]
word_features = buildVocabulary(processed_tweets)
print(word_features)
trainingFeatures=nltk.classify.apply_features(extract_features,processed_tweets)
"""
processed_tweets2 = []
print(len(y))
index = 0
for sentence in X:
    processed_tweets2.append((sentence, y[index]))
    index = index + 1

print(len(processed_tweets2))     
    
word_features2 = buildVocabulary2(processed_tweets2)
#print(word_features2)
trainingFeatures=nltk.classify.apply_features(extract_features,processed_tweets2)

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
                                                 
NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in processed_tweets2]

# ------------------------------------------------------------------------

# get the majority vote
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
else: 
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
                                                 



















"""
class_choice = ['positive', 'negative', 'neutral']
classification = []
probability = (5, 6, 6)

print(probability)
classification.append(class_choice[np.argmax(probability)])
print(classification)
"""