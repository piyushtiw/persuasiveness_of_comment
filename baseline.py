import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re, string, unicodedata, math
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import contractions
from typing import List, Union
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
stop = stopwords.words("english")
stemmer = LancasterStemmer()
encoding="utf-8"
lemmatizer = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer()

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class LoadHeadgeWords():
    def __init__(self):
        self.hedge_word_set = None

    def hedge_words(self):
        words = []
        with open('data/hedge_words.txt', 'r') as f:
            for word in f.readlines():
                words.append(word.rstrip('\n').lower())
        self.hedge_word_set = set(words)

    def hedge_score_with_sentence(self, sentence):
        sentence_set = set(sentence)
        if len(self.hedge_word_set.intersection(sentence_set)) > 0:
            return len(self.hedge_word_set.intersection(sentence_set))
        else:
            return 0

class LoadData():
    def __init__(self):
        pass

    def load_data_file(self, file):
        self.data = pd.read_csv(file)
        self.data['text'] = self.data.text.astype(str)
        self.data = self.data.drop(self.data[self.data['text'] == 'nan'].index)
        self.data['label'] = self.data.apply(
            lambda row: 1 if row['author_score'] and row['author_score'] > 5 else 0, axis=1)
        self.data['original'] = self.data.apply(lambda row: row['text'], axis=1)
        self.data.set_index("comment_id", inplace=True)

    def test_train_data(self):
        msk = np.random.rand(len(self.data)) < 1.0
        train = self.data[msk]
        test = self.data[~msk]
        return train, test

class NormaliseData():
    def __init__(self, data):
        self.data = data

    def remove_between_square_brackets(self, text):
        text = re.sub('\[[^]]*\]', '', text)
        #text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text, flags=re.MULTILINE)
        return text


    def preprocess(self, column):
        self.data[column] = self.data[column].apply(
            lambda x: [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in x])
        self.data[column] = self.data[column].apply(lambda x: [word.lower() for word in x])
        self.data[column] = self.data[column].apply(lambda x: [i for i in x if not i.isdigit()])
        self.data[column] = self.data[column].apply(lambda x: [item for item in x if item not in string.punctuation])
        self.data[column] = self.data[column].apply(lambda x: [item for item in x if item not in stop])
        #self.data[column] = self.data[column].apply(lambda x: [stemmer.stem(y) for y in x])
        self.data[column] = self.data[column].apply(lambda x: [lemmatizer.lemmatize(word, pos='v') for word in x])

    def tokenize(self):
        self.data['text'] = self.data.apply(lambda row: self.remove_between_square_brackets(row['text']), axis=1)
        self.data['text'] = self.data.apply(lambda row: contractions.fix(row['text']), axis=1)
        self.data['text'] = self.data.apply(lambda row: word_tokenize(row['text']), axis=1)
        self.preprocess('text')
        return self.data


class AddFeatures():
    def __init__(self, data, hedgeWordObj):
        self.data = data
        self.hedgeWordObj = hedgeWordObj

    def similarity_between_cc_op(self, cc, replyTo):
        try:
            op = self.data.loc[replyTo, 'text']
            s=" "
            tfidf_matrix = tfidf_vectorizer.fit_transform((s.join(op), s.join(cc)))
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0,0]
        except:
            return 0.0

    def sentiment_analyzer_scores(self, sentence):
        s = " "
        score = analyser.polarity_scores(s.join(sentence))
        return score['pos']

    def hedge_word_count(self, sentence):
        return self.hedgeWordObj.hedge_score_with_sentence(sentence)

    def baseline_features(self):
        self.data['length'] = self.data.apply(lambda row: len(row['text']), axis=1)
        self.data['similarity'] = self.data.apply(
           lambda row: self.similarity_between_cc_op(row['text'], row['reply_to']), axis=1)
        self.data['sentiment_score'] = self.data.apply(
           lambda row: self.sentiment_analyzer_scores(row['text']), axis=1)
        self.data['hedge_score'] = self.data.apply(
            lambda row: self.hedge_word_count(row['text']), axis=1)
        return self.data

class AdditionalFeatures():
    def __init__(self, data):
        self.data = data

    def get_url_count(self, str):
        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] | [! * \(\),] | (?: %[0-9a-fA-F][0-9a-fA-F]))+', str)
        return len(url)

    def add_features(self):
        self.data['users_score'] = self.data.apply(lambda row: row['author_score'], axis=1)
        self.data['number_of_urls'] = self.data.apply(lambda row: self.get_url_count(row['original']), axis=1)
        return self.data

class SvmClassification():
    def __init__(self):
        self.clf = None

    def train(self, features, processed_data):
        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(wor_to_vec, processed_data['Label'])
        self.clf = clf

    def predict(self, event):
        return self.clf.predict(event)


if __name__ == '__main__':
    obj = LoadData()
    file = 'data/P2_Training_Dataset.csv'
    obj.load_data_file(file)
    data, _ = obj.test_train_data()

    obj1 = NormaliseData(data)
    data = obj1.tokenize()

    hedgeWordObj = LoadHeadgeWords()
    hedgeWordObj.hedge_words()

    obj2 = AddFeatures(data, hedgeWordObj)
    data = obj2.baseline_features()

    baseline_features = data[['length', 'similarity', 'sentiment_score', 'hedge_score']]
    np.savetxt('baseline_features.csv', baseline_features, delimiter=',')

    obj3 = AdditionalFeatures(data)
    data = obj3.add_features()

    baseline_and_additional_features = data[['length', 'similarity', 'sentiment_score', 'hedge_score', 'users_score', 'number_of_urls']]
    np.savetxt('baseline_and_additional_features.csv', baseline_and_additional_features, delimiter=',')
    additional_features = data[['users_score', 'number_of_urls']]
    np.savetxt('additional_features.csv', additional_features, delimiter=',')

    print("Features extraction completed")
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(baseline_and_additional_features, data['label'])
    print("Model Trained")
    obj = LoadData()
    file = 'data/P2_Testing_Dataset.csv'
    obj.load_data_file(file)
    test_data, _ = obj.test_train_data()

    obj1 = NormaliseData(test_data)
    test_data = obj1.tokenize()

    obj2 = AddFeatures(test_data, hedgeWordObj)
    test_data = obj2.baseline_features()

    obj3 = AdditionalFeatures(test_data)
    test_data = obj3.add_features()

    baseline_and_additional_features_for_test = test_data[
        ['length', 'similarity', 'sentiment_score', 'hedge_score', 'users_score', 'number_of_urls']]

    print("Prediction started")
    predictions = text_classifier.predict(baseline_and_additional_features_for_test)
    np.savetxt('prediction.csv', predictions, delimiter=',')
    print(confusion_matrix(test_data['label'], predictions))
    print(classification_report(test_data['label'], predictions))
    print(accuracy_score(test_data['label'], predictions))