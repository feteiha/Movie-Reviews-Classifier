# -*- coding: utf-8 -*-
"""
Created on Mon May 24 01:20:20 2021

@author: Hussien Ashraf, ID:20170093
@author: Hatem Mamdoh, ID:20170085
"""
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from sklearn import svm

classifier = svm.SVC()
loaded_vec = TfidfVectorizer()


def calc_accuracy(first, second):
    counter = 0
    for i in range(len(first)):
        if first[i] == second[i]:
            counter += 1
    return counter / len(first)


def train():
    global classifier
    folder = load_files(r"./txt_sentoken", shuffle=False)
    reviews, labels = folder.data, folder.target

    # 0 for negative
    # 1 for positive

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    x = vectorizer.fit_transform(reviews)
    vocab = vectorizer.vocabulary_

    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=3)

    print('training set length:', len(y_train))
    print('+ve samples:', np.count_nonzero(y_train))
    print('percentage of +ve to dataset:', np.count_nonzero(y_train) / len(y_train) * 100)

    print('started training...')
    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)
    print('finished training')

    y_pred = classifier.predict(x_test)

    print('test accuracy:', calc_accuracy(y_test, y_pred))
    print('')

    with open('classifier.txt', 'wb') as file:
        pickle.dump(classifier, file)

    with open('vocab.txt', 'wb') as file:
        pickle.dump(vocab, file)
    load()


def load():
    global classifier
    global loaded_vec

    with open('vocab.txt', 'rb') as file:
        vocab = pickle.load(file)

    with open('classifier.txt', 'rb') as file:
        classifier = pickle.load(file)

    loaded_vec = TfidfVectorizer(vocabulary=vocab)


def predict(review):
    global loaded_vec
    global classifier
    to_predict = loaded_vec.fit_transform([review])
    result = classifier.predict(to_predict)
    if result == 0:
        print('This is a negative review.')
    else:
        print('This is a positive review')


print('1)Train')
print('2)Load')
choice = input('choice: ')
if choice == '1':
    train()
elif choice == '2':
    load()
else:
    exit(0)

while True:
    review = input("Write your review: ")
    if review == '0':
        break
    predict(review)