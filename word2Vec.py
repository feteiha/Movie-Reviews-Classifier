'''
authors: Hatem Mamdoh, Hussien Ashraf
ID: 20170085, 20170093
Group: CS_DS_4
'''
from sklearn.datasets import load_files
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np

def calc_accuracy(first, second):
    counter = 0
    for i in range(len(first)):
        if first[i] == second[i]:
            counter += 1
    return counter / len(first)


folder = load_files(r"./txt_sentoken", shuffle=False)
reviews, labels = folder.data, folder.target

tokenized_reviews = []
classifier = None
x_train = []
x_test = []
y_train = []
y_test = []

stop_words = set(stopwords.words('english'))

for review in reviews:
    review = review.decode('utf-8')
    result = nltk.word_tokenize(review)
    filtered_sentence = [w for w in result if not w.lower() in stop_words]
    tokenized_reviews.append(filtered_sentence)

#model = Word2Vec(tokenized_reviews, window=8, min_count=1, size = vector_length)
model = Word2Vec()

def vectorize_review(review):
    global model
    review_vector = vector_length*[0]
    count = 0
    for word in review:
       review_vector += model.wv[word]
       count += 1
    review_vector /= count
    return review_vector


vectorized_reviews = []

def vectorize_all(vector_length):
    global model, vectorized_reviews
    vectorized_reviews = []
    model = Word2Vec(tokenized_reviews, window=8, min_count=1, size = vector_length, sg=1)

    for review in tokenized_reviews:
        vectorized = vectorize_review(review)
        vectorized_reviews.append(vectorized)

def train_test_split_function():
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(vectorized_reviews, labels, test_size=0.2, shuffle=True, random_state=3)
    print('training set length:', len(y_train))
    print('+ve samples:', np.count_nonzero(y_train))
    print('percentage of +ve to dataset:', np.count_nonzero(y_train) / len(y_train) * 100)
    print(' ')


def trainSVM():
    global classifier, x_train, y_train, x_test, y_test

    print('started SVM training...')
    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)
    print('finished SVM training')

    y_pred = classifier.predict(x_test)

    print('SVM test accuracy:', calc_accuracy(y_test, y_pred))
    print('')

    
def trainNaiveBayes():
    global classifier, x_train, y_train, x_test, y_test

    print('started Naive Bayes training...')
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print('finished Naive Bayes training')

    y_pred = classifier.predict(x_test)

    print('Naive Bayes test accuracy:', calc_accuracy(y_test, y_pred))
    print('')
    
def trainLogisticRegression():
    global classifier, x_train, y_train, x_test, y_test
    
    print('started Logistic Regression training...')
    classifier = LogisticRegression(max_iter= 1000)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print('finished Logistic Regression training')

    y_pred = classifier.predict(x_test)

    print('Logistic Regression test accuracy:', calc_accuracy(y_test, y_pred))
    print('')

'''
for i in range (10, 200, 10):
    vector_length = i
    print("---------------------------------------")
    print("Vector Length = ", vector_length)
    vectorize_all(vector_length)
    train_test_split_function()
    trainSVM()
    trainNaiveBayes()
    trainLogisticRegression()
'''
vector_length = 100
vectorize_all(vector_length)
train_test_split_function()
trainSVM()
trainNaiveBayes()
trainLogisticRegression()