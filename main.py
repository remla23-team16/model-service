from flask import Flask, request, jsonify
import re
import nltk
import pickle
import joblib
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

app = Flask(__name__)

classifier_path = os.environ['CLASSIFIER_PATH']
bow_path = os.environ['BOW_PATH']

@app.route('/<data>', methods=['GET'])
def fetch_model(data):
    X = pre_process([data])
    return jsonify(predict(X).tolist())


def transform(corpus):
    with open(os.path.join(bow_path, 'sentiment-model-1'), 'rb') as f:
        cv = pickle.load(f)
    return cv.transform(corpus).toarray()


def pre_process(data):
    nltk.download('stopwords')
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []

    for i in range(len(data)):
        review = re.sub('[^a-zA-Z]', ' ', data[i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return transform(corpus)


def predict(X):
    model = joblib.load(os.path.join(bow_path, 'classifier-model-1'))
    y = model.predict(X)
    return y


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
